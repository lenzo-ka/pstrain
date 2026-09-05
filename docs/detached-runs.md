# Detached runs

Use `scripts/procctl.py` for long-running commands that must survive the shell or SSH
connection that launched them. It creates one JSON fingerprint per attempt; it is not a
process registry.

On the remote host, run:

```bash
python scripts/procctl.py launch \
  --attempt /absolute/path/to/attempt \
  --fingerprint /absolute/path/to/attempt/process.json \
  --log /absolute/path/to/attempt/process.log \
  -- command arg1 arg2
```

The fingerprint records the host, operating-system user, PID, process-group ID, the
finest process start time available from the operating system, the requested command,
the settled command reported by `ps`, and the attempt path (the process working
directory). On Linux the start time is `/proc/<pid>/stat` field 22, expressed in clock
ticks since boot. On macOS it is `kinfo_proc.p_starttime`, obtained with
`KERN_PROC_PID` and expressed as seconds plus microseconds. The process-group ID equals
the PID because the launcher creates a fresh session led by the launched process. The
launcher records an observed identity only after consecutive equal samples span at least
one second. A sample that differs from the one before it opens a new confirmation
window, which is what a wrapper that replaces itself with `exec` needs; three such
changes are tolerated. Counting windows rather than wall-clock seconds is what lets a
machine loaded enough to make `ps` or the working-directory reader take longer than a
window settle later instead of refusing. A reader failure interrupts the window and the
wait restarts, but it is not counted as a change, because failing to read an identity is
not evidence that it differs; those failures are retried for up to two seconds after the
last reading that succeeded, except that a second attempt is always made, so one slow
failing reader cannot decide a launch by itself. Because a first attempt may consume up
to the reader cap, that exception can carry the retry past the two seconds.

Identity settling is bounded: it produces a settled identity or refuses within sixty
seconds, plus whatever it costs to reap the last reader subprocess. No reader is started
once that deadline has passed, each is capped at thirty seconds and clamped again to the
time left, and a reading that finishes late is refused rather than accepted, so no
identity is confirmed on evidence gathered after the deadline. The cap is a hang
detector, not a latency threshold: `lsof` was measured taking almost seven seconds on a
loaded twelve-core machine at `nice` 19, and a cap near that figure would refuse healthy
launches. Earlier releases documented a two-second bound the launcher could not honor,
because the reads it was waiting on were themselves spending the budget; sixty seconds
is the bound settling now enforces. A refusal reports the last identity read, the last
reader error, how long settling took, and how many distinct identities, interrupted
windows, and reader failures lay behind them.

Settling is not the whole of a failed launch. When it refuses, the launcher still has a
detached child to dispose of, and the group-wide `SIGTERM` to `SIGKILL` escalation waits
up to two seconds at each of four points: for the leader after `SIGTERM`, for the rest
of its group, then for both again after `SIGKILL`. That is at most eight seconds of
cleanup on top of the settling bound, so a refused launch returns within about
sixty-eight seconds. A launch that succeeds pays none of it. Keep the fingerprint beside
the attempt outputs and copy it along with them.

To stop the run, execute the helper on the same remote host:

```bash
python scripts/procctl.py stop /absolute/path/to/attempt/process.json
```

Before signaling, the helper re-reads the live process. Host, PID, high-precision start
time, user, and working directory form the identity and must all match exactly. The
recorded and live commands are diagnostic only: an arbitrary wrapper may change its
command with `exec`, so command text is never part of the kill decision. A mismatch, a
missing process, a malformed file, or an inability to inspect the working directory or
start time causes a refusal without signaling. The helper never accepts a name,
command, or pattern in place of a fingerprint.

Stop sends `SIGTERM` to the verified process group and waits up to the configured
timeout. If any group member remains, it sends `SIGKILL` to the group and waits once
more. Launch-failure cleanup uses the same group-wide TERM-to-KILL escalation. Because
launch creates a fresh session, these signals cover the wrapper and its children without
reaching unrelated process groups.

On an unsupported platform, procctl falls back to the second-granularity start time
reported by `ps`. That fallback has an explicit residual: a PID reused within the same
wall-clock second with the same host, user, and working directory cannot be
distinguished safely. Command matching is not reintroduced to cover that case.
