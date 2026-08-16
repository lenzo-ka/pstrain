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
launcher samples the live identity for up to two seconds and records an observed
identity only after consecutive equal samples span at least one second. Temporary `ps`
or working-directory-reader failures are retried within that budget. Keep the
fingerprint beside the attempt outputs and copy it along with them.

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
