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

The fingerprint records the host, operating-system user, PID, process start time, the
requested command, the settled command reported by `ps`, and the attempt path (the
process working directory). The launcher samples the live identity for up to two seconds
and records an observed identity only after consecutive equal samples span at least one
second. Temporary `ps` or working-directory-reader failures are retried within that
budget. Keep the fingerprint beside the attempt outputs and copy it along with them.

To stop the run, execute the helper on the same remote host:

```bash
python scripts/procctl.py stop /absolute/path/to/attempt/process.json
```

Before sending `SIGTERM`, the helper re-reads the live process. PID, start time, and user
are the primary identity core; host and working directory must also match. The live
command may match either the requested command or the settled observed command. Any
other mismatch, missing process, malformed file, or inability to inspect the working
directory causes a refusal and no signal. The helper never accepts a name or pattern in
place of a fingerprint. A successful stop waits up to ten seconds for exit; it does not
escalate to `SIGKILL` automatically.

The two-second observation window is the contract boundary. A process that changes its
command with `exec` after that window may be refused later. Recovery is human
verification followed by a manual kill of the verified PID; there is deliberately no
procctl force flag.

Accepting either command has a deliberate discrimination residual. Compared with
accepting only the earlier wrapper command, the settled generic form can admit an
unrelated process that has a reused PID, the same second-granularity start time, the same
user, command, and working directory. Start time remains the primary PID-reuse guard,
but its platform-provided granularity cannot eliminate that case.
