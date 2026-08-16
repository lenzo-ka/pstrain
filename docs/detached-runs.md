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

The fingerprint records the host, operating-system user, PID, process-group ID, process
start time, the requested command, the settled command reported by `ps`, and the attempt
path (the process working directory). The process-group ID equals the PID because the
launcher creates a fresh session led by the launched process. The launcher samples the
live identity for up to two seconds and records an observed identity only after
consecutive equal samples span at least one second. Temporary `ps` or
working-directory-reader failures are retried within that budget. Keep the fingerprint
beside the attempt outputs and copy it along with them.

To stop the run, execute the helper on the same remote host:

```bash
python scripts/procctl.py stop /absolute/path/to/attempt/process.json
```

Before signaling, the helper re-reads the live process. Host, PID, start time, user, and
working directory form the strong identity and must all match. The live command must
also match the recorded settled wrapper command, the requested command, or the expected
post-exec target derived from the requested command's argv tail. This permits a wrapper
to exec its launched target after the settlement window, but refuses an unrelated
command even when every strong-identity field matches. A mismatch, a missing process, a
malformed file, or an inability to inspect the working directory causes a refusal
without signaling. The helper never accepts a name, command, or pattern in place of a
fingerprint.

Stop sends `SIGTERM` to the verified process group and waits up to the configured
timeout. If any group member remains, it sends `SIGKILL` to the group and waits once
more. Launch-failure cleanup uses the same group-wide TERM-to-KILL escalation. Because
launch creates a fresh session, these signals cover the wrapper and its children without
reaching unrelated process groups.

Start time remains the primary PID-reuse guard, and command validation protects the
same-second reuse case when the replacement command is unrelated. No command match can
overcome a strong-identity mismatch.
