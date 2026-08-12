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

The fingerprint records the host, operating-system user, PID, process start time, full
command reported by `ps`, and attempt path (the process working directory). The launcher
observes these values from the running child rather than reconstructing them from its
inputs. Keep the fingerprint beside the attempt outputs and copy it along with them.

To stop the run, execute the helper on the same remote host:

```bash
python scripts/procctl.py stop /absolute/path/to/attempt/process.json
```

Before sending `SIGTERM`, the helper re-reads every field from the live process. Any
mismatch, missing process, malformed file, or inability to inspect the working directory
causes a refusal and no signal. The helper never accepts a name or pattern in place of a
fingerprint. A successful stop waits up to ten seconds for exit; it does not escalate to
`SIGKILL` automatically.
