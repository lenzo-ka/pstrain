# Procctl framework-Python main fix

## Outcome

`procctl launch` records the finest process start time exposed by the operating system:
Linux `/proc` clock ticks or macOS `kinfo_proc.p_starttime` microseconds. `procctl stop`
requires that value, PID, user, host, and working directory to match exactly. Requested,
observed, and live commands are diagnostic only because an arbitrary wrapper's eventual
`exec` target cannot be predicted safely.

The fingerprint also records the fresh session's process-group ID. Normal stop and
launch cleanup send SIGTERM to the group, wait for a bounded period, escalate the group
to SIGKILL if necessary, confirm exit, and report the action. A failure to confirm exit
is reported as such.

## Verification

- `python scripts/run_verified_tests.py tests/test_procctl.py`: 14 passed.
- `PYTHONPATH=. make verified`: passed on macOS; 692 tests passed, 1 skipped, and 1 deselected
  in the main verified suite, followed by 41 configuration tests passing, CTest
  6/6 passing, floating-point contraction verification passing, generated-file
  checks passing, Ruff passing, mypy passing, and Ruff format checks passing.

## Workflow event and matrix finding

Pull requests run Python 3.11 and 3.13 on Ubuntu, but default-branch pushes run Python
3.11, 3.12, and 3.13 on both Ubuntu and macOS. The macOS C-test pull-request leg does
not exercise this Python-only path. Workflow triggers and matrices are unchanged.

## Round 2 — 2026-08-16

Settlement now means consecutive equal live-identity samples spanning at least one
second within the full two-second observation budget, rather than two adjacent samples
100 milliseconds apart. A deterministic mocked wrapper, target, target sequence proves
the distinction without scheduler timing. Reader hiccups are retryable during the
budget.

Fingerprints contain requested and settled-observed commands, and stop accepts either
only after the remaining identity fields match. Dead PID, start-time mismatch, and
genuine command mismatch constructions continue to refuse without signaling.

The availability gain has a deliberate residual: accepting a settled generic command
can admit an unrelated process with a reused PID, the same second-granularity start
time, the same user, command, and working directory. Start time is the primary reuse
guard, not a proof against that indistinguishable case. An exec after the two-second
window is outside the contract and requires human verification and a manual kill by
PID; procctl provides no force flag.

## Round 3 — 2026-08-16

Stop now treats host, PID, start time, user, and working directory as the strong process
identity. When those fields still match, live command drift is accepted as a legitimate
wrapper-to-target `exec`, even if it occurs after launch settlement. Dead processes and
every strong-identity mismatch still refuse without signaling; command text cannot
override those checks.

Launch records the fresh session's process-group ID, which must equal its leader PID.
Normal stop and failed-launch cleanup signal that complete group. Both paths send
SIGTERM, wait for a bounded period, escalate surviving group members to SIGKILL, and
wait again to confirm the group has exited. The process regression uses a TERM-ignoring
wrapper and child and requires that no group member survive stop; the late-exec
regression uses a deterministic mocked identity sequence.

## Round 4 — 2026-08-16

Command drift is accepted only when the live command matches the recorded settled
wrapper, the requested command, or the expected post-exec target derived from the
requested argv tail. The target is derived from the recorded launch request rather than
the settlement observation, so a wrapper that remains visible throughout settlement
and execs later can still be stopped safely.

Deterministic regressions cover both sides of the rule: a late exec to the launched
target is terminated, while an unrelated same-second PID-reuse imposter with matching
host, PID, start time, user, and working directory is refused without a signal. Dead
PIDs and strong-identity mismatches continue to refuse without signaling, and group
termination continues to reap a TERM-ignoring wrapper and child.

## Round 5 — 2026-08-16

The PID-reuse guard now uses the highest-precision process start time available. Linux
reads `/proc/<pid>/stat` field 22 as clock ticks since boot; macOS reads
`kinfo_proc.p_starttime` through `KERN_PROC_PID` as a seconds-and-microseconds pair. An
unsupported platform falls back to the second-granularity `ps` value with the residual
same-second reuse ambiguity documented explicitly.

Stop no longer predicts an arbitrary wrapper's post-`exec` command. Once host, PID,
user, working directory, and precise start time all match, it terminates the recorded
process group regardless of live command. Requested and observed commands remain only
for diagnostics. Deterministic regressions terminate both resolved-wrapper and Python
wrapper command transitions, refuse a same-second PID reuse with a different
subsecond start time without any signal, preserve zero-signal refusal for dead or
mismatched identities, and retain group cleanup of a TERM-ignoring wrapper and child.
