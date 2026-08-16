# Procctl framework-Python main fix

## Outcome

`procctl launch` records both the requested command and a settled observed live command.
The original implementation on `main` already recorded one live observation, but that
single immediate sample could capture a pre-exec wrapper. The fix samples for up to two
seconds and requires consecutive equal observations spanning at least one second;
transient reader failures are retried inside that budget. `procctl stop` retains PID,
start time, and user as its primary identity core and accepts a live command matching
either recorded command.

If fingerprint creation fails, launch cleanup now sends SIGTERM, waits for a bounded
period, escalates to SIGKILL if necessary, confirms exit, and reports the action. A
failure to confirm exit is reported as such.

## Verification

- `python scripts/run_verified_tests.py tests/test_procctl.py`: 10 passed.
- `make verified`: passed on macOS; 686 tests passed, 1 skipped, and 1 deselected
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
