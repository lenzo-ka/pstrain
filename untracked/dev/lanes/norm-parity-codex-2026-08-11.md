# Norm parity lane report — 2026-08-11

## Invariants implemented

- BW construction requires an explicit unobserved-Gaussian policy.
- Pipeline stages select `zero`, matching upstream norm's fresh allocation.
- `retain` remains an intentional library choice for sparse-training safety.
- Normalized variance artifacts contain direct, unfloored results.
- Evaluation uses a separate copied representation with the `1e-4` floor and
  reciprocal precomputation; saving never round-trips through reciprocals.

## Discriminating coverage

The engineered constant-observation fixture creates both occupied and empty
codebooks. It proves exact zero artifacts under `zero`, prior preservation under
`retain`, identical occupied cells across policies, and an exact direct
one-pass variance result of zero (rather than an artifact-time floor).

## Golden governance

The checked-in numeric golden did not move. One invariant assertion changed:
saved variances are no longer required to be at least `1e-4`, because that
would enforce the removed artifact-time floor. The replacement explicitly
checks the prescribed load/evaluation floor transformation. No unrelated
numeric expectation changed.

## Gates

- Native CMake rebuild: passed.
- CTest: 2/2 passed.
- Full pytest after rebuild: 411 passed, 2 skipped, 1 expected failure.
- Ruff check and format: passed.
- Mypy with the repository's missing-import policy: passed.
- Pre-commit (all files): passed.
