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
- Full pytest after rebuild: 414 passed, 2 skipped, 1 expected failure. The
  sandboxed run first blocked 12 process-control tests with `EPERM`; the full
  unsandboxed rerun passed. Goldens were unchanged.
- Ruff check and format: passed.
- Mypy with the repository's missing-import policy: passed.
- Pre-commit (all files): passed.

## Fugu fold-in

- Retain now reloads the serialized, unfloored variance representation before
  evaluation flooring. Empty cells containing both `5e-5` and `0` survive a
  retain pass exactly.
- Zero policy exempts graph-marked fallback senones. The multipron regression
  includes fallback branches with zero posterior mass, proves their Gaussian
  cells remain identical to the prior pass, and trains the utterance from every
  later checkpoint.
- The C boundary directly rejects null, INVALID, and out-of-range policies.
- Evaluation invariance is scoped to a given loaded float; save/reload behavior
  intentionally changes because the lossless value now reaches the next pass.
- Follow-up only: add a discriminating decode test for PocketSphinx handling of
  zero/unfloored exports versus upstream artifacts; parity is currently matched
  by upstream's zero-writing convention but is not directly tested.

## Fold-in #2

- Fallback senone marking now runs only for multipron graph updates; successful
  text and MFCC updates in linear mode leave the marking set empty.
- The zero-occupancy fixture expands its copied flat model to two densities and
  runs linear training with `topn=1`. It asserts no fallback senones are marked
  and that zero-policy output zeros every zero-posterior Gaussian cell exactly.
- Native rebuild, full pytest, CTest, Ruff check/format, mypy with the repository
  missing-import policy, and pre-commit all passed. The numeric golden was
  unchanged.
