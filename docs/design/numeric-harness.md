# Numerical-correctness harness

## Purpose

The normal recognition tests answer whether a model is useful. They do not
identify the training stage at which a small numerical change first appeared.
This harness pins the pre-PP4 behavior at the five BASIS choke points so a
numerical change is reviewed at its source instead of being absorbed by WER.

The program lives in `tests/test_numeric_harness.py`. It uses three utterances
for the CI trajectory and one shared ten-utterance `mini_arctic` build for the
full 1→2→4→8 path. The complete module is expected to take less than five
seconds on a development machine and remains in the normal test suite.

## Golden trajectory

`tests/golden/numeric_bw.json` records three fixed CI Baum-Welch passes. Each
pass contains total log-likelihood, per-frame log probability and pass-to-pass
per-frame delta, frames, input
utterances, successful utterances, retried utterances, and skipped utterances.
The same file anchors one extracted feature file by frame count and SHA-256.

The floating values use `rtol=1e-12` and `atol=1e-8`. This is intentionally a
tight same-machine tolerance, not a cross-platform promise. A toolchain,
architecture, or engine change may require a reviewed golden update.

Regenerate from the repository root with:

```console
python scripts/regenerate_numeric_golden.py
```

The script creates a fresh project, records seed 42, builds the fixed flat
model, runs exactly three passes, and replaces the JSON. Review the numerical
diff; do not regenerate merely to make an unexplained failure green.

## Five choke points

1. **Feature extraction.** Every mini fixture produces a non-empty 13-wide,
   finite MFC array. `arctic_a0001` has an exact frame count and byte checksum.
   Front-end dithering is disabled in this exercised path, so it has no hidden
   stochastic state.
2. **Aggregation.** The same three utterances are accumulated separately and
   together. Utterance count, frame count, and total log-likelihood must
   conserve exactly (within floating addition tolerance).
3. **BW and recombination.** First-class `TrainingIteration` telemetry checks
   `processed + skipped == input` on every pass and records retry counts. The
   clean fixture requires zero skips. A retry remains a second attempt for the
   same input, not another accounted utterance.
4. **Per-pass update.** Means, variances, mixture weights, and transition
   matrices remain finite. Mixture-weight and populated transition rows sum to
   one, variances remain at or above the native `1e-4` floor, and every density
   has positive reported occupancy.
5. **Splitting and propagation.** The shared full build checks the exact
   1→2→4→8 schedule after training each stage. Senone counts remain fixed,
   density/count dimensions match the scheduled value, occupancies are
   present, parameters are finite, and normalization remains intact.

## Acceptance debts

The multipron test supplies `author K` as variant 1 and the acoustically
matching `author(2) AO TH ER` as variant 2 for the real `arctic_a0001` audio.
With multipron disabled the AO CI states have exactly zero occupancy; with it
enabled they receive positive occupancy. This is a stable, phone-specific
proof that posterior summation changes the trained sufficient statistics.

The tied-state check does not call the native tie implementation. It parses
the committed question syntax and pruned trees, reproduces preorder leaf
labels, independently walks a sample of at least 60 triphone states, and
compares the result with the tied mdef assignment.

## Reproducibility

The exercised corpus split records seed 42. Feature dithering is off and
Gaussian splitting is deterministic; there are no other stochastic operations
in these paths. Two independent one-pass BW runs must produce byte-identical
means, variances, mixture weights, transition matrices, and density counts.
The golden JSON also records the seed, providing the configuration seam for
future PP5 stochastic stages.
