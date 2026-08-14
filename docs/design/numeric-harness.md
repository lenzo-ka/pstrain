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

`tests/golden/numeric_bw.json` records three fixed CI Baum-Welch passes. The
values are conditional on the build's declared floating-point contraction
policy (`off`). Like the rest of this golden, they certify self-conformance to
pstrain's output at regeneration time, not independent numerical correctness.
The first pass uses one-pass variance accumulation, matching upstream
`20.ci_hmm`; later passes use centered two-pass accumulation. Each pass
contains total log-likelihood, per-frame log probability and pass-to-pass
per-frame delta, frames, input
utterances, successful utterances, retried utterances, and skipped utterances.
The same file anchors one extracted feature file by frame/value count and a
portable numerical envelope (minimum, maximum, mean, standard deviation, and
L2 norm). It also records a same-machine SHA-256.

The always-on portable tier uses `rtol=1e-6, atol=1e-4` for trajectory totals
and `rtol=1e-4, atol=1e-3` for the feature envelope. These bounds allow normal
floating reassociation across compilers without accepting material engine
movement. At the first-pass total near -167,937, the allowed error is about
0.168 log units, so the approximately 100-log-unit E1 regression is roughly
595 times larger than the catch envelope.

Set `PSTRAIN_GOLDEN_STRICT=1` for the strict developer/regeneration tier. It
adds the exact feature SHA-256 and uses `rtol=1e-12, atol=1e-8` for trajectory
values. Strict failures may reflect architecture or toolchain differences;
portable failures require investigation.

The Ubuntu Python 3.11 CI lane and `make test` enable this strict tier. A tolerance wider than the
effect a golden is meant to pin does not pin that effect: the portable
trajectory tolerance is roughly three orders wider than the contraction
delta, so it remains a portability/regression envelope and is not evidence
that the contraction-specific values are unchanged.

## Floating-point contraction checks

`scripts/check_fp_contract.py build` disassembles the executable, static-library,
and shared-library artifacts in a CMake build's canonical `bin/` and `lib/`
directories. The Tests workflow applies that check to its own CMake build trees.
The Build workflow separately runs `scripts/check_fp_contract.py --wheels
wheelhouse` after cibuildwheel finishes. That mode extracts every wheel, finds
every ELF, Mach-O, and native archive by file signature, and checks each
reported architecture. A missing training executable or `libpstrainc`, an
unavailable disassembler, an object with no reported architecture, or a failed
disassembly is an explicit `unchecked` error rather than a silent skip.

`tests/test_fp_contract_gate.py::test_contraction_enabled_build_makes_gate_red`
is the re-runnable negative control. It compiles `a * b + c` with contraction
enabled, presents copies as the required build artifacts, and requires the
shipped gate to reject the fused instruction. This proves the detector's red
path on the test host; it does not reproduce the separate training trajectory
or benchmark measurements reported separately.

Regenerate from the repository root with:

```console
python scripts/regenerate_numeric_golden.py
```

The script creates a fresh project, records seed 42, builds the fixed flat
model, runs exactly three passes, and replaces the JSON. Review the numerical
diff. Regeneration is legitimate only after an understood, intentional
fixture/toolchain baseline change, never to paper over an engine change. The
regenerating commit must state why the baseline changed.

## Five choke points

1. **Feature extraction.** Every mini fixture produces a non-empty 13-wide,
   finite MFC array. `arctic_a0001` has portable count/envelope checks; strict
   mode additionally checks its exact bytes.
   Front-end dithering and DC removal use their canonical enabled defaults in
   this path. Two independent fresh-project regenerations must produce the same
   feature bytes before the checksum is updated.
2. **Aggregation.** The same three utterances are accumulated separately and
   together. Utterance count, frame count, and total log-likelihood must
   conserve exactly (within floating addition tolerance).
3. **BW and recombination.** First-class `TrainingIteration` telemetry checks
   `processed + skipped == input` on every pass and records retry counts. The
   clean fixture requires zero skips. A retry remains a second attempt for the
   same input, not another accounted utterance.
4. **Per-pass update.** Means, variances, mixture weights, and transition
   matrices remain finite after every retained BW pass, including CI passes.
   Mixture-weight and populated transition rows sum to one, variances remain at
   or above the native `1e-4` floor, and every senone/codebook density has
   positive reported occupancy.
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

## Per-pass utterance exclusions

`training.exclusion_schedule` is an experiment/parity instrument, not a
production corpus-filtering feature. It maps pipeline stage names to one-based
BW pass numbers (or `"*"`) and then to utterance IDs. For example:

```yaml
training:
  exclusion_schedule:
    ci-1g:
      5: [arctic_a0587]
      6: [arctic_a0587]
    cd-untied:
      "*": [arctic_a0587]
```

Matching utterances are removed only at the BW accumulation boundary and are
reported as `excluded_by_schedule`; decode evaluation continues to use the
normal test split. The active mapping is retained in training provenance so a
parity run can be reproduced exactly. Leave the knob absent for ordinary
training.

## Reproducibility

The exercised corpus split records seed 42. Feature dithering is repeatable
across independent fresh-project builds, Gaussian splitting is deterministic,
and there are no other stochastic operations in these paths. Two independent
one-pass BW runs must produce byte-identical means, variances, mixture weights,
transition matrices, and density counts. The golden JSON also records the seed,
providing the configuration seam for future PP5 stochastic stages.
