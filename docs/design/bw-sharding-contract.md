# Baum-Welch sharding contract

Non-multipron Baum-Welch training partitions the canonical manifest into contiguous ranges and
reduces shard artifacts in ascending shard ID through the vendored accumulator reducers.

<!-- BEGIN GENERATED GATE SCOPE -->
## Generated gate scope

This section is generated from `@contract_scope` declarations on the named test gates.

1. At exactly `2` shards and 3 passes, `tests/test_numeric_harness.py::test_seeded_bw_shards_are_reproducible_and_discrete_state_is_partition_independent` repeats the same seeded manifest twice. It compares produced model files `means`, `variances`, `mixture_weights`, and `transition_matrices`, the copied input `mdef`, and per-shard files `artifact.json`, `gauden_counts`, `mixw_counts`, and `tmat_counts` byte-for-byte. The scope is the architecture and operating system executing the test; it does not compare across architectures or operating systems.

2. Across shard counts `1` and `2`, `tests/test_numeric_harness.py::test_seeded_bw_shards_are_reproducible_and_discrete_state_is_partition_independent` compares `assigned_ids`, `processed_ids`, `retried_ids`, `skipped`, `total_frames`, and `stop_decision` for exactly 3 passes on the same seeded manifest. It makes no cross-count floating-parameter comparison.

3. At exactly `1` shard and 3 passes, `tests/test_numeric_harness.py::test_one_shard_reducer_matches_established_in_process_bw` compares the reducer path with the in-process reference. It compares files `gauden_counts`, `mdef`, `means`, `mixture_weights`, `transition_matrices`, and `variances` byte-for-byte and telemetry fields `total_frames` and `stop_decision` value-for-value.

4. `tests/test_bw_sharding.py::test_model_comparison_surfaces_effective_bw_shard_count` compares model directories whose `provenance.json` records effective shard counts `1` and `2` and requires the comparison to report that file as different.

5. For a request of `4` shards, `tests/test_bw_sharding.py::test_multipron_multiple_shards_falls_back_loudly` requires multipron training to select `1` effective shard and emit reason `fallback_senone`; it also requires non-multipron training to retain the request.

6. `tests/test_bw_sharding.py::test_artifacts_reject_missing_or_stale_payload` exercises rejection of `missing metadata` and `stale accumulator payload` in the shard-artifact validator. These are unit-level mutations, not externally supplied production artifacts.

<!-- END GENERATED GATE SCOPE -->

## Hand-written limitations and interpretation

The reproducibility gate detects unstable ordering, races, and uninitialized bytes, but does not
independently test arbitrary completion-order permutations. Floating parameters are not claimed
equal across shard counts. Binary32 partial-sum regrouping changes the pass-one seed, and multi-pass
re-estimation amplifies that difference. No tolerance is offered for cross-count parameter
comparisons.

Effective BW shard count is an experimental variable recorded in training provenance beside the
requested job count, host, and architecture. A run comparison that does not pin the effective count
is invalid: downstream tree construction can amplify flat-direction parameter differences into
different senone compositions.

The cross-count discrete-state gate uses the one-shard reducer path. Together with the separate
in-process comparison it exposes a defect shared by all reducer shard counts, but neither gate
certifies upstream mathematical correctness. Defects present in the serial reference remain
invisible.

The assigned/outcomes/coverage reconciliation in `_validate_shard_artifacts` is the active runtime
guard: it checks the real shard results for complete, unique manifest coverage and consistent
outcomes. The pass/model/config/manifest fingerprints and `payload_sha256` are defensive scaffolding
today. The coordinator deletes and rewrites the pass directory, then writes and validates those
values in one call, so production cannot currently present stale, swapped, or externally modified
artifacts. Mutation tests exercise those rejection paths; retaining them is inexpensive and makes
them useful if artifacts later become externally supplied.

## Cross-count characterization

The retained nine-utterance, one-versus-two-shard experiment is characterization, not an acceptance
tolerance. After three passes, means differed by up to 2.5% relative, with 1,235 of 4,212 elements
outside the earlier arithmetic-derived budget. Maximum relative raw-mean-numerator differences were
`1.79804e-5`, `4.53897e-2`, and `2.47423e-2` across passes, including roughly 2,500-fold amplification
from pass one to pass two. The per-frame likelihood gaps were `0`, `3.964e-7`, and `2.960e-6`; both
arms improved monotonically. These three observations do not establish convergence to a common
value—the likelihood gap was still growing at the iteration cap.
