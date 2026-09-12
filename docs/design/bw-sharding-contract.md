# Baum-Welch sharding contract

Baum-Welch training with multiple workers partitions the canonical manifest into contiguous ranges and
reduces shard artifacts in ascending shard ID through the vendored accumulator reducers.

<!-- BEGIN GENERATED GATE SCOPE -->
## Generated gate scope

`CHECKED` entries are derived from assertion-helper calls in the named gates. `DESCRIBES` entries come from decorators and certify nothing. No comprehensive `CONSUMES` inventory is claimed because the harness does not mechanically trace inputs.

1. `tests/test_numeric_harness.py::test_seeded_bw_shards_are_reproducible_and_discrete_state_is_partition_independent`. CHECKED (mechanically asserted): produced files `means`, `variances`, `mixture_weights`, `transition_matrices`, `artifact.json`, `gauden_counts`, `mixw_counts`, and `tmat_counts`; copied inputs `mdef`. DESCRIBES (certifies nothing): kind `fixed-count-reproducibility`, declared shard counts `2`, and 3 passes. It makes no cross-architecture or cross-operating-system comparison.

2. `tests/test_numeric_harness.py::test_seeded_bw_shards_are_reproducible_and_discrete_state_is_partition_independent`. CHECKED (mechanically asserted): fields `assigned_ids`, `processed_ids`, `retried_ids`, `skipped`, `total_frames`, and `stop_decision`. DESCRIBES (certifies nothing): kind `cross-count-discrete-state`, declared shard counts `1` and `2`, and 3 passes. It makes no cross-count floating-parameter comparison.

3. `tests/test_numeric_harness.py::test_one_shard_reducer_matches_established_in_process_bw`. CHECKED (mechanically asserted): produced files `gauden_counts`, `means`, `mixture_weights`, `transition_matrices`, and `variances`; copied inputs `mdef`; fields `total_frames` and `stop_decision`. DESCRIBES (certifies nothing): kind `one-shard-reference`, declared shard counts `1`, and 3 passes. The compared `mdef` is copied from the same input model in both arms.

4. `tests/test_bw_sharding.py::test_model_comparison_surfaces_effective_bw_shard_count`. CHECKED (mechanically asserted): nothing. DESCRIBES (certifies nothing): kind `provenance-comparison`, declared shard counts `1` and `2`.

5. `tests/test_bw_sharding.py::test_multipron_multiple_shards_preserves_requested_parallelism`. CHECKED (mechanically asserted): nothing. DESCRIBES (certifies nothing): kind `multipron-sharding`, declared shard counts `4`.

6. `tests/test_bw_sharding.py::test_artifacts_reject_missing_or_stale_payload`. CHECKED (mechanically asserted): nothing. DESCRIBES (certifies nothing): kind `artifact-validation`, declared shard counts `not declared`. The declared mutations are orientation, not a certified artifact inventory.

<!-- END GENERATED GATE SCOPE -->

## Hand-written limitations and interpretation

The contract guarantees remain:

1. Repeating a run with the same manifest and shard count produces byte-identical accumulator
   artifacts and model files. The executable evidence exercises shard count 2 for exactly three
   passes; other shard counts and pass counts are untested.
2. Assigned, processed, retried, and skipped utterance identities, accepted frame counts, and
   per-pass stop decisions are exactly partition-independent. The executable evidence compares
   shard counts 1 and 2 for exactly three passes in the non-multipron gate. The additional
   multipron fixture exercises 1, 2, and 4 as described below; other cases are untested.
3. Floating parameters are not claimed equal across shard counts.
4. Effective BW shard count is recorded in training provenance and model-directory comparison
   includes `provenance.json`.
5. Multipron training uses the requested shard count. Each shard writes raw sufficient statistics
   and a success-only CI fallback activation bitmap. Reduction sums the statistics and unions the
   bitmaps before normalization seeds the survival prior once globally. Zero-occupancy activated
   states retain the same parameters as in the serial path. Normalization clears the bitmap.
6. A single requested worker never constructs a process pool. Multipron uses the existing serial
   loop; non-multipron executes its one-shard transport inline with native containment.

The reproducibility gate detects unstable ordering, races, and uninitialized bytes, but does not
independently test arbitrary completion-order permutations. Floating parameters are not claimed
equal across shard counts. Binary32 partial-sum regrouping changes the pass-one seed, and multi-pass
re-estimation amplifies that difference. No tolerance is offered for cross-count parameter
comparisons.

Effective BW shard count is an experimental variable recorded in training provenance beside the
requested job count, host, and architecture. The training fingerprint hashes the requested and
effective counts as resolved execution values, but retains host only as diagnostic provenance.
Its declared identity keys are the pstrain version, execution architecture, native library
identity, and standalone native-program identities. Version, architecture, or native-artifact
changes conservatively invalidate the cache because they can change the numeric trajectory; a run
comparison that does not pin the effective count is invalid because downstream tree construction
can amplify flat-direction parameter differences into different senone compositions.

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


## Multipron activation and worker lifetime

The `fallback_senones` sidecar contains the ASCII version line
`PSTRAIN_FALLBACK_SENONES_V1`, the decimal tied-state count on its own line, and exactly one
binary byte (zero or one) per tied state. Multipron reduction requires a sidecar even for empty
shards and all-zero masks. Native validation checks every sidecar before merging any counts;
Python artifact validation includes the sidecar in the payload digest and binds it to the same
model, configuration, pass, and manifest identities as the raw counts. Non-multipron artifacts
retain their original three-file format.

The native numerical test uses actual successful fallback graphs and asserts both observed and
unobserved activated states. It compares against the existing serial native path, checks one
unit of survival prior rather than one per shard, and verifies reset after normalization in both
variance modes. The full-loop multipron fixture compares one, two, and four workers for three
passes, including retry/skip accounting, actual overlapping worker lifetimes, and byte-identical
repeated two-shard artifacts. Its small float32 error budget characterizes that fixture only;
it does not replace the broader cross-count limitations above. Failed attempts contribute no
counts or activation flags; skipped utterances remain eligible on the next pass.

Worker PIDs and lifetimes are diagnostic telemetry, excluded from deterministic artifacts.
Contained serial native CPU usage is reported as unavailable when it cannot be measured; parent
CPU alone is not reported as BW worker CPU. A failed or interrupted parallel pass terminates and
reaps only its owned pool workers before propagating the failure. Each pool worker also watches
its parent sentinel and exits if the coordinator dies, including during a native update.
Incomplete artifacts are not reduced or promoted into a model.
