# Baum-Welch sharding contract lane — 2026-08-14

## Outcome

- **[measured]** Training provenance now records `requested_jobs` separately from the effective
  `bw_shard_count`. With three requested jobs, the provenance tests observe effective counts of
  three for `multipron_training=false` and one for `multipron_training=true`.
- **[measured]** `compare_models` now discovers and compares `provenance.json`; its negative case
  reports `provenance.json: DIFFER (text)` when requested jobs are equal but effective BW shard
  counts differ.
- **[measured]** The established pre-sharding in-process accumulation loop remains callable only
  through the private `_in_process_reference` test switch. A three-pass gate compares it with the
  one-shard dump, restore, and reducer path.
- **[measured]** The established-reference gate observed byte-identical final `mdef`, `means`,
  `variances`, `mixture_weights`, `transition_matrices`, and top-level `gauden_counts`; it also
  observed identical accepted frame counts and stop decisions on all three passes.
- **[measured]** Clause 1 now states that ascending shard-ID reduction is deterministic by
  construction. This was chosen instead of adding an artificial completion-order test because the
  reducer is given an already sorted directory list and does not consume completion order.

## Contract and gates

- **[read]** Non-multipron training partitions the canonical manifest into contiguous ranges, runs
  every partition through the existing BW driver and native accumulator dump, sorts results by
  shard ID, and reduces through the vendored `rdacc_mixw`, `rdacc_tmat`, and `rdacc_den` paths.
- **[measured]** Repeating the seeded three-pass two-shard run produced byte-identical shard
  metadata, accumulator payloads, and final model files.
- **[measured]** The one-versus-two-shard gate observed exact equality of assigned, processed,
  retried, and skipped utterance identities, accepted frame counts, and every pass stop decision.
- **[measured]** The clause-2 negative control removed utterance `b` from both assigned and
  processed identities and produced:

  ```text
  BW discrete-state mismatch: {'total_frames': 9, 'stop_decision': 'continued', 'assigned_ids': ['a', 'b'], 'processed_ids': ['a', 'b'], 'retried_ids': [], 'skipped': []} != {'total_frames': 9, 'stop_decision': 'continued', 'assigned_ids': ['a'], 'processed_ids': ['a'], 'retried_ids': [], 'skipped': []}
  ```

- **[read]** The assigned/outcomes/coverage reconciliation in `_validate_shard_artifacts` is the
  active runtime guard. It checks real shard results for unique, complete manifest coverage and for
  outcomes that exactly reconcile with assigned identities.
- **[read]** Pass/model/config/manifest fingerprints and `payload_sha256` are defensive scaffolding,
  not active protection in the current production flow: each pass directory is deleted and
  rewritten, then the same coordinator writes and validates the values in one call.
- **[measured]** Mutation tests reject missing metadata, tampered payloads, duplicate shard IDs,
  overlapping or absent coverage, duplicate manifest identities, wrong fingerprints, and
  incompatible parameter shapes. These tests exercise future externally supplied or persisted
  artifact failure modes; they do not imply that those states arise in today's production flow.
- **[measured]** Multipron with four requested shards resolves to one effective shard and logs
  `fallback_senone`; non-multipron retains four.
- **[inference]** Floating tensors intentionally have no cross-count equality or tolerance contract
  because binary32 shard-local partial-sum regrouping changes the first-pass seed and later passes
  amplify it.

## Host and architecture fingerprint finding

- **[read]** `host` and `architecture` are inside the canonical training provenance payload used to
  calculate the content-addressed training provenance filename.
- **[inference]** Identical configuration on a second host produces a different training provenance
  path. Because training tasks consume that path, a newly written second-host provenance input is
  newer than existing model outputs and invalidates those tasks rather than reusing the first-host
  outputs.
- **[read]** There is no separate cross-host BW artifact cache in this path; shard pass directories
  are local transient outputs and are deleted before every pass.
- **[inference]** Model comparison now intentionally reports the two `provenance.json` files as
  different when host or architecture differs, even if parameter files match. Consumers must inspect
  the component result rather than treating `all_match` as parameter-only equality.
- **[inference]** This is a problem for deliberate second-machine replication if the goal is
  cross-host task reuse or an `all_match` comparison based only on model content. It does not prevent
  training or per-component comparison. No fix was made because separating execution metadata from
  the stage fingerprint changes broader provenance and invalidation semantics beyond this contained
  shard-count correction.

## Retained cross-count characterization

- **[read]** The retained nine-utterance one-versus-two-shard run ended with maximum mean relative
  difference `0.02495766`, maximum absolute difference `0.00448418`, and 1,235 of 4,212 mean elements
  outside the former arithmetic-derived gate.
- **[read]** Failing states were occupied: minimum occupancy `5.2367449`, median `18.9594345`, and
  1,055 of 1,235 failing elements were in `(10,100]`; the sole near-empty state had no failures.
- **[read]** Maximum relative raw-mean-numerator differences were `1.79804e-5`, `4.53897e-2`, and
  `2.47423e-2` across passes, approximately 2,500-fold amplification from pass one to pass two.
- **[read]** Per-frame likelihood gaps were `0`, `3.96436775e-7`, and `2.96006957e-6`; both arms
  improved monotonically at every observed pass.
- **[inference]** Three monotonic passes do not establish convergence to a common value because the
  likelihood gap was still growing at the iteration cap.

## Edge coverage

- **[measured]** Empty partitions are covered at the partition boundary: five shards over four
  utterances retain an explicit empty fifth shard.
- **[measured]** Top-level merged `gauden_counts` is now covered byte-for-byte by the established
  in-process versus one-shard reducer gate.
- **[read]** The existing end-to-end training suite exercises default multipron training, and the
  fallback resolver is tested with more than one requested shard, but there is no end-to-end
  pipeline run combining default multipron with `-j>1`.

## Validation

- **[measured]** `ruff check pstrain tests`: passed.
- **[measured]** `ruff format --check pstrain tests`: passed; 148 files checked.
- **[measured]** `mypy pstrain`: passed; 101 source files checked.
- **[measured]** CTest: 5/5 passed.
- **[measured]** Full pytest with the native library required: 522 passed, one benchmark deselected,
  23 warnings, 48.51 seconds.
- **[measured]** `make config-check`: 32 passed; generated configuration reference unchanged.
- **[measured]** Focused comparison, provenance, artifact, negative-control, reproducibility, and
  established-reference gates passed.
- **[measured]** The first full-pytest attempt exposed an environment-sensitive niceness assertion:
  a process already at nice 17 saturates at 19 rather than reaching 22. The assertion now models the
  POSIX upper bound; its focused rerun and the subsequent full suite passed.

## Declined work

- **[measured]** Declined changing any numerical budget or adding a cross-count tensor tolerance;
  the contract makes no such equality claim, and a tolerance would not be a valid invariant.
- **[measured]** Declined a float64 accumulator path because it requires changes across live
  accumulation, serialization, restoration, reduction, format versioning, and normalization.
- **[measured]** Declined claiming common-limit convergence or downstream WER parity because neither
  was established by the retained three-pass data.
- **[measured]** Declined an end-to-end default-multipron `-j>1` test in this lane because the
  requested/effective provenance cases and loud resolver fallback are directly covered, while such a
  pipeline run would duplicate the expensive model-building suite without exercising a second
  reducer path.
- **[measured]** Declined changing host/architecture fingerprint semantics because it affects all
  training provenance paths and task invalidation, not only BW sharding.
- **[measured]** Declined pushing to or merging `main`, force-pushing, opening a new pull request, or
  merging the feature branch; this lane is limited to a normal push to the existing branch.
