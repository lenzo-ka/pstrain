# Baum-Welch sharding contract

Non-multipron Baum-Welch training partitions the canonical manifest into contiguous ranges and
reduces shard artifacts in ascending shard ID through the vendored accumulator reducers. The
contract is:

1. Repeating a run with the same manifest and shard count produces byte-identical accumulator
   artifacts and model files. This detects unstable ordering, races, uninitialized bytes, and
   completion-order-dependent reduction.
2. Assigned, processed, retried, and skipped utterance identities, accepted frame counts, and
   per-pass stop decisions are exactly partition-independent.
3. Floating parameters are not claimed equal across shard counts. Binary32 partial-sum regrouping
   changes the pass-one seed, and multi-pass re-estimation amplifies that difference. No tolerance
   is offered for cross-count parameter comparisons.
4. Shard count is an experimental variable recorded in training provenance beside host and
   architecture. A run comparison that does not pin shard count is invalid: downstream tree
   construction can amplify flat-direction parameter differences into different senone
   compositions.
5. Multipron training requested with more than one shard falls back loudly to one shard and logs
   `fallback_senone` as the reason.

The gates certify conformance to pstrain's own serial path at shard count one. They are blind to a
defect shared with that reference and do not certify upstream mathematical correctness.

## Cross-count characterization

The retained nine-utterance, one-versus-two-shard experiment is characterization, not an acceptance
tolerance. After three passes, means differed by up to 2.5% relative, with 1,235 of 4,212 elements
outside the earlier arithmetic-derived budget. Maximum relative raw-mean-numerator differences were
`1.79804e-5`, `4.53897e-2`, and `2.47423e-2` across passes, including roughly 2,500-fold amplification
from pass one to pass two. The per-frame likelihood gaps were `0`, `3.964e-7`, and `2.960e-6`; both
arms improved monotonically. These three observations do not establish convergence to a common
value—the likelihood gap was still growing at the iteration cap.
