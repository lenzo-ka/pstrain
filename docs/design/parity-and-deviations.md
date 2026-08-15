# Parity and declared deviations

pstrain rebuilds stock SphinxTrain with a vendored C core and reimplemented
orchestration. A parity experiment compares the two engines under pinned
conditions. Configuration equality is not input equality: stage inputs,
shard membership, reduction order, front-end settings, inventory policy, and
checkpoint convention must be recorded or made identical for the claim being
tested.

The [Arctic benchmark pin](../benchmarks/arctic-pin.md) records the end-to-end
product measurement. The design records linked below define the narrower
contracts behind each classified difference.

## Declared register

| Surface | Classification | pstrain posture | Stock posture | A parity experiment must pin |
|---|---|---|---|---|
| Untied inventory | Deliberate deviation; ratification pending | `transcript-reachable` includes contexts reachable through alternative pronunciations when multipron training is on | First-pronunciation-observed inventory | Multipron mode, inventory policy, dictionary, transcripts, and the produced untied mdef |
| Training front end | Deliberate deviation | Dither and per-frame DC removal enabled | Neither enabled | Input audio, complete feature configuration, extractor identity, and produced feature bytes |
| Alignment retry | Deliberate deviation | One wider-beam retry after a forward-final-state pruning failure | No retry | Normal beam, retry factor, failure policy, retry telemetry, and skipped-utterance identities |
| Optional final silence | Deliberate deviation | Final transcript silence may consume zero frames | Final silence must consume a frame | Optional-final-silence setting, transcripts, alignment outcomes, and accepted frame counts |
| Baum-Welch checkpoint representation | Deliberate representation convention with bounded floating-point effect | Normalized model checkpoints | Raw-count checkpoints; loaders normalize them | Common pre-reload inputs, checkpoint representation, loader path, and post-reload arrays |
| Shard partition and reduction order | Deliberate orchestration convention with inherent floating-point sensitivity | Canonical contiguous partitioning and ascending-shard reduction | Different remainder placement and accumulation order | Manifest order, shard count and membership, accumulator inputs, and reduction order |

## Inventory under multiple pronunciations

`transcript-reachable` inventory follows the same pronunciation graph consumed
by multipron Baum-Welch. Stock SphinxTrain instead observes the first
pronunciation while producing its untied inventory, even when later training
uses pronunciation alternatives. The pstrain posture is coherent with its
multipron default, but it remains a declared decision awaiting ratification,
not an upstream-equivalent result. The inventory producer and its interaction
with training mode are specified in
[multiple-pronunciation training](multi-pron-training.md).

## Training front-end boundary

pstrain trains with dither and DC removal enabled; stock SphinxTrain trains
with neither. A controlled boundary measurement attributed a 0.547-point WER
movement to the combined boundary, with a confidence interval spanning zero.
Resetting the flags restored the historical exact-zero-codebook anchor, but
that artifact mechanism explained only about 35% of the WER movement. The
remaining movement may pass through different tree inputs. The measurement
therefore supports two mechanisms and does not justify attributing the entire
WER change to Gaussian artifacts.

## Alignment recovery and final silence

The product defaults recover once with a wider forward beam and allow final
transcript silence to consume zero frames. Stock SphinxTrain has no retry and
requires final silence to consume a frame. Under pstrain's defaults,
`arctic_a0587` trains without a skip. Experiments intended to mirror stock must
disable retry and optional final silence; those are comparison controls, not
the product posture. The runtime guarantees are detailed in
[failed-alignment policy](failed-alignment-policy.md) and
[optional final silence](optional-final-silence.md).

## Checkpoints, sharding, and reduction

Stock checkpoints raw counts while pstrain checkpoints normalized parameters;
both loaders normalize before use. From a common input, the raw-to-normalized
round trip seeds a 2-ULP difference in 20 of 480 transition entries at the
second-pass reload. The difference is bounded at that boundary, can amplify
downstream, and did not change a convergence decision in the measured common-
start trajectory. This is a representation deviation plus inherent finite-
precision behavior, not an estimator discrepancy. The normalization contract
is recorded in [Baum-Welch normalization policy](bw-normalization-policy.md).

The engines also place a manifest remainder on different sides of their shard
partition. In the measured 1,042-utterance input this moved 13 utterances
between shards. When membership and reduction order are matched, accumulation
is byte-identical. With different grouping, non-associative floating-point
addition supplies a real but controllable seed. pstrain's guarantees and
limits are recorded in [Baum-Welch sharding contract](bw-sharding-contract.md).
