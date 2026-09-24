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
| Untied inventory | Deliberate deviation with upstream-compatible and exhaustive alternatives | Shipped default and live benchmark profile: `transcript-reachable` when multipron training is on; multipron-off configurations without an explicit policy select `linear`; explicit `linear` and `all-triphone` remain available | First-pronunciation-observed inventory, equivalent to pstrain's `linear` policy | Multipron mode, inventory policy, dictionary, transcripts, and the produced untied mdef |
| Training front end | Deliberate deviation | Dither and per-frame DC removal enabled | Neither enabled | Input audio, complete feature configuration, extractor identity, and produced feature bytes |
| Alignment retry | Deliberate deviation | One wider-beam retry after a forward-final-state pruning failure | No retry | Normal beam, retry factor, failure policy, retry telemetry, and skipped-utterance identities |
| Optional final silence | Deliberate deviation | Final transcript silence may consume zero frames | Final silence must consume a frame | Optional-final-silence setting, transcripts, alignment outcomes, and accepted frame counts |
| Baum-Welch checkpoint representation | Aligned | Raw mixture-weight and transition-matrix accumulators are serialized; checkpoints copy those files unchanged | Raw-count checkpoints; loaders normalize them | Common pre-reload inputs, checkpoint bytes, loader path, and post-reload arrays |
| Shard partition position | Shipped default plus upstream-compatible alternative | Shipped default `remainder-first` distributes extra utterances across leading shards; opt-in `remainder-last` matches stock | Floor-sized leading shards and the entire remainder in the last shard | `sharding.partition_position`, manifest order, shard count, and produced partition manifest |
| Shard reduction order | Aligned | Ascending shard-index reduction, independent of worker completion order | Ascending partition-index reduction | Accumulator inputs and reduction order |
| Vendored byte swapping | Aligned safety fix | Exact-width unsigned intermediates avoid undefined signed shifts while preserving output bits | Signed shift expressions rely on implementation behavior | Serialized model and feature bytes |
| Alignment dictionary, dropped base pronunciation | Deliberate deviation | A word whose unsuffixed pronunciation is dropped for a phone the model does not define takes its first surviving alternative as its base entry and aligns over all of its surviving alternatives, with a warning, as multiple-pronunciation training does | The aligner ends the process on the surviving alternative, so no utterance aligns | Dictionary, model phone inventory, and the alignment outcomes of utterances using the word |
| Vendored log-math conversion | Aligned safety fix | Log-to-linear conversions scale by 2^shift with `ldexp`, preserving output at every shift pstrain uses | Left shift of a negative log value, which is undefined | Aligner model load and alignment scores |
| Aligner CI-senone count | Aligned safety fix | The count of leading CI senones stops at the model's senone count, with no output change | Reads past the end of the senone table for a model with only CI senones | Model definition |
| Library alignment frame limit | Deliberate deviation | The library path aligns up to 32,768 frames per utterance and refuses a longer one before any work, with a message naming its frame count and the limit | `sphinx3_align` stops at 15,000 frames | Utterance lengths, and which utterances are refused |
| Forced-alignment retry acceptance | Deliberate deviation | After a final-state failure the library aligner retries once at the nominal beam divided by `alignment.retry_beam_factor` (1e-200 by default), and keeps what the retry recovers only if its mean score per speech frame reaches a threshold calibrated at the retry beam: by default the 5% quantile of the run's own first-pass alignments realigned there. First-pass alignments are never checked | `sphinx3_align` does not retry | Nominal beam, retry factor, acceptance target or supplied threshold, the calibration sample, and which recoveries were accepted or rejected |
| Alignment score range | Deliberate deviation | Phone, word and total scores beyond the int32 range saturate at the range edge | Such scores wrap around | Segment lengths and the reported phone, word and total scores |

## Inventory under multiple pronunciations

The shipped pstrain default is `transcript-reachable` when multipron training
is enabled, as it is by default, and the live Arctic evidence is measured under
that policy. It follows the pronunciation graphs consumed by multipron
Baum-Welch, covering contexts from every usable pronunciation without
allocating the complete phoneset cross-product produced by `all-triphone`.
Unlike stock SphinxTrain's first-pronunciation-observed inventory, represented
by pstrain's `linear` policy, it does not omit contexts that are reachable only
through alternative pronunciations. Stock SphinxTrain does observe only the
first pronunciation while producing its untied inventory, even when later
training uses pronunciation alternatives. Within a configuration layer, an
explicit inventory is honored. However, a later layer that sets
`multipron_training=false` without also setting an inventory triggers the
resolver's `linear` auto-substitution and can override the earlier explicit
choice. This is a known pre-existing resolver behavior tracked as a follow-up.
The inventory producer and its interaction with training mode are specified in
[multiple-pronunciation training](multi-pron-training.md).

## Training front-end boundary

pstrain trains with dither and DC removal enabled; stock SphinxTrain trains
with neither. A localized boundary effect was measured out-of-tree, so its
numeric estimate is not part of the checked-in evidence. The remainder is
UNEXPLAINED. Different tree inputs and Gaussian artifacts remain hypotheses,
not separately demonstrated mechanisms; the available measurement does not
justify attributing the WER movement to either one.

## Alignment recovery and final silence

The product defaults recover once with a wider forward beam and allow final
transcript silence to consume zero frames. Stock SphinxTrain has no retry and
requires final silence to consume a frame. Under pstrain's defaults,
`arctic_a0587` trains without a skip. Experiments intended to mirror stock must
set `training.retry_beam_factor=1`, `training.failed_alignment=omit`, and
`training.optional_final_silence=false`. The `omit` policy is required because
disabling retry under the shipped `recover` policy aborts on a final-state
failure instead of reproducing stock's report-and-omit behavior. These are
comparison controls, not the product posture. The runtime guarantees are
detailed in
[failed-alignment policy](failed-alignment-policy.md) and
[optional final silence](optional-final-silence.md).

## Checkpoints, sharding, and reduction

Current pstrain serializes raw mixture-weight and transition-matrix
accumulators, and checkpoint creation copies those model files unchanged.
Loaders normalize them before use, matching the stock checkpoint convention.
The normalization contract is recorded in
[Baum-Welch normalization policy](bw-normalization-policy.md).

The shipped pstrain `sharding.partition_position=remainder-first` policy gives
one extra utterance to each leading shard. Stock SphinxTrain uses floor-sized
leading shards and assigns the entire remainder to the last shard; parity arms
can select that behavior with `remainder-last`. The default remains unchanged.
In the measured 1,042-utterance, eight-shard input, the two shapes were
`131,131,130,130,130,130,130,130` and
`130,130,130,130,130,130,130,132`, moving 13 utterances between adjacent
shards without changing manifest order or coverage.

Production reduction was already deterministic before this option: worker
results and accumulator directories are ordered by ascending shard index before
the vendored reducer is called, independently of completion order. When shard
membership and reduction order are matched, accumulation is byte-identical.
With different grouping, non-associative floating-point addition supplies a
real but controllable seed. pstrain's guarantees and limits are recorded in
[Baum-Welch sharding contract](bw-sharding-contract.md).

## Dropped base pronunciations in the alignment dictionary

The alignment dictionary loader drops a pronunciation that uses a phone the
acoustic model does not define. Upstream, when the dropped pronunciation is a
word's unsuffixed one and an alternative such as `word(2)` survives, the loader
ends the process on that alternative ("Missing base word"), so no utterance
aligns, including utterances that never use the word. The training loader
behaves differently: it keeps the surviving alternatives under the word, and
multiple-pronunciation training, the default, trains over all of them.

pstrain's alignment loader deliberately follows training. The first surviving
alternative read after the dropped line becomes the word's base entry, the
unsuffixed spelling in a transcript reaches it, and later alternatives link to
it as usual. The word therefore aligns over all of its surviving alternatives
and the best-matching one wins, as in multiple-pronunciation training. The
loader warns that it has done so. The pre-run phone report still lists the
dropped pronunciation and says the word is aligned over its surviving
alternatives.

The change is confined to a base that was read and dropped from the same
dictionary file as the alternative, so a main-dictionary word never lands in
the filler range. An alternative read before any unsuffixed line for its word,
including a dictionary that never has one, or one whose dropped base is in the
other file, still ends the aligner as it does upstream.
