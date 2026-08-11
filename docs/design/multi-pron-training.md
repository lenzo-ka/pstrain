# Multi-pronunciation training

## Status

Landed. Multi-pronunciation Baum-Welch training is enabled by default
as of commit `d68c17e`. Phase 6 (data-estimated pronunciation
probabilities) is still open as future work.

## What it does in one paragraph

Per-utterance HMMs are built as graphs in which each word with `k`
pronunciation variants in the dictionary contributes `k` parallel
phone paths. Forward-backward sums acoustic posteriors across these
paths during every iteration, so every variant gets training mass
proportional to its acoustic fit instead of zero. Variant arc
weights are initialized uniformly (`1/k`) at the start of training,
removing dictionary-order bias. The output models, viewed through
PocketSphinx forced alignment of held-out data, pick non-default
pronunciation variants on 6.4% of word tokens in our test corpus
vs. the same model trained without multipron (see "Empirical
signal" below).

## Why the prior trainer was wrong

The Baum-Welch lexicon (`csrc/libs/libcommon/lexicon.c`) used to
treat `reading` and `reading(2)` as two unrelated hash entries.
`mk_phone_list` looked each word up by its bare ortho and returned
exactly one entry — so for every multi-pron word the transcript
carried (`reading`), the trainer always used pron[1]. `(N)`
variants in the dictionary received zero training data unless the
user manually disambiguated the transcript. This meant the
**dictionary's row order** silently picked the acoustic targets for
every multi-pron word for the entire life of the model.

## How it works

### The key reuse: forward-backward is already a DAG engine

The state structure already supports arbitrary topology:

```68:99:csrc/include/s3/state.h
typedef struct state_s {
    /* ... */
    uint32 n_prior;
    uint32 *prior_state;
    float32 *prior_tprob;

    uint32 n_next;
    uint32 *next_state;
    float32 *next_tprob;
    /* ... */
}
```

`forward.c` traverses via these adjacency lists, never by positional
index, and `backward.c` is symmetric on `prior_state[]`. So the BW
math is already a generalized forward-backward over any acyclic
state graph. The linearity assumption used to live entirely in
`mk_phone_list` and `state_seq_make`. Multi-pron training is just
"build a wider graph and let the existing engine run on it."

### Inspiration from Kaldi, without OpenFST

Kaldi compiles per-utterance training HMMs by composing FSTs:
`HCLG_per_utt = HMM ∘ Context ∘ Lexicon ∘ WordSequence`, where the
lexicon FST has parallel arcs per pronunciation variant. We don't
need OpenFST; our per-utterance HMM is already a graph and a small
graph-builder achieves the same effect:

1. **Pronunciations are parallel paths in the training graph**, not
   independent dict entries. Forward-backward sums posteriors across
   them. There is no Viterbi pick anywhere in training — the soft
   distribution naturally re-weights variants as the model learns.
2. **Optional: pronunciation probabilities are data-estimated.**
   Tracked but deferred to phase 6. See "Future work".

## As-built layout

### New C modules

| File | Role | Lines |
|---|---|---|
| `csrc/include/s3/phone_graph.h` | Phone-graph type + builder declarations | ~110 |
| `csrc/libs/libcommon/phone_graph.c` | `mk_phone_graph` + `phone_graph_alloc/free` | ~290 |
| `csrc/libs/libcommon/phone_graph_triphone.c` | `phone_graph_split_contexts` + `cvt2triphone_graph` | ~290 |
| `csrc/include/s3/state_seq_graph.h` | Graph-aware state-sequence declarations | ~50 |
| `csrc/libs/libcommon/state_seq_graph.c` | `state_seq_make_graph` | ~330 |

### Changes to existing modules

| File | Change |
|---|---|
| `csrc/include/s3/lexicon.h`<br>`csrc/libs/libcommon/lexicon.c` | Add base-word index `base_ht` and `next_variant` linked list on `lex_entry_t`; new `lexicon_lookup_variants()` + accessors. Existing call sites unchanged. |
| `csrc/libs/libpstrain/pstrain_bw.{h,c}` | New `int32 multipron` flag on the BW context; new `pstrain_bw_set_multipron()`; `build_utt_state_seq()` dispatches to `next_utt_states` or `next_utt_states_graph`. |
| `csrc/programs/bw/next_utt_states.{c,h}` | New `next_utt_states_graph()` alongside the existing linear `next_utt_states()`; shared by the standalone `bw` binary and the CFFI BW context. |
| `csrc/programs/bw/main.c`<br>`csrc/programs/bw/train_cmd_ln.c` | New `-multipron` argv flag (default `no`) on the standalone `bw` binary; matches the same flag in upstream sphinxtrain PR #58. |
| `pstrain/lib/_cffi/cdef.py` | Declare `pstrain_bw_set_multipron`. |
| `pstrain/lib/bw.py` | `BWConfig.multipron` (default `True`); `BWTrainer.__init__` calls the setter. |
| `pstrain/lib/steps/train.py` | `run_bw_training(..., multipron=True)` flows into the per-iteration `BWConfig`. |
| `pstrain/lib/pipeline/context.py` | `TrainParams.multipron_training` (default `True`). |
| `pstrain/lib/pipeline/tasks.py` | Both BW-training task builders pass `ctx.train.multipron_training`. |

### Untouched

The BW math: `forward.c`, `backward.c`, `viterbi.c`, `accum.c`,
`baum_welch.c`. Zero changes. The wide-graph topology surfaces as
`state[i].n_next > 1` / `state[i].n_prior > 1` and the existing
loops handle it.

The linear path: `mk_phone_list`, `cvt2triphone`, `state_seq_make`,
`next_utt_states`. All kept verbatim and called when
`multipron=false`.

## Pipeline graph shape

For a transcript `"A B"` where word A has 2 variants and word B has 1
variant:

```
slot:  0    1    2     3    4    5     6    7
phone: A1a  A1b  A1c   A2a  A2b  A2c   B1   B2
       └ variant 1 ┘   └ variant 2 ┘

edges: 0 -> 1 -> 2 ──┐
       3 -> 4 -> 5 ──┴──> 6 -> 7
```

`mk_phone_graph` constructs this; `phone_graph_split_contexts`
duplicates non-fillers over the two-sided predecessor/successor context
cross-product so triphone resolution is unambiguous;
`cvt2triphone_graph` writes triphone ids in place; finally
`state_seq_make_graph` produces a state-level HMM that
forward/backward consumes unchanged.

## Operations

### Default behavior

Multi-pron training is **on** by default. No configuration changes
needed. The pipeline picks it up automatically:

```bash
pstrain build cd-8g           # uses multipron training
```

Every named config in `etc/configs.yaml` inherits the default.

### Opting out (legacy / SphinxTrain parity)

Set `training.multipron_training: false` for a config:

```yaml
sphinxtrain:
  description: "Matched to SphinxTrain defaults for comparison"
  features: { ... }
  training:
    n_state: 3
    n_senones: 200
    max_iterations: 10
    multipron_training: false
```

The default untied inventory remains the full dictionary-producible set
introduced for M4b. A PP3g-style multipron cell selects the exact set of
contexts reachable in its training pronunciation graphs:

```yaml
training:
  multipron_training: true
  untied_inventory: transcript-reachable
```

`transcript-reachable` expands every pronunciation variant and is valid only
with `multipron_training: true`, where inventory generation and Baum-Welch
share the same graph construction. Configuration resolution rejects it in
linear mode because the equivalent runtime-reachable inventory is already the
upstream-compatible `linear` first-pronunciation occurrence policy. The other
policy is `all-triphone`. Inventory misses always back off to a trainable CI
state; the inventory choice controls parameter allocation, not whether an
utterance can train.

### CI fallback survival prior

When a reachable CD context is absent from the inventory, its emitting state
backs off to the corresponding CI senone. At normalization, every active
non-filler fallback senone receives one full-distribution pseudo-count of its
parameters from the previous pass: the old mixture weights contribute total
mass one, and the matching old mean and variance moments are added with that
mass. Consequently, accumulator mass is not pure Baum-Welch posterior while
this fallback prior is active. The fallback still learns from every pass's
posterior, but a persistently low-occupancy state can remain prior-dominated.

This is deliberate and has no upstream analogue. Upstream's occurrence-based
linear inventory cannot omit a context that occurs in its own training path,
whereas graph-reachable inventories can expose a missing context through a
pronunciation branch. Numeric coverage exercises both repeated posterior
movement and the low-occupancy, prior-dominated case. With a complete CD
inventory, non-filler CI accumulators stay zero, the prior is not entered, and
the established numeric golden remains byte-for-byte unchanged.

Then `pstrain build cd-8g --config sphinxtrain` falls through to the
legacy `mk_phone_list` + `cvt2triphone` + `state_seq_make` path.
Output is bit-identical to pstrain's pre-multipron behavior.

### Mixing models across runs

Different configs write to different `shared/models/{target}/{config}/`
directories, so multipron and non-multipron models can coexist on
disk for A/B testing. The included
`scripts/compare_multipron_alignments.py` runs forced alignment of
held-out audio against two models and reports which variants each
model picks per word.

## What we don't do

* **OpenFST or K2-style FST machinery.** Overkill; our per-utterance
  graph is small.
* **Touch the BW math.** The DAG engine already exists.

## M4b SLT growth measurement

The preserved 1,043-utterance CMU Arctic SLT parity run changed from 9,786
to 10,052 untied triphone rows (+266, +2.72%). Parameter files grew from
9,309,224 to 9,561,400 bytes (+252,176, +2.71%): mdef 491,359→504,926;
means and variances 4,598,636→4,723,124 each; mixture weights
117,976→121,168; transition matrices stayed 1,984 bytes. The measured
CD-untied stage stayed within run noise (2.4s before, 2.2s after); peak RSS
was not captured by the lane runner.

For the engineered two-variant boundary, the phone graph changes from 12
slots/14 directed edges under the former one-sided split to 14 slots/16
directed edges under the two-sided cross-product. The two added slots and
edges are position-local; this is additive across the two ambiguous boundary
positions, not a corpus-wide pronunciation cross-product.
* **SphinxTrain-style hard-Viterbi disambiguation transcript stage.**
  That was always a workaround for the lack of soft-posterior
  multi-pron, and produces the very bias trap we're fixing. The
  `align` pipeline target (Tier 1 in `pipeline-runner.md`) emits the
  variant each word token won, but it's an output for inspection /
  TextGrids, not used for retraining.

## Empirical signal

Sanity check on the CMU Arctic test corpus (55 utterances, single
speaker, controlled speech). Two ci-1g models trained on the same
data — one with `multipron_training=true`, one with `false`.
PocketSphinx forced alignment against both, comparing which variant
`seg().word` reported per word position.

`scripts/compare_multipron_alignments.py` runs this end to end:

```bash
python scripts/compare_multipron_alignments.py \
    /path/to/ci-1g.multipron/default \
    /path/to/ci-1g.linear/default
```

Results:

| Metric | Value |
|---|---|
| Test utterances aligned (both models) | 55 / 55 |
| Content-word tokens compared | 512 |
| Same variant chosen | 479 (93.6%) |
| Different variant chosen | 33 (6.4%) |
| Base-word mismatches (sanity gate) | 0 |
| Utterances with ≥ 1 disagreement | 25 / 55 |

The disagreements concentrate in high-frequency function words —
exactly the words where dictionary multi-pron entries matter most.
The multipron-trained model picks variant 2 of `and` 14 times
vs. the linear model's 6 (2.3×). It picks `to(2)` four times where
the linear model never does. The reverse happens too (linear picks
`can(2)` more than multipron for one utterance), but the dominant
direction is multipron exploiting variant entries that the linear
model, having only ever trained on pron[1] acoustics, can't score
confidently.

CMU Arctic is a small, single-speaker, controlled corpus, so 6.4%
is a lower bound on what we'd see on a real multi-speaker corpus
with more pronunciation variation. The fact that we see a
structured, non-trivial difference even here is a good sanity check
that multipron training is materially changing what the model
learns about variant phones.

## Commit history

| Phase | Commit | What |
|---|---|---|
| Scope | `2420b25` | Design doc with the as-planned shape. |
| 1 | `51f40a9` | Lexicon base-word variant index. |
| 2 | `25209c9` | `phone_graph_t` + `mk_phone_graph`. |
| 3 | `db5faa5` | Graph-aware utterance HMM builder (`state_seq_make_graph`, `phone_graph_split_contexts`, `cvt2triphone_graph`). |
| 4 | `4a05db5` | Wire into BW behind the `multipron` config knob. |
| Flip | `d68c17e` | Enable multipron training by default. |
| Eval | `0b0691c` | Empirical comparison script + results. |
| Fix | `115dd02` | Repair forced alignment to use single-pass `seg()` (and preserve variant suffixes). |
| Upstream-align | `7e2bc43` | Hoist `build_utt_state_seq` into `next_utt_states_graph()` to mirror upstream PR #58. |
| Upstream-align | `9fe0e38` | Add `-multipron` argv flag to the standalone `bw` binary. |

## Upstream port

The C surface here is also the subject of
[cmusphinx/sphinxtrain PR #58](https://github.com/cmusphinx/sphinxtrain/pull/58)
("Bake multiple lexical pronunciation into baum-welch"). The two
branches converge on the same function names, file locations, and
signatures (`mk_phone_graph`, `phone_graph_split_contexts`,
`cvt2triphone_graph`, `state_seq_make_graph`, `next_utt_states_graph`,
the shared private `state_seq_internal.h`, and the `-multipron` argv
flag on `bw`) so future fixes can be cherry-picked in either
direction with minimal manual translation.

The two **defaults** differ on purpose:

| Layer | pstrain default | Upstream PR #58 default |
|---|---|---|
| C-level `-multipron` flag on `bw` argv | `no` (parity with prior behavior) | `no` (parity with prior behavior) |
| BW session API used by Python pipeline | `pstrain_bw_set_multipron(ctx, 1)` — **on** by default | n/a (no CFFI layer) |
| Recipe / config knob | `TrainParams.multipron_training: true` in `etc/configs.yaml` (default `true`) | `CFG_MULTIPRON_TRAINING = 'yes'` in `etc/sphinx_train.cfg` (default `yes`) |

So both projects ship multi-pron training **on** at the layer real
users drive (Python pipeline / Perl recipes), while keeping the
underlying C entry point default-off for anyone running `bw`
directly from the shell.

## Future work

### Phase 6: data-estimated pronunciation probabilities

Kaldi tracks per-variant arc probabilities and re-estimates them
from forward-backward posteriors at the end of each iteration. We
ship phase 1-5 with uniform `1/k` arc weights and no re-estimation.

Adding this is additive:
1. Per-variant accumulator alongside the existing gauden / mixw /
   tmat accumulators.
2. Normalize step at the end of each BW iteration: arc weight =
   (accumulated posterior on this arc) / (sum over all arcs from
   the same source).
3. Side-output file (`prons.txt` analog) recording the per-variant
   probabilities for diagnostic / publication use.

The accumulator is cheap (one float per dictionary entry); the
challenge is plumbing it through the existing C accumulator path
without breaking parity. Estimated ~1-2 days when motivated.

### Right-context split symmetry

See "What we don't do" above. Mirror of `phone_graph_split_contexts`
that splits a slot when its successors have different CI phones.
Bounded ~14% extra work on multi-pron-rich corpora; closes the last
known approximation in the triphone resolution.
