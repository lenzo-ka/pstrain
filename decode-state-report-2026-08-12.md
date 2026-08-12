# DECODE-STATE report — 2026-08-12

## Result so far

The retained state is the front end's **spectral-subtraction noise estimate**.
It is not search, language-model, lattice, backpointer, allocator, or CMN state.

Evidence:

- Both `off` and `on` pinned acoustic models configure PocketSphinx with
  `remove_noise=True` (while CMN is the already-established `batch`).
- PocketSphinx 5.1.1 documents `ps_start_stream()` as having exactly one effect:
  it resets noise-removal statistics that otherwise persist across utterances.
- The existing probe's `start_stream()`-before-each-utterance comparison therefore
  isolates this state without reconstructing the decoder or resetting search/LM
  structures.

PocketSphinx's implementation makes the mechanism concrete. `ps_start_stream()`
only calls `fe_reset_noisestats()`, which marks the estimate undefined. On the
next frame, `fe_remove_noise()` initializes per-filter smoothed power, noise,
signal-floor, and temporal-peak arrays from that frame. Thereafter every frame
updates them with exponential lower-envelope and temporal-masking rules and
applies the resulting smoothed gains to the Mel spectrum before cepstra and
acoustic scores are computed. `ps_start_utt()` clears the old lattice and
hypothesis and starts fresh acoustic/search utterance state, but does not reset
those front-end arrays. Thus the preceding WAVs alter the target WAV's features,
which alter its acoustic path scores; in the known `clb/arctic_b0438` case the
changed scores move the best path across the boundary between a leading `the`
insertion and no insertion.

The wrapper calls `start_utt()` / `process_raw(..., full_utt=True)` / `end_utt()`
for every WAV but never calls `start_stream()`. PocketSphinx intentionally treats
successive utterances as one audio stream for noise estimation. Reusing a decoder
for unrelated corpus files without marking stream boundaries is therefore an
artifact of how `score_model()` drives the API, though persistence within a real
continuous stream is intended PocketSphinx behavior.

## Deliberately untouched

No decoder configuration, stream-reset behavior, corpus ordering, hypotheses,
scoring, or bootstrap implementation has been changed.

## Reordering sensitivity

These are hypothesis-string differences, not merely score differences. Reverse
order is a deterministic, plausible reordering; per-WAV `start_stream()` isolates
the named state. Completed results:

| Mode | Corpus | Utterances | Canonical vs reverse | Canonical vs reset each WAV |
|---|---|---:|---:|---:|
| off | slt55 | 55 | 6 (10.9%) | 10 (18.2%) |
| on | slt55 | 55 | 6 (10.9%) | 8 (14.5%) |
| off | big | 3,395 | 645 (19.0%) | 716 (21.1%) |
| on | big | 3,395 | 609 (17.9%) | 712 (21.0%) |

The full detached shrub chain completed at `2026-08-12T12:04:09-04:00`. It ran
one cell at a time. Its parent and spawned-worker imports were verified to
resolve to `/mnt/shrub-data/decode-state-tree`, with PocketSphinx 5.1.1 from that
checkout's dedicated `.venv`. Machine-readable outputs are committed under
`decode-state-results/`.

## Preliminary answers

1. An order-independent decode cannot in general preserve the present serial
   hypotheses: the serial result is a function of the ordered audio prefix.
   `start_stream()` makes WAVs independent specifically by changing their
   features and, observably, hypotheses. Replaying the canonical prefix can
   reconstruct a target's serial state, but is neither an order-independent
   exactly-once decode nor the proposed speedup.
2. If the current serial hypothesis identity remains pinned, the corpus order
   and one continuously reused decoder/noise-estimator stream must be declared
   as part of measurement identity. Unlike PP5 accumulation order, this is not
   a free canonicalization: the order is load-bearing for decoded text.
3. The utterance-level matched-pair bootstrap's exchangeability assumption is
   false for this decode path. An observation at position N is a function of its
   own WAV and the noise-estimator state produced by positions 1 through N-1.
   This is not negligible: reverse order changes 17.9% to 19.0% of `big`
   hypotheses, and isolating each WAV changes about 21%.

   The published bands may still be described as **conditional resampling
   summaries of the already-decoded, fixed canonical sequence**. They must not
   be described as iid-bootstrap confidence intervals with nominal coverage for
   repetitions in which utterances are sampled or ordered differently: the
   bootstrap holds each stored hypothesis fixed while moving its utterance, so
   it omits the decoder-state response that the experiment directly observes.
   The practical remedy is therefore to restate every affected band and its
   measurement identity, not merely append a one-in-twenty footnote. A valid
   unconditional interval would require a newly specified estimand and a
   dependence-aware procedure that re-decodes each sampled/ordered corpus;
   choosing that procedure would change the measurement and is outside this
   investigation.

## Intended behavior versus driver artifact

PocketSphinx's state retention is intentional for successive utterances in one
continuous audio stream. The 5.1.1 public header says that noise-removal
statistics are retained across utterances and that `ps_start_stream()` resets
them. Calling only `start_utt()` is valid API usage and correctly resets search,
lattice, hypothesis, and acoustic utterance structures.

Treating thousands of independent ARCTIC WAV files and different speakers as a
single stream is the driver artifact. `score_model()` creates one decoder and
calls `decode_file()` repeatedly; `decode_file()` marks utterance boundaries but
never stream boundaries. Consequently the benchmark's canonical file ordering
and decoder reuse define a synthetic noise-adaptation stream.

## Conclusions for the owner

1. **No, not while preserving the current serial hypotheses and decoding each
   WAV once.** The load-bearing state is reconstructible only by replaying the
   exact canonical audio prefix (or by an unsupported snapshot/restore of the
   private noise arrays). Resetting the public state gives order independence
   but demonstrably changes 14.5%–21.1% of hypotheses depending on the cell.
2. **Yes: if the pinned serial path remains authoritative, declare ordering and
   stream reuse as measurement identity.** State the exact transcript insertion
   order, one reused decoder per mode/corpus cell, and no `start_stream()` between
   files. This resembles PP5's canonical-order remedy but is materially less
   benign because changing the order changes decoded words, not just floating
   accumulation details.
3. **Restate the bootstrap bands as conditional summaries, not nominal iid
   confidence intervals.** The observed 17.9%–19.0% reverse-order sensitivity on
   the pin's large corpora makes the coverage caveat material.

## Items found and deliberately untouched

- No decoder option or reset call was changed.
- No attempt was made to disable `remove_noise`, call `start_stream()` in the
  benchmark, construct a decoder per WAV, snapshot private front-end state, or
  replay prefixes in parallel workers.
- Search, LM, lattice, backpointer, allocator, and utterance-counter hypotheses
  were not separately mutated because the public single-effect reset isolates
  the noise state, while `ps_start_utt()` source shows the old lattice and
  hypothesis are cleared for every file.
- No scoring, ordering, bootstrap, record, or published band was modified.
