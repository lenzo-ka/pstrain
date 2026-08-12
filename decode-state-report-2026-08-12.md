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

The detached shrub chain is still supplying both `big` rows. It runs one cell at
a time, and its parent and spawned-worker imports were verified to resolve to
the dedicated shrub checkout and its PocketSphinx 5.1.1 virtual environment.

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
   false for this decode path. On `slt55`, a reverse order changes 10.9% of the
   hypotheses in either mode, already large enough that the dependence should
   be disclosed rather than dismissed as a single boundary anomaly. Final CI
   wording awaits the much larger `big` cells.
