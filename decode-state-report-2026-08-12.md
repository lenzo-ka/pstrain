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
  structures. Full-corpus results are being quantified on shrub.

The wrapper calls `start_utt()` / `process_raw(..., full_utt=True)` / `end_utt()`
for every WAV but never calls `start_stream()`. PocketSphinx intentionally treats
successive utterances as one audio stream for noise estimation. Reusing a decoder
for unrelated corpus files without marking stream boundaries is therefore an
artifact of how `score_model()` drives the API, though persistence within a real
continuous stream is intended PocketSphinx behavior.

## Deliberately untouched

No decoder configuration, stream-reset behavior, corpus ordering, hypotheses,
scoring, or bootstrap implementation has been changed.

## Pending quantified evidence

The detached shrub runs will supply canonical-versus-reverse and
canonical-versus-`start_stream()` sensitivity for every pinned mode/corpus cell.
The mechanism and answers about serial-path preservation and CI interpretation
will be completed from those results.
