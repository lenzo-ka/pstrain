# F10 multipron alignment stage design

Status: design only (Lane A7c, 2026-08-10). This is intentionally not
implemented in the parity-residual lane.

## Scope assessment

F10 is a full pipeline feature, not a local parameter correction. In the
audited SphinxTrain fork, stage 21 runs a corpus-wide `sphinx3_align` pass after
CI training when multipron mode is enabled. It prepares alignment dictionaries,
removes silence tokens and existing pronunciation suffixes from the input,
aligns every training utterance with the CI model, writes Viterbi-selected
dictionary word variants such as `WORD(2)`, records alignment fallbacks, and
makes that generated transcription the input to later CD training. Pstrain has
an in-process aligner, but it has no pipeline task or artifact contract for this
workflow and every BW task currently reads `etc/train.transcription` directly.

Building this safely therefore requires new generated artifacts, dependency
edges, variant-label serialization, fallback semantics, configuration, and
end-to-end parity tests. It is too large for A7c.

## Proposed contract

Add an on-mode-only `multipron-align` task after the CI model used to initialize
CD training and before `cd-untied-init`. Its inputs are the CI model, training
fileids, original training transcription, `.mfc` files, main dictionary, filler
dictionary, feature parameters, and training provenance. Its outputs are:

- `models/multipron-align/train.transcription`, containing exactly one line for
  every training fileid and preserving selected dictionary variant suffixes;
- `models/multipron-align/skipped.fileids`, listing fallback alignments;
- `models/multipron-align/summary.json`, containing counts, beam/configuration,
  CI model provenance, and per-utterance failure reasons.

When `multipron_training` is false, do not register the task and retain
`etc/train.transcription`. When it is true, all CD stages must consume the
generated transcription through one context-level resolver rather than
hardcoded paths. CI stages continue to use the original transcription.

Use the existing long-lived native `Aligner` with `align_mfc_file` so alignment
and BW consume the same precomputed cepstra. Before alignment, reproduce the
upstream preparation rules: merge non-silence fillers into the main alignment
dictionary, keep silence-only entries in the filler dictionary, remove silence
words from alignment input, strip pre-existing numeric variant suffixes, and
use the upstream stage beam/configuration. Serialize the word labels returned
by the aligner, removing only sentence-boundary/silence labels and retaining
pronunciation suffixes.

The failure policy must be settled from an upstream fixture before coding. The
audited fork can emit an `-outsent` fallback and separately records those
utterances; pstrain must either reproduce that original-transcript fallback
exactly or fail the task. It must never silently drop a training utterance or
reuse a partial alignment.

## Test and acceptance plan

1. Unit-test dictionary preparation, silence removal, variant-suffix stripping,
   output serialization, one-output-per-fileid validation, and fallback report
   generation.
2. Add a two-pronunciation synthetic corpus where the acoustic evidence selects
   the second variant; verify the generated transcript contains `WORD(2)` and
   the downstream CD task receives that artifact.
3. Add dependency tests proving off-mode has no alignment task and consumes the
   original transcript, while on-mode places alignment between CI and CD and
   invalidates CD outputs when its artifact changes.
4. Compare generated transcripts, selected variants, fallback IDs, and failure
   counts against the upstream stage on mini_arctic before evaluating WER.
5. Run the full native, Python, formatting, type, pre-commit, and fresh CMake /
   CTest gates because the native aligner and training pipeline are both in
   scope.

Acceptance is artifact parity with upstream for the same CI model and corpus,
not merely a successful alignment pass or a WER movement.
