# Optional boundary isolation measurement — 2026-08-14

## Decision result first

- **[measured]** With normalized entry and exit mass, `arctic_a0587` **does recover under final-only** in both `multipron=false` and `multipron=true`.
- **[measured]** With normalized mass, `arctic_a0587` **does not recover under initial-only** in either pronunciation mode.
- **[measured]** The recovery therefore did not disappear in all normalized arms; it is present in final-only and both, and absent in off and initial-only.
- **[inference]** This measurement supports reducing PR #81's potential scope to final-only. Initial optionality supplies none of the observed failure recovery and only a small minority of the aggregate `SIL` occupancy movement.

## Per-arm results

The occupancy delta is `arm - off`; negative values mean less `SIL` occupancy than off. The percentage denominator is the corresponding mode's off-arm total `SIL` occupancy.

| multipron | arm | `arctic_a0587` recovers? | BW failures | changed failure state vs off | new failures | total `SIL` occupancy | delta from off | percent of off |
|---|---|---:|---:|---|---|---:|---:|---:|
| false | off | **[measured]** no | **[measured]** 174 | **[measured]** none | **[measured]** none | **[measured]** 32534.72265625 | **[measured]** 0 | **[measured]** 0% |
| false | final-only | **[measured]** yes | **[measured]** 173 | **[measured]** recovered: `arctic_a0587` | **[measured]** none | **[measured]** 31875.52734375 | **[measured]** -659.1953125 | **[measured]** -2.0261285749% |
| false | initial-only | **[measured]** no | **[measured]** 174 | **[measured]** none | **[measured]** none | **[measured]** 32476.775390625 | **[measured]** -57.947265625 | **[measured]** -0.1781089891% |
| false | both | **[measured]** yes | **[measured]** 173 | **[measured]** recovered: `arctic_a0587` | **[measured]** none | **[measured]** 31817.578125 | **[measured]** -717.14453125 | **[measured]** -2.2042435672% |
| true | off | **[measured]** no | **[measured]** 38 | **[measured]** none | **[measured]** none | **[measured]** 37099.3671875 | **[measured]** 0 | **[measured]** 0% |
| true | final-only | **[measured]** yes | **[measured]** 37 | **[measured]** recovered: `arctic_a0587` | **[measured]** none | **[measured]** 36343.1328125 | **[measured]** -756.234375 | **[measured]** -2.0384023565% |
| true | initial-only | **[measured]** no | **[measured]** 38 | **[measured]** none | **[measured]** none | **[measured]** 37040.4140625 | **[measured]** -58.953125 | **[measured]** -0.1589060123% |
| true | both | **[measured]** yes | **[measured]** 37 | **[measured]** recovered: `arctic_a0587` | **[measured]** none | **[measured]** 36284.18359375 | **[measured]** -815.18359375 | **[measured]** -2.1972978397% |

- **[measured]** Across all eight cells, `arctic_a0587` is the only utterance whose BW success/failure state changes relative to off.
- **[measured]** No utterance changes from success in off to failure in any enabled arm.
- **[inference]** In single-pron mode, final-only accounts for 659.1953125 / 717.14453125, or about 91.92%, of both-arm occupancy reduction; initial-only accounts for about 8.08%.
- **[inference]** In multipron mode, final-only accounts for 756.234375 / 815.18359375, or about 92.77%, of both-arm occupancy reduction; initial-only accounts for about 7.23%.

## Method and provenance

- **[measured]** The requested branch was clean when inspected, and its starting head was `e6502444713014a80efba64595e73b0c6976b9e5` on `feat/optional-boundary-silence`.
- **[read]** Before changing or measuring anything, I read `/Volumes/experiments/pstrain-parity/optsilprice-report-2026-08-14.md` and `/Volumes/experiments/pstrain-parity/fugu81b-verdict.md`.
- **[read]** The stopped prior lane had already committed normalized competing entry and exit alternatives in `5c5e1d8`; the retained and bypass alternatives each have mass 0.5, so each boundary's total alternative mass is one.
- **[measured]** The focused `bw_state_seq_build` CTest passed after the measurement split was added, confirming the existing normalized both/off topology gate still passes.
- **[measured]** Each arm performed exactly one BW accumulation pass over the same 1,132-entry SLT transcription, with `a_beam=1e-90`, `b_beam=1e-10`, `pass2var=true`, `unobserved_gaussian_policy=zero`, and no retry or retraining.
- **[measured]** The transcription was `/Volumes/experiments/arctic/slt/etc/all.transcription` (SHA-256 `01450cc1393ecf2e66a6f5e288643b5d6d9793392090430287dbbc2187cff1af`).
- **[measured]** The model was `/Volumes/experiments/pstrain-parity/pstrain-cmu-arctic-slt/shared/models/cd-1g/parity`; the feature directory was `/Volumes/experiments/pstrain-parity/pstrain-cmu-arctic-slt/shared/features/parity`.
- **[read]** These are the available parity CD-1g substitute assets identified in the prior report, not the missing ladder4 assets used by the earliest headline result.
- **[measured]** The emitted JSON records the absolute model and feature paths and absolute paths plus SHA-256 for the transcription, model parameter files, dictionary, and filler dictionary.
- **[read]** The four-arm selector is explicitly temporary and measurement-only. The existing public boolean remains `false=off`, `true=both`; no schema/default or shipped configuration surface was split.

## Artifacts

- **[measured]** Machine-readable result: `/Volumes/experiments/pstrain-parity/optsil-tree/optsiliso-measurement-2026-08-14.json`, SHA-256 `77170226de52a5b252d334ecbd04621056eede06fce0a0bc9ca5d174e0eff2a1`.
- **[measured]** Per-utterance rows: `/Volumes/experiments/pstrain-parity/optsil-tree/optsiliso-measurement-2026-08-14.csv`, SHA-256 `8fa5688ea34288668ef06441e828131e62984f2263368415f3bf908e924d6232`.
- **[measured]** The CSV has 2,264 data rows (1,132 utterances × two pronunciation modes) plus one header. Every row contains failure status and `SIL` occupancy for all four arms.
- **[measured]** The JSON retains the complete failure list per arm, the recovered/new-failure set differences, per-utterance measurements, denominator labels, and the CSV path/hash.

## Declined work

- **[measured]** I omitted trailing-silence classification and sensitivity output entirely; the measurement script no longer reads audio or emits the rejected stratification.
- **[read]** I declined the deferred initial-side `state_seq[0]` precondition enumeration.
- **[measured]** I did not change beams, dither, metrics, evaluator behavior, unrelated defaults, or the training schema default.
- **[measured]** I did not retrain a model; all cells are one BW pass against the same existing model.
- **[measured]** I did not merge, push, create a PR, or update PR #81.
- **[inference]** I did not implement a shippable final-only configuration surface because this lane is a decision measurement; the result supplies the scope decision, not the feature push.
