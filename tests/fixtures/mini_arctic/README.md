# mini_arctic — tiny end-to-end training fixture

A 10-utterance slice of the **CMU ARCTIC** `slt` (US English female) speech
database, used by `tests/test_e2e_training.py` to exercise the full
`features → flat → ci-1g` training pipeline on real audio in CI.

## Provenance / license

CMU ARCTIC databases (http://festvox.org/cmu_arctic/) were constructed at the
Carnegie Mellon University Language Technologies Institute and are distributed
under a permissive, free-to-use license (attribution appreciated). Only a
handful of utterances are included here, downsampled to what the trainer
needs, purely as a test asset.

## Contents

- `wav/arctic_a00NN.wav` — 10 utterances, 16 kHz / 16-bit / mono PCM.
- `transcription.txt` — simple `<fileid> <words...>` format (the format
  `pstrain setup --transcription` expects).
- `dictionary.dict` — pronunciation dictionary subset to exactly the
  vocabulary in `transcription.txt`.
- `phoneset.txt` — the phones used by `dictionary.dict` **plus `SIL`** (the
  filler phone). Every phone here is observed in training, so the trained
  model has no unoccupied — hence NaN-prone — states.
- `filler.dict` — maps `<s>`, `</s>`, `<sil>` to `SIL`.

## Regenerating / expanding

Pick fileids from a full ARCTIC `slt` setup, copy their wavs, build a simple
`<fileid> <words>` transcription, subset the dictionary to the spoken
vocabulary, and derive the phoneset from **dictionary + filler** pronunciations
(so `SIL` is included). See the commit that introduced this fixture.
