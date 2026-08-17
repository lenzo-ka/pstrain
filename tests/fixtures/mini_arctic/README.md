# mini_arctic — tiny end-to-end training fixture

A 10-utterance slice from a single speaker in the CMU_ARCTIC corpus: **CMU
ARCTIC `slt`**, a US English female speaker. It is used by
`tests/test_e2e_training.py` to exercise the full
`features → flat → ci-1g` training pipeline on real audio in CI.
The same suite also trains `ci-1g`, builds a language model from these ten
transcripts, decodes all ten WAVs through the production decoder, and scores
the hypotheses against `tests/golden/mini_arctic_ci_1g_decode.json`. The
gate requires exactly the expected utterance IDs, checks each live hypothesis
with an independent word edit-distance implementation, and checks aggregate
errors and WER against those rows. The golden records a coarse reference tuple:
Linux x86-64, Python 3.11, the declared FP-contraction policy (`off`), and one
native job. The classifier matches only that platform and declared-policy tuple;
it does not identify the compiler or hash the native library. A tuple match
requires exact hypotheses, per-utterance errors, aggregate counts, and WER. A
nonmatching tuple permits at most two aggregate word errors around the 15-error
golden. That portable band cannot mask a matching-tuple regression; it is a
provisional allowance until a cross-platform dispersion matrix can refine the
threshold. `native_jobs` records the deterministic execution policy but does not
affect the measurement result. FP contraction being off is not merely assumed:
`scripts/check_fp_contract.py` verifies the built native library repo-wide from
the Makefile and CI test jobs, so a contraction-enabled library cannot pass the
repository gates. Because compiler identity is outside the tuple, changing the
compiler on the reference runner requires regenerating this golden as an
operational step. Set `PSTRAIN_STRICT_MEASUREMENT_GOLDEN=1` to require exact
results for any tuple.

## Provenance / license

Source: http://festvox.org/cmu_arctic/

CMU ARCTIC was constructed at the Carnegie Mellon University Language
Technologies Institute. The corpus is free for any use; attribution is
appreciated. Only these utterances are included, prepared in the format the
trainer needs, purely as a test asset.

## Included utterances

- `arctic_a0001` — author of the danger trail philip steels etc
- `arctic_a0002` — not at this particular case tom apologized whittemore
- `arctic_a0003` — for the twentieth time that evening the two men shook hands
- `arctic_a0004` — lord but i'm glad to see you again phil
- `arctic_a0005` — will we ever forget it
- `arctic_a0006` — god bless em i hope i'll go on seeing them forever
- `arctic_a0007` — and you always want to see it in the superlative degree
- `arctic_a0008` — gad your letter came just in time
- `arctic_a0009` — he turned sharply and faced gregson across the table
- `arctic_a0010` — i'm playing a single hand in what looks like a losing game

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
