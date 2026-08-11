# CMU Arctic BM1 benchmark

This harness downloads and authenticates the CMU Arctic SLT, BDL, RMS, and
CLB packed voices, prepares two acoustic-model projects, loads the committed
canonical unigram LM, trains the multipron-off and multipron-on
PIN configurations through 8 Gaussians, decodes SLT-55 and the 3,395-utterance
cross-speaker set, reports WER and OOV counts, audits every inter-pass
likelihood delta, and compares each result cell with a benchmark record.

**The PIN band uses the committed Arctic dictionary and committed unigram LM.**
Every recorded number, including the ratified `off/big` +1.0-point floor, was
measured with those exact resources. The earlier pip-dictionary banked intent
was “matched at decode time”; the dictionary actually matched for every
measurement was the Arctic dictionary. This implementation honors that intent
over the letter of the earlier note. Kevin reviews this interpretation at the
pin PR. The pip PocketSphinx `en-us` dictionary remains available only as the
explicitly labeled alternative band `--band=pip-en-us`; it is not the PIN band.

Install the project with its test dependencies, then run:

```console
python scripts/bench_arctic.py --record docs/benchmarks/<record>.json
```

The committed record is produced by the PIN run with
`--emit-record docs/benchmarks/<record>.json`. Until that pin run has been
performed, use `--no-compare` for an exploratory run or `--emit-record` for the
pin candidate; there is no placeholder record to compare against. Comparison
authenticates all corpus archives, transcript files, dictionary and LM, the
complete training/decode conditions, and engine identity before examining WER.
Each record retains compact per-utterance word/error rows. Comparison aligns
their IDs with the current run and bootstraps the cross-run paired WER delta
(100,000 percentile resamples, speaker-stratified for the cross-speaker cell).
The upper confidence bound must be at or below parity, except for the ratified
one-point `off/big` floor. Engine identity includes dirty tracked-source state,
the native library, Python, PocketSphinx, and the selected decode dictionary.
Engine drift requires the explicit `--allow-engine-drift` override.

The gate keeps zero failed decode alignments as a hard requirement. Named,
documented terminal training skips are permitted only through the benchmark's
known-skips manifest and are always reported in each affected record cell.
The measured on-mode parity anchor itself carried nine terminal skip events;
these documented skips are a property of the measured band, not a new
concession.

`PSTRAIN_BENCH_CACHE`
selects the archive cache; authenticated archives there are reused. The work
tree defaults to `.pstrain-benchmark/arctic`. Expect approximately 8–16 hours
on a current laptop and 8 GB of free disk space. `-j N` controls feature and
tree parallelism. Cache reuse verifies the full WAV name/size inventory and a
deterministic 32-file hash sample per voice; `--deep-verify` hashes every WAV.

`data/pin-train.transcription` and `data/pin-train.fileids` are the exact
1,043-utterance upstream training inputs used by the parity program (SHA-256
`28788cd1ce2269d344b50420d74007fa8c443778680724f6334e2712ea110959` and
`8ce9a55c5929f6f86579ee1b244c38fd4d0a9d41e436e2057337f74c1bb4d631`).
`data/full-slt.transcription` retains the normalized 1,132-prompt SLT set for
corpus-wide feature extraction and provenance (SHA-256
`cce3d341c02445d3aa91453f1d7f0fa3097f6ab6df76264336277ca2c54f3085`), but
the harness trains only on the 1,043-utterance pin corpus.
`data/slt55.transcription` is the frozen same-speaker held-out cell: none of
its 55 IDs is in the 1,043-utterance training set. It remains matched on both
engines throughout the parity program and comes from
`upstream-cmu-arctic-slt/etc/cmu_arctic_slt_test.transcription` (SHA-256
`1de4e31c934ea6c5cd414307b8cf4f71c0d846adfd68edf04c4ececaafa4c532`).
`data/big.transcription` is the frozen BDL/RMS/CLB reference from
`experiments/v5/eval/big.transcription` (SHA-256
`5737c7296d491df39aaa2db24ab03e6acf6e058c10bd7d6b331ffdbf242c5b58`).
These normalized files are committed because each voice archive contains only
Scheme-formatted, unnormalized `etc/txt.done.data`.

`data/cmu_arctic_slt.dict` is the ratified CMUdict-derived Arctic dictionary
(SHA-256 `24ff2852a707b63f499fd968294d5e4c02d44e0eb1ec511e40be1f380d785846`).
Its CMU license is included at `csrc/LICENSE.sphinx`. `data/training-unigram.lm`
is the canonical measured-band LM (SHA-256
`2cf11ab0474a0bdd165cbee59db674b05764fdb00bf6f9824c0dccce571637b5`).
The current `arpabo` builder remains tested against the 1,043-utterance pin
training corpus but produces `2c75cacb19b4...`, not the preserved canonical
LM's bytes; the committed measured-band LM therefore remains canonical.

The committed serializations match their consumers. The pin training transcript
uses normalized leading-ID form (`fileid text`), as required by the split,
flat-initialization control, and Baum-Welch path. Decoder references use Sphinx
form (`<s> text </s> (fileid)`), including a voice prefix for the cross-speaker
cell. The retained full set also uses normalized leading-ID form. Consumption
tests enforce the training boundary while the decoder-format tests remain.

An empty `exclusion_schedule` is correct for the pin band: it records pstrain's
honest whole-set-training behavior, while composition emulation was only a
diagnostic instrument.
