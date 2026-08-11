# CMU Arctic BM1 benchmark

This harness downloads and authenticates the CMU Arctic SLT, BDL, RMS, and
CLB packed voices, prepares two acoustic-model projects, builds a unigram LM
from the committed training prompts, trains the multipron-off and multipron-on
PIN configurations through 8 Gaussians, decodes SLT-55 and the 3,395-utterance
cross-speaker set, reports WER and OOV counts, audits every inter-pass
likelihood delta, and compares each result cell with a benchmark record.

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
Engine drift requires the explicit `--allow-engine-drift` override.

`PSTRAIN_BENCH_CACHE`
selects the archive cache; authenticated archives there are reused. The work
tree defaults to `.pstrain-benchmark/arctic`. Expect approximately 8–16 hours
on a current laptop and 8 GB of free disk space. `-j N` controls feature and
tree parallelism.

`data/train.transcription` is the normalized 1,132-prompt SLT training corpus
from the parity workspace's SLT prompt set (SHA-256
`cce3d341c02445d3aa91453f1d7f0fa3097f6ab6df76264336277ca2c54f3085`).
`data/slt55.transcription` is the frozen same-speaker resubstitution cell by
design: all 55 IDs are in the SLT training set, matched on both engines
throughout the parity program, as in the upstream Arctic recipes. It is from
`upstream-cmu-arctic-slt/etc/cmu_arctic_slt_test.transcription` (SHA-256
`1de4e31c934ea6c5cd414307b8cf4f71c0d846adfd68edf04c4ececaafa4c532`).
`data/big.transcription` is the frozen BDL/RMS/CLB reference from
`experiments/v5/eval/big.transcription` (SHA-256
`5737c7296d491df39aaa2db24ab03e6acf6e058c10bd7d6b331ffdbf242c5b58`).
These normalized files are committed because each voice archive contains only
Scheme-formatted, unnormalized `etc/txt.done.data`.

The two committed serializations are intentional. `train.transcription` uses
the trainer's leading-ID form (`fileid text`). Decoder references use Sphinx
form (`<s> text </s> (fileid)`), including a voice prefix for the cross-speaker
cell. Format tests enforce each file's contract.
