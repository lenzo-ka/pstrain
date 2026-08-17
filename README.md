# pstrain — Peace Train

[![Tests](https://github.com/lenzo-ka/pstrain/actions/workflows/tests.yml/badge.svg)](https://github.com/lenzo-ka/pstrain/actions/workflows/tests.yml)

The name **pstrain** is short for *PocketSphinx train*; it is pronounced *"peace train."*

pstrain is a toolkit for training continuous-density HMM/GMM acoustic models.
It combines a C library for the numerical work with a Python interface for
project setup, configuration, and training pipelines.

pstrain descends from
[CMU SphinxTrain](https://github.com/cmusphinx/sphinxtrain) and the wider
[CMU Sphinx](https://cmusphinx.github.io/) lineage. We are grateful to the
developers, researchers, institutions, and community members whose work made
this project possible.

pstrain rebuilds SphinxTrain with its C core vendored and its orchestration
reimplemented. On the shared CMU Arctic benchmark, pstrain under its shipped
defaults except for `split.test_count=0` (including multiple-pronunciation
training) shows no statistically significant regression on either cell versus
stock SphinxTrain. On SLT-55, aggregate WER and error count are identical even
though 26 of 55 per-utterance error rows differ. The
[benchmark pin](https://github.com/lenzo-ka/pstrain/blob/main/docs/benchmarks/arctic-pin.md) and its
[machine-readable record](https://github.com/lenzo-ka/pstrain/blob/main/docs/benchmarks/arctic-pin/record.json) preserve the
evidence and measurement conditions. Run `make verified` for the repository's
aggregate verification command; its Arctic gate authenticates the pinned
measurement conditions against the current checkout and re-derives the recorded
paired statistics from the stored per-utterance rows without re-decoding. The
end-to-end pstrain-versus-baseline WER comparison is produced by the full
`scripts/bench_arctic.py` benchmark run, not by `make verified`.

## What pstrain does

pstrain provides an in-process training pipeline for:

- acoustic feature extraction and corpus splitting;
- flat initialization and Baum–Welch training;
- context-independent and context-dependent models;
- Gaussian splitting, decision-tree state tying, and model packaging;
- forced alignment and PocketSphinx decoding with WER/CER evaluation; and
- multiple-pronunciation training, where pronunciation variants participate as
  parallel paths in each utterance's training graph.

pstrain supports macOS and Linux. Windows support is planned; today, WSL
provides a Linux environment on Windows. See the
[support policy](https://github.com/lenzo-ka/pstrain/blob/main/docs/support.md) for details.

## Quickstart

Install from PyPI:

```bash
pip install pstrain
```

The commands below use the small CMU ARCTIC fixture bundled in the
[repository](https://github.com/lenzo-ka/pstrain) for a complete local run; from
a checkout, install with the decoding extra instead:

```bash
python -m pip install ".[test]"
```

The repository includes a small CMU ARCTIC fixture for a complete local run.
Set up a project and train the default eight-Gaussian context-dependent model:

```bash
pstrain train /tmp/pstrain-demo \
  --audio tests/fixtures/mini_arctic/wav \
  --prompts tests/fixtures/mini_arctic/transcription.txt \
  --dictionary tests/fixtures/mini_arctic/dictionary.dict \
  --phoneset tests/fixtures/mini_arctic/phoneset.txt \
  --filler-dict tests/fixtures/mini_arctic/filler.dict \
  -j 1
```

The command stores separate typed training and decoder transcripts. Decode the
held-out set without a language model directly—no transcript conversion is needed:

```bash
pstrain test cd-8g --project-dir /tmp/pstrain-demo --no-lm
```

For a project of your own, `pstrain train --help` describes the accepted audio,
prompt, dictionary, phoneset, and configuration inputs. The lower-level
`setup`, `validate`, and `build` commands remain available for decomposed workflows. The
[getting-started guide](https://github.com/lenzo-ka/pstrain/blob/main/docs/getting-started.md) continues from project setup.

The default `pstrain train` target is `cd-8g`. Use `--target ci-1g` when only
the faster context-independent bootstrap model is wanted.

## Benchmark pin

The [ARCTIC benchmark pin](https://github.com/lenzo-ka/pstrain/blob/main/docs/benchmarks/arctic-pin.md) freezes the corpus,
training modes, model identities, decoder, and per-utterance measurements used
as the comparison baseline.

## Documentation

- [Getting started](https://github.com/lenzo-ka/pstrain/blob/main/docs/getting-started.md)
- [Input formats](https://github.com/lenzo-ka/pstrain/blob/main/docs/input-formats.md)
- [Examples](https://github.com/lenzo-ka/pstrain/blob/main/docs/examples.md)
- [API reference](https://github.com/lenzo-ka/pstrain/blob/main/docs/api/index.rst)
- [Design documents](https://github.com/lenzo-ka/pstrain/blob/main/docs/design/README.md)
- [Development guide](https://github.com/lenzo-ka/pstrain/blob/main/docs/development.md)
- [Support and dependency policy](https://github.com/lenzo-ka/pstrain/blob/main/docs/support.md)

The documentation can be built locally with `make -C docs html`.

## License

New pstrain code is available under the BSD 2-Clause license. CMU-derived code
retains its applicable CMU BSD-style license alongside the license for pstrain
modifications. See [LICENSE](https://github.com/lenzo-ka/pstrain/blob/main/LICENSE),
[csrc/LICENSE.sphinx](https://github.com/lenzo-ka/pstrain/blob/main/csrc/LICENSE.sphinx), and
[THIRD_PARTY_NOTICES.md](https://github.com/lenzo-ka/pstrain/blob/main/THIRD_PARTY_NOTICES.md) for the complete terms and
credits.
