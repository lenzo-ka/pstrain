# pstrain — the Peace Train

[![Tests](https://github.com/lenzo-ka/pstrain/actions/workflows/tests.yml/badge.svg)](https://github.com/lenzo-ka/pstrain/actions/workflows/tests.yml)

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
[benchmark pin](docs/benchmarks/arctic-pin.md) and its
[machine-readable record](docs/benchmarks/arctic-pin/record.json) preserve the
evidence and measurement conditions. Run `make verified` for the repository's
aggregate verification command; its Arctic gate compares current pstrain with
the pinned pstrain baseline, not with upstream SphinxTrain.

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
[support policy](docs/support.md) for details.

## Quickstart

From a checkout, install pstrain and the decoding dependencies:

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
[getting-started guide](docs/getting-started.md) continues from project setup.

The default `pstrain train` target is `cd-8g`. Use `--target ci-1g` when only
the faster context-independent bootstrap model is wanted.

## Benchmark pin

The [ARCTIC benchmark pin](docs/benchmarks/arctic-pin.md) freezes the corpus,
training modes, model identities, decoder, and per-utterance measurements used
as the comparison baseline.

## Documentation

- [Getting started](docs/getting-started.md)
- [Examples](docs/examples.md)
- [API reference](docs/api/index.rst)
- [Design documents](docs/design/README.md)
- [Development guide](docs/development.md)
- [Support and dependency policy](docs/support.md)

The documentation can be built locally with `make -C docs html`.

## License

New pstrain code is available under the BSD 2-Clause license. CMU-derived code
retains its applicable CMU BSD-style license alongside the license for pstrain
modifications. See [LICENSE](LICENSE),
[csrc/LICENSE.sphinx](csrc/LICENSE.sphinx), and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the complete terms and
credits.
