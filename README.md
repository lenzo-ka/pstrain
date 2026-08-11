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
Set up a project and train a one-Gaussian context-independent model:

```bash
pstrain setup /tmp/pstrain-demo \
  --audio tests/fixtures/mini_arctic/wav \
  --transcription tests/fixtures/mini_arctic/transcription.txt \
  --dictionary tests/fixtures/mini_arctic/dictionary.dict \
  --phoneset tests/fixtures/mini_arctic/phoneset.txt \
  --filler-dict tests/fixtures/mini_arctic/filler.dict \
  --validate

pstrain build ci-1g --project-dir /tmp/pstrain-demo -j 1
```

Prepare the held-out transcript in the decoder's Sphinx transcript format,
then decode it without a language model:

```bash
awk '{ id=$1; $1=""; sub(/^ /, ""); print "<s> " $0 " </s> (" id ")" }' \
  /tmp/pstrain-demo/experiments/default/etc/test.transcription \
  > /tmp/pstrain-demo/experiments/default/etc/test.sphinx.transcription
mv /tmp/pstrain-demo/experiments/default/etc/test.sphinx.transcription \
  /tmp/pstrain-demo/experiments/default/etc/test.transcription

pstrain test ci-1g --project-dir /tmp/pstrain-demo --no-lm
```

For a project of your own, `pstrain setup --help` describes the accepted audio,
transcription, dictionary, phoneset, and configuration inputs. The
[getting-started guide](docs/getting-started.md) continues from project setup.

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
