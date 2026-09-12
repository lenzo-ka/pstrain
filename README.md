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

## What pstrain does

pstrain provides a training pipeline for:

- acoustic feature extraction and corpus splitting;
- flat initialization and Baum–Welch training;
- context-independent and context-dependent models;
- Gaussian splitting, decision-tree state tying, and model packaging;
- forced alignment and PocketSphinx decoding with WER/CER evaluation; and
- multiple-pronunciation training, where pronunciation variants participate as
  parallel paths in each utterance's training graph.

## Quickstart

Start with the [terminal walkthrough](https://github.com/lenzo-ka/pstrain/blob/main/docs/getting-started.md):
it covers Python and compiler prerequisites, an isolated installation, and a
small training run using the included audio, followed by decoding and packaging.
No Python programming or corpus download is needed for that example.

Training requires **macOS or Linux and Python 3.11+**. Native Windows supports
the package and native tools, but cannot run the training pipeline; see the
[support policy](https://github.com/lenzo-ka/pstrain/blob/main/docs/support.md).

The guide distinguishes building this checkout from installing a published
release, and starts with the faster `ci-1g` model before the default `cd-8g`.

## Benchmark pin

The [ARCTIC benchmark pin](https://github.com/lenzo-ka/pstrain/blob/main/docs/benchmarks/arctic-pin.md) freezes the corpus,
training modes, model identities, decoder, and per-utterance measurements used
as the comparison baseline.

pstrain rebuilds SphinxTrain with its C core vendored and its orchestration
reimplemented. On the shared CMU Arctic benchmark, pstrain under its shipped
defaults except for `split.test_count=0` (including multiple-pronunciation
training) shows no statistically significant regression on either cell versus
the preserved upstream oracle, historically attributed to stock SphinxTrain;
see its [provenance limits](https://github.com/lenzo-ka/pstrain/blob/main/docs/benchmarks/oracle-provenance.md).
The arms share one matched decode path but are **NOT COMPARABLE** for
implementation attribution because the oracle's producing lineage is unknown.
The two cells carry different weight. The 3,395-utterance set is cross-speaker,
so it measures what the model does on voices it was not trained on: there
pstrain records 22,703 errors (75.7221% WER) against 22,638 (75.5053%) for that
oracle — behind by 0.2168 percentage points, a margin the data cannot separate
from zero (95% paired interval [-0.2414, +0.6713]). SLT-55 is a
same-speaker held-out cell of 55 utterances; there pstrain is ahead, 139 errors
(27.0955% WER) against 144 (28.0702%), with 23 of 55 per-utterance error rows
differing, but its interval of [-3.6965, +1.6575] is consistent with anything
from a 3.7-point gain to a 1.7-point regression and so settles little on its
own. The larger cross-speaker set is where the evidence is. The
[benchmark pin](https://github.com/lenzo-ka/pstrain/blob/main/docs/benchmarks/arctic-pin.md) and its
[machine-readable record](https://github.com/lenzo-ka/pstrain/blob/main/evidence/arctic-pin/record.json) preserve the
evidence and measurement conditions. Run `make verified` for the repository's
aggregate verification command; its Arctic gate authenticates the pinned
measurement conditions against the current checkout and re-derives the recorded
paired statistics from the stored per-utterance rows without re-decoding. The
end-to-end pstrain-versus-baseline WER comparison is produced by the full
`scripts/bench_arctic.py` benchmark run, not by `make verified`.

## Documentation

- [Getting started](https://github.com/lenzo-ka/pstrain/blob/main/docs/getting-started.md)
- [Package replacement safety](https://github.com/lenzo-ka/pstrain/blob/main/docs/package-safety.md)
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
