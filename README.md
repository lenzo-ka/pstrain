# pstrain — the Peace Train

[![Tests](https://github.com/lenzo-ka/pstrain/actions/workflows/tests.yml/badge.svg)](https://github.com/lenzo-ka/pstrain/actions/workflows/tests.yml)

pstrain is a toolkit for training HMM/GMM acoustic models, in the lineage of CMU
SphinxTrain but rebuilt to be cleaner and well organized: an efficient C
substrate (`libpstrainc`) driven in-process from Python via CFFI, orchestrated by a
small Python pipeline runner. No shell-outs, no Perl.

> **Status: alpha.** The continuous-model training backbone works end to end;
> APIs and on-disk layouts may still change. See [Known issues](#known-issues).

## What it does

The full continuous-density training pipeline runs in-process against the C
library:

```
features → flat init → CI (1→2→4→8 Gaussians)
        → CD-untied (triphones) → questions → decision trees → state tying
        → CD-tied (1→2→…→32 Gaussians) → package
```

Plus forced alignment, a language-model build, and PocketSphinx-based decoding
for WER/CER evaluation. The heavy numerical work (Baum-Welch, Gaussian
splitting, decision-tree clustering, feature extraction) lives in `libpstrainc`;
Python owns orchestration, configuration, and I/O.

## Install

Building the wheel compiles the C library via
[scikit-build-core](https://scikit-build-core.readthedocs.io/) (needs CMake ≥
3.16 and a C compiler):

```bash
pip install .              # or: pip install -e ".[dev,test]" for development
```

Optional extras: `test` (PocketSphinx + jiwer, for `pstrain test`), `dev` (pytest,
ruff, mypy), `docs` (Sphinx).

## Quickstart

Point `pstrain setup` at audio, a `<fileid> <words…>` transcription, a
pronunciation dictionary, and optionally a phoneset and filler dictionary,
then build a target:

```bash
pstrain setup myproject \
    --audio        wav/ \
    --transcription transcription.txt \
    --dictionary   cmudict.dict

pstrain build ci-1g --project-dir myproject     # monophone, 1 Gaussian/state
pstrain build cd-8g --project-dir myproject     # full CD pipeline, 8 Gaussians
```

Useful flags: `--dry-run` prints the resolved task plan, `-j N` parallelizes
the feature-extraction fan-out, `--config <name>` selects a named profile from
`etc/configs.yaml` (`default`, `wideband`, `telephone`, …).

When `--phoneset` is omitted, setup extracts one covering both the dictionary
and filler dictionary (notably `SIL`) so that every model state is trainable.

`tests/fixtures/mini_arctic/` is a tiny, self-contained example corpus (used by
the end-to-end test).

## Command surface

| Command | Purpose |
|---|---|
| `pstrain setup` | Scaffold a project from audio + transcription + dictionary |
| `pstrain build <target>` | Build a model target (`ci-1g`…`ci-8g`, `cd-untied`, `cd-1g`…`cd-32g`) |
| `pstrain features` | Extract MFCC features |
| `pstrain split` | Train/test split |
| `pstrain flat` | Flat (uniform) model initialization |
| `pstrain align` | Forced alignment against a trained model |
| `pstrain test` | Decode and report WER/CER |
| `pstrain compare` / `pstrain info` / `pstrain validate-project` | Inspection and validation |

## Development

```bash
make build-c            # configure + build libpstrainc into build/
pip install -e ".[dev,test]"
make test               # pytest
make lint               # ruff + mypy
```

Set `PSTRAIN_REQUIRE_CLIB=1` when running the tests to turn "C library not built"
from a skip into a hard failure (used in CI so the CFFI/parity tier can't be
silently skipped). The C smoke tests run under `ctest --test-dir build`.

## Repository layout

```
pstrain/          Python package (cli/, api/, lib/ with the pipeline + CFFI bridge)
csrc/         C sources: libs/libpstrain (the new session layer) + vendored
              SphinxTrain/SphinxBase/Sphinx-3 under libs/ and programs/
tests/        Unit, CFFI, parity, and end-to-end training tests
docs/         Design notes and reference documentation
etc/          Named configuration profiles (configs.yaml)
```

## Known issues

This is an early alpha; a few rough edges are known and tracked:

- The train/test split extracts the file id as the first whitespace token, so a
  Sphinx-format transcription (`<s> … </s> (id)`) is mis-parsed even though the
  transcription *reader* accepts that format. Use `<fileid> <words…>`.
- Two configuration systems coexist; the pydantic `etc/config.yaml` does not yet
  drive training (the pipeline reads `etc/configs.yaml`).
- **Platforms:** Linux and macOS are supported (wheels + CI). Windows is not yet
  supported — the vendored CMU Sphinx C does not build under MSVC (POSIX-only
  `drand48` and some unresolved symbols); build from source on WSL for now.

## Acknowledgements

pstrain builds on decades of work by the [CMU Sphinx](https://github.com/cmusphinx)
project. The vendored C under `csrc/` derives from CMU SphinxTrain, SphinxBase,
and Sphinx-3, and is used under the CMU BSD-style license.

## License

The pstrain Python package and the new C session layer (`csrc/libs/libpstrain/`) are
licensed under the BSD 2-Clause license — see [LICENSE](LICENSE).

The vendored CMU Sphinx C code under `csrc/` has been modified as part of pstrain
and is dual-licensed: the original portions under the CMU BSD-style license
(see [`csrc/LICENSE.sphinx`](csrc/LICENSE.sphinx)) and the pstrain modifications
under BSD 2-Clause. See [`csrc/NOTICE.md`](csrc/NOTICE.md) for the full breakdown.
