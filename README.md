# pstrain — the Peace Train

[![Tests](https://github.com/lenzo-ka/pstrain/actions/workflows/tests.yml/badge.svg)](https://github.com/lenzo-ka/pstrain/actions/workflows/tests.yml)

pstrain is a toolkit for training HMM/GMM acoustic models, in the lineage of CMU
SphinxTrain but rebuilt to be cleaner and well organized: an efficient C
substrate (`libpstrainc`) driven in-process from Python via CFFI, orchestrated by a
small Python pipeline runner.

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
the feature-extraction fan-out (the default leaves two CPUs free; pass an
explicit full CPU count to opt into the whole machine), `--config <name>` selects a named profile from
`etc/configs.yaml` (`default`, `wideband`, `telephone`, …).

Named profiles may set `runner.jobs` and `runner.nice` (default 5; 0 disables
the POSIX niceness increment). Ctrl-C, SIGTERM, or `Pipeline.cancel()` aborts
an active fan-out: pending work is cancelled and worker process groups,
including native helpers, receive SIGTERM followed by SIGKILL after a one-second
grace period. Completed tasks retain their completion markers, so a rerun starts
at the first unfinished task. Cancellation is checked between inline tasks;
an inline task already running with `jobs=1` finishes before cancellation.

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
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_CLI=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure --no-tests=error
pip install -e ".[dev,test,docs]"
PSTRAIN_REQUIRE_CLIB=1 pytest
ruff check pstrain tests && ruff format --check pstrain tests
mypy pstrain
pre-commit run --all-files
```

These repository-root CMake commands are the canonical native build entry
point; `make build-c`, `make test`, and `make lint` are shortcuts for the same
build and checks.

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
- **Platforms:** Linux and macOS are supported (wheels + CI). Windows/MSVC is
  explicitly future work; build through WSL for now. See
  [support and dependency policy](docs/support.md).

## Acknowledgements

pstrain descends from decades of generous work by the
[CMU Sphinx](https://github.com/cmusphinx) community and carries CMU
SphinxTrain, SphinxBase, and Sphinx-3 code forward in its C implementation. We
are grateful to the people and institutions that made that lineage possible;
see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for its notices and credits.

## License

New pstrain code is licensed under the BSD 2-Clause license. CMU-derived code
carries both layers: CMU's BSD-style license applies to the base, and Kevin
Lenzo's BSD 2-Clause license applies to the pstrain modifications, with git
history recording those changes. See [LICENSE](LICENSE),
[csrc/LICENSE.sphinx](csrc/LICENSE.sphinx), and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the full structure.
