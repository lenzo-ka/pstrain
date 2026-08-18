# Changelog

Release-relevant changes are recorded here. The project is currently an alpha;
the version in `pyproject.toml` is authoritative.

## 0.2.0 - 2026-08-18

- Added a structured `pstrain.api` as the supported programmatic entry point;
  the command-line interface now reaches the training library only through it,
  enforced by a boundary check in the verified build (2026-08-18).
- Added a `bench` extra providing PocketSphinx for the `pip-en-us` benchmark
  band, and corrected the install hint the benchmark error printed (2026-08-17).
- Split the README quickstart into "from PyPI" and "from a checkout" so a
  pip-installed user can run it without the repository fixtures (2026-08-17).
- `make_quests` now fails loudly on degraded input statistics instead of writing
  a plausible-looking question file (2026-08-17).
- Relocated the Arctic benchmark pin evidence under `evidence/` so a record
  change runs its authenticating gate, and added a generated coverage statement
  and an oracle-provenance note (2026-08-17, 2026-08-18).
- The package version now falls back to `0.0.0+unknown` instead of raising at
  import when the package is relocated without its metadata (2026-08-17).

## 0.1.1 - 2026-08-17

- Added self-contained Windows wheels with bundled native dependencies
  (2026-08-17).
- Extended build and PyPI release workflows to cover Windows wheels
  (2026-08-17).
- Added PE artifact checks to the native floating-point contract gate
  (2026-08-17).

## 0.1.0 - 2026-08-17

- Imported the original pstrain/SphinxTrain-derived codebase (2026-07-03).
- Adopted the BSD 2-Clause license for new pstrain code while preserving the
  notices and license terms of the CMU-derived C sources (2026-07-08).
- Added and pinned the reproducible CMU Arctic benchmark baseline and its
  immutable corpus metadata (2026-08-11).
- Contained fatal native CFFI operations behind a reusable worker boundary and
  completed the audited CFFI operation surface (2026-08-10 through 2026-08-11).
