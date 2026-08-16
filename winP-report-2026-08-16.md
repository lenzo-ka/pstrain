# Windows PocketSphinx pipe-link report — August 16, 2026

## Scope and parity classification

This is a build-only compatibility change. It does not change training
arithmetic, algorithms, model data, runtime behavior, output formats, or any
parity-sensitive path. There is no numeric or parity impact.

## MSVC link fix

The pinned PocketSphinx v5.1.1 dependency compiles `src/util/pio.c` into its
`pocketsphinx` shared-library target. That file calls the POSIX spellings
`popen` and `pclose`, but the MSVC CRT exports `_popen` and `_pclose`. The
integration now adds private `popen=_popen` and `pclose=_pclose` compile
definitions to the `pocketsphinx` target when the compiler is MSVC.

The workaround is limited to the dependency target that compiles `pio.c`. It
does not alter pstrain's compatibility headers, patch the verified fetched
source, affect MinGW or non-Windows builds, or use a force-link option.

## Upstream routing

A local handoff note is recorded at
`untracked/upstream/pocketsphinx-popen.md` for Kevin to route through
`pocketsphinx-prs`. No upstream issue or pull request was filed.

## Validation

- macOS AppleClang: `make verified` passed, including 718 selected Python tests
  (with 1 skipped and 1 deselected), all 10 CTest tests, the floating-point
  contraction gate, configuration and generated-document checks, Arctic pin
  checks, Ruff, mypy, and format checks. The bundled PocketSphinx shared
  library compiled and linked successfully on the unaffected non-MSVC path.
- Windows link confirmation: CI-only. The shrub environment cannot perform the
  complete Windows link, so the coordinator should dispatch the Windows probe
  after this branch is pushed.
