# Development

## Canonical verification command

Run the complete local verification verdict:

```bash
make verified
```

`make verified` is **the** command to cite when reporting a branch as green. It
first brings the native build up to date, then runs the subject-identity-aware
runtime suite (`verified-test`), configuration and generated-file checks
(`config-check`), Ruff lint, mypy, and the Ruff format check. The aggregate is
fail-fast: a red verdict identifies the first failing constituent, not every
failure that may be present. Ruff deliberately covers `pstrain` and `tests`,
matching the blocking CI lint job; repository-wide Ruff also includes
pre-existing vendored and utility scripts outside that scope.

The aggregate invokes the checks directly instead of running pre-commit. This
keeps verification non-mutating and avoids hook-environment setup and unrelated
repository-hygiene hooks. The tradeoff is that the Ruff format command and its
scope are repeated in the Makefile and CI and must remain aligned.

## Individual build and test commands

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_CLI=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure --no-tests=error
pip install -e ".[dev,test,docs]"
PSTRAIN_REQUIRE_CLIB=1 pytest
```

Always configure CMake from the repository root. The Makefile provides
`build-c` and `test` shortcuts for these individual steps. `make test` is a
runtime-suite shortcut, not a complete verification verdict; use `make
verified` for that verdict.

## Changing the native interface

Python reaches `libpstrainc` through the declarations in the `CDEF` string in
`pstrain/lib/_cffi/cdef.py`. Because cffi resolves symbols lazily, a library
built from older sources can load and then fail much later. Two load-time
checks in `pstrain/lib/_cffi/core.py` reject such a stale library instead:

- **Interface fingerprint.** `make cffi-exports-gen` writes a hash of the exact
  `CDEF` text into `csrc/libs/libpstrain/pstrain_interface_fingerprint.h`,
  alongside the linker export lists. The library returns it from
  `pstrain_interface_fingerprint()`, and Python compares it with the hash of the
  `CDEF` it loaded. After any edit to `CDEF`, rerun `make cffi-exports-gen` and
  commit the regenerated files; `make config-check` fails until you do. No
  version bump is needed for an added, removed, or re-typed function or a
  changed struct.
- **ABI version.** `PSTRAIN_ABI_VERSION` in
  `csrc/libs/libpstrain/pstrain_align.h` must equal the same constant in
  `core.py`. Bump both only for a change of meaning behind unchanged
  declarations, such as a function whose signature stays the same while its
  return contract, ownership rule, or units change. The fingerprint cannot see
  that kind of change.

## Building documentation

```bash
make docs
```

This runs the existing Sphinx HTML build. Configuration-reference generation is
available separately as `make docs-gen`.

## Code Quality

```bash
# Linting
ruff check pstrain tests

# Type checking
mypy pstrain

# Formatting
ruff format --check pstrain tests

# All repository hooks
pre-commit run --all-files
```
