# Development

## Canonical verification command

Build the C library, then run the complete local verification verdict:

```bash
make build-c
make verified
```

`make verified` is **the** command to cite when reporting a branch as green. It
runs the subject-identity-aware runtime suite (`verified-test`), configuration
and generated-file checks (`config-check`), Ruff lint, mypy, and the Ruff format
check. Ruff deliberately covers `pstrain` and `tests`, matching the blocking CI
lint job; repository-wide Ruff also includes pre-existing vendored and utility
scripts outside that scope.

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
