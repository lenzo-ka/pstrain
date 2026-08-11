# Development

## Canonical build and test commands

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_CLI=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure --no-tests=error
pip install -e ".[dev,test,docs]"
PSTRAIN_REQUIRE_CLIB=1 pytest
```

Always configure CMake from the repository root. The Makefile provides
equivalent `build-c` and `test` shortcuts.

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
