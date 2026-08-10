# Development

## Building Documentation

```bash
cd docs
make html
```

Or using sphinx-build directly:

```bash
sphinx-build -b html docs docs/_build/html
```

## Running Tests

```bash
pytest
```

## Code Quality

```bash
# Linting
ruff check .

# Type checking
mypy pstrain

# Formatting
ruff format .
```
