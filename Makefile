# Build configuration
BUILD_DIR := build
CMAKE_BUILD_TYPE ?= Release

# Default target: build everything
.PHONY: all
all: build-c install-dev

# Build the C library (libpstrainc)
.PHONY: build-c
build-c:
	@echo "Building C library..."
	cmake -S csrc -B $(BUILD_DIR) \
		-DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE) \
		-DBUILD_SHARED_LIBS=ON
	cmake --build $(BUILD_DIR) --parallel

# Verify C library was built
.PHONY: check-c
check-c:
	@if [ ! -f $(BUILD_DIR)/libpstrainc.dylib ] && [ ! -f $(BUILD_DIR)/libpstrainc.so ]; then \
		echo "Error: C library not built. Run 'make build-c' first."; \
		exit 1; \
	fi
	@echo "C library found at $(BUILD_DIR)/"

# Test CFFI bindings work
.PHONY: check-cffi
check-cffi: check-c
	@echo "Testing CFFI bindings..."
	python -c "from pstrain.lib import _pstrainc; lib = _pstrainc.get_lib(); print('CFFI OK: loaded', lib)"

.PHONY: install
install: build-c
	pip install -e .

.PHONY: install-dev
install-dev: build-c
	pip install -e ".[dev]"

.PHONY: test
test: check-c
	pytest

.PHONY: lint
lint:
	ruff check pstrain tests
	mypy pstrain

.PHONY: format
format:
	ruff format pstrain tests
	ruff check --fix pstrain tests

.PHONY: clean
clean:
	rm -rf $(BUILD_DIR) dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf docs/_build

.PHONY: clean-c
clean-c:
	rm -rf $(BUILD_DIR)

.PHONY: docs-gen
docs-gen:
	python -c "from pstrain.lib.config import generate_rst_docs; open('docs/api/config-reference.rst', 'w').write(generate_rst_docs())"

.PHONY: docs
docs: docs-gen
	cd docs && $(MAKE) html

.PHONY: docs-clean
docs-clean:
	cd docs && $(MAKE) clean
