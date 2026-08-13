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
	cmake -S . -B $(BUILD_DIR) \
		-DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE) \
		-DBUILD_SHARED_LIBS=ON
	cmake --build $(BUILD_DIR) --parallel

# Verify C library was built
.PHONY: check-c
check-c:
	@if [ ! -f $(BUILD_DIR)/lib/libpstrainc.dylib ] && [ ! -f $(BUILD_DIR)/lib/libpstrainc.so ]; then \
		echo "Error: C library not built. Run 'make build-c' first."; \
		exit 1; \
	fi
	@echo "C library found at $(BUILD_DIR)/lib/"

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
	ctest --test-dir $(BUILD_DIR) --output-on-failure --no-tests=error
	PSTRAIN_REQUIRE_CLIB=1 pytest

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

.PHONY: config-check
config-check:
	pytest tests/test_config.py \
		tests/test_pipeline_runner.py::test_config_reference_names_runner_keys_used_by_context \
		tests/test_decoder_config.py \
		tests/test_features.py::TestFeatureExtractor::test_new_front_end_options_change_produced_features
	$(MAKE) docs-gen
	git diff --exit-code -- docs/api/config-reference.rst

.PHONY: docs
docs:
	cd docs && $(MAKE) html

.PHONY: docs-clean
docs-clean:
	cd docs && $(MAKE) clean
