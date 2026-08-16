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
	python scripts/check_fp_contract.py $(BUILD_DIR)
	PSTRAIN_REQUIRE_CLIB=1 pytest

# Disable any shared pstrain editable-install finder, then let the suite's
# subject identity gate prove both the Python and native artifact paths.
.PHONY: verified-test
verified-test: check-c
	ctest --test-dir $(BUILD_DIR) --output-on-failure --no-tests=error
	python scripts/check_fp_contract.py $(BUILD_DIR)
	PSTRAIN_REQUIRE_CLIB=1 python scripts/run_verified_tests.py

.PHONY: fp-contract-check
fp-contract-check: check-c
	python scripts/check_fp_contract.py $(BUILD_DIR)

.PHONY: lint
lint:
	ruff check pstrain tests
	mypy pstrain

# Canonical local merge-gate verification. Keep the Ruff format scope aligned
# with the blocking lint job in .github/workflows/tests.yml.
.PHONY: verified
verified: ambient-import-check build-c verified-test config-check lint
	ruff format --check pstrain tests

# This is intentionally part of the in-tree verified flow, not the installed-
# package CI flow.  Installed-wheel jobs legitimately resolve from site-packages.
.PHONY: ambient-import-check
ambient-import-check:
	python scripts/check_ambient_import.py

# These verified legs may run concurrently with one another, but none may begin
# until the ambient-import gate has completed successfully.
build-c verified-test config-check lint: | ambient-import-check

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
	python scripts/run_verified_tests.py --exec "from pstrain.lib.config import generate_rst_docs; open('docs/api/config-reference.rst', 'w').write(generate_rst_docs())"

.PHONY: cffi-exports-gen
cffi-exports-gen:
	python scripts/generate_cffi_exports.py

.PHONY: cffi-exports-check
cffi-exports-check:
	python scripts/generate_cffi_exports.py --check

docs-gen cffi-exports-check: | ambient-import-check

.PHONY: config-check
config-check: cffi-exports-check
	python scripts/run_verified_tests.py tests/test_config.py \
		tests/test_pipeline_runner.py::test_config_reference_names_runner_keys_used_by_context \
		tests/test_decoder_config.py \
		tests/test_features.py::TestFeatureExtractor::test_new_front_end_options_change_produced_features
	$(MAKE) docs-gen
	git diff --exit-code -- docs/api/config-reference.rst
	python scripts/check_arctic_pin.py
	python scripts/regenerate_arctic_paired_analysis.py --check

.PHONY: contract-docs-gen
contract-docs-gen:
	python -c "from pstrain.lib.contract_docs import write_bw_sharding_contract; write_bw_sharding_contract()"

.PHONY: contract-check
contract-check:
	pytest tests/test_contract_docs.py
	$(MAKE) contract-docs-gen
	git diff --exit-code -- docs/design/bw-sharding-contract.md

.PHONY: docs
docs:
	cd docs && $(MAKE) html

.PHONY: docs-clean
docs-clean:
	cd docs && $(MAKE) clean
