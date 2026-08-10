"""Pytest configuration and shared fixtures.

This module centralizes test configuration, including:
- C library availability detection
- Common fixtures for test data
- Skip markers for tests requiring the C library
"""

from __future__ import annotations

from pathlib import Path

import pytest

# Single source of truth for libpstrainc availability. Re-exported here so tests
# and other conftests can `from tests.clib import ...` or rely on the marker.
from tests.clib import (  # noqa: F401
    C_LIBRARY_AVAILABLE,
    c_library_available,
    require_clib_env,
    requires_c_library,
)

_PROJECT_ROOT = Path(__file__).parent.parent


# =============================================================================
# CI gate: fail loudly when the C library is required but missing
# =============================================================================


def pytest_configure(config: pytest.Config) -> None:
    """Abort collection if PSTRAIN_REQUIRE_CLIB is set but libpstrainc can't load.

    Without this, a misconfigured CI job (lib not built, wrong path) would
    silently skip the entire CFFI/parity tier and still report green.
    """
    if require_clib_env() and not c_library_available():
        raise pytest.UsageError(
            "PSTRAIN_REQUIRE_CLIB is set but libpstrainc could not be loaded. "
            "Build it first (e.g. 'make build-c') or unset PSTRAIN_REQUIRE_CLIB."
        )


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def project_root() -> Path:
    """Return the project root directory."""
    return _PROJECT_ROOT


@pytest.fixture
def sample_data_dir() -> Path:
    """Return the sample data directory."""
    return _PROJECT_ROOT / "pstrain" / "data" / "sample"


@pytest.fixture
def sample_audio(sample_data_dir: Path) -> Path:
    """Return path to sample audio file."""
    return sample_data_dir / "kevin-alice-16k.wav"


@pytest.fixture
def sample_transcript(sample_data_dir: Path) -> Path:
    """Return path to sample transcript file."""
    return sample_data_dir / "kevin-alice-16k.txt"


@pytest.fixture
def temp_model_dir(tmp_path: Path) -> Path:
    """Create and return a temporary directory for model files."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    return model_dir


@pytest.fixture
def basic_phones() -> list[str]:
    """Return a basic phone set for testing."""
    return ["SIL", "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D"]
