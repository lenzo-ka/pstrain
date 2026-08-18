"""Tests for the public environment diagnostics API."""

import pytest

from pstrain.api import diagnostics


def test_native_library_available_when_library_is_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(diagnostics, "_find_library", lambda: object())

    assert diagnostics.native_library_available() is True


def test_native_library_available_when_library_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_library() -> None:
        raise RuntimeError("not found")

    monkeypatch.setattr(diagnostics, "_find_library", missing_library)

    assert diagnostics.native_library_available() is False
