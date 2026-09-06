"""Tests for the public diagnostics API."""

import json
from pathlib import Path

import pytest

from pstrain.api import diagnostics

_PASS = {
    "pass": 1,
    "total_log_likelihood": -10.0,
    "total_frames": 5,
    "per_frame_log_likelihood": -2.0,
    "signed_convergence_delta": None,
    "stop_decision": "continued",
}


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


def test_load_bw_telemetry(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    document = {"schema_version": 2, "passes": [_PASS]}
    (model_dir / "bw_telemetry.json").write_text(json.dumps(document))

    assert diagnostics.load_bw_telemetry(model_dir) == document


def test_load_bw_telemetry_accepts_schema_version_one(tmp_path: Path) -> None:
    document = {"schema_version": 1, "passes": [_PASS]}
    (tmp_path / "bw_telemetry.json").write_text(json.dumps(document))

    assert diagnostics.load_bw_telemetry(tmp_path) == document


def test_load_bw_telemetry_reports_missing_and_malformed_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="telemetry file not found"):
        diagnostics.load_bw_telemetry(tmp_path)

    (tmp_path / "bw_telemetry.json").write_text("not JSON")
    with pytest.raises(ValueError, match="Could not read Baum-Welch telemetry"):
        diagnostics.load_bw_telemetry(tmp_path)

    (tmp_path / "bw_telemetry.json").write_text(
        json.dumps({"schema_version": 2, "passes": "wrong"})
    )
    with pytest.raises(ValueError, match="expected a passes list"):
        diagnostics.load_bw_telemetry(tmp_path)


@pytest.mark.parametrize(
    ("document", "message"),
    [
        ({"passes": []}, "Unsupported.*schema"),
        ({"schema_version": 3, "passes": []}, "Unsupported.*schema"),
        ({"schema_version": 2, "passes": [1]}, "pass 1 must be an object"),
        ({"schema_version": 2, "passes": [{"pass": 1}]}, "pass 1 is missing"),
    ],
)
def test_load_bw_telemetry_rejects_invalid_schema(
    tmp_path: Path, document: object, message: str
) -> None:
    (tmp_path / "bw_telemetry.json").write_text(json.dumps(document))

    with pytest.raises(ValueError, match=message):
        diagnostics.load_bw_telemetry(tmp_path)


def test_learning_curves_uses_pipeline_run_order(tmp_path: Path) -> None:
    for stage, pass_number in (("cd-2g", 3), ("ci-1g", 1), ("cd-untied", 2)):
        directory = tmp_path / "shared" / "models" / stage / "custom"
        directory.mkdir(parents=True)
        (directory / "bw_telemetry.json").write_text(
            json.dumps({"schema_version": 2, "passes": [{**_PASS, "pass": pass_number}]})
        )

    assert diagnostics.learning_curves(tmp_path, "custom") == [
        ("ci-1g", [_PASS]),
        ("cd-untied", [{**_PASS, "pass": 2}]),
        ("cd-2g", [{**_PASS, "pass": 3}]),
    ]


def test_learning_curves_returns_empty_when_no_models_are_trained(tmp_path: Path) -> None:
    assert diagnostics.learning_curves(tmp_path) == []
