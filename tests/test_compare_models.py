"""Model-level comparison gates for recorded build provenance."""

from pathlib import Path

from pstrain.lib.compare import compare_models


def _model(directory: Path, policy: str | None) -> None:
    directory.mkdir()
    (directory / "mdef").write_text("same\n")
    if policy is not None:
        (directory / "provenance.json").write_text(
            '{"native_library":{"fp_contract_declared":"' + policy + '"}}\n'
        )


def test_model_comparison_consults_fp_contract_provenance(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _model(first, "off")
    _model(second, "fast")

    result = compare_models(first, second)

    assert not result.all_match
    assert result.components["provenance.json"].text_match is False


def test_model_comparison_rejects_missing_provenance(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _model(first, "off")
    _model(second, None)

    result = compare_models(first, second)

    assert not result.all_match
    assert not result.components["provenance.json"].exists_b


def test_model_comparison_reports_both_policies_undeclared(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _model(first, None)
    _model(second, None)

    result = compare_models(first, second)

    assert not result.all_match
    provenance = result.components["provenance.json"]
    assert provenance.status == (
        "contraction policy undeclared for both models; comparability cannot be established"
    )
    assert provenance.summary() == provenance.status
    assert result.to_dict()["components"]["provenance.json"]["status"] == provenance.status


def test_model_comparison_reports_both_legacy_policies_undeclared(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _model(first, None)
    _model(second, None)
    (first / "provenance.json").write_text('{"legacy":true}\n')
    (second / "provenance.json").write_text('{"legacy":true}\n')

    result = compare_models(first, second)

    assert not result.all_match
    assert "comparability cannot be established" in result.summary()
    assert result.components["provenance.json"].text_match is True
