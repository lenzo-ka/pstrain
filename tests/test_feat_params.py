"""Regression tests for training and packaged ``feat.params`` files."""

from pathlib import Path

import pytest

from pstrain.lib.filetypes import FileType, detect_file_type
from pstrain.lib.model import MODEL_FILES_REQUIRED
from pstrain.lib.model import require_complete_model as real_require_complete_model
from pstrain.lib.pipeline.context import FeatParams
from pstrain.lib.pipeline.feat_params import feat_params_lines, write_feat_params
from pstrain.lib.steps.package import package_model


def _parse_feat_params(path: Path) -> dict[str, str]:
    return dict(line.split(maxsplit=1) for line in path.read_text().splitlines())


def test_default_feat_params_contains_complete_training_front_end() -> None:
    params = dict(line.rstrip().split(maxsplit=1) for line in feat_params_lines(FeatParams()))

    assert params == {
        "-samprate": "16000",
        "-ncep": "13",
        "-ceplen": "13",
        "-nfilt": "25",
        "-nfft": "512",
        "-lowerf": "130.0",
        "-upperf": "6800.0",
        "-alpha": "0.97",
        "-dither": "yes",
        "-seed": "-1",
        "-remove_dc": "yes",
        "-remove_noise": "yes",
        "-frate": "100",
        "-wlen": "0.025625",
        "-feat": "1s_c_d_dd",
        "-transform": "dct",
        "-lifter": "22",
        "-agc": "none",
        "-cmn": "batch",
        "-cmninit": "40,3,-1",
        "-varnorm": "no",
        "-unit_area": "yes",
        "-round_filters": "yes",
    }


def test_default_feat_params_writer_emits_25_filters(tmp_path: Path) -> None:
    output = write_feat_params(tmp_path / "feat.params", FeatParams())

    assert _parse_feat_params(output)["-nfilt"] == "25"


def test_packaging_copies_trained_feat_params_despite_config_drift(tmp_path: Path) -> None:
    trained_model = tmp_path / "shared" / "models" / "ci-8g" / "telephone"
    trained_model.mkdir(parents=True)
    for filename in MODEL_FILES_REQUIRED:
        (trained_model / filename).write_text(filename)

    trained_profile = FeatParams(
        samprate=8000,
        ncep=12,
        nfilt=25,
        nfft=256,
        lowerf=200,
        upperf=3500,
        lifter=17,
    )
    training_path = write_feat_params(trained_model / "feat.params", trained_profile)
    expected = training_path.read_bytes()

    # The active profile may drift after training; packaging must not consult it.
    _drifted_profile = FeatParams(samprate=16000, lifter=22)

    result = package_model(
        model_dir=trained_model,
        output_dir=tmp_path / "dist" / "models",
        model_name="ci-8g-telephone",
        include_dict=False,
    )

    assert result["feat_params"].read_bytes() == expected
    assert "Created:" not in result["readme"].read_text()


def test_packaging_fails_if_required_file_disappears_after_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    for filename in MODEL_FILES_REQUIRED:
        (model_dir / filename).write_text(filename)
    write_feat_params(model_dir / "feat.params", FeatParams())
    removed = model_dir / MODEL_FILES_REQUIRED[-1]

    def validate_then_remove(path: Path) -> Path:
        feat_params = real_require_complete_model(path)
        removed.unlink()
        return feat_params

    monkeypatch.setattr("pstrain.lib.steps.package.require_complete_model", validate_then_remove)

    with pytest.raises(FileNotFoundError, match=str(removed)):
        package_model(model_dir, tmp_path / "dist")


def test_packaging_allows_absent_optional_dictionaries(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    for filename in MODEL_FILES_REQUIRED:
        (model_dir / filename).write_text(filename)
    write_feat_params(model_dir / "feat.params", FeatParams())

    result = package_model(
        model_dir,
        tmp_path / "dist",
        dictionary_path=tmp_path / "missing.dict",
        filler_dict_path=tmp_path / "missing.filler",
    )

    assert "dictionary" not in result
    assert "filler_dict" not in result
    assert result["noisedict"].read_text() == "<sil> SIL\n<s> SIL\n</s> SIL\n"


@pytest.mark.parametrize(
    ("field", "requested"),
    [
        ("feat_type", "s2_4x"),
        ("agc", "max"),
        ("cmn", "live"),
        ("cmninit", "12,1,-3"),
        ("varnorm", "yes"),
    ],
)
def test_feat_params_rejects_values_hardcoded_by_training(field: str, requested: str) -> None:
    feat = FeatParams(**{field: requested})

    with pytest.raises(
        ValueError,
        match=rf"{field}=.*{requested}.*training engine hardcodes {field}=",
    ):
        feat_params_lines(feat)


def test_packaging_requires_trained_feat_params(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    with pytest.raises(
        FileNotFoundError,
        match=(
            rf"feat\.params.*{model_dir}.*decode-time front end.*"
            r"silently differ.*feature shape and basis"
        ),
    ):
        package_model(model_dir, tmp_path / "dist")


def test_model_under_construction_does_not_require_feat_params(tmp_path: Path) -> None:
    """Training's five-file model contract remains valid before feat.params exists."""
    model_dir = tmp_path / "model-under-construction"
    model_dir.mkdir()
    for filename in MODEL_FILES_REQUIRED:
        (model_dir / filename).write_text(filename)

    assert "feat.params" not in MODEL_FILES_REQUIRED
    assert detect_file_type(model_dir) is FileType.MODEL
