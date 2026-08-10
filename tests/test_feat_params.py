"""Regression tests for training and packaged ``feat.params`` files."""

from pathlib import Path

import pytest

from st2.lib.model import MODEL_FILES_REQUIRED
from st2.lib.pipeline.context import FeatParams
from st2.lib.pipeline.feat_params import feat_params_lines, write_feat_params
from st2.lib.steps.package import package_model


def _parse_feat_params(path: Path) -> dict[str, str]:
    return dict(line.split(maxsplit=1) for line in path.read_text().splitlines())


def test_default_feat_params_contains_complete_training_front_end() -> None:
    params = dict(line.rstrip().split(maxsplit=1) for line in feat_params_lines(FeatParams()))

    assert params == {
        "-samprate": "16000",
        "-ncep": "13",
        "-nfilt": "40",
        "-nfft": "512",
        "-lowerf": "130",
        "-upperf": "6800",
        "-feat": "1s_c_d_dd",
        "-transform": "dct",
        "-lifter": "22",
        "-agc": "none",
        "-cmn": "batch",
        "-varnorm": "no",
        "-unit_area": "yes",
        "-round_filters": "yes",
        "-remove_dc": "no",
        "-remove_noise": "yes",
    }


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
        cmn="current",
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


@pytest.mark.parametrize(
    ("field", "requested"),
    [
        ("transform", "legacy"),
        ("feat_type", "s2_4x"),
        ("agc", "max"),
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

    with pytest.raises(FileNotFoundError, match=r"trained model directory lacks feat\.params"):
        package_model(model_dir, tmp_path / "dist")
