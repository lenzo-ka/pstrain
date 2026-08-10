"""Regression tests for training and packaged ``feat.params`` files."""

from pathlib import Path

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
        "-remove_dc": "yes",
    }


def test_packaged_feat_params_matches_training_profile(tmp_path: Path) -> None:
    trained_model = tmp_path / "shared" / "models" / "ci-8g" / "telephone"
    trained_model.mkdir(parents=True)
    for filename in MODEL_FILES_REQUIRED:
        (trained_model / filename).write_text(filename)

    profile = FeatParams(
        samprate=8000,
        ncep=12,
        nfilt=25,
        nfft=256,
        lowerf=200,
        upperf=3500,
        feat_type="s2_4x",
        transform="legacy",
        lifter=17,
        agc="max",
        cmn="current",
        varnorm="yes",
    )
    training_path = write_feat_params(trained_model / "feat.params", profile)

    result = package_model(
        model_dir=trained_model,
        output_dir=tmp_path / "dist" / "models",
        model_name="ci-8g-telephone",
        feat_params=profile,
        include_dict=False,
    )

    assert _parse_feat_params(result["feat_params"]) == _parse_feat_params(training_path)
    assert result["feat_params"].read_text() == training_path.read_text()
