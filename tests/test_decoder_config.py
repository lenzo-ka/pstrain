"""Effect-level coverage for schema-owned decoder front-end settings."""

import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from pstrain.lib.config.models import FeatureConfig
from pstrain.lib.pipeline.context import FeatParams
from pstrain.lib.pipeline.feat_params import write_feat_params
from pstrain.lib.testing.decoder import Decoder
from tests.conftest import requires_c_library

pytestmark = requires_c_library


@pytest.mark.parametrize(
    ("field", "value", "native_name", "expected"),
    [
        ("agc", "max", "agc", "max"),
        ("cmn", "batch", "cmn", "batch"),
        ("cmninit", "12,1,-3", "cmninit", "12,1,-3"),
        ("varnorm", "yes", "varnorm", 1),
        ("samprate", 22050, "samprate", 22050),
    ],
)
def test_nondefault_feature_config_reaches_live_decoder(
    tmp_path: Path, field: str, value: object, native_name: str, expected: object
) -> None:
    """Each profile value differs from the engine default if its assignment is deleted."""
    fixture = Path(__file__).parent / "fixtures" / "multipron_final_state"
    model = tmp_path / "model"
    shutil.copytree(fixture / "model", model)
    feat_params = model / "feat.params"
    feat_params.write_text(
        "\n".join(
            line
            for line in feat_params.read_text().splitlines()
            if not line.startswith(f"-{native_name} ")
            and not (native_name == "samprate" and line.startswith("-nfft "))
        )
        + "\n"
    )
    decoder = Decoder(
        model,
        fixture / "dictionary.dict",
        fixture / "filler.dict",
        feature_config=FeatureConfig(**{field: value}),
    )
    try:
        name = native_name.encode()
        if isinstance(expected, str):
            actual = decoder._ffi.string(
                decoder._lib.pstrain_decoder_config_str(decoder._decoder, name)
            ).decode()
        else:
            actual = decoder._lib.pstrain_decoder_config_int(decoder._decoder, name)
        assert actual == expected
    finally:
        decoder.close()


@pytest.mark.parametrize("remove_noise", [False, True])
def test_remove_noise_reaches_live_decoder_when_feat_params_omits_it(
    tmp_path: Path, remove_noise: bool
) -> None:
    """Gate conditional profile routing, with an intentionally asymmetric oracle.

    With ``-remove_noise`` omitted from an otherwise present ``feat.params``, the
    True arm redlines a deleted native assignment because PocketSphinx defaults
    to false.  The False arm agrees with that default even without the assignment;
    it instead guards against an always-true or inverted assignment.
    """
    fixture = Path(__file__).parent / "fixtures" / "multipron_final_state"
    model = tmp_path / "model"
    shutil.copytree(fixture / "model", model)
    feat_params = model / "feat.params"
    feat_params.write_text(
        "\n".join(
            line
            for line in feat_params.read_text().splitlines()
            if not line.startswith("-remove_noise ")
        )
        + "\n"
    )
    decoder = Decoder(
        model,
        fixture / "dictionary.dict",
        fixture / "filler.dict",
        feature_config=FeatureConfig(remove_noise=remove_noise),
    )
    try:
        actual = decoder._lib.pstrain_decoder_config_int(decoder._decoder, b"remove_noise")
        assert actual == remove_noise
    finally:
        decoder.close()


def test_decoder_rejects_model_without_feat_params(tmp_path: Path) -> None:
    fixture = Path(__file__).parent / "fixtures" / "multipron_final_state"
    model = tmp_path / "model"
    shutil.copytree(
        fixture / "model",
        model,
        ignore=shutil.ignore_patterns("feat.params"),
    )

    with pytest.raises(
        FileNotFoundError,
        match=(
            rf"feat\.params.*{model}.*decode-time front end.*"
            r"silently differ.*feature shape and basis"
        ),
    ):
        Decoder(model, fixture / "dictionary.dict", fixture / "filler.dict")


@pytest.mark.parametrize("remove_noise", [False, True])
def test_decoder_completes_two_utterances_with_each_remove_noise_setting(
    tmp_path: Path, remove_noise: bool
) -> None:
    fixture = Path(__file__).parent / "fixtures" / "multipron_final_state"
    model = tmp_path / "model"
    shutil.copytree(fixture / "model", model)
    write_feat_params(model / "feat.params", replace(FeatParams(), remove_noise=remove_noise))
    audio = Path(__file__).parent / "fixtures" / "mini_arctic" / "wav" / "arctic_a0001.wav"
    lm = Path(__file__).parent.parent / "benchmarks" / "arctic" / "data" / "training-unigram.lm"

    decoder = Decoder(model, fixture / "dictionary.dict", fixture / "filler.dict", lm=lm)
    try:
        results = [decoder.decode_file(audio) for _ in range(2)]
        assert all(result.success for result in results), [result.error for result in results]
    finally:
        decoder.close()
