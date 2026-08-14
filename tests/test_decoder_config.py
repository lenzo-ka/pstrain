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
        ("cmn", "live", "cmn", "live"),
        ("cmninit", "12,1,-3", "cmninit", "12,1,-3"),
        ("varnorm", "yes", "varnorm", 1),
        ("samprate", 22050, "samprate", 22050),
    ],
)
def test_nondefault_feature_config_reaches_live_decoder(
    field: str, value: object, native_name: str, expected: object
) -> None:
    fixture = Path(__file__).parent / "fixtures" / "multipron_final_state"
    decoder = Decoder(
        fixture / "model",
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
