"""Effect-level coverage for schema-owned decoder front-end settings."""

import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from pstrain.lib.config.models import FeatureConfig
from pstrain.lib.native_worker import PstrainError
from pstrain.lib.pipeline.context import FeatParams
from pstrain.lib.pipeline.feat_params import write_feat_params
from pstrain.lib.testing.decoder import Decoder
from tests.conftest import requires_c_library

pytestmark = requires_c_library


def test_density_probe_failure_warns_and_uses_default_topn(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    fixture = Path(__file__).parent / "fixtures" / "multipron_final_state"
    observed_topn: list[int] = []

    from pstrain.lib._cffi.core import get_lib

    lib = get_lib()

    class RecordingLib:
        def __getattr__(self, name: str) -> object:
            return getattr(lib, name)

        def pstrain_decoder_create(self, config: Any) -> object:
            observed_topn.append(config.topn)
            return lib.pstrain_decoder_create(config)

    def fail_probe(path: str) -> object:
        raise PstrainError("density probe broke")

    monkeypatch.setattr("pstrain.lib._cffi.core.get_lib", lambda: RecordingLib())
    monkeypatch.setattr("pstrain.lib.testing.decoder.read_gau", fail_probe)

    with caplog.at_level("WARNING", logger="pstrain.lib.testing.decoder"):
        decoder = Decoder(
            fixture / "model",
            fixture / "dictionary.dict",
            fixture / "filler.dict",
        )
    try:
        assert observed_topn == [0]
        assert "density probe broke" in caplog.text
        assert "PocketSphinx's default top-N" in caplog.text
    finally:
        decoder.close()


def test_density_probe_programming_error_is_not_swallowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = Path(__file__).parent / "fixtures" / "multipron_final_state"

    def fail_probe(path: str) -> object:
        raise TypeError("density probe programming error")

    monkeypatch.setattr("pstrain.lib.testing.decoder.read_gau", fail_probe)

    with pytest.raises(RuntimeError, match="density probe programming error") as error:
        Decoder(
            fixture / "model",
            fixture / "dictionary.dict",
            fixture / "filler.dict",
        )
    assert isinstance(error.value.__cause__, TypeError)


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
def test_complete_feat_params_reaches_live_decoder(
    tmp_path: Path, field: str, value: object, native_name: str, expected: object
) -> None:
    """The complete trained record wins over a conflicting active profile."""
    fixture = Path(__file__).parent / "fixtures" / "multipron_final_state"
    model = tmp_path / "model"
    shutil.copytree(fixture / "model", model)
    feat_params = model / "feat.params"
    feat_params.write_text(
        "\n".join(
            (
                f"-{native_name} {value}"
                if line.startswith(f"-{native_name} ")
                else "-nfft 1024"
                if native_name == "samprate" and line.startswith("-nfft ")
                else line
            )
            for line in feat_params.read_text().splitlines()
        )
        + "\n"
    )
    decoder = Decoder(
        model,
        fixture / "dictionary.dict",
        fixture / "filler.dict",
        feature_config=FeatureConfig(),
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
def test_remove_noise_in_complete_feat_params_wins_over_profile(
    tmp_path: Path, remove_noise: bool
) -> None:
    """Both boolean values in the trained record override a conflicting profile."""
    fixture = Path(__file__).parent / "fixtures" / "multipron_final_state"
    model = tmp_path / "model"
    shutil.copytree(fixture / "model", model)
    feat_params = model / "feat.params"
    feat_params.write_text(
        "\n".join(
            f"-remove_noise {'yes' if remove_noise else 'no'}"
            if line.startswith("-remove_noise ")
            else line
            for line in feat_params.read_text().splitlines()
        )
        + "\n"
    )
    decoder = Decoder(
        model,
        fixture / "dictionary.dict",
        fixture / "filler.dict",
        feature_config=FeatureConfig(remove_noise=not remove_noise),
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
