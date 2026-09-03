"""Tests for model testing: total decode failure is not a score."""

from pathlib import Path

import pytest

from pstrain.lib.model import MODEL_FILES_REQUIRED
from pstrain.lib.native_worker import PstrainError
from pstrain.lib.pipeline.context import FeatParams
from pstrain.lib.pipeline.feat_params import feat_params_lines
from pstrain.lib.testing.decoder import DecodingResult
from pstrain.lib.testing.test import _DecodeConfig
from pstrain.lib.testing.test import test_model as run_test_model

FIXTURE = Path(__file__).parent / "fixtures" / "mini_arctic"


def _model_dir(tmp_path: Path, *, complete: bool = True) -> Path:
    """A model directory. Testing decodes, so by default it is a COMPLETE model."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    for filename in MODEL_FILES_REQUIRED:
        (model_dir / filename).touch()
    if complete:
        (model_dir / "feat.params").write_text("".join(feat_params_lines(FeatParams())))
    return model_dir


def test_model_raises_when_every_submitted_utterance_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("pstrain.lib.testing.test.check_pocketsphinx", lambda: (True, "ok"))

    def fail_decodes(
        config: _DecodeConfig, files: list[tuple[str, Path]], _jobs: int | None
    ) -> list[tuple[str, DecodingResult]]:
        assert config.lm is None
        return [
            (
                utterance_id,
                DecodingResult(
                    utterance_id=utterance_id,
                    hypothesis="",
                    success=False,
                    error="Failed to start utterance processing",
                ),
            )
            for utterance_id, _ in files
        ]

    monkeypatch.setattr("pstrain.lib.testing.test._decode_files", fail_decodes)

    with pytest.raises(PstrainError) as raised:
        run_test_model(
            model_dir=_model_dir(tmp_path),
            test_audio_dir=FIXTURE / "wav",
            test_transcripts={"arctic_a0001": "AUTHOR OF THE DANGER TRAIL"},
            dict_file=FIXTURE / "dictionary.dict",
            filler_dict=FIXTURE / "filler.dict",
            lm=None,
            jobs=1,
        )

    message = str(raised.value)
    assert "Nothing decoded: all 1 requested utterances failed" in message
    assert "arctic_a0001: Failed to start utterance processing" in message


def test_model_allows_empty_transcript_set(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("pstrain.lib.testing.test.check_pocketsphinx", lambda: (True, "ok"))

    result = run_test_model(
        model_dir=_model_dir(tmp_path),
        test_audio_dir=FIXTURE / "wav",
        test_transcripts={},
        dict_file=FIXTURE / "dictionary.dict",
        filler_dict=FIXTURE / "filler.dict",
        lm=None,
        jobs=1,
    )

    assert result.n_utterances == 0
    assert result.n_decoded == 0


def test_model_raises_when_all_audio_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing audio decodes exactly as little as failed audio, and must say so.

    Utterances whose WAV is absent never reach the decoder, so a guard written
    only around submitted files leaves this path returning a word error rate of
    1.0 over zero reference words -- the very artifact this contract removes.
    """
    monkeypatch.setattr("pstrain.lib.testing.test.check_pocketsphinx", lambda: (True, "ok"))

    with pytest.raises(PstrainError) as raised:
        run_test_model(
            model_dir=_model_dir(tmp_path),
            test_audio_dir=tmp_path / "no_such_audio_dir",
            test_transcripts={"arctic_a0001": "AUTHOR OF THE DANGER TRAIL", "arctic_a0002": "NOT"},
            dict_file=FIXTURE / "dictionary.dict",
            filler_dict=FIXTURE / "filler.dict",
            lm=None,
            jobs=1,
        )

    message = str(raised.value)
    assert "Nothing decoded: all 2 requested utterances failed" in message
    assert "arctic_a0001: audio file not found" in message
    assert "arctic_a0002: audio file not found" in message


def test_model_requires_a_complete_model_even_with_nothing_to_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The complete-model contract does not depend on there being work to do.

    Model completeness used to be established as a side effect of constructing
    the decoder, so skipping construction when there is nothing to decode would
    let an incomplete model through without complaint.
    """
    monkeypatch.setattr("pstrain.lib.testing.test.check_pocketsphinx", lambda: (True, "ok"))

    with pytest.raises(FileNotFoundError, match=r"feat\.params"):
        run_test_model(
            model_dir=_model_dir(tmp_path, complete=False),
            test_audio_dir=FIXTURE / "wav",
            test_transcripts={},
            dict_file=FIXTURE / "dictionary.dict",
            filler_dict=FIXTURE / "filler.dict",
            lm=None,
            jobs=1,
        )
