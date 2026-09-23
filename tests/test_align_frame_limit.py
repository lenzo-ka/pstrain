"""Long utterances through the library aligner.

The library aligner accepts up to 32,768 frames in one utterance. Its
per-frame score buffer used to hold only 15,000, so a longer utterance wrote
past the end of a heap buffer and could kill the native worker or corrupt its
results. These run in the native worker, so a crash fails the test instead of
taking pytest down with it.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from pstrain.lib.alignment import Aligner
from pstrain.lib.features import read_sphinx_mfc

_FIXTURE = Path(__file__).parent / "fixtures" / "multipron_final_state"
_UTTERANCE = "arctic_a0257"
_TRANSCRIPT = "they are coming ashore whoever they are"
_LIMIT = 32768


def _tiled(n_frames: int) -> np.ndarray:
    """The fixture utterance repeated end to end to exactly ``n_frames`` frames."""
    mfcc = read_sphinx_mfc(_FIXTURE / f"{_UTTERANCE}.mfc", veclen=13)
    copies = -(-n_frames // mfcc.shape[0])
    return np.ascontiguousarray(np.tile(mfcc, (copies, 1))[:n_frames], dtype=np.float32)


@pytest.fixture
def aligner(tmp_path: Path):
    model = tmp_path / "model"
    shutil.copytree(_FIXTURE / "model", model)
    with Aligner(model, _FIXTURE / "dictionary.dict", filler_dict=_FIXTURE / "filler.dict") as a:
        yield a


@pytest.mark.parametrize("n_frames", [15001, _LIMIT])
def test_utterance_past_15000_frames_aligns(aligner: Aligner, n_frames: int) -> None:
    # The fixture model is a flat snapshot, so where the words land says
    # nothing; the alignment must cover every frame.
    result = aligner.align_mfcc(_tiled(n_frames), _TRANSCRIPT, "long")
    assert result.n_frames == n_frames
    assert result.words[0].start_frame == 0
    assert result.words[-1].end_frame == n_frames - 1


def test_utterance_over_the_limit_is_refused(aligner: Aligner) -> None:
    with pytest.raises(
        RuntimeError,
        match=(
            r"utterance too_long has 32769 frames, more than the 32768 frames "
            r"the aligner accepts for one utterance; split it into shorter utterances"
        ),
    ):
        aligner.align_mfcc(_tiled(_LIMIT + 1), _TRANSCRIPT, "too_long")

    # The refusal leaves the aligner usable.
    assert aligner.align_mfcc(_tiled(1000), _TRANSCRIPT, "after").n_frames == 1000
