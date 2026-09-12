"""Reject malformed public BW buffers before native pointer marshalling."""

from pathlib import Path

import numpy as np
import pytest

from pstrain.lib.bw import BW_CEPSTRAL_LENGTH, BW_FEATURE_LENGTH, BWConfig, BWTrainer
from pstrain.lib.features import read_sphinx_mfc
from tests.clib import requires_c_library

pytestmark = requires_c_library
FIXTURE = Path(__file__).parent / "fixtures" / "multipron_final_state"


@pytest.fixture(scope="module")
def trainer() -> BWTrainer:
    model = FIXTURE / "model"
    result = BWTrainer(
        *(
            model / name
            for name in ("mdef", "means", "variances", "mixture_weights", "transition_matrices")
        ),
        BWConfig(pass2var=True, unobserved_gaussian_policy="zero", a_beam=1e-200),
    )
    result.set_dict(FIXTURE / "dictionary.dict", FIXTURE / "filler.dict")
    return result


@pytest.mark.parametrize(
    "method,width",
    [
        ("process_utterance_text", BW_FEATURE_LENGTH),
        ("process_utterance", BW_FEATURE_LENGTH),
        ("process_utterance_mfcc", BW_CEPSTRAL_LENGTH),
    ],
)
@pytest.mark.parametrize("shape_kind", ["flat", "narrow", "wide", "rank3", "empty"])
def test_feature_shape_rejected(
    trainer: BWTrainer,
    method: str,
    width: int,
    shape_kind: str,
) -> None:
    shapes = {
        "flat": (width,),
        "narrow": (2, width - 1),
        "wide": (2, width + 1),
        "rank3": (2, width, 1),
        "empty": (0, width),
    }
    values = np.zeros(shapes[shape_kind], dtype=np.float32)
    argument = np.array([0], dtype=np.uint32) if method == "process_utterance" else "author"
    with pytest.raises(ValueError, match="feature"):
        getattr(trainer, method)(values, argument)
    assert trainer.get_stats().total_utts == 0


@pytest.mark.parametrize("ids", [np.array([[0]]), np.array([-1]), np.array([0.5]), np.array([])])
def test_phone_id_array_validation(trainer: BWTrainer, ids: np.ndarray) -> None:
    with pytest.raises(ValueError, match="Phone IDs"):
        trainer.process_utterance(np.zeros((2, BW_FEATURE_LENGTH), dtype=np.float32), ids)


def test_native_phone_id_bounds(trainer: BWTrainer) -> None:
    assert not trainer.process_utterance(
        np.zeros((2, BW_FEATURE_LENGTH), dtype=np.float32),
        np.array([np.iinfo(np.uint32).max], dtype=np.uint32),
    )
    assert trainer.get_stats().total_utts == 0


def test_valid_mfcc_still_reaches_native_training(trainer: BWTrainer) -> None:
    mfcc = read_sphinx_mfc(FIXTURE / "arctic_a0257.mfc")
    assert trainer.process_utterance_mfcc(mfcc, "<s> they are coming ashore whoever they are </s>")
    stats = trainer.get_stats()
    assert stats.total_utts == 1
    assert stats.total_frames == len(mfcc)
