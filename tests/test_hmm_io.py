"""Composed model I/O contracts, including storage and runtime shapes."""

from pathlib import Path

import numpy as np
import pytest

from pstrain.lib import _pstrainc
from pstrain.lib.bw import HMM
from tests.clib import requires_c_library

pytestmark = requires_c_library


def _model() -> HMM:
    return HMM(
        np.zeros((2, 1, 2), dtype=np.float32),
        np.ones((2, 1, 2), dtype=np.float32),
        np.ones((2, 1), dtype=np.float32),
        np.array(
            [
                [[2, 6, 0], [0, 3, 1], [0, 0, 0]],
                [[1, 3, 0], [0, 1, 9], [0, 0, 0]],
            ],
            dtype=np.float32,
        ),
    )


@pytest.mark.parametrize("square", [False, True])
def test_transition_storage_roundtrip(tmp_path: Path, square: bool) -> None:
    original = _model().tmat
    supplied = original if square else original[:, :-1, :]
    before = supplied.copy()
    path = str(tmp_path / "tmat")
    assert _pstrainc.write_tmat(path, supplied) == 0
    stored, count, states = _pstrainc.read_tmat_counts(path)
    assert (count, states) == (2, 3)
    np.testing.assert_array_equal(stored, original[:, :-1, :])
    np.testing.assert_array_equal(supplied, before)
    probabilities, _, _ = _pstrainc.read_tmat(path)
    np.testing.assert_allclose(probabilities, stored / stored.sum(axis=-1, keepdims=True))
    # A rectangular read result can be written directly, without inventing
    # an exit row or flattening matrix boundaries.
    assert _pstrainc.write_tmat(path, stored) == 0
    np.testing.assert_array_equal(_pstrainc.read_tmat_counts(path)[0], stored)


@pytest.mark.parametrize("shape", [(2, 3), (1, 2, 4), (1, 4, 2), (0, 2, 3), (1, 1, 1)])
def test_transition_writer_rejects_invalid_shape(tmp_path: Path, shape: tuple[int, ...]) -> None:
    path = tmp_path / "tmat"
    with pytest.raises(ValueError, match="[Tt]ransition"):
        _pstrainc.write_tmat(str(path), np.zeros(shape, dtype=np.float32))
    assert not path.exists()


def test_hmm_load_save_preserves_runtime_parameters(tmp_path: Path) -> None:
    model = _model()
    model.save(tmp_path / "first")
    first = HMM.load(tmp_path / "first")
    assert first.tmat.shape == (2, 2, 3)
    first.save(tmp_path / "second")
    second = HMM.load(tmp_path / "second")
    for name in ("means", "variances", "mixw", "tmat"):
        np.testing.assert_array_equal(getattr(second, name), getattr(first, name))


def test_transition_reader_checks_stored_row_count(tmp_path: Path) -> None:
    # S3 mixture and transition files share the same version/checksum/3D
    # envelope. Use the real 3D writer to produce a valid-checksum file with
    # a transition-incompatible row count, avoiding a hand-rolled serializer.
    path = str(tmp_path / "bad_rows")
    assert _pstrainc.write_mixw(path, np.ones((2, 1, 4), dtype=np.float32)) == 0
    with pytest.raises(RuntimeError, match="Failed to read tmat"):
        _pstrainc.read_tmat_counts(path)
