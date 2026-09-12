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


@pytest.mark.parametrize(
    "failed_parameter", ["means", "variances", "mixture_weights", "transition_matrices"]
)
def test_hmm_save_preserves_model_on_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failed_parameter: str
) -> None:
    model = _model()
    model.save(tmp_path)
    originals = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    writers = {
        "means": "write_gau",
        "variances": "write_gau",
        "mixture_weights": "write_mixw",
        "transition_matrices": "write_tmat",
    }
    writer_name = writers[failed_parameter]
    real_writer = getattr(_pstrainc, writer_name)

    def fail_one(filename: str, values: np.ndarray) -> int:
        if Path(filename).name == failed_parameter:
            return -1
        return int(real_writer(filename, values))

    monkeypatch.setattr(_pstrainc, writer_name, fail_one)
    model.means += 1
    with pytest.raises(RuntimeError, match=failed_parameter):
        model.save(tmp_path)
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == originals


def test_hmm_save_reports_existing_output_directory(tmp_path: Path) -> None:
    (tmp_path / "means").mkdir()
    with pytest.raises(RuntimeError, match="means"):
        _model().save(tmp_path)
    assert list(tmp_path.iterdir()) == [tmp_path / "means"]


@pytest.mark.parametrize("existing", [False, True])
def test_hmm_save_rolls_back_replacement_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, existing: bool
) -> None:
    model = _model()
    if existing:
        model.save(tmp_path)
    # Model directories may also contain an mdef, dictionaries, or provenance.
    (tmp_path / "mdef").write_text("preserved metadata")
    originals = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    real_replace = Path.replace

    def fail_replacement(path: Path, target: Path) -> Path:
        if path.name == "variances" and path.parent.name.startswith(".hmm-save-"):
            raise OSError("injected replacement failure")
        return real_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_replacement)
    model.means += 1
    with pytest.raises(OSError, match="injected"):
        model.save(tmp_path)
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == originals


def test_gaussian_reader_preserves_variable_stream_lengths(tmp_path: Path) -> None:
    ffi, lib = _pstrainc.get_ffi(), _pstrainc.get_lib()
    n_codebooks, n_densities = 2, 2
    widths = [2, 3]
    root = ffi.new("float32***[]", n_codebooks)
    streams = [ffi.new("float32**[]", len(widths)) for _ in range(n_codebooks)]
    densities = [[ffi.new("float32*[]", n_densities) for _ in widths] for _ in range(n_codebooks)]
    values = np.arange(n_codebooks * n_densities * sum(widths), dtype=np.float32)
    expected = np.zeros((n_codebooks, len(widths), n_densities, max(widths)), dtype=np.float32)
    offset = 0
    for m in range(n_codebooks):
        root[m] = streams[m]
        for f, width in enumerate(widths):
            streams[m][f] = densities[m][f]
            for d in range(n_densities):
                densities[m][f][d] = ffi.cast("float32 *", values.ctypes.data) + offset
                expected[m, f, d, :width] = values[offset : offset + width]
                offset += width
    path = str(tmp_path / "gaussians")
    assert (
        lib.s3gau_write(
            path.encode(), root, n_codebooks, len(widths), n_densities, ffi.new("uint32[]", widths)
        )
        == 0
    )
    result, actual_m, actual_f, actual_d, actual_widths = _pstrainc.read_gau(path)
    assert (actual_m, actual_f, actual_d, actual_widths) == (
        n_codebooks,
        len(widths),
        n_densities,
        widths,
    )
    np.testing.assert_array_equal(result, expected)
    # Exercise another native allocation after the reader freed its source;
    # the first returned array must own its data independently.
    _pstrainc.read_gau(path)
    np.testing.assert_array_equal(result, expected)
