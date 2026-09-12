"""Fixed-reference variance bounds using real raw model I/O and BW updates."""

from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from pstrain.lib import _pstrainc
from pstrain.lib.bw import BW_FEATURE_LENGTH, BWConfig, BWTrainer
from pstrain.lib.config.models import TrainingConfig
from pstrain.lib.model import MODEL_PARAMETER_FILES
from pstrain.lib.steps.train import _save_iteration_model, run_bw_training
from pstrain.lib.steps.variance import load_variance_floor
from pstrain.lib.telemetry import load_bw_telemetry
from tests.clib import requires_c_library


def _write(path: Path, values: np.ndarray) -> Path:
    assert _pstrainc.write_gau(str(path), values) == 0
    return path


def _variances(codebooks: int = 2, densities: int = 1) -> np.ndarray:
    return np.ones((codebooks, 1, densities, BW_FEATURE_LENGTH), dtype=np.float32)


@pytest.mark.parametrize(
    "fraction", [-1, 1.01, float("nan"), float("inf"), -float("inf"), True, "0.2"]
)
def test_fraction_schema_rejects_invalid_or_implicit_opt_in(fraction: object) -> None:
    with pytest.raises(ValidationError):
        TrainingConfig(split_variance_floor_fraction=fraction)


def test_regularization_defaults_off_and_does_not_read_reference(tmp_path: Path) -> None:
    assert TrainingConfig().split_variance_floor_fraction == 0
    assert load_variance_floor(tmp_path / "missing", 0, tmp_path / "also-missing") is None
    for fraction in (-1, 2, float("nan"), float("inf"), True):
        with pytest.raises(ValueError, match="finite and between"):
            load_variance_floor(None, fraction, tmp_path / "missing")
    with pytest.raises(ValueError, match="reference is required"):
        load_variance_floor(None, 0.25, tmp_path / "missing")


@requires_c_library
def test_raw_floor_persists_all_densities_and_keeps_zero_reference_unobserved(
    tmp_path: Path,
) -> None:
    reference = _variances()
    reference[0, 0, 0] = np.arange(1, BW_FEATURE_LENGTH + 1)
    reference[1] = 0
    source = _write(tmp_path / "reference", reference)
    candidate = np.repeat(reference, 3, axis=2)
    candidate[0, :, 0] = 0
    candidate[0, :, 1] *= 0.1
    candidate[1] = 0
    target = _write(tmp_path / "candidate", candidate)
    floor = load_variance_floor(source, 0.25, target)
    assert floor is not None
    expected = np.maximum(candidate, (reference.astype(np.float64) * 0.25).astype(np.float32))
    changed = floor.apply(target)
    stored = _pstrainc.read_gau(str(target))[0]
    np.testing.assert_array_equal(stored, expected)
    assert changed == 2 * BW_FEATURE_LENGTH
    assert np.count_nonzero(stored[1]) == 0
    # Reapplying never uses the preceding candidate as a new reference.
    assert floor.apply(target) == 0


@requires_c_library
@pytest.mark.parametrize("invalid", [-1.0, float("nan"), float("inf")])
def test_invalid_candidate_is_rejected_before_maximum(tmp_path: Path, invalid: float) -> None:
    reference = _write(tmp_path / "reference", _variances())
    candidate = _write(tmp_path / "candidate", _variances(densities=2))
    floor = load_variance_floor(reference, 0.25, candidate)
    assert floor is not None
    bad = _variances(densities=2)
    bad[0, 0, 0, 0] = invalid
    _write(candidate, bad)
    before = candidate.read_bytes()
    with pytest.raises(ValueError, match="finite and nonnegative"):
        floor.apply(candidate)
    assert candidate.read_bytes() == before


@requires_c_library
@pytest.mark.parametrize("shape", [(2, 1, 2, 39), (2, 2, 1, 39), (2, 1, 1, 13), (3, 1, 1, 39)])
def test_reference_requires_matching_one_density_native_layout(
    tmp_path: Path, shape: tuple[int, ...]
) -> None:
    reference = _write(tmp_path / "reference", np.ones(shape, dtype=np.float32))
    candidate = _write(tmp_path / "candidate", _variances(densities=2))
    with pytest.raises(ValueError):
        load_variance_floor(reference, 0.25, candidate)


@requires_c_library
@pytest.mark.parametrize("fault", ["writer", "interrupt"])
def test_regularized_update_keeps_old_parameters_and_counts_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:

    output = tmp_path / "output"
    output.mkdir()
    names = (*MODEL_PARAMETER_FILES, "gauden_counts")
    for name in names:
        (output / name).write_bytes(("old-" + name).encode())
    before = {name: (output / name).read_bytes() for name in names}
    from pstrain.lib.features import read_sphinx_mfc

    fixture = Path(__file__).parent / "fixtures" / "multipron_final_state"
    variances = fixture / "model" / "variances"
    reference = _write(tmp_path / "reference", _pstrainc.read_gau(str(variances))[0] * 4)
    floor = load_variance_floor(reference, 0.5, variances)
    assert floor is not None
    trainer = BWTrainer(
        *(fixture / "model" / name for name in ("mdef", *MODEL_PARAMETER_FILES)),
        BWConfig(pass2var=False, unobserved_gaussian_policy="zero", a_beam=1e-200),
    )
    trainer.set_dict(fixture / "dictionary.dict", fixture / "filler.dict")
    assert trainer.process_utterance_mfcc(
        read_sphinx_mfc(fixture / "arctic_a0257.mfc"),
        "<s> they are coming ashore whoever they are </s>",
    )
    assert trainer.get_stats().total_frames > 0
    real_replace = Path.replace

    def failed_write(path: str, values: np.ndarray) -> int:
        return -1

    def interrupted(path: Path, target: Path) -> Path:
        result = real_replace(path, target)
        if path.name == "gauden_counts" and path.parent.name.startswith(".hmm-save-"):
            raise KeyboardInterrupt("injected after counts publication")
        return result

    if fault == "writer":
        monkeypatch.setattr(_pstrainc, "write_gau", failed_write)
        expected_error = RuntimeError
    else:
        monkeypatch.setattr(Path, "replace", interrupted)
        expected_error = KeyboardInterrupt
    with pytest.raises(expected_error):
        _save_iteration_model(trainer, output, 1, floor)
    assert {name: (output / name).read_bytes() for name in names} == before


@requires_c_library
def test_real_bw_passes_reload_fixed_floor_and_default_off_remains_unstaged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = Path(__file__).parent / "fixtures" / "multipron_final_state"
    source = _pstrainc.read_gau(str(fixture / "model" / "variances"))[0]
    # A fixed external reference larger than the initial model makes clamping
    # observable, without fabricating likelihoods or training accumulators.
    reference = _write(tmp_path / "reference", source * 4)
    reference_bytes = reference.read_bytes()
    fileids = tmp_path / "train.fileids"
    transcript = tmp_path / "train.transcription"
    fileids.write_text("arctic_a0257\n")
    transcript.write_text("arctic_a0257 they are coming ashore whoever they are\n")
    reads: list[Path] = []
    real_read = _pstrainc.read_gau

    def observe_read(path: str):
        reads.append(Path(path))
        return real_read(path)

    monkeypatch.setattr(_pstrainc, "read_gau", observe_read)
    real_init = BWTrainer.__init__
    loaded: list[tuple[Path, np.ndarray]] = []

    def observe_init(self: BWTrainer, *args, **kwargs) -> None:
        variance_path = Path(kwargs["vars_path"])
        loaded.append((variance_path, real_read(str(variance_path))[0]))
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(BWTrainer, "__init__", observe_init)
    results = {}
    for label, fraction in [("implicit", 0), ("default", 0), ("regularized", 0.5)]:
        output = tmp_path / label
        results[label] = run_bw_training(
            model_dir=fixture / "model",
            output_dir=output,
            features_dir=fixture,
            train_fileids=fileids,
            transcription=transcript,
            dictionary=fixture / "dictionary.dict",
            filler_dict=fixture / "filler.dict",
            first_pass_2passvar=False,
            config=BWConfig(pass2var=False, unobserved_gaussian_policy="zero", a_beam=1e-200),
            n_iter=3,
            min_iterations=3,
            checkpoint_iterations=True,
            **(
                {}
                if label == "implicit"
                else {
                    "variance_floor_reference": reference if fraction else tmp_path / "absent",
                    "variance_floor_fraction": fraction,
                }
            ),
        )
    assert results["default"].trajectory == results["implicit"].trajectory
    for name in (*MODEL_PARAMETER_FILES, "gauden_counts"):
        assert (tmp_path / "default" / name).read_bytes() == (
            tmp_path / "implicit" / name
        ).read_bytes()
    assert reads.count(reference) == 1
    assert reference.read_bytes() == reference_bytes
    floor = source * 2
    rows = load_bw_telemetry(tmp_path / "regularized")["passes"]
    assert len(rows) == 3
    assert any(row["variance_regularization"]["clamped_coordinates"] > 0 for row in rows)
    hashes = {row["variance_regularization"]["reference_sha256"] for row in rows}
    assert len(hashes) == 1
    next_pass_inputs = [
        values for path, values in loaded if path == tmp_path / "regularized" / "variances"
    ]
    assert len(next_pass_inputs) == 2
    for index, values in enumerate(next_pass_inputs, 1):
        previous_checkpoint = tmp_path / "regularized" / "iterations" / f"{index:02d}" / "variances"
        np.testing.assert_array_equal(values, real_read(str(previous_checkpoint))[0])
        assert np.all(values >= floor)
    for index in range(1, 4):
        path = tmp_path / "regularized" / "iterations" / f"{index:02d}" / "variances"
        values = _pstrainc.read_gau(str(path))[0]
        assert np.all(values >= floor)
    assert "variance_regularization" not in load_bw_telemetry(tmp_path / "default")["passes"][0]
