"""End-to-end training smoke test on a tiny real corpus.

Runs the actual training pipeline — feature extraction → flat init → ci-1g
Baum-Welch — on a 10-utterance CMU ARCTIC slice (tests/fixtures/mini_arctic)
and asserts it produces a valid, finite CI acoustic model.

This is the safety net Phase 1 exists to provide: a PR that breaks BW
training, flat init, or feature extraction turns this test red. Everything
here runs in-process against libpstrainc via CFFI (no CLI binaries needed), so
the only requirement is a built C library.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from tests.clib import requires_c_library

FIXTURE = Path(__file__).parent / "fixtures" / "mini_arctic"
MODEL_FILES = (
    "mdef",
    "means",
    "variances",
    "mixture_weights",
    "transition_matrices",
    "feat.params",
)


def _mdef_counts(path: Path) -> dict[str, int]:
    """Read the integer count fields from a text mdef header."""
    counts: dict[str, int] = {}
    with path.open() as handle:
        for line in handle:
            fields = line.split()
            if len(fields) == 2 and fields[0].isdigit():
                counts[fields[1]] = int(fields[0])
    return counts


def _mdef_senone_assignments(path: Path) -> tuple[list[int], list[int]]:
    """Read CI and CD emitting-state senone IDs from a text mdef."""
    ci_senones: list[int] = []
    cd_senones: list[int] = []
    with path.open() as handle:
        for line in handle:
            fields = line.split()
            if len(fields) < 8 or fields[0].startswith("#"):
                continue
            assignments = [int(field) for field in fields[6:] if field != "N"]
            if fields[1:3] == ["-", "-"]:
                ci_senones.extend(assignments)
            else:
                cd_senones.extend(assignments)
    return ci_senones, cd_senones


@requires_c_library
def test_build_ci_1g_produces_finite_model(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """features → flat → ci-1g yields a finite, converged CI model."""
    from pstrain.lib.pipeline import PipelineContext
    from pstrain.lib.pipeline.tasks import build_pipeline
    from pstrain.lib.setup import setup_project

    project_dir = tmp_path / "proj"

    setup_project(
        project_dir,
        transcription_path=FIXTURE / "transcription.txt",
        audio_path=FIXTURE / "wav",
        dictionary_path=FIXTURE / "dictionary.dict",
        phoneset_path=FIXTURE / "phoneset.txt",
        filler_dict_path=FIXTURE / "filler.dict",
    )

    ctx = PipelineContext.from_config(project_dir)
    with caplog.at_level("WARNING"):
        rc = build_pipeline(ctx).run("ci-1g", jobs=2)
    assert rc == 0, "pipeline run of ci-1g failed"
    assert "multipron_training=true" in caplog.text
    assert "fallback_senone" in caplog.text
    execution = ctx.provenance_document("training")["execution"]
    assert execution["requested_jobs"] == 2
    assert execution["bw_shard_count"] == 1

    model_dir = ctx.model_dir("ci-1g")
    for name in ("mdef", "means", "variances", "mixture_weights", "transition_matrices"):
        assert (model_dir / name).exists(), f"missing model file: {name}"

    # Every model parameter must be finite. Unobserved/degenerate states are
    # the classic way BW produces NaN/inf; this is the assertion that catches
    # a broken trainer or a phoneset with untrained phones.
    from pstrain.lib import _pstrainc

    means = _pstrainc.read_gau(str(model_dir / "means"))[0]
    variances = _pstrainc.read_gau(str(model_dir / "variances"))[0]
    mixw = _pstrainc.read_mixw_counts(str(model_dir / "mixture_weights"))[0]

    assert np.isfinite(means).all(), "non-finite values in means"
    assert np.isfinite(variances).all(), "non-finite values in variances"
    assert np.isfinite(mixw).all(), "non-finite values in mixture_weights"
    # Saved variances are upstream-style raw normalization output. Decode/BW
    # load applies the floor, including to exact-zero unobserved cells.
    assert (np.maximum(variances, np.float32(1e-4)) > 0).all()


@requires_c_library
def test_bw_preserves_extreme_forward_density_scale(tmp_path: Path) -> None:
    """Synthesized finite observations exercise BW below log(DBL_MIN)."""
    from pstrain.lib import _pstrainc
    from pstrain.lib.bw import BWConfig, BWTrainer
    from pstrain.lib.pipeline import PipelineContext
    from pstrain.lib.pipeline.tasks import build_pipeline
    from pstrain.lib.setup import setup_project

    project_dir = tmp_path / "proj"
    setup_project(
        project_dir,
        transcription_path=FIXTURE / "transcription.txt",
        audio_path=FIXTURE / "wav",
        dictionary_path=FIXTURE / "dictionary.dict",
        phoneset_path=FIXTURE / "phoneset.txt",
        filler_dict_path=FIXTURE / "filler.dict",
    )
    ctx = PipelineContext.from_config(project_dir)
    assert build_pipeline(ctx).run("flat", jobs=1) == 0

    model_dir = ctx.model_dir("flat")
    means = _pstrainc.read_gau(str(model_dir / "means"))[0]
    variances = _pstrainc.read_gau(str(model_dir / "variances"))[0]
    observation = np.full(39, float(means.max()) + 1000.0, dtype=np.float32)

    # This is the diagonal-Gaussian calculation used by gauden_compute_log
    # after its 0.0001 variance floor. Prove every codebook's real forward
    # offset (best log density minus MAX_LOG_DEN=10) crosses log(DBL_MIN).
    effective_vars = np.maximum(variances[:, 0, 0, :], 0.0001)
    normalizers = -0.5 * (
        observation.size * np.log(2.0 * np.pi) + np.log(effective_vars).sum(axis=1)
    )
    log_densities = normalizers - (
        np.square(observation - means[:, 0, 0, :]) / (2.0 * effective_vars)
    ).sum(axis=1)
    assert float(log_densities.max() - 10.0) < np.log(np.finfo(np.float64).tiny)

    features = np.repeat(observation[None, :], 160, axis=0)
    trainer = BWTrainer(
        mdef_path=model_dir / "mdef",
        means_path=model_dir / "means",
        vars_path=model_dir / "variances",
        mixw_path=model_dir / "mixture_weights",
        tmat_path=model_dir / "transition_matrices",
        config=BWConfig(
            pass2var=True,
            unobserved_gaussian_policy="zero",
            a_beam=1e-200,
            multipron=False,
        ),
    )
    trainer.set_dict(ctx.shared_dir / "dictionary.dict", ctx.filler_dict)

    assert trainer.process_utterance_text(features, "<s> author </s>")
    stats = trainer.get_stats()
    assert stats.total_utts == 1
    assert stats.total_frames == len(features)
    assert np.isfinite(stats.total_log_lik)


@requires_c_library
def test_features_extracted_for_every_utterance(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Feature extraction fans out to one .mfc per fixture utterance."""
    from pstrain.lib.pipeline import PipelineContext
    from pstrain.lib.pipeline.tasks import build_pipeline
    from pstrain.lib.setup import setup_project

    project_dir = tmp_path / "proj"
    setup_project(
        project_dir,
        transcription_path=FIXTURE / "transcription.txt",
        audio_path=FIXTURE / "wav",
        dictionary_path=FIXTURE / "dictionary.dict",
        phoneset_path=FIXTURE / "phoneset.txt",
        filler_dict_path=FIXTURE / "filler.dict",
    )

    ctx = PipelineContext.from_config(project_dir)
    rc = build_pipeline(ctx).run("features", jobs=2, verbose=True)
    assert rc == 0

    n_wav = len(list((FIXTURE / "wav").glob("*.wav")))
    mfcs = list(ctx.features_dir.glob("*.mfc"))
    assert len(mfcs) == n_wav, f"expected {n_wav} .mfc files, got {len(mfcs)}"
    assert list((project_dir / ".pstrain" / "timings").glob("*.json"))
    assert "Pipeline timings" in capsys.readouterr().out


@requires_c_library
def test_build_cd_8g_produces_genuine_tied_model(tmp_path: Path) -> None:
    """The complete CI → CD pipeline builds trees and trains all 8 densities."""
    from pstrain.lib import _pstrainc
    from pstrain.lib.pipeline import PipelineContext
    from pstrain.lib.pipeline.tasks import build_pipeline
    from pstrain.lib.setup import setup_project
    from pstrain.lib.steps.cd_pipeline import filter_tree_phones

    project_dir = tmp_path / "proj"
    setup_project(
        project_dir,
        transcription_path=FIXTURE / "transcription.txt",
        audio_path=FIXTURE / "wav",
        dictionary_path=FIXTURE / "dictionary.dict",
        phoneset_path=FIXTURE / "phoneset.txt",
        filler_dict_path=FIXTURE / "filler.dict",
    )

    ctx = PipelineContext.from_config(project_dir)
    # A8's saved-stage beam regression uses this deliberately wide value;
    # the public/upstream-compatible default remains 1e-90.
    ctx = replace(
        ctx,
        train=replace(ctx.train, a_beam=1e-200, optional_final_silence=False),
    )
    # Measured locally at 1.7 seconds (Apple M-series, Python 3.12, jobs=2).
    rc = build_pipeline(ctx).run("cd-8g", jobs=2)
    assert rc == 0, "pipeline run of cd-8g failed"

    for stage in ("cd-untied", "cd-1g-init", "cd-1g"):
        for name in MODEL_FILES:
            assert (ctx.model_dir(stage) / name).exists(), f"missing {stage} model file: {name}"

    questions = ctx.trees_dir / "questions"
    alltriphones = ctx.architecture_dir / "alltriphones.mdef"
    assert questions.exists(), "missing phonetic questions"
    assert alltriphones.exists(), "missing all-triphones mdef"

    phones = filter_tree_phones(ctx.shared_dir / "phoneset.txt")
    expected_trees = {
        f"{phone}-{state}.dtree" for phone in phones for state in range(ctx.train.n_state)
    }
    unpruned = {path.name: path for path in (ctx.trees_dir / "unpruned").glob("*.dtree")}
    pruned = {path.name: path for path in (ctx.trees_dir / "pruned").glob("*.dtree")}
    assert set(unpruned) == expected_trees, "missing or unexpected unpruned decision trees"
    assert set(pruned) == expected_trees, "missing or unexpected pruned decision trees"

    # build_tree_one() masks a C tree-build failure with this one-line stub.
    stub_prefix = "# Trivial tree for "
    for tree in (*unpruned.values(), *pruned.values()):
        assert not tree.read_text().startswith(stub_prefix), f"stub decision tree: {tree.name}"

    ci_counts = _mdef_counts(ctx.model_dir("ci-1g") / "mdef")
    alltri_counts = _mdef_counts(alltriphones)
    tied_mdef = ctx.model_dir("cd-1g") / "mdef"
    tied_counts = _mdef_counts(tied_mdef)
    ci_senones = ci_counts["n_tied_state"]
    tied_senones = tied_counts["n_tied_state"]
    assert ci_senones < tied_senones <= ctx.train.n_senones + ci_senones
    assert 1 < tied_senones < alltri_counts["n_tied_state"], "state tying is identity or degenerate"
    # Fixed seed + fixture + pruning target make this exact count deterministic:
    # 108 CI senones are retained and the trees contribute the requested 200.
    assert tied_senones == 308

    ci_assignments, cd_assignments = _mdef_senone_assignments(tied_mdef)
    # Exact CI identity coverage catches remapped, duplicated, or missing CI states.
    assert ci_assignments == list(range(ci_senones)), "CI senone assignments are not identities"
    expected_cd_senones = set(range(ci_senones, tied_senones))
    actual_cd_senones = set(cd_assignments)
    # Exact CD range coverage rejects both a constant mapping and the modulo
    # placeholder, which leaks assignments into the CI block.
    assert actual_cd_senones == expected_cd_senones, "CD senones do not cover the tied-state range"

    model_dir = ctx.model_dir("cd-1g")
    means, n_mgau, n_feat, n_density, veclen = _pstrainc.read_gau(str(model_dir / "means"))
    ci_means_info = _pstrainc.read_gau(str(ctx.model_dir("ci-1g") / "means"))
    variances = _pstrainc.read_gau(str(model_dir / "variances"))[0]
    mixw, n_mixw, mixw_feat, mixw_density = _pstrainc.read_mixw_counts(
        str(model_dir / "mixture_weights")
    )
    transition_matrices, n_tmat, n_state = _pstrainc.read_tmat_counts(
        str(model_dir / "transition_matrices")
    )[:3]

    assert (n_mgau, n_mixw) == (tied_senones, tied_senones)
    assert (n_feat, n_density, veclen) == ci_means_info[2:]
    assert (mixw_feat, mixw_density) == (n_feat, n_density)
    assert ctx.feat.ncep == 13
    assert n_state == ctx.train.n_state + 1
    assert n_tmat == ci_counts["n_tied_tmat"]
    assert transition_matrices.shape[1:] == (ctx.train.n_state, n_state)
    assert np.isfinite(means).all()
    assert np.isfinite(variances).all()
    assert np.isfinite(mixw).all()

    cd1_counts = _pstrainc.read_dnom(str(model_dir / "gauden_counts"))[0]
    zero_parameter_cells = np.all(means == 0, axis=-1) & np.all(variances == 0, axis=-1)
    assert np.all(cd1_counts[zero_parameter_cells] == 0)

    # Upstream stores the raw BW accumulators in these model files.  At one
    # density the mixture value is therefore the state occupancy, not 1.0.
    # The following CD stages successfully reload this model through
    # mod_inv_read_{mixw,tmat}, which normalize and floor it for evaluation.
    np.testing.assert_allclose(mixw, cd1_counts, rtol=1e-5, atol=1e-6)
    assert np.any(mixw.sum(axis=-1) > 1.0)
    assert np.any(transition_matrices.sum(axis=-1) > 1.0)

    counts, _, _, n_density = _pstrainc.read_dnom(str(ctx.model_dir("cd-8g") / "gauden_counts"))
    assert n_density == 8
    # topn=1 can update only one density for each state.  This assertion
    # proves the tied-stage BW pass accumulates posterior mass in several.
    assert np.any(np.count_nonzero(counts > 0, axis=2) > 1)
