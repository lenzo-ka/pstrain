"""Pre-pin numerical-correctness program for the five BASIS choke points."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pstrain.lib import _pstrainc
from pstrain.lib.bw import BWConfig, BWTrainer
from pstrain.lib.features import read_sphinx_mfc
from pstrain.lib.pipeline import PipelineContext
from pstrain.lib.pipeline.tasks import TARGETS
from pstrain.lib.steps.train import run_bw_training
from tests.clib import requires_c_library
from tests.numeric_harness import (
    GOLDEN,
    GOLDEN_FILEIDS,
    SEED,
    create_project,
    golden_payload,
    read_model_arrays,
    sha256,
    strict_golden_enabled,
    train_golden,
    write_golden_subset,
)

_CHECKPOINT_MODEL_FILES = {
    "mdef",
    "means",
    "variances",
    "mixture_weights",
    "transition_matrices",
    "gauden_counts",
}
_M4_FIXTURE = Path(__file__).parent / "fixtures" / "multipron_final_state"


@pytest.fixture(scope="module")
def flat_project(tmp_path_factory: pytest.TempPathFactory) -> PipelineContext:
    """One fixed flat model shared by the BW-level invariants."""
    return create_project(tmp_path_factory.mktemp("numeric-flat") / "project")


@pytest.fixture(scope="module")
def full_project(tmp_path_factory: pytest.TempPathFactory) -> PipelineContext:
    """One full 1→2→4→8 run shared by split and tree invariants."""
    return create_project(
        tmp_path_factory.mktemp("numeric-full") / "project",
        "cd-8g",
        checkpoint_iterations=True,
    )


def _trainer(ctx: PipelineContext, *, multipron: bool = True) -> BWTrainer:
    model = ctx.model_dir("flat")
    trainer = BWTrainer(
        model / "mdef",
        model / "means",
        model / "variances",
        model / "mixture_weights",
        model / "transition_matrices",
        BWConfig(a_beam=1e-200, multipron=multipron),
    )
    trainer.set_dict(ctx.shared_dir / "dictionary.dict", ctx.filler_dict)
    return trainer


@requires_c_library
def test_feature_frames_finiteness_and_golden_checksum(flat_project: PipelineContext) -> None:
    """Choke point A: portable feature shape/envelope, plus optional bytes."""
    expected = json.loads(GOLDEN.read_text())["feature"]
    paths = sorted(flat_project.features_dir.glob("*.mfc"))
    assert len(paths) == 10
    for path in paths:
        features = read_sphinx_mfc(path)
        assert features.shape[0] > 0
        assert features.shape[1] == 13
        assert np.isfinite(features).all()
    anchor = flat_project.features_dir / f"{expected['fileid']}.mfc"
    values = read_sphinx_mfc(anchor)
    assert values.shape[0] == expected["frames"]
    assert values.size == expected["values"]
    observed = [values.min(), values.max(), values.mean(), values.std(), np.linalg.norm(values)]
    reference = [
        expected["minimum"],
        expected["maximum"],
        expected["mean"],
        expected["stddev"],
        expected["l2_norm"],
    ]
    tolerance = json.loads(GOLDEN.read_text())["feature_tolerance"]
    np.testing.assert_allclose(observed, reference, **tolerance)
    if strict_golden_enabled():
        assert sha256(anchor) == expected["sha256"]


@requires_c_library
def test_bw_golden_trajectory_and_accounting(flat_project: PipelineContext, tmp_path: Path) -> None:
    """Choke points B/C: BW numerics and utterance conservation cannot drift."""
    expected = json.loads(GOLDEN.read_text())
    result = train_golden(flat_project, tmp_path / "trained")
    actual = golden_payload(flat_project, result)
    tolerance = expected["strict_tolerance" if strict_golden_enabled() else "portable_tolerance"]
    assert len(actual["trajectory"]) == len(expected["trajectory"]) == 3
    for observed, golden in zip(actual["trajectory"], expected["trajectory"], strict=True):
        for key in (
            "iteration",
            "frames",
            "input_utts",
            "processed_utts",
            "retried_utts",
            "skipped_utts",
        ):
            assert observed[key] == golden[key]
        assert (
            observed["processed_utts"] + observed["retried_utts"] + observed["skipped_utts"]
            == observed["input_utts"]
        )
        assert observed["skipped_utts"] == 0
        np.testing.assert_allclose(
            [observed["total_log_lik"], observed["avg_log_prob"]],
            [golden["total_log_lik"], golden["avg_log_prob"]],
            **tolerance,
        )
        if golden["per_frame_delta"] is None:
            assert observed["per_frame_delta"] is None
        else:
            assert observed["per_frame_delta"] == pytest.approx(
                golden["per_frame_delta"], rel=tolerance["rtol"], abs=tolerance["atol"]
            )


@requires_c_library
def test_per_utterance_aggregation_conserves_totals(flat_project: PipelineContext) -> None:
    """Choke point B: utterance contributions sum to the batch accumulator."""
    transcripts = {
        line.split(maxsplit=1)[0]: line.split(maxsplit=1)[1]
        for line in (flat_project.project_dir / "etc" / "all.transcription")
        .read_text()
        .splitlines()
    }
    individual = []
    for fileid in GOLDEN_FILEIDS:
        trainer = _trainer(flat_project)
        mfcc = read_sphinx_mfc(flat_project.features_dir / f"{fileid}.mfc")
        assert trainer.process_utterance_mfcc(mfcc, f"<s> {transcripts[fileid]} </s>")
        individual.append(trainer.get_stats())

    combined = _trainer(flat_project)
    for fileid in GOLDEN_FILEIDS:
        mfcc = read_sphinx_mfc(flat_project.features_dir / f"{fileid}.mfc")
        assert combined.process_utterance_mfcc(mfcc, f"<s> {transcripts[fileid]} </s>")
    total = combined.get_stats()
    assert total.total_utts == sum(item.total_utts for item in individual)
    assert total.total_frames == sum(item.total_frames for item in individual)
    assert total.total_log_lik == pytest.approx(
        sum(item.total_log_lik for item in individual), rel=1e-14, abs=1e-8
    )


@requires_c_library
def test_real_training_retry_is_accounted_once_and_conserves_stats(
    flat_project: PipelineContext, tmp_path: Path
) -> None:
    """A recovered real retry is one input and one accumulated contribution."""
    fileid = "arctic_a0001"
    fileids = tmp_path / "retry.fileids"
    transcription = tmp_path / "retry.transcription"
    fileids.write_text(f"{fileid}\n")
    text = "author of the danger trail philip steels etc"
    transcription.write_text(f"{fileid} {text}\n")
    result = run_bw_training(
        flat_project.model_dir("flat"),
        tmp_path / "retried-model",
        flat_project.features_dir,
        fileids,
        transcription,
        flat_project.shared_dir / "dictionary.dict",
        flat_project.filler_dict,
        n_iter=1,
        config=BWConfig(a_beam=1e-1),
        retry_beam_factor=1e199,
    )
    row = result.trajectory[0]
    assert (row.input_utts, row.processed_utts, row.retried_utts, row.skipped_utts) == (1, 0, 1, 0)
    assert row.processed_utts + row.retried_utts + row.skipped_utts == row.input_utts

    direct = _trainer(flat_project)
    mfcc = read_sphinx_mfc(flat_project.features_dir / f"{fileid}.mfc")
    assert direct.process_utterance_mfcc(mfcc, f"<s> {text} </s>")
    expected = direct.get_stats()
    assert (row.frames, result.final_frames, result.final_utts) == (
        expected.total_frames,
        expected.total_frames,
        expected.total_utts,
    )
    assert row.total_log_lik == pytest.approx(expected.total_log_lik, rel=1e-14, abs=1e-8)


def _assert_normalized_model(model_dir: Path) -> None:
    arrays = read_model_arrays(model_dir)
    for name, values in arrays.items():
        assert np.isfinite(values).all(), name
    assert (arrays["variances"] >= 1e-4).all()
    np.testing.assert_allclose(arrays["mixture_weights"].sum(axis=-1), 1.0, rtol=1e-6, atol=1e-6)
    tmat_sums = arrays["transition_matrices"].sum(axis=-1)
    np.testing.assert_allclose(tmat_sums[tmat_sums > 0], 1.0, rtol=1e-6, atol=1e-6)


@requires_c_library
def test_iteration_checkpoints_are_opt_in_and_replace_stale_passes(
    flat_project: PipelineContext, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Normal training emits no checkpoints; the diagnostic mode is stage-clean."""
    fileids, transcription = write_golden_subset(flat_project)
    kwargs = {
        "model_dir": flat_project.model_dir("flat"),
        "features_dir": flat_project.features_dir,
        "train_fileids": fileids,
        "transcription": transcription,
        "dictionary": flat_project.shared_dir / "dictionary.dict",
        "filler_dict": flat_project.filler_dict,
        "min_iterations": 4,
        "config": BWConfig(a_beam=1e-200),
    }

    monkeypatch.delenv("PSTRAIN_BW_CHECKPOINTS", raising=False)
    default_output = tmp_path / "default-model"
    run_bw_training(output_dir=default_output, n_iter=1, **kwargs)
    assert not (default_output / "iterations").exists()

    enabled_output = tmp_path / "checkpointed-model"
    stale = enabled_output / "iterations" / "99"
    stale.mkdir(parents=True)
    (stale / "stale").write_text("stale")
    monkeypatch.setenv("PSTRAIN_BW_CHECKPOINTS", "1")
    run_bw_training(output_dir=enabled_output, n_iter=2, **kwargs)
    assert [path.name for path in sorted((enabled_output / "iterations").iterdir())] == [
        "01",
        "02",
    ]
    for checkpoint in (enabled_output / "iterations").iterdir():
        assert {path.name for path in checkpoint.iterdir()} == set(_CHECKPOINT_MODEL_FILES)


@requires_c_library
def test_updates_and_split_schedule_preserve_invariants(full_project: PipelineContext) -> None:
    """Choke points D/E: each pass/split stays valid at exactly 1→2→4→8."""
    exercised_specs = TARGETS[
        : next(i for i, spec in enumerate(TARGETS) if spec.name == "cd-8g") + 1
    ]
    bw_stages = [
        spec.name for spec in exercised_specs if spec.kind in {"ci", "cd"} and spec.name != "flat"
    ]
    senones: int | None = None
    for stage in bw_stages:
        model = full_project.model_dir(stage)
        checkpoints = sorted((model / "iterations").iterdir())
        assert checkpoints, f"no per-pass checkpoints for {stage}"
        for checkpoint in checkpoints:
            _assert_normalized_model(checkpoint)
        _assert_normalized_model(model)

    for density in (1, 2, 4, 8):
        model = full_project.model_dir(f"cd-{density}g")
        mixw, n_mixw, _, actual_density = _pstrainc.read_mixw(str(model / "mixture_weights"))
        assert actual_density == density
        if senones is None:
            senones = n_mixw
        assert n_mixw == senones
        counts, n_cb, _, count_density = _pstrainc.read_dnom(str(model / "gauden_counts"))
        assert (n_cb, count_density) == (n_mixw, density)
        observed = counts > 0
        # The report is intentionally explicit in assertion output: an added
        # unobserved density is a reviewable numerical event, never hidden.
        unobserved_report = np.argwhere(~observed).tolist()
        assert observed.all(), f"unobserved densities at {density}g: {unobserved_report}"


def _ci_state_ids(mdef: Path, phone: str) -> list[int]:
    for line in mdef.read_text().splitlines():
        fields = line.split()
        if len(fields) >= 8 and fields[:4] == [phone, "-", "-", "-"]:
            return [int(value) for value in fields[6:] if value != "N"]
    raise AssertionError(f"missing CI phone {phone}")


@requires_c_library
def test_multipron_second_variant_receives_occupancy(
    flat_project: PipelineContext, tmp_path: Path
) -> None:
    """M2: a matching second pronunciation changes stable phone occupancy."""
    engineered = tmp_path / "engineered.dict"
    base = (flat_project.shared_dir / "dictionary.dict").read_text()
    base = re.sub(r"^author .*$", "author K", base, flags=re.MULTILINE)
    engineered.write_text(base + "author(2) AO TH ER\n")
    mfcc = read_sphinx_mfc(flat_project.features_dir / "arctic_a0001.mfc")

    occupancies: dict[bool, np.ndarray[Any, Any]] = {}
    for enabled in (False, True):
        trainer = _trainer(flat_project, multipron=enabled)
        trainer.set_dict(engineered, flat_project.filler_dict)
        assert trainer.process_utterance_mfcc(mfcc, "<s> author </s>")
        counts_path = tmp_path / f"counts-{enabled}"
        assert trainer.save_density_counts(counts_path)
        occupancies[enabled] = _pstrainc.read_dnom(str(counts_path))[0]

    ao_states = _ci_state_ids(flat_project.model_dir("flat") / "mdef", "AO")
    off = float(occupancies[False][ao_states].sum())
    on = float(occupancies[True][ao_states].sum())
    assert off == 0.0
    assert on > 1.0


@requires_c_library
def test_multipron_variants_share_utterance_final_state(
    flat_project: PipelineContext, tmp_path: Path
) -> None:
    """M4: every last-word pronunciation can terminate through one </s>."""
    engineered = tmp_path / "terminal-variants.dict"
    base = (flat_project.shared_dir / "dictionary.dict").read_text()
    # The recording says the normal AO TH ER pronunciation. Put a deliberately
    # bad pronunciation last: before M4, context expansion duplicated </s> and
    # made only this last (K-ending) branch lead to n_state - 1.
    engineered.write_text(base + f"author(2) {' '.join(['K'] * 150)}\n")
    mfcc = read_sphinx_mfc(flat_project.features_dir / "arctic_a0001.mfc")

    trainer = _trainer(flat_project)
    trainer.set_dict(engineered, flat_project.filler_dict)
    n_state = trainer._ffi.new("uint32 *")
    states = trainer._lib.pstrain_bw_build_state_seq(trainer._ctx, b"<s> author </s>", n_state)
    assert states != trainer._ffi.NULL
    try:
        terminal_exits = [index for index in range(n_state[0]) if states[index].n_next == 0]
        assert terminal_exits == [n_state[0] - 1]

        terminal_exit = terminal_exits[0]
        terminal_phone = states[terminal_exit].phn
        terminal_starts = [
            index
            for index in range(n_state[0])
            if states[index].phn == terminal_phone and states[index].m_state == 0
        ]
        assert len(terminal_starts) == 2  # initial <s> and one terminal </s>
        terminal_start = terminal_starts[-1]
        branch_count = 2  # author and author(2)
        fan_in = [
            states[terminal_start].prior_state[offset]
            for offset in range(states[terminal_start].n_prior)
            if states[states[terminal_start].prior_state[offset]].phn != terminal_phone
        ]
        assert len(fan_in) == branch_count

        reachable = {terminal_exit}
        pending = [terminal_exit]
        while pending:
            state_id = pending.pop()
            for offset in range(states[state_id].n_prior):
                predecessor = states[state_id].prior_state[offset]
                if predecessor not in reachable:
                    reachable.add(predecessor)
                    pending.append(predecessor)
        assert reachable == set(range(n_state[0]))
    finally:
        trainer._lib.pstrain_bw_free_state_seq(states, n_state[0])

    assert trainer.process_utterance_mfcc(mfcc, "<s> author </s>")
    assert not trainer.final_state_not_reached


@requires_c_library
def test_cd_variant_boundaries_expand_both_triphone_contexts(
    full_project: PipelineContext,
) -> None:
    """M4b: fan-out and fan-in both select the matching untied triphone."""
    model = full_project.model_dir("cd-untied")
    trainer = BWTrainer(
        model / "mdef",
        model / "means",
        model / "variances",
        model / "mixture_weights",
        model / "transition_matrices",
        BWConfig(a_beam=1e-200, topn=1, multipron=True),
    )
    trainer.set_dict(full_project.shared_dir / "dictionary.dict", full_project.filler_dict)

    mdef_rows = list(_mdef_rows(model / "mdef"))
    expected_boundary_models = {
        ("AH", "SIL", "AH", "s"),
        ("AH", "SIL", "AE", "s"),
        ("EY", "SIL", "AH", "s"),
        ("EY", "SIL", "AE", "s"),
    }
    assert expected_boundary_models <= {tuple(row[:4]) for row in mdef_rows}

    # `a` has AH/EY variants and `and` has AH/AE initial phones.  Thus the
    # word boundary requires two right-context copies of each `a` variant and
    # two left-context copies of each `and` variant.  There are 14 phone HMMs:
    # two shared SILs, four `a` copies, four first-phone `and` copies, and the
    # four remaining phones in the two `and` variants.
    n_state = trainer._ffi.new("uint32 *")
    states = trainer._lib.pstrain_bw_build_state_seq(trainer._ctx, b"<s> a and </s>", n_state)
    assert states != trainer._ffi.NULL
    try:
        phone_starts = [index for index in range(n_state[0]) if states[index].m_state == 0]
        assert len(phone_starts) == 14
        graph_models = [tuple(mdef_rows[states[index].phn][:4]) for index in phone_starts]
        assert {model for model in graph_models if model in expected_boundary_models} == (
            expected_boundary_models
        )
        assert all(graph_models.count(model) == 1 for model in expected_boundary_models)
        terminal_exits = [index for index in range(n_state[0]) if states[index].n_next == 0]
        assert terminal_exits == [n_state[0] - 1]
    finally:
        trainer._lib.pstrain_bw_free_state_seq(states, n_state[0])


@requires_c_library
@pytest.mark.parametrize("fileid", ["arctic_a0257", "arctic_a0336", "arctic_b0424"])
def test_m4_real_utterances_reach_shared_final_state(
    fileid: str,
) -> None:
    """M4: representative shared and pstrain-only SLT failures train."""
    model = _M4_FIXTURE / "model"
    transcripts = {
        fields[0]: fields[1]
        for line in (_M4_FIXTURE / "transcription.txt").read_text().splitlines()
        if (fields := line.split(maxsplit=1))
    }
    trainer = BWTrainer(
        model / "mdef",
        model / "means",
        model / "variances",
        model / "mixture_weights",
        model / "transition_matrices",
        BWConfig(a_beam=1e-90, multipron=True),
    )
    trainer.set_dict(_M4_FIXTURE / "dictionary.dict", _M4_FIXTURE / "filler.dict")
    mfcc = read_sphinx_mfc(_M4_FIXTURE / f"{fileid}.mfc")

    assert trainer.process_utterance_mfcc(mfcc, f"<s> {transcripts[fileid]} </s>")
    assert not trainer.final_state_not_reached


def _mdef_rows(path: Path) -> Iterator[list[str]]:
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) >= 8 and not fields[0].isdigit() and not fields[0].startswith("#"):
            yield fields


def _tree_leaf(
    tree_path: Path, questions: dict[str, set[str]], base: str, left: str, right: str, pos: str
) -> int:
    lines = tree_path.read_text().splitlines()[1:]
    nodes = {int(fields[0]): fields for fields in map(str.split, lines)}
    leaf_labels: dict[int, int] = {}

    def label(node_id: int) -> None:
        fields = nodes[node_id]
        if fields[1] == "-":
            leaf_labels[node_id] = len(leaf_labels)
            return
        label(int(fields[1]))
        label(int(fields[2]))

    label(0)
    phone_by_context = {-1: left, 0: base, 1: right}
    position_by_name = {"WDBNDRY_B": "b", "WDBNDRY_E": "e", "WDBNDRY_S": "s", "WDBNDRY_I": "i"}
    node = 0
    while nodes[node][1] != "-":
        fields = nodes[node]
        expression = " ".join(fields[5:])
        terms = re.findall(r"(!?[A-Za-z0-9_]+) (-?\d+)", expression)
        matched = True
        for raw_name, raw_context in terms:
            negated = raw_name.startswith("!")
            name = raw_name.removeprefix("!")
            if name in position_by_name:
                value = pos == position_by_name[name]
            else:
                value = phone_by_context[int(raw_context)] in questions[name]
            matched &= not value if negated else value
        node = int(fields[1] if matched else fields[2])
    return leaf_labels[node]


@requires_c_library
def test_tied_assignments_match_independent_tree_walk(full_project: PipelineContext) -> None:
    """Declined-A1 debt: sampled mdef assignments equal independently walked leaves."""
    question_path = full_project.trees_dir / "questions"
    questions = {
        fields[0]: set(fields[1:])
        for fields in map(str.split, question_path.read_text().splitlines())
        if fields and not fields[0].startswith("WDBNDRY_")
    }
    ci_count = len(
        [
            state
            for row in _mdef_rows(full_project.model_dir("ci-1g") / "mdef")
            for state in row[6:]
            if state != "N"
        ]
    )
    offsets: dict[tuple[str, int], int] = {}
    next_id = ci_count
    for phone in (full_project.shared_dir / "phoneset.txt").read_text().splitlines():
        if phone.startswith("+") or phone == "SIL":
            continue
        for state in range(3):
            tree = full_project.trees_dir / "pruned" / f"{phone}-{state}.dtree"
            offsets[(phone, state)] = next_id
            next_id += sum(
                1 for line in tree.read_text().splitlines()[1:] if line.split()[1] == "-"
            )

    checked = 0
    for row in _mdef_rows(full_project.model_dir("cd-1g") / "mdef"):
        base, left, right, pos = row[:4]
        if left == "-" or (base, 0) not in offsets:
            continue
        for state, assigned in enumerate(row[6:-1]):
            leaf = _tree_leaf(
                full_project.trees_dir / "pruned" / f"{base}-{state}.dtree",
                questions,
                base,
                left,
                right,
                pos,
            )
            assert int(assigned) == offsets[(base, state)] + leaf
            checked += 1
        if checked >= 60:
            break
    assert checked >= 60


@requires_c_library
def test_seeded_bw_reruns_are_bit_identical(flat_project: PipelineContext, tmp_path: Path) -> None:
    """PP5 groundwork: the recorded seed produces bit-identical exercised outputs."""
    assert flat_project.split.seed == SEED
    fileids, transcription = write_golden_subset(flat_project)
    outputs = [tmp_path / "one", tmp_path / "two"]
    for output in outputs:
        run_bw_training(
            flat_project.model_dir("flat"),
            output,
            flat_project.features_dir,
            fileids,
            transcription,
            flat_project.shared_dir / "dictionary.dict",
            flat_project.filler_dict,
            n_iter=1,
            config=BWConfig(a_beam=1e-200),
        )
    for filename in (
        "means",
        "variances",
        "mixture_weights",
        "transition_matrices",
        "gauden_counts",
    ):
        assert (outputs[0] / filename).read_bytes() == (outputs[1] / filename).read_bytes(), (
            filename
        )
