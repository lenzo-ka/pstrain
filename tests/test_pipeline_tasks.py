"""Tests for pstrain.lib.pipeline.tasks.

Validates that the task graph builds cleanly, every registered target
resolves to a known output, and every task's declared inputs are produced
by some other task or treated as required external files.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import replace
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from pstrain.lib.pipeline import PipelineContext
from pstrain.lib.pipeline.context import DEFAULT_CONFIGS, FeatParams, RunnerParams, SplitParams
from pstrain.lib.pipeline.feat_params import (
    feat_params_lines,
    feature_extractor_config_from_record,
)
from pstrain.lib.pipeline.tasks import DEFAULT_TARGET, TARGETS, build_pipeline
from tests.clib import C_LIBRARY_AVAILABLE


@pytest.fixture
def empty_project(tmp_path: Path) -> Path:
    """A minimal project layout: empty fileids, presence of the required
    'shared' files so paths resolve. No actual audio/training data."""
    project = tmp_path / "proj"
    (project / "shared").mkdir(parents=True)
    (project / "audio").mkdir(parents=True)
    (project / "etc").mkdir(parents=True)
    (project / "experiments" / "default" / "etc").mkdir(parents=True)

    (project / "shared" / "phoneset.txt").write_text("AA\nAE\nB\nSIL\n")
    (project / "shared" / "dictionary.dict").write_text("HELLO HH EH L OW\n")
    (project / "etc" / "all.transcription").write_text("")
    (project / "audio" / "placeholder.wav").write_text("fake-wav")

    return project


def test_pipeline_builds_without_error(empty_project: Path) -> None:
    ctx = PipelineContext.from_config(empty_project)
    pl = build_pipeline(ctx)
    assert len(pl.tasks()) > 0
    assert len(pl.targets()) > 0


def test_all_registered_targets_have_producers(empty_project: Path) -> None:
    ctx = PipelineContext.from_config(empty_project)
    pl = build_pipeline(ctx)
    for target_name, target_path in pl.targets().items():
        # Every registered target's sentinel path must be produced by some task.
        all_outputs = {Path(o) for task in pl.tasks().values() for o in task.outputs}
        assert target_path in all_outputs, (
            f"target {target_name!r} resolves to {target_path}, which no task produces"
        )


def test_every_target_in_TARGETS_is_registered(empty_project: Path) -> None:
    ctx = PipelineContext.from_config(empty_project)
    pl = build_pipeline(ctx)
    registered = set(pl.targets())
    declared = {spec.name for spec in TARGETS}
    missing = declared - registered
    # `flat` is intentionally a build step but not currently registered as a
    # standalone target since it always runs as a dependency of ci-1g; allow it.
    missing.discard("flat")
    assert not missing, f"declared but not registered: {sorted(missing)}"


def test_target_registry_declares_one_default() -> None:
    defaults = [spec.name for spec in TARGETS if spec.default]
    assert defaults == ["cd-8g"]
    assert defaults[0] == DEFAULT_TARGET


def test_can_plan_each_ci_and_cd_target(empty_project: Path) -> None:
    """Plan every CI/CD target. With empty fileids, feature files are
    absent, so the plan will mark them stale. We just want a clean plan
    (no cycles, no missing producers)."""
    ctx = PipelineContext.from_config(empty_project)
    pl = build_pipeline(ctx)
    for spec in TARGETS:
        if spec.kind in {"ci", "cd"}:
            plan = pl.plan(spec.name)
            assert plan, f"empty plan for {spec.name}"
            # Every plan ends in a task that produces the target sentinel.
            last = plan[-1]
            sentinel = pl.targets()[spec.name]
            assert sentinel in {Path(p) for p in last.task.outputs}


def test_cd_8g_plan_includes_full_chain(empty_project: Path) -> None:
    """Sanity: building cd-8g should require flat, ci-1g, cd-untied, etc."""
    ctx = PipelineContext.from_config(empty_project)
    pl = build_pipeline(ctx)
    plan = pl.plan("cd-8g")
    names = [e.task.name for e in plan]
    for required in [
        "flat",
        "ci-1g",
        "cd-untied-init",
        "cd-untied",
        "questions",
        "trees",
        "prune-trees",
        "alltriphones-mdef",
        "cd-1g-init",
        "cd-1g",
        "cd-2g",
        "cd-4g",
        "cd-8g",
    ]:
        assert required in names, f"missing {required!r} in plan: {names}"

    # And the order respects dependencies for a few key pairs.
    assert names.index("flat") < names.index("ci-1g")
    assert names.index("ci-1g") < names.index("cd-untied-init")
    assert names.index("cd-untied") < names.index("trees")
    assert names.index("prune-trees") < names.index("cd-1g-init")
    assert names.index("cd-1g") < names.index("cd-2g") < names.index("cd-8g")


def test_fanout_tasks_share_parallel_group(empty_project: Path) -> None:
    """Add two audio files and ensure their extract tasks share the same
    parallel group. Note: extract tasks now derive from `audio/*.wav`
    (corpus-wide) rather than train.fileids, so the split task does not
    need to have run for fan-out planning."""
    for fid in ["utt_a", "utt_b"]:
        (empty_project / "audio" / f"{fid}.wav").write_text("fake-wav")

    ctx = PipelineContext.from_config(empty_project)
    pl = build_pipeline(ctx)
    tasks = pl.tasks()
    extracts = [t for name, t in tasks.items() if name.startswith("extract:")]
    assert len(extracts) == 3
    groups = {t.parallel_group for t in extracts}
    assert groups == {"features"}


def test_extract_task_forwards_lifter(empty_project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def capture(_audio: Path, _output: Path, params: dict[str, object]) -> None:
        captured.update(params)

    monkeypatch.setattr("pstrain.lib.pipeline.tasks._extract_features_worker", capture)
    ctx = PipelineContext(project_dir=empty_project, feat=FeatParams(lifter=17))
    build_pipeline(ctx).tasks()["extract:placeholder"].fn()

    assert captured["lifter"] == 17


def test_extract_task_forwards_every_recorded_waveform_field(empty_project: Path) -> None:
    """Both consumers receive declared non-defaults, checked independently of the schema."""
    feat = FeatParams(remove_noise=False, transform="legacy", frate=80, wlen=0.02)
    ctx = PipelineContext(project_dir=empty_project, feat=feat)
    task = build_pipeline(ctx).tasks()["extract:placeholder"]

    assert isinstance(task.fn, partial)
    params = task.fn.args[2]
    assert params["frate"] == 80
    assert params["transform"] == "legacy"
    record = dict(line.rstrip().split(maxsplit=1) for line in feat_params_lines(feat))
    alignment_params = feature_extractor_config_from_record(record)
    assert alignment_params["frate"] == 80
    assert alignment_params["transform"] == "legacy"


def test_extract_task_forwards_preemphasis_alpha(empty_project: Path) -> None:
    (empty_project / "etc" / "configs.yaml").write_text("custom:\n  features:\n    alpha: 0.42\n")
    ctx = PipelineContext.from_config(empty_project, config_name="custom")

    extract_task = build_pipeline(ctx).tasks()["extract:placeholder"]

    assert isinstance(extract_task.fn, partial)
    assert extract_task.fn.args[2]["alpha"] == 0.42


def test_meaningful_feature_config_change_rebuilds_features(
    empty_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs: list[float] = []

    def extract(_audio: Path, output: Path, params: dict[str, object]) -> None:
        runs.append(float(params["alpha"]))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("features")

    monkeypatch.setattr("pstrain.lib.pipeline.tasks._extract_features_worker", extract)
    configs = empty_project / "etc" / "configs.yaml"
    configs.write_text("custom:\n  features:\n    alpha: 0.42\n")
    first = build_pipeline(PipelineContext.from_config(empty_project, config_name="custom"))
    assert first.run("features", jobs=1) == 0
    old_feature = empty_project / "shared" / "features" / "custom" / "placeholder.mfc"
    assert old_feature.read_text() == "features"

    configs.write_text("custom:\n  features:\n    alpha: 0.21\n")
    changed = build_pipeline(PipelineContext.from_config(empty_project, config_name="custom"))
    stale = {entry.task.name for entry in changed.plan("features") if entry.stale}

    assert "provenance:features" in stale
    assert "extract:placeholder" in stale
    assert changed.run("features", jobs=1) == 0
    assert runs == [0.42, 0.21]


def test_reverting_feature_config_rebuilds_features(
    empty_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs: list[float] = []

    def extract(_audio: Path, output: Path, params: dict[str, object]) -> None:
        runs.append(float(params["alpha"]))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("features")

    monkeypatch.setattr("pstrain.lib.pipeline.tasks._extract_features_worker", extract)
    configs = empty_project / "etc" / "configs.yaml"

    for alpha in (0.42, 0.21, 0.42):
        configs.write_text(f"custom:\n  features:\n    alpha: {alpha}\n")
        pipeline = build_pipeline(PipelineContext.from_config(empty_project, config_name="custom"))
        assert pipeline.run("features", jobs=1) == 0

    assert runs == [0.42, 0.21, 0.42]


def test_irrelevant_config_edit_does_not_rebuild_features(
    empty_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs: list[str] = []

    def extract(_audio: Path, output: Path, _params: dict[str, object]) -> None:
        runs.append("ran")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("features")

    monkeypatch.setattr("pstrain.lib.pipeline.tasks._extract_features_worker", extract)
    configs = empty_project / "etc" / "configs.yaml"
    configs.write_text("custom:\n  features:\n    alpha: 0.42\n")
    pipeline = build_pipeline(PipelineContext.from_config(empty_project, config_name="custom"))
    assert pipeline.run("features", jobs=1) == 0

    configs.write_text(
        "# formatting and an unrelated profile changed\n"
        "unrelated:\n  training:\n    ci: {max_iterations: 3}\n"
        "custom: {features: {alpha: 0.42}}\n"
    )
    unchanged = build_pipeline(PipelineContext.from_config(empty_project, config_name="custom"))

    plan = unchanged.plan("features")
    assert not any(entry.stale for entry in plan), [
        (entry.task.name, entry.reason) for entry in plan if entry.stale
    ]
    assert unchanged.run("features", jobs=1) == 0
    assert runs == ["ran"]


def test_model_and_package_copy_build_provenance(
    empty_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = PipelineContext.from_config(empty_project)
    pipeline = build_pipeline(ctx)
    tasks = pipeline.tasks()
    tasks["provenance:training"].fn()

    def init_flat_model(_phones: list[str], output_dir: Path, **_kwargs: object) -> None:
        for name in ("mdef", "means", "variances", "mixture_weights", "transition_matrices"):
            (output_dir / name).write_text(name)

    def package_model(
        *, model_dir: Path, output_dir: Path, model_name: str, **_kwargs: object
    ) -> None:
        package_dir = output_dir / model_name
        acoustic = package_dir / "acoustic"
        acoustic.mkdir(parents=True)
        for name in (
            "feat.params",
            "mdef",
            "means",
            "variances",
            "mixture_weights",
            "transition_matrices",
        ):
            shutil.copyfile(model_dir / name, acoustic / name)
        (acoustic / "noisedict").write_text("")
        (package_dir / "README.txt").write_text("")

    monkeypatch.setattr("pstrain.lib.flat.init_flat_model", init_flat_model)
    monkeypatch.setattr("pstrain.lib.steps.package.package_model", package_model)
    tasks["flat"].fn()

    package_task = tasks["package-ci-8g"]
    ci_8g_dir = ctx.model_dir("ci-8g")
    ci_8g_dir.mkdir(parents=True)
    for source in ctx.model_files("flat"):
        shutil.copyfile(source, ci_8g_dir / source.name)
    package_task.fn()

    expected = ctx.provenance_document("training")
    assert json.loads((ctx.model_dir("flat") / "provenance.json").read_text()) == expected
    package_provenance = ctx.dist_dir / "ci-8g-default" / "provenance.json"
    assert json.loads(package_provenance.read_text()) == expected


def test_stage_fingerprints_cover_only_effective_relevant_values(empty_project: Path) -> None:
    base = PipelineContext.from_config(empty_project)
    feature_change = replace(base, feat=FeatParams(alpha=0.5))
    training_change = replace(
        base, train=replace(base.train, ci=replace(base.train.ci, max_iterations=3))
    )
    split_change = replace(base, split=SplitParams(seed=99))

    assert feature_change.provenance_path("features") != base.provenance_path("features")
    assert training_change.provenance_path("features") == base.provenance_path("features")
    assert split_change.provenance_path("features") == base.provenance_path("features")
    assert split_change.provenance_path("split") != base.provenance_path("split")
    assert training_change.provenance_path("split") == base.provenance_path("split")
    for changed in (feature_change, training_change, split_change):
        assert changed.provenance_path("training") != base.provenance_path("training")

    document = base.provenance_document("training")
    assert document["fingerprint"] in base.provenance_path("training").name


@pytest.mark.parametrize(
    ("multipron", "effective_shards"),
    [(False, 3), (True, 1)],
)
def test_training_provenance_declares_requested_and_effective_bw_shard_count(
    empty_project: Path, monkeypatch: pytest.MonkeyPatch, multipron: bool, effective_shards: int
) -> None:
    """Record requested jobs separately from the effective BW shard count."""
    monkeypatch.setattr("pstrain.lib.pipeline.context.socket.gethostname", lambda: "training-host")
    monkeypatch.setattr("pstrain.lib.pipeline.context.platform.machine", lambda: "test-arch")
    base = PipelineContext.from_config(empty_project)
    ctx = replace(
        base,
        runner=RunnerParams(jobs=3),
        train=replace(base.train, multipron_training=multipron),
    )

    execution = ctx.provenance_document("training")["execution"]

    assert execution == {
        "host": "training-host",
        "architecture": "test-arch",
        "requested_jobs": 3,
        "bw_shard_count": effective_shards,
    }


def test_exclusion_schedule_config_and_provenance_are_verbatim(empty_project: Path) -> None:
    schedule = {
        "ci-1g": {5: ["arctic_a0587"], 6: ["arctic_a0587"]},
        "cd-untied": {"*": ["arctic_a0587"]},
    }
    (empty_project / "etc" / "configs.yaml").write_text(
        yaml.safe_dump({"scheduled": {"training": {"exclusion_schedule": schedule}}})
    )

    ctx = PipelineContext.from_config(empty_project, config_name="scheduled")

    assert ctx.train.exclusion_schedule == schedule
    assert ctx.provenance_payload("training")["training"]["exclusion_schedule"] == schedule


def test_exclusion_schedule_does_not_change_decode_eval_inputs(empty_project: Path) -> None:
    (empty_project / "etc" / "configs.yaml").write_text(
        "scheduled:\n  training:\n    exclusion_schedule:\n      ci-8g: {'*': [arctic_a0587]}\n"
    )
    ctx = PipelineContext.from_config(empty_project, config_name="scheduled")
    task = build_pipeline(ctx).tasks()["test-ci-8g"]

    assert ctx.etc_dir / "test.transcription" in task.inputs
    assert ctx.etc_dir / "train.fileids" not in task.inputs
    assert ctx.etc_dir / "train.transcription" not in task.inputs


def test_native_library_identity_is_content_based_not_path_based(
    empty_project: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first_lib = tmp_path / "first" / "libpstrainc.so"
    second_lib = tmp_path / "second" / "libpstrainc.so"
    first_lib.parent.mkdir()
    second_lib.parent.mkdir()
    library_bytes = b"identical build"
    first_lib.write_bytes(library_bytes)
    second_lib.write_bytes(library_bytes)
    selected = first_lib
    monkeypatch.setattr("pstrain.lib.pipeline.context.get_lib_path", lambda: selected)
    monkeypatch.setattr("pstrain.lib.pipeline.context._fp_contract_policy", lambda: "off")

    ctx = PipelineContext.from_config(empty_project)
    first_path = ctx.provenance_path("training")
    selected = second_lib
    second_path = ctx.provenance_path("training")

    assert first_path == second_path
    assert ctx.provenance_payload("training")["native_library"] == {
        "sha256": hashlib.sha256(library_bytes).hexdigest(),
        "fp_contract_declared": "off",
    }
    assert ctx.provenance_document("training")["native_library"] == {
        "path": str(second_lib.resolve()),
        "sha256": hashlib.sha256(library_bytes).hexdigest(),
        "fp_contract_declared": "off",
    }


def test_native_library_content_changes_fingerprint(
    empty_project: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    library = tmp_path / "libpstrainc.so"
    library.write_bytes(b"first build")
    monkeypatch.setattr("pstrain.lib.pipeline.context.get_lib_path", lambda: library)
    monkeypatch.setattr("pstrain.lib.pipeline.context._fp_contract_policy", lambda: "off")
    ctx = PipelineContext.from_config(empty_project)

    first_path = ctx.provenance_path("training")
    library.write_bytes(b"second build")
    second_path = ctx.provenance_path("training")

    assert first_path != second_path


def test_missing_native_library_is_recorded(
    empty_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("pstrain.lib.pipeline.context.get_lib_path", lambda: None)

    payload = PipelineContext.from_config(empty_project).provenance_payload("features")

    assert payload["native_library"] == {"state": "absent"}


def test_provenance_rejects_non_finite_config(empty_project: Path) -> None:
    ctx = PipelineContext(project_dir=empty_project, feat=FeatParams(alpha=float("nan")))

    with pytest.raises(ValueError, match="Out of range float values"):
        ctx.provenance_path("features")


def test_nested_and_flat_audio_fanout_uses_relative_fileids(empty_project: Path) -> None:
    (empty_project / "audio" / "placeholder.wav").unlink()
    for relative_path in ["flat.wav", "spk1/utt2.wav", "spk1/utt1.wav", "spk2/utt1.wav"]:
        path = empty_project / "audio" / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fake-wav")

    ctx = PipelineContext.from_config(empty_project)
    tasks = build_pipeline(ctx).tasks()

    assert sorted(name for name in tasks if name.startswith("extract:")) == [
        "extract:flat",
        "extract:spk1/utt1",
        "extract:spk1/utt2",
        "extract:spk2/utt1",
    ]
    assert tasks["extract:spk1/utt1"].outputs == (ctx.features_dir / "spk1" / "utt1.mfc",)


def test_audio_fileids_are_recursive_sorted_relative_posix_paths(empty_project: Path) -> None:
    (empty_project / "audio" / "placeholder.wav").unlink()
    for relative_path in ["z.wav", "spk2/b.wav", "spk1/c.wav", "spk1/a.wav"]:
        path = empty_project / "audio" / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    ctx = PipelineContext.from_config(empty_project)

    assert ctx.audio_fileids() == ["spk1/a", "spk1/c", "spk2/b", "z"]
    assert all("\\" not in fileid for fileid in ctx.audio_fileids())


def test_empty_audio_directory_fails_during_pipeline_construction(
    empty_project: Path,
) -> None:
    (empty_project / "audio" / "placeholder.wav").unlink()
    ctx = PipelineContext.from_config(empty_project)

    with pytest.raises(ValueError, match=r"No audio files found.*\*\*/\*\.wav"):
        build_pipeline(ctx)


def test_missing_audio_directory_fails_during_pipeline_construction(
    empty_project: Path,
) -> None:
    (empty_project / "audio" / "placeholder.wav").unlink()
    (empty_project / "audio").rmdir()
    ctx = PipelineContext.from_config(empty_project)

    with pytest.raises(ValueError, match=r"No audio files found.*\*\*/\*\.wav"):
        build_pipeline(ctx)


def test_split_task_produces_fileid_files(empty_project: Path) -> None:
    """Split should be registered as a task with the typed expected outputs
    and as a target whose sentinel is train.fileids."""
    ctx = PipelineContext.from_config(empty_project)
    pl = build_pipeline(ctx)

    split = pl.tasks()["split"]
    output_names = {p.name for p in split.outputs}
    assert output_names == {
        "train.fileids",
        "test.fileids",
        "train.transcription",
        "test.transcription",
        "test.decoder.transcription",
        ".split.generated.json",
        ".split.validated.json",
    }
    assert pl.targets()["split"].name == ".split.validated.json"


def test_split_runs_end_to_end_and_partitions(tmp_path: Path) -> None:
    """The split task should write all four files when invoked."""
    project = tmp_path / "proj"
    (project / "etc").mkdir(parents=True)
    (project / "shared").mkdir(parents=True)
    (project / "audio").mkdir(parents=True)
    (project / "experiments" / "default" / "etc").mkdir(parents=True)
    (project / "shared" / "phoneset.txt").write_text("AA\nB\n")
    (project / "shared" / "dictionary.dict").write_text("HI HH AY\n")
    (project / "audio" / "placeholder.wav").write_text("fake-wav")

    transcripts = "\n".join(f"utt_{i:03d} HELLO WORLD" for i in range(20)) + "\n"
    (project / "etc" / "all.transcription").write_text(transcripts)

    ctx = PipelineContext.from_config(project)
    pl = build_pipeline(ctx)
    assert pl.run("split") == 0

    etc = project / "experiments" / "default" / "etc"
    train_ids = (etc / "train.fileids").read_text().splitlines()
    test_ids = (etc / "test.fileids").read_text().splitlines()
    assert len(train_ids) + len(test_ids) == 20
    assert set(train_ids).isdisjoint(set(test_ids))


def test_editing_persistent_split_revalidates_and_changes_membership(tmp_path: Path) -> None:
    """A consistent edit becomes authoritative and invalidates the split marker."""
    project = tmp_path / "proj"
    (project / "etc").mkdir(parents=True)
    (project / "shared").mkdir()
    (project / "audio").mkdir()
    etc = project / "experiments" / "default" / "etc"
    etc.mkdir(parents=True)
    (project / "shared" / "phoneset.txt").write_text("AA\n")
    (project / "shared" / "dictionary.dict").write_text("WORD W ER D\n")
    rows = {f"utt_{i}": f"WORD {i}" for i in range(4)}
    (project / "etc" / "all.transcription").write_text(
        "".join(f"{fileid} {text}\n" for fileid, text in rows.items())
    )
    for fileid in rows:
        (project / "audio" / f"{fileid}.wav").write_bytes(b"wav")

    assert build_pipeline(PipelineContext.from_config(project)).run("split") == 0
    original_train = (etc / "train.fileids").read_text().splitlines()
    original_test = (etc / "test.fileids").read_text().splitlines()
    moved = original_train[-1]
    new_train = original_train[:-1]
    new_test = [moved, *original_test]
    (etc / "train.fileids").write_text("".join(f"{fileid}\n" for fileid in new_train))
    (etc / "test.fileids").write_text("".join(f"{fileid}\n" for fileid in new_test))
    (etc / "train.transcription").write_text(
        "".join(f"{fileid} {rows[fileid]}\n" for fileid in new_train)
    )
    (etc / "test.transcription").write_text(
        "".join(f"{fileid} {rows[fileid]}\n" for fileid in new_test)
    )
    before = {
        name: (etc / name).read_bytes()
        for name in ("train.fileids", "test.fileids", "train.transcription", "test.transcription")
    }

    rerun = build_pipeline(PipelineContext.from_config(project))
    plan = rerun.plan("split")
    assert plan[-1].stale
    assert plan[-1].reason == "missing completion marker"
    assert rerun.run("split") == 0

    assert {name: (etc / name).read_bytes() for name in before} == before
    assert json.loads((etc / ".split.validated.json").read_text())["mode"] == "external"
    assert (etc / "train.fileids").read_text().splitlines() == new_train
    assert (etc / "test.fileids").read_text().splitlines() == new_test

    # Once external mode is established, a further ordered edit invalidates
    # the existing completion marker through the files' declared-input mtimes.
    reordered_train = list(reversed(new_train))
    (etc / "train.fileids").write_text("".join(f"{fileid}\n" for fileid in reordered_train))
    (etc / "train.transcription").write_text(
        "".join(f"{fileid} {rows[fileid]}\n" for fileid in reordered_train)
    )
    edited = build_pipeline(PipelineContext.from_config(project))
    edited_plan = edited.plan("split")
    assert edited_plan[-1].stale
    assert edited_plan[-1].reason == "inputs not older than outputs"
    assert edited.run("split") == 0
    assert (etc / "train.fileids").read_text().splitlines() == reordered_train


def test_tree_building_is_fanned_out(empty_project: Path) -> None:
    """Each (phone, state) gets its own task in the 'trees' parallel
    group, plus a sentinel 'trees' task that depends on all of them."""
    # The empty_project fixture writes a 4-phone phoneset; tree building
    # skips SIL, leaving 3 phones x 3 states = 9 per-tree tasks.
    ctx = PipelineContext.from_config(empty_project)
    pl = build_pipeline(ctx)
    tasks = pl.tasks()

    per_tree = [t for name, t in tasks.items() if name.startswith("tree:")]
    assert len(per_tree) == 9
    assert all(t.parallel_group == "trees" for t in per_tree)

    sentinel = tasks["trees"]
    assert sentinel.parallel_group == ""
    tree_outputs = {p for t in per_tree for p in t.outputs}
    assert tree_outputs.issubset(set(sentinel.inputs))


def test_model_tasks_depend_on_split_outputs(empty_project: Path) -> None:
    """flat and ci-Ng tasks should depend on train.fileids etc, which the
    split task produces. Planning cd-1g without prior split should plan
    split first."""
    ctx = PipelineContext.from_config(empty_project)
    pl = build_pipeline(ctx)
    # Wipe the post-split files so split is stale.
    for fname in ["train.fileids", "test.fileids", "train.transcription"]:
        (empty_project / "experiments" / "default" / "etc" / fname).unlink(missing_ok=True)
    plan = pl.plan("cd-1g")
    names = [e.task.name for e in plan]
    assert "split" in names
    assert names.index("split") < names.index("flat")


def test_named_config_overrides_defaults(empty_project: Path) -> None:
    """The 'telephone' config should change sample rate and filter count."""
    ctx = PipelineContext.from_config(empty_project, config_name="telephone")
    assert ctx.feat.samprate == 8000
    assert ctx.feat.nfilt == 15
    assert ctx.feat.lowerf == 200
    assert ctx.feat.upperf == 3500
    # And the features dir reflects the config name.
    assert ctx.features_dir.name == "telephone"


def test_unknown_config_raises(empty_project: Path) -> None:
    with pytest.raises(ValueError, match="unknown config"):
        PipelineContext.from_config(empty_project, config_name="nonsense")


def test_every_shipped_profile_loads(tmp_path: Path) -> None:
    project = tmp_path / "project"
    (project / "etc").mkdir(parents=True)
    shipped_configs = Path(__file__).parents[1] / "etc" / "configs.yaml"
    shutil.copyfile(shipped_configs, project / "etc" / "configs.yaml")
    profiles = yaml.safe_load(shipped_configs.read_text())["profiles"]

    for profile in profiles:
        context = PipelineContext.from_config(project, config_name=profile)
        assert context.config_name == profile


def test_shipped_profiles_equal_builtin_defaults() -> None:
    shipped_configs = Path(__file__).parents[1] / "etc" / "configs.yaml"
    document = yaml.safe_load(shipped_configs.read_text())
    assert document["config_version"] == 1
    assert set(document["profiles"]) == set(DEFAULT_CONFIGS)


@pytest.mark.parametrize(
    ("block", "key", "expected"),
    [
        ("features", "alphaa", "feature"),
        ("training", "iterationz", "training"),
        ("split", "sead", "split"),
    ],
)
def test_unknown_profile_parameter_names_context(
    empty_project: Path, block: str, key: str, expected: str
) -> None:
    (empty_project / "etc" / "configs.yaml").write_text(f"sphinxtrain:\n  {block}:\n    {key}: 1\n")

    with pytest.raises(
        ValueError,
        match=rf"unknown {expected} parameter '{key}' in profile 'sphinxtrain'",
    ):
        PipelineContext.from_config(empty_project, config_name="sphinxtrain")


def test_multipron_training_defaults_on(empty_project: Path) -> None:
    """Multi-pron training is on by default at every layer."""
    ctx = PipelineContext.from_config(empty_project)
    assert ctx.train.multipron_training is True
    assert ctx.train.untied_inventory == "all-triphone"


def test_optional_final_silence_defaults_on_and_can_be_disabled(empty_project: Path) -> None:
    """The schema declaration owns both the corrected default and its legacy gate."""
    assert PipelineContext.from_config(empty_project).train.optional_final_silence is True
    (empty_project / "etc" / "configs.yaml").write_text(
        "default:\n  training:\n    optional_final_silence: false\n"
    )
    assert PipelineContext.from_config(empty_project).train.optional_final_silence is False


def test_failed_alignment_position_defaults_to_recover_and_resolves_omit(
    empty_project: Path,
) -> None:
    assert PipelineContext.from_config(empty_project).train.failed_alignment == "recover"
    (empty_project / "etc" / "configs.yaml").write_text(
        "default:\n  training:\n    failed_alignment: omit\n"
    )
    assert PipelineContext.from_config(empty_project).train.failed_alignment == "omit"


def test_linear_training_defaults_to_occurrence_inventory(empty_project: Path) -> None:
    """An omitted inventory policy retains the pre-PR31 linear behavior."""
    (empty_project / "etc" / "configs.yaml").write_text(
        "default:\n  training:\n    multipron_training: false\n"
    )

    ctx = PipelineContext.from_config(empty_project)

    assert ctx.train.multipron_training is False
    assert ctx.train.untied_inventory == "linear"


@pytest.mark.parametrize(
    ("multipron", "policy"),
    [
        (True, "linear"),
        (True, "all-triphone"),
        (False, "linear"),
        (False, "all-triphone"),
    ],
)
def test_explicit_untied_inventory_is_honored(
    empty_project: Path, multipron: bool, policy: str
) -> None:
    (empty_project / "etc" / "configs.yaml").write_text(
        "default:\n  training:\n"
        f"    multipron_training: {str(multipron).lower()}\n"
        f"    untied_inventory: {policy}\n"
    )

    assert PipelineContext.from_config(empty_project).train.untied_inventory == policy


def test_resolved_untied_inventory_appears_in_provenance(empty_project: Path) -> None:
    (empty_project / "etc" / "configs.yaml").write_text(
        "default:\n  training:\n    multipron_training: false\n"
    )

    payload = PipelineContext.from_config(empty_project).provenance_payload("training")

    assert payload["training"]["untied_inventory"] == "linear"


@pytest.mark.skipif(not C_LIBRARY_AVAILABLE, reason="libpstrainc not built")
def test_linear_default_untied_stage_builds_occurrence_inventory(
    empty_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The untied-init stage retains the mini fixture's pre-PR31 count."""
    (empty_project / "shared" / "phoneset.txt").write_text("AA\nAE\nAH\nB\nD\nSIL\n")
    (empty_project / "shared" / "dictionary.dict").write_text(
        "BAD B AE D\nDAD D AE D\nADD AE D\nBAA B AA\n"
    )
    (empty_project / "shared" / "filler.dict").write_text("<s> SIL\n</s> SIL\n<sil> SIL\n")
    (empty_project / "experiments" / "default" / "etc" / "train.transcription").write_text(
        "<s> BAD DAD </s> (utt1)\n<s> ADD BAA </s> (utt2)\n"
    )
    (empty_project / "etc" / "configs.yaml").write_text(
        "default:\n  training:\n    multipron_training: false\n"
    )
    monkeypatch.setattr("pstrain.lib.steps.cd_pipeline.run_init_cd_untied", lambda **_kwargs: None)
    monkeypatch.setattr("pstrain.lib.pipeline.tasks._record_model_provenance", lambda *_args: None)

    ctx = PipelineContext.from_config(empty_project)
    build_pipeline(ctx).tasks()["cd-untied-init"].fn()

    mdef_lines = (ctx.model_dir("cd-untied-init") / "mdef").read_text().splitlines()
    assert "10 n_tri" in mdef_lines


def test_transcript_reachable_untied_inventory_can_be_selected(empty_project: Path) -> None:
    """The PP3g inventory dial is opt-in under graph training."""
    (empty_project / "etc" / "configs.yaml").write_text(
        "default:\n  description: reachable\n  training:\n"
        "    untied_inventory: transcript-reachable\n"
    )
    ctx = PipelineContext.from_config(empty_project)
    assert ctx.train.untied_inventory == "transcript-reachable"
    assert ctx.train.multipron_training is True


def test_transcript_reachable_untied_inventory_rejects_linear_mode(
    empty_project: Path,
) -> None:
    """Linear training uses the equivalent occurrence-based inventory policy."""
    (empty_project / "etc" / "configs.yaml").write_text(
        "default:\n  description: redundant\n  training:\n"
        "    multipron_training: false\n"
        "    untied_inventory: transcript-reachable\n"
    )

    with pytest.raises(
        ValueError,
        match=(
            r"training\.untied_inventory 'transcript-reachable' requires "
            r"training\.multipron_training: true; linear mode's equivalent is the "
            r"'linear' policy"
        ),
    ):
        PipelineContext.from_config(empty_project)


def test_multipron_training_can_be_disabled(empty_project: Path) -> None:
    """Per-config opt-out via etc/configs.yaml."""
    (empty_project / "etc" / "configs.yaml").write_text(
        "default:\n  description: legacy\n  training:\n    multipron_training: false\n"
    )
    ctx = PipelineContext.from_config(empty_project)
    assert ctx.train.multipron_training is False


def test_bw_config_multipron_default_on() -> None:
    """BWConfig also defaults multipron on so library callers
    get the new behavior unless they explicitly opt out."""
    from pstrain.lib.bw import BWConfig

    assert BWConfig(pass2var=True, unobserved_gaussian_policy="zero").multipron is True
    assert (
        BWConfig(pass2var=True, unobserved_gaussian_policy="zero", multipron=False).multipron
        is False
    )


def test_bw_mixture_floor_is_higher_only_for_tied_cd_stages() -> None:
    from pstrain.lib.pipeline.tasks import _mixw_floor_for_stage

    assert _mixw_floor_for_stage("ci-8g") == 1e-8
    assert _mixw_floor_for_stage("cd-untied") == 1e-8
    assert _mixw_floor_for_stage("cd-1g") == 1e-5
    assert _mixw_floor_for_stage("cd-8g") == 1e-5


def test_bw_transition_floor_mapping_covers_every_training_stage() -> None:
    from pstrain.lib.pipeline.tasks import _tmat_floor_for_stage

    bw_stages = [spec.name for spec in TARGETS if spec.kind in {"ci", "cd"} and spec.name != "flat"]
    actual = {stage: _tmat_floor_for_stage(stage) for stage in bw_stages}
    assert actual == {
        "ci-1g": 1e-4,
        "ci-2g": 1e-4,
        "ci-4g": 1e-4,
        "ci-8g": 1e-4,
        "cd-untied": 1e-4,
        "cd-1g": 1e-5,
        "cd-2g": 1e-5,
        "cd-4g": 1e-5,
        "cd-8g": 1e-5,
        "cd-16g": 1e-5,
        "cd-32g": 1e-5,
    }


def test_slt_profile_resolves_per_stage_schedule(empty_project: Path) -> None:
    """The standard parity profile exposes the intended effective schedule."""
    from pstrain.lib.pipeline.tasks import _bw_policy_for_stage

    ctx = PipelineContext.from_config(empty_project, config_name="sphinxtrain")
    ci, ci_first_2pass = _bw_policy_for_stage(ctx, "ci-1g")
    untied, untied_first_2pass = _bw_policy_for_stage(ctx, "cd-untied")
    tied, tied_first_2pass = _bw_policy_for_stage(ctx, "cd-8g")

    assert (ci.max_iterations, ci.min_iterations, ci.convergence_ratio) == (10, 1, 0.001)
    assert (untied.max_iterations, untied.min_iterations, untied.convergence_ratio) == (
        6,
        1,
        0.001,
    )
    assert (tied.max_iterations, tied.min_iterations, tied.convergence_ratio) == (10, 1, 0.001)
    assert (ci_first_2pass, untied_first_2pass, tied_first_2pass) == (False, True, False)


def test_stage_variance_policy_reaches_first_engine_iteration(empty_project: Path) -> None:
    """CI is one-pass and untied is centered at the orchestration/engine seam."""
    from pstrain.lib.bw import BWConfig
    from pstrain.lib.pipeline.tasks import _bw_policy_for_stage
    from pstrain.lib.steps.train import _config_for_iteration

    ctx = PipelineContext.from_config(empty_project)
    base = BWConfig(pass2var=True, unobserved_gaussian_policy="zero")
    observed = {}
    for stage in ("ci-1g", "cd-untied"):
        _, first_pass_2passvar = _bw_policy_for_stage(ctx, stage)
        observed[stage] = _config_for_iteration(
            base,
            multipron=True,
            iteration=1,
            first_pass_2passvar=first_pass_2passvar,
        ).pass2var

    assert observed == {"ci-1g": False, "cd-untied": True}


def test_bw_config_requires_explicit_normalization_policies() -> None:
    """Library callers cannot inherit free-floating normalization defaults."""
    from pstrain.lib.bw import BWConfig

    with pytest.raises(TypeError, match="pass2var"):
        BWConfig()  # type: ignore[call-arg]

    with pytest.raises(TypeError, match="unobserved_gaussian_policy"):
        BWConfig(pass2var=True)  # type: ignore[call-arg]


def test_configured_bw_parameters_reach_training_call(
    empty_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The public training profile must drive the actual BW call."""
    from pstrain.lib.steps.train import TrainingResult

    (empty_project / "etc" / "configs.yaml").write_text(
        "default:\n  training:\n    a_beam: 1e-123\n    b_beam: 1e-9\n"
        "    ci: {max_iterations: 7, convergence_ratio: 0.004, min_iterations: 3}\n"
        "    max_skip_fraction: 0.02\n    retry_beam_factor: 1e12\n"
        "    failed_alignment: omit\n"
        "    bw_checkpoint_iterations: true\n"
    )
    ctx = PipelineContext.from_config(empty_project)
    flat = ctx.model_dir("flat")
    flat.mkdir(parents=True)
    for name in ("mdef", "means", "variances", "mixture_weights", "transition_matrices"):
        (flat / name).write_text(name)

    captured: dict[str, object] = {}

    class FakeFFI:
        NULL = None

        @staticmethod
        def new(cdecl: str) -> SimpleNamespace:
            assert cdecl == "pstrain_bw_config_t *"
            return SimpleNamespace()

    class FakeLib:
        def pstrain_bw_init(self, *args: object) -> object:
            captured["c_config"] = args[-1]
            return object()

        @staticmethod
        def pstrain_bw_set_multipron(ctx: object, enabled: int) -> int:
            return 0

        @staticmethod
        def pstrain_bw_free(ctx: object) -> None:
            pass

    monkeypatch.setattr("pstrain.lib._pstrainc._init", lambda: (FakeFFI(), FakeLib()))

    def fake_bw(**kwargs: object) -> TrainingResult:
        from pstrain.lib.bw import BWTrainer
        from pstrain.lib.steps.train import _config_for_iteration

        captured.update(kwargs)
        model = Path(kwargs["model_dir"])  # type: ignore[arg-type]
        engine_config = _config_for_iteration(
            kwargs["config"],  # type: ignore[arg-type]
            multipron=bool(kwargs["multipron"]),
            iteration=1,
            first_pass_2passvar=bool(kwargs["first_pass_2passvar"]),
        )
        with monkeypatch.context() as local_patch:
            # This unit test deliberately substitutes the CFFI constructor;
            # exercise that child-side implementation without spawning.
            local_patch.setattr("pstrain.lib.native_worker.in_worker", lambda: True)
            trainer = BWTrainer(
                model / "mdef",
                model / "means",
                model / "variances",
                model / "mixture_weights",
                model / "transition_matrices",
                config=engine_config,
            )
        del trainer
        output = Path(kwargs["output_dir"])  # type: ignore[arg-type]
        output.mkdir(parents=True, exist_ok=True)
        for name in ("means", "variances", "mixture_weights", "transition_matrices"):
            (output / name).write_text(name)
        return TrainingResult(1, False, -1.0, 1, 1)

    monkeypatch.setattr("pstrain.lib.steps.train.run_bw_training", fake_bw)
    tasks = build_pipeline(ctx).tasks()
    tasks["provenance:training"].fn()
    tasks["ci-1g"].fn()

    config: Any = captured["config"]
    assert config.a_beam == 1e-123
    assert config.b_beam == 1e-9
    assert config.topn == 1
    assert config.mixw_floor == 1e-8
    assert config.tmat_floor == 1e-4
    assert config.unobserved_gaussian_policy == "zero"
    c_config: Any = captured["c_config"]
    assert c_config.a_beam == 1e-123
    assert c_config.b_beam == 1e-9
    assert c_config.topn == 1
    assert c_config.mixw_floor == 1e-8
    assert c_config.tmat_floor == 1e-4
    assert c_config.unobserved_gaussian_policy == 1
    assert c_config.pass2var == 0
    assert c_config.optional_final_silence == 1
    assert captured["convergence_ratio"] == 0.004
    assert captured["min_iterations"] == 3
    assert captured["n_iter"] == 7
    assert captured["first_pass_2passvar"] is False
    assert captured["max_skip_fraction"] == 0.02
    assert captured["retry_beam_factor"] == 1e12
    assert captured["failed_alignment"] == "omit"
    assert captured["checkpoint_iterations"] is True


def test_configured_untied_schedule_and_variance_reach_training_call(
    empty_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Untied schedule values and first-pass 2passvar cross orchestration."""
    from pstrain.lib.steps.train import TrainingResult

    (empty_project / "etc" / "configs.yaml").write_text(
        "default:\n  training:\n"
        "    untied: {max_iterations: 4, min_iterations: 2, convergence_ratio: 0.02}\n"
    )
    ctx = PipelineContext.from_config(empty_project)
    source = ctx.model_dir("cd-untied-init")
    source.mkdir(parents=True)
    for name in ("mdef", "means", "variances", "mixture_weights", "transition_matrices"):
        (source / name).write_text(name)

    captured: dict[str, object] = {}

    def fake_bw(**kwargs: object) -> TrainingResult:
        captured.update(kwargs)
        output = Path(kwargs["output_dir"])  # type: ignore[arg-type]
        output.mkdir(parents=True)
        for name in ("means", "variances", "mixture_weights", "transition_matrices"):
            (output / name).write_text(name)
        return TrainingResult(1, False, -1.0, 1, 1)

    monkeypatch.setattr("pstrain.lib.steps.train.run_bw_training", fake_bw)
    tasks = build_pipeline(ctx).tasks()
    tasks["provenance:training"].fn()
    tasks["cd-untied"].fn()

    assert captured["n_iter"] == 4
    assert captured["min_iterations"] == 2
    assert captured["convergence_ratio"] == 0.02
    assert captured["first_pass_2passvar"] is True
