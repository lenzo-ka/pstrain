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
from pstrain.lib.pipeline.context import DEFAULT_CONFIGS, FeatParams, SplitParams, TrainParams
from pstrain.lib.pipeline.tasks import TARGETS, build_pipeline


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
    (project / "experiments" / "default" / "etc" / "train.fileids").write_text("")
    (project / "experiments" / "default" / "etc" / "test.fileids").write_text("")
    (project / "experiments" / "default" / "etc" / "train.transcription").write_text("")
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
        "unrelated:\n  training:\n    max_iterations: 3\n"
        "custom: {features: {alpha: 0.42}}\n"
    )
    unchanged = build_pipeline(PipelineContext.from_config(empty_project, config_name="custom"))

    assert not any(entry.stale for entry in unchanged.plan("features"))
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
    training_change = replace(base, train=TrainParams(max_iterations=3))
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

    ctx = PipelineContext.from_config(empty_project)
    first_path = ctx.provenance_path("training")
    selected = second_lib
    second_path = ctx.provenance_path("training")

    assert first_path == second_path
    assert ctx.provenance_payload("training")["native_library"] == {
        "sha256": hashlib.sha256(library_bytes).hexdigest()
    }
    assert ctx.provenance_document("training")["native_library"] == {
        "path": str(second_lib.resolve()),
        "sha256": hashlib.sha256(library_bytes).hexdigest(),
    }


def test_native_library_content_changes_fingerprint(
    empty_project: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    library = tmp_path / "libpstrainc.so"
    library.write_bytes(b"first build")
    monkeypatch.setattr("pstrain.lib.pipeline.context.get_lib_path", lambda: library)
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
    """Split should be registered as a task with the four expected outputs
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
    }
    assert pl.targets()["split"].name == "train.fileids"


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
    profiles = yaml.safe_load(shipped_configs.read_text())

    for profile in profiles:
        context = PipelineContext.from_config(project, config_name=profile)
        assert context.config_name == profile


def test_shipped_profiles_equal_builtin_defaults() -> None:
    shipped_configs = Path(__file__).parents[1] / "etc" / "configs.yaml"
    assert yaml.safe_load(shipped_configs.read_text()) == DEFAULT_CONFIGS


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

    assert BWConfig().multipron is True
    assert BWConfig(multipron=False).multipron is False


def test_bw_mixture_floor_is_higher_only_for_tied_cd_stages() -> None:
    from pstrain.lib.pipeline.tasks import _mixw_floor_for_stage

    assert _mixw_floor_for_stage("ci-8g") == 1e-8
    assert _mixw_floor_for_stage("cd-untied") == 1e-8
    assert _mixw_floor_for_stage("cd-1g") == 1e-5
    assert _mixw_floor_for_stage("cd-8g") == 1e-5


def test_configured_bw_parameters_reach_training_call(
    empty_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The public training profile must drive the actual BW call."""
    from pstrain.lib.steps.train import TrainingResult

    (empty_project / "etc" / "configs.yaml").write_text(
        "default:\n  training:\n    a_beam: 1e-123\n    b_beam: 1e-9\n"
        "    convergence_ratio: 0.004\n    min_iterations: 3\n"
        "    max_skip_fraction: 0.02\n    retry_beam_factor: 1e12\n"
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

        captured.update(kwargs)
        model = Path(kwargs["model_dir"])  # type: ignore[arg-type]
        trainer = BWTrainer(
            model / "mdef",
            model / "means",
            model / "variances",
            model / "mixture_weights",
            model / "transition_matrices",
            config=kwargs["config"],  # type: ignore[arg-type]
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
    c_config: Any = captured["c_config"]
    assert c_config.a_beam == 1e-123
    assert c_config.b_beam == 1e-9
    assert c_config.topn == 1
    assert c_config.mixw_floor == 1e-8
    assert captured["convergence_ratio"] == 0.004
    assert captured["min_iterations"] == 3
    assert captured["max_skip_fraction"] == 0.02
    assert captured["retry_beam_factor"] == 1e12
