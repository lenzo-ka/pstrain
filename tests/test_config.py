"""Canonical configuration contract tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from pstrain.lib.config.models import FeatureConfig, Profile, TrainingConfig
from pstrain.lib.config.resolver import (
    CONSUMER_TOUCHES,
    CONSUMERS,
    migrate_project,
    resolve_config,
    validate_consumer_coverage,
)
from pstrain.lib.config.schema import get_parameter
from pstrain.lib.pipeline.context import DEFAULT_CONFIGS, PipelineContext


def test_tree_intermediate_dumps_are_off_by_default() -> None:
    assert TrainingConfig().tree_intermediate_dumps is False


def test_bw_iteration_checkpoints_are_off_by_default() -> None:
    assert TrainingConfig().bw_checkpoint_iterations is False


def test_tree_semantic_fixes_are_on_by_default() -> None:
    training = TrainingConfig()
    assert training.tree_rotate_state_weights is True
    assert training.tree_directional_questions is True


def test_literal_parameter_types_list_admissible_values() -> None:
    parameter = get_parameter("training.failed_alignment")
    assert parameter is not None
    assert parameter.type == "recover | abort | omit"


def test_sharding_partition_positions_are_declared_and_default_unchanged() -> None:
    parameter = get_parameter("sharding.partition_position")
    assert parameter is not None
    assert parameter.type == "remainder-first | remainder-last"
    assert Profile().sharding.partition_position == "remainder-first"


def test_tree_state_weight_count_must_match_emitting_states() -> None:
    with pytest.raises(ValueError, match="expected 4, got 3"):
        TrainingConfig(n_state=4)


def test_active_names_and_defaults_are_canonical() -> None:
    profile = Profile()
    assert profile.features.ncep == 13
    assert profile.features.samprate == 16000
    assert profile.training.n_state == 3
    assert profile.runner.jobs is None
    assert profile.alignment.verbatim_tokens is False
    with pytest.raises(ValueError):
        FeatureConfig.model_validate({"num_ceps": 26})
    with pytest.raises(ValueError):
        TrainingConfig.model_validate({"n_states": 5})


def test_alignment_verbatim_tokens_resolves_opt_in(tmp_path: Path) -> None:
    (tmp_path / "etc").mkdir()
    (tmp_path / "etc" / "config.yaml").write_text(
        "config_version: 1\nalignment:\n  verbatim_tokens: true\n"
    )
    resolved = resolve_config(tmp_path)
    assert resolved.profile.alignment.verbatim_tokens is True
    assert resolved.fields["alignment.verbatim_tokens"].consumer == "cli.align"


@pytest.mark.parametrize(
    ("lower", "higher", "expected_kind"),
    [
        ("built-in", "user", "user"),
        ("user", "project-profile", "project-profile"),
        ("project-profile", "project", "project"),
        ("project", "experiment", "experiment"),
        ("experiment", "cli", "cli"),
    ],
)
def test_each_adjacent_layer_wins_in_isolation(
    tmp_path: Path, lower: str, higher: str, expected_kind: str
) -> None:
    project = tmp_path / "project"
    (project / "etc").mkdir(parents=True)
    (project / "experiments" / "trial").mkdir(parents=True)
    user = tmp_path / "user.yaml"
    values = {
        "built-in": 13,
        "user": 17,
        "project-profile": 26,
        "project": 39,
        "experiment": 52,
        "cli": 65,
    }
    active = {lower, higher}
    if "user" in active:
        user.write_text(f"config_version: 1\nfeatures:\n  ncep: {values['user']}\n")
    if "project-profile" in active:
        (project / "etc" / "configs.yaml").write_text(
            "config_version: 1\nprofiles:\n  custom:\n    extends: default\n"
            f"    features:\n      ncep: {values['project-profile']}\n"
        )
    if "project" in active:
        (project / "etc" / "config.yaml").write_text(
            f"config_version: 1\nfeatures:\n  ncep: {values['project']}\n"
        )
    if "experiment" in active:
        (project / "experiments" / "trial" / "config.yaml").write_text(
            f"config_version: 1\nfeatures:\n  ncep: {values['experiment']}\n"
        )
    resolved = resolve_config(
        project,
        profile_name="custom" if "project-profile" in active else "default",
        experiment="trial",
        cli_overrides={"features": {"ncep": values["cli"]}} if "cli" in active else None,
        user_config_path=user,
    )
    assert resolved.profile.features.ncep == values[higher]
    assert resolved.fields["features.ncep"].winner.source_kind == expected_kind


def test_fugu_project_profile_beats_user_reproduction(tmp_path: Path) -> None:
    project = tmp_path / "project"
    (project / "etc").mkdir(parents=True)
    user = tmp_path / "user.yaml"
    user.write_text("config_version: 1\nfeatures:\n  ncep: 17\n")
    (project / "etc" / "configs.yaml").write_text(
        "config_version: 1\nprofiles:\n  custom:\n    extends: default\n"
        "    features:\n      ncep: 26\n"
    )
    resolved = resolve_config(project, profile_name="custom", user_config_path=user)
    assert resolved.profile.features.ncep == 26
    assert resolved.fields["features.ncep"].winner.source_kind == "project-profile"


def test_cli_multipron_off_preserves_project_explicit_inventory(tmp_path: Path) -> None:
    (tmp_path / "etc").mkdir()
    (tmp_path / "etc" / "config.yaml").write_text(
        "config_version: 1\ntraining:\n  untied_inventory: all-triphone\n"
    )

    resolved = resolve_config(
        tmp_path,
        cli_overrides={"training": {"multipron_training": False}},
        user_config_path=tmp_path / "absent-user.yaml",
    )

    assert resolved.profile.training.untied_inventory == "all-triphone"
    assert resolved.fields["training.untied_inventory"].winner.source_kind == "project"


def test_later_multipron_on_restores_default_inventory(tmp_path: Path) -> None:
    (tmp_path / "etc").mkdir()
    (tmp_path / "experiments" / "trial").mkdir(parents=True)
    (tmp_path / "etc" / "config.yaml").write_text(
        "config_version: 1\ntraining:\n  multipron_training: false\n"
    )
    (tmp_path / "experiments" / "trial" / "config.yaml").write_text(
        "config_version: 1\ntraining:\n  multipron_training: true\n"
    )

    resolved = resolve_config(
        tmp_path,
        experiment="trial",
        user_config_path=tmp_path / "absent-user.yaml",
    )

    assert resolved.profile.training.multipron_training is True
    assert resolved.profile.training.untied_inventory == "transcript-reachable"
    assert resolved.fields["training.untied_inventory"].winner.source_kind == "schema-default"


def test_deep_merge_requires_extends(tmp_path: Path) -> None:
    project = tmp_path
    (project / "etc").mkdir()
    (project / "etc" / "configs.yaml").write_text(
        "config_version: 1\nprofiles:\n  default:\n    features:\n      ncep: 26\n"
    )
    with pytest.raises(ValueError, match="has no extends and is incomplete"):
        resolve_config(project)


def test_legacy_resolution_is_equivalent_and_warns(tmp_path: Path) -> None:
    (tmp_path / "etc").mkdir()
    (tmp_path / "etc" / "configs.yaml").write_text(
        "custom:\n  features:\n    samprate: 8000\n  training:\n    n_senones: 99\n"
    )
    with pytest.warns(FutureWarning, match="pstrain config migrate"):
        resolved = resolve_config(tmp_path, profile_name="custom")
    assert resolved.profile.features.samprate == 8000
    assert resolved.profile.features.ncep == 13
    assert resolved.profile.training.n_senones == 99


def test_migration_check_is_read_only_and_write_keeps_backup(tmp_path: Path) -> None:
    (tmp_path / "etc").mkdir()
    path = tmp_path / "etc" / "configs.yaml"
    path.write_text("default:\n  features:\n    ncep: 13\n")
    _, rendered, backup = migrate_project(tmp_path, check=True)
    assert backup is None
    assert "config_version: 1" in rendered
    assert "config_version" not in path.read_text()
    _, _, backup = migrate_project(tmp_path, check=False)
    assert backup is not None and backup.exists()
    assert yaml.safe_load(path.read_text())["config_version"] == 1


def _leaf_paths(value: object, prefix: str = "") -> set[str]:
    if prefix == "training.exclusion_schedule":
        return {prefix}
    if isinstance(value, dict):
        result: set[str] = set()
        for key, child in value.items():
            result |= _leaf_paths(child, f"{prefix}.{key}" if prefix else str(key))
        return result
    return {prefix}


def test_every_semantic_field_has_registered_consumer() -> None:
    fields = _leaf_paths(Profile().model_dump(mode="python"))
    validate_consumer_coverage(fields)
    assert fields == set(CONSUMERS) == set(CONSUMER_TOUCHES)


def test_phantom_schema_field_fails_consumer_coverage() -> None:
    fields = _leaf_paths(Profile().model_dump(mode="python")) | {"training.phantom"}
    with pytest.raises(ValueError, match=r"unregistered=.*training\.phantom.*unproven"):
        validate_consumer_coverage(fields)


def test_registered_touch_proofs_reach_runtime_projection(tmp_path: Path) -> None:
    context = PipelineContext.from_config(tmp_path)
    expected = context.resolved_config.as_dict()  # type: ignore[union-attr]
    for field_path, runtime_path in CONSUMER_TOUCHES.items():
        expected_value: object = expected
        for part in field_path.split("."):
            expected_value = expected_value[part]  # type: ignore[index]
        actual: object = context
        for part in runtime_path.split("."):
            actual = getattr(actual, part)
        if isinstance(expected_value, list) and isinstance(actual, tuple):
            expected_value = tuple(expected_value)
        assert actual == expected_value, field_path


def test_every_field_nondefault_reaches_runtime_projection(tmp_path: Path) -> None:
    """Anti-recurrence: canonical leaves project to immutable runtime values."""
    from dataclasses import asdict

    from pstrain.lib.pipeline.context import PipelineContext

    (tmp_path / "etc").mkdir()
    (tmp_path / "etc" / "configs.yaml").write_text(
        "config_version: 1\nprofiles:\n  custom:\n    extends: default\n"
        "    features:\n      ncep: 14\n    training:\n      n_senones: 201\n"
        "    split:\n      seed: 43\n    runner:\n      nice: 6\n"
    )
    context = PipelineContext.from_config(tmp_path, config_name="custom")
    assert asdict(context.feat)["ncep"] == 14
    assert asdict(context.train)["n_senones"] == 201
    assert asdict(context.split)["seed"] == 43
    assert asdict(context.runner)["nice"] == 6


def test_repo_profiles_resolve_equivalently_before_and_after_refactor(tmp_path: Path) -> None:
    """Every maintained fixture profile keeps its pre-C2 resolved dictionary."""
    legacy = tmp_path / "legacy"
    (legacy / "etc").mkdir(parents=True)
    (legacy / "etc" / "configs.yaml").write_text(yaml.safe_dump(DEFAULT_CONFIGS))
    maintained_projects = [
        Path(__file__).parents[1],
        Path(__file__).parent / "fixtures/mini_arctic",
    ]
    for name, body in DEFAULT_CONFIGS.items():
        expected = Profile.model_validate(body).model_dump(mode="json")
        before = resolve_config(
            legacy, profile_name=name, user_config_path=tmp_path / "absent-user.yaml"
        ).as_dict()
        for project in maintained_projects:
            after = resolve_config(
                project, profile_name=name, user_config_path=tmp_path / "absent-user.yaml"
            ).as_dict()
            assert before == expected == after


def test_extends_preserves_builtin_leaf_provenance(tmp_path: Path) -> None:
    (tmp_path / "etc").mkdir()
    (tmp_path / "etc" / "configs.yaml").write_text(
        "config_version: 1\nprofiles:\n  custom:\n    extends: default\n    description: custom\n"
    )
    resolved = resolve_config(tmp_path, profile_name="custom")
    assert resolved.fields["features.ncep"].winner.source_kind == "built-in"
    assert resolved.fields["description"].winner.source_kind == "project-profile"


def test_inactive_legacy_overlay_is_ignored_with_loud_warning(tmp_path: Path) -> None:
    (tmp_path / "etc").mkdir()
    path = tmp_path / "etc" / "config.yaml"
    path.write_text("features:\n  num_ceps: 20\n")
    with pytest.warns(FutureWarning, match=str(path)):
        resolved = resolve_config(tmp_path)
    assert resolved.profile.features.ncep == 13
    assert resolved.warnings and str(path) in resolved.warnings[0]


def test_effective_legacy_overlay_provenance_is_labeled_legacy(tmp_path: Path) -> None:
    user = tmp_path / "user.yaml"
    user.write_text("features:\n  num_ceps: 20\n")
    with pytest.warns(FutureWarning):
        resolved = resolve_config(tmp_path, user_config_path=user)
    assert resolved.profile.features.ncep == 20
    assert resolved.fields["features.ncep"].winner.source_kind == "legacy"


def test_legacy_warning_is_deduplicated_once_per_run(tmp_path: Path) -> None:
    (tmp_path / "etc").mkdir()
    path = tmp_path / "etc" / "config.yaml"
    path.write_text("features:\n  num_ceps: 20\n")
    with pytest.warns(FutureWarning) as caught:
        resolve_config(tmp_path)
        resolve_config(tmp_path)
    assert len(caught) == 1
