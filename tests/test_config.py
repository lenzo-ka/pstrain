"""Canonical configuration contract tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from pstrain.lib.config.models import FeatureConfig, Profile, TrainingConfig
from pstrain.lib.config.resolver import CONSUMERS, migrate_project, resolve_config
from pstrain.lib.pipeline.context import DEFAULT_CONFIGS


def test_active_names_and_defaults_are_canonical() -> None:
    profile = Profile()
    assert profile.features.ncep == 13
    assert profile.features.samprate == 16000
    assert profile.training.n_state == 3
    assert profile.runner.jobs is None
    with pytest.raises(ValueError):
        FeatureConfig.model_validate({"num_ceps": 26})
    with pytest.raises(ValueError):
        TrainingConfig.model_validate({"n_states": 5})


def test_layer_precedence_and_sources(tmp_path: Path) -> None:
    project = tmp_path / "project"
    (project / "etc").mkdir(parents=True)
    (project / "experiments" / "trial").mkdir(parents=True)
    user = tmp_path / "user.yaml"
    user.write_text("config_version: 1\nfeatures:\n  ncep: 14\n")
    (project / "etc" / "configs.yaml").write_text(
        "config_version: 1\nprofiles:\n  custom:\n    extends: default\n"
        "    features:\n      ncep: 15\n"
    )
    (project / "etc" / "config.yaml").write_text("config_version: 1\nfeatures:\n  ncep: 16\n")
    (project / "experiments" / "trial" / "config.yaml").write_text(
        "config_version: 1\nfeatures:\n  ncep: 17\n"
    )
    resolved = resolve_config(
        project,
        profile_name="custom",
        experiment="trial",
        cli_overrides={"features": {"ncep": 18}},
        user_config_path=user,
    )
    assert resolved.profile.features.ncep == 18
    assert [item.source_kind for item in resolved.fields["features.ncep"].overridden] == [
        "schema-default",
        "project-profile",
        "user",
        "project",
        "experiment",
    ]
    assert resolved.fields["features.ncep"].winner.source_kind == "cli"


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
    assert _leaf_paths(Profile().model_dump(mode="python")) == set(CONSUMERS)


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
    canonical = Path(__file__).parents[1]
    for name, body in DEFAULT_CONFIGS.items():
        expected = Profile.model_validate(body).model_dump(mode="json")
        with pytest.warns(FutureWarning):
            before = resolve_config(
                legacy,
                profile_name=name,
                user_config_path=tmp_path / "absent-user.yaml",
            ).as_dict()
        after = resolve_config(
            canonical,
            profile_name=name,
            user_config_path=tmp_path / "absent-user.yaml",
        ).as_dict()
        assert before == expected == after
