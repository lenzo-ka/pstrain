"""Source-aware resolution and bounded legacy migration."""

from __future__ import annotations

import copy
import os
import tempfile
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from pstrain.lib.config.models import (
    CURRENT_CONFIG_VERSION,
    FeatureConfig,
    OverlayDocument,
    Profile,
    ProfilesDocument,
    RunnerConfig,
    SplitConfig,
    TrainingConfig,
    TrainingScheduleConfig,
)

LEGACY_WARNING = (
    "legacy pstrain configuration read from {path}; run "
    "'pstrain config migrate --project-dir {project}' to convert it (support ends after "
    "the next minor release)"
)


@dataclass(frozen=True)
class Candidate:
    source_kind: str
    source: str
    field_path: str
    value: Any


@dataclass(frozen=True)
class FieldExplanation:
    field_path: str
    value: Any
    canonical_type: str
    winner: Candidate
    overridden: tuple[Candidate, ...]
    default: Any
    constraints: dict[str, Any]
    consumer: str
    provenance_scope: str


@dataclass(frozen=True)
class ResolvedConfig:
    profile: Profile
    profile_name: str
    config_version: int
    fields: dict[str, FieldExplanation]
    warnings: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return self.profile.model_dump(mode="json")


# Registration is intentionally data, not convention. Coverage tests compare
# this set with every leaf in Profile.model_json_schema().
CONSUMERS: dict[str, tuple[str, str]] = {
    "description": ("pipeline.metadata", "run"),
    **{
        f"features.{name}": ("pipeline.features", "features") for name in FeatureConfig.model_fields
    },
    **{
        f"training.{name}": ("pipeline.training", "training")
        for name in TrainingConfig.model_fields
        if name not in {"ci", "untied", "tied"}
    },
    **{
        f"training.{stage}.{name}": ("pipeline.baum_welch", "training")
        for stage in ("ci", "untied", "tied")
        for name in TrainingScheduleConfig.model_fields
    },
    **{f"split.{name}": ("pipeline.split", "split") for name in SplitConfig.model_fields},
    **{f"runner.{name}": ("pipeline.runner", "runner") for name in RunnerConfig.model_fields},
}


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _leaves(value: Any, prefix: str = "") -> Iterable[tuple[str, Any]]:
    if prefix == "training.exclusion_schedule":
        yield prefix, value
        return
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            yield from _leaves(child, path)
    else:
        yield prefix, value


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"configuration layer {path} must be a mapping")
    return data


def _field_constraints(field_path: str) -> dict[str, Any]:
    schema = Profile.model_json_schema()
    node: dict[str, Any] = schema
    for part in field_path.split("."):
        while "$ref" in node:
            node = schema["$defs"][node["$ref"].rsplit("/", 1)[-1]]
        node = node.get("properties", {}).get(part, {})
    while "$ref" in node:
        node = schema["$defs"][node["$ref"].rsplit("/", 1)[-1]]
    keys = {
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "minItems",
        "maxItems",
        "enum",
    }
    constraints = {key: value for key, value in node.items() if key in keys}
    for variant in node.get("anyOf", []):
        constraints.update({key: value for key, value in variant.items() if key in keys})
    return constraints


def _warn_legacy(path: Path, project_dir: Path) -> str:
    message = LEGACY_WARNING.format(path=path, project=project_dir)
    warnings.warn(message, FutureWarning, stacklevel=3)
    return message


def _builtin_profiles() -> dict[str, dict[str, Any]]:
    # Imported lazily to avoid a models/context cycle during migration.
    from pstrain.lib.pipeline.context import DEFAULT_CONFIGS

    return copy.deepcopy(DEFAULT_CONFIGS)


def _profile_documents(
    project_dir: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, str], list[str]]:
    profiles = _builtin_profiles()
    origins = dict.fromkeys(profiles, "built-in")
    messages: list[str] = []
    path = project_dir / "etc" / "configs.yaml"
    if not path.exists():
        return profiles, origins, messages
    raw = _load_yaml(path)
    if "config_version" not in raw:
        messages.append(_warn_legacy(path, project_dir))
        # Exact pre-C2 semantics: a legacy named profile replaces its built-in
        # namesake, and omitted fields use canonical (formerly dataclass) defaults.
        for name, body in raw.items():
            if not isinstance(body, dict):
                raise ValueError(f"legacy profile {name!r} in {path} must be a mapping")
            profiles[str(name)] = body
            origins[str(name)] = str(path)
        return profiles, origins, messages

    try:
        document = ProfilesDocument.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(f"invalid configuration layer {path}: {exc}") from exc
    definitions = document.profiles
    resolving: set[str] = set()

    def materialize(name: str) -> dict[str, Any]:
        definition = definitions[name]
        body = definition.model_dump(exclude_none=True, exclude={"extends"})
        if definition.extends is None:
            expected = Profile().model_dump(mode="python")
            missing = sorted(
                path for path, _ in _leaves(expected) if path not in dict(_leaves(body))
            )
            if missing:
                raise ValueError(
                    f"profile {name!r} in {path} has no extends and is incomplete; "
                    f"missing {missing[0]!r}"
                )
            return body
        if name in resolving:
            raise ValueError(f"profile inheritance cycle involving {name!r} in {path}")
        parent = definition.extends
        if parent not in definitions and parent not in profiles:
            raise ValueError(f"profile {name!r} extends unknown profile {parent!r}")
        resolving.add(name)
        base = materialize(parent) if parent in definitions else profiles[parent]
        resolving.remove(name)
        return _deep_merge(base, body)

    for name in definitions:
        profiles[name] = materialize(name)
        origins[name] = str(path)
    return profiles, origins, messages


def _legacy_inactive_overlay(raw: dict[str, Any], path: Path) -> dict[str, Any]:
    """Map only inactive-schema keys with unambiguous active consumers."""
    mapped: dict[str, Any] = {}
    translations = {
        ("audio", "sample_rate"): ("features", "samprate"),
        ("features", "num_ceps"): ("features", "ncep"),
        ("features", "num_filters"): ("features", "nfilt"),
        ("features", "fft_size"): ("features", "nfft"),
        ("features", "lower_freq"): ("features", "lowerf"),
        ("features", "upper_freq"): ("features", "upperf"),
        ("parallel", "n_jobs"): ("runner", "jobs"),
        ("parallel", "nice"): ("runner", "nice"),
    }
    for old, new in translations.items():
        section = raw.get(old[0])
        if not isinstance(section, dict) or old[1] not in section:
            continue
        value = section[old[1]]
        if old == ("parallel", "n_jobs") and value == -1:
            value = None
        mapped.setdefault(new[0], {})[new[1]] = value
    # Values without a canonical runtime consumer are deliberately ignored and
    # never represented as having affected historical training.
    return mapped


def _reject_legacy_collision(
    effective: dict[str, Any], overlay: dict[str, Any], source: Path
) -> None:
    current = dict(_leaves(effective))
    for path, value in _leaves(overlay):
        if path in current and current[path] != value:
            raise ValueError(
                f"ambiguous legacy configuration in {source}: {path!r} is {value!r}, "
                f"but the active profile declares {current[path]!r}; run 'pstrain config "
                "migrate --check' and choose one value"
            )


def _overlay(path: Path, project_dir: Path, kind: str) -> tuple[dict[str, Any], list[str]]:
    if not path.exists():
        return {}, []
    raw = _load_yaml(path)
    if "config_version" not in raw:
        return _legacy_inactive_overlay(raw, path), [_warn_legacy(path, project_dir)]
    try:
        doc = OverlayDocument.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(f"invalid configuration layer {path}: {exc}") from exc
    return doc.model_dump(exclude_none=True, exclude={"config_version", "profile"}), []


def resolve_config(
    project_dir: Path | str,
    *,
    profile_name: str = "default",
    experiment: str = "default",
    cli_overrides: dict[str, Any] | None = None,
    user_config_path: Path | None = None,
) -> ResolvedConfig:
    """Resolve built-in < user < project < experiment < CLI."""
    project = Path(project_dir).resolve()
    profiles, origins, messages = _profile_documents(project)
    if profile_name not in profiles:
        raise ValueError(
            f"unknown config {profile_name!r}; available: {', '.join(sorted(profiles))}"
        )

    default = Profile().model_dump(mode="python")
    candidates: dict[str, list[Candidate]] = {}
    for path, value in _leaves(default):
        candidates.setdefault(path, []).append(Candidate("schema-default", "Profile", path, value))

    layers: list[tuple[str, str, dict[str, Any]]] = [
        (
            "built-in" if origins[profile_name] == "built-in" else "project-profile",
            origins[profile_name],
            profiles[profile_name],
        )
    ]
    user_path = user_config_path or Path(
        os.environ.get("PSTRAIN_USER_CONFIG", Path.home() / ".pstrain" / "config.yaml")
    )
    user, warns = _overlay(user_path, project, "user")
    if warns:
        _reject_legacy_collision(_deep_merge(default, profiles[profile_name]), user, user_path)
    messages.extend(warns)
    layers.append(("user", str(user_path), user))
    project_path = project / "etc" / "config.yaml"
    project_overlay, warns = _overlay(project_path, project, "project")
    if warns:
        _reject_legacy_collision(
            _deep_merge(_deep_merge(default, profiles[profile_name]), user),
            project_overlay,
            project_path,
        )
    messages.extend(warns)
    layers.append(("project", str(project_path), project_overlay))
    experiment_path = project / "experiments" / experiment / "config.yaml"
    experiment_overlay, warns = _overlay(experiment_path, project, "experiment")
    if warns:
        effective = _deep_merge(_deep_merge(default, profiles[profile_name]), user)
        _reject_legacy_collision(
            _deep_merge(effective, project_overlay), experiment_overlay, experiment_path
        )
    messages.extend(warns)
    layers.append(("experiment", str(experiment_path), experiment_overlay))
    if cli_overrides:
        layers.append(
            (
                "cli",
                "--jobs"
                if cli_overrides
                == {"runner": {"jobs": cli_overrides.get("runner", {}).get("jobs")}}
                else "command line",
                cli_overrides,
            )
        )

    merged = default
    for kind, source, layer in layers:
        if not layer:
            continue
        training_layer = layer.get("training")
        if (
            isinstance(training_layer, dict)
            and training_layer.get("multipron_training") is False
            and "untied_inventory" not in training_layer
        ):
            layer = copy.deepcopy(layer)
            layer["training"]["untied_inventory"] = "linear"
        merged = _deep_merge(merged, layer)
        for path, value in _leaves(layer):
            candidates.setdefault(path, []).append(Candidate(kind, source, path, value))
    try:
        profile = Profile.model_validate(merged)
    except ValidationError as exc:
        for error in exc.errors():
            if error["type"] == "extra_forbidden" and len(error["loc"]) >= 2:
                block, key = str(error["loc"][0]), str(error["loc"][1])
                parameter = block.removesuffix("s")
                raise ValueError(
                    f"unknown {parameter} parameter {key!r} in profile {profile_name!r}"
                ) from exc
        raise ValueError(f"invalid resolved profile {profile_name!r}: {exc}") from exc
    values = profile.model_dump(mode="python")
    explanations: dict[str, FieldExplanation] = {}
    for path, value in _leaves(values):
        if path not in CONSUMERS:
            raise ValueError(f"declared configuration field {path!r} has no runtime consumer")
        history = candidates[path]
        consumer, scope = CONSUMERS[path]
        explanations[path] = FieldExplanation(
            field_path=path,
            value=value,
            canonical_type=type(value).__name__,
            winner=history[-1],
            overridden=tuple(history[:-1]),
            default=history[0].value,
            constraints=_field_constraints(path),
            consumer=consumer,
            provenance_scope=scope,
        )
    return ResolvedConfig(
        profile, profile_name, CURRENT_CONFIG_VERSION, explanations, tuple(messages)
    )


def canonical_profiles_document(project_dir: Path) -> dict[str, Any]:
    """Return a deterministic migration of the legacy active profiles."""
    path = project_dir / "etc" / "configs.yaml"
    raw = _load_yaml(path) if path.exists() else _builtin_profiles()
    if "config_version" in raw:
        ProfilesDocument.model_validate(raw)
        return raw
    profiles = {
        name: Profile.model_validate(body).model_dump(mode="json") for name, body in raw.items()
    }
    return {"config_version": CURRENT_CONFIG_VERSION, "profiles": profiles}


def migrate_project(project_dir: Path, *, check: bool) -> tuple[Path, str, Path | None]:
    """Render or atomically write the canonical profiles document."""
    project = project_dir.resolve()
    path = project / "etc" / "configs.yaml"
    document = canonical_profiles_document(project)
    rendered = yaml.safe_dump(document, sort_keys=False)
    overlays: list[tuple[Path, str]] = []
    overlay_paths = [project / "etc" / "config.yaml"]
    experiments = project / "experiments"
    if experiments.exists():
        overlay_paths.extend(sorted(experiments.glob("*/config.yaml")))
    for overlay_path in overlay_paths:
        if not overlay_path.exists():
            continue
        raw = _load_yaml(overlay_path)
        if "config_version" in raw:
            OverlayDocument.model_validate(raw)
            continue
        converted = {
            "config_version": CURRENT_CONFIG_VERSION,
            **_legacy_inactive_overlay(raw, overlay_path),
        }
        overlays.append((overlay_path, yaml.safe_dump(converted, sort_keys=False)))
    if check:
        extra = "".join(f"\n# {item}\n{text}" for item, text in overlays)
        return path, rendered + extra, None

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    backup: Path | None = None
    for destination, text in [(path, rendered), *overlays]:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and "config_version" not in _load_yaml(destination):
            item_backup = destination.with_name(f"{destination.name}.{stamp}.bak")
            item_backup.write_bytes(destination.read_bytes())
            backup = backup or item_backup
        _atomic_write(destination, text)
    return path, rendered, backup


def _atomic_write(path: Path, content: str) -> None:
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def list_profiles(project_dir: Path) -> list[dict[str, Any]]:
    profiles, origins, _ = _profile_documents(project_dir.resolve())
    builtins = _builtin_profiles()
    return [
        {
            "name": name,
            "description": Profile.model_validate(body).description,
            "origin": origins[name],
            "schema_version": CURRENT_CONFIG_VERSION,
            "shadows_builtin": name in builtins and origins[name] != "built-in",
        }
        for name, body in sorted(profiles.items())
    ]
