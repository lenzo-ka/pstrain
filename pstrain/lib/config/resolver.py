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
    OverlayDocument,
    Profile,
    ProfilesDocument,
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

    def benchmark_document(self) -> dict[str, Any]:
        """Return a deterministic snapshot of values and winning source kinds."""
        return {
            "profile": self.as_dict(),
            "profile_name": self.profile_name,
            "config_version": self.config_version,
            "field_source_kinds": {
                path: explanation.winner.source_kind
                for path, explanation in sorted(self.fields.items())
            },
        }


# Registration is intentionally explicit data, not schema-derived convention.
# CONSUMER_TOUCHES names the concrete PipelineContext attribute touched by each
# field; coverage tests compare both ledgers with every semantic schema leaf.
CONSUMERS: dict[str, tuple[str, str]] = {
    "description": ("pipeline.metadata", "run"),
    "features.samprate": ("pipeline.features", "features"),
    "features.ncep": ("pipeline.features", "features"),
    "features.nfilt": ("pipeline.features", "features"),
    "features.nfft": ("pipeline.features", "features"),
    "features.lowerf": ("pipeline.features", "features"),
    "features.upperf": ("pipeline.features", "features"),
    "features.alpha": ("pipeline.features", "features"),
    "features.dither": ("pipeline.features", "features"),
    "features.remove_dc": ("pipeline.features", "features"),
    "features.remove_noise": ("pipeline.features", "features"),
    "features.frate": ("pipeline.features", "features"),
    "features.wlen": ("pipeline.features", "features"),
    "features.lifter": ("pipeline.features", "features"),
    "features.transform": ("pipeline.features", "features"),
    "features.agc": ("pipeline.features", "features"),
    "features.cmn": ("pipeline.features", "features"),
    "features.cmninit": ("pipeline.features", "features"),
    "features.varnorm": ("pipeline.features", "features"),
    "features.feat_type": ("pipeline.features", "features"),
    "training.n_state": ("pipeline.training", "training"),
    "training.n_senones": ("pipeline.training", "training"),
    "training.a_beam": ("pipeline.training", "training"),
    "training.b_beam": ("pipeline.training", "training"),
    "training.max_skip_fraction": ("pipeline.training", "training"),
    "training.retry_beam_factor": ("pipeline.training", "training"),
    "training.failed_alignment": ("pipeline.training", "training"),
    "training.bw_checkpoint_iterations": ("pipeline.training", "training"),
    "training.arctic_a0302_zero_codebook_band": ("pipeline.training", "training"),
    "training.accept_arctic_a0587_known_skip": ("pipeline.training", "training"),
    "training.tree_state_weights": ("pipeline.training", "training"),
    "training.tree_rotate_state_weights": ("pipeline.training", "training"),
    "training.tree_directional_questions": ("pipeline.training", "training"),
    "training.tree_ssplitmax": ("pipeline.training", "training"),
    "training.tree_ssplitthr": ("pipeline.training", "training"),
    "training.tree_csplitmax": ("pipeline.training", "training"),
    "training.tree_csplitthr": ("pipeline.training", "training"),
    "training.tree_mwfloor": ("pipeline.training", "training"),
    "training.tree_intermediate_dumps": ("pipeline.training", "training"),
    "training.question_npermute": ("pipeline.training", "training"),
    "training.question_quests_per_state": ("pipeline.training", "training"),
    "training.question_niter": ("pipeline.training", "training"),
    "training.multipron_training": ("pipeline.training", "training"),
    "training.optional_final_silence": ("pipeline.training", "training"),
    "training.untied_inventory": ("pipeline.training", "training"),
    "training.exclusion_schedule": ("pipeline.training", "training"),
    "training.ci.max_iterations": ("pipeline.baum_welch", "training"),
    "training.ci.min_iterations": ("pipeline.baum_welch", "training"),
    "training.ci.convergence_ratio": ("pipeline.baum_welch", "training"),
    "training.untied.max_iterations": ("pipeline.baum_welch", "training"),
    "training.untied.min_iterations": ("pipeline.baum_welch", "training"),
    "training.untied.convergence_ratio": ("pipeline.baum_welch", "training"),
    "training.tied.max_iterations": ("pipeline.baum_welch", "training"),
    "training.tied.min_iterations": ("pipeline.baum_welch", "training"),
    "training.tied.convergence_ratio": ("pipeline.baum_welch", "training"),
    "split.train_ratio": ("pipeline.split", "split"),
    "split.test_count": ("pipeline.split", "split"),
    "split.seed": ("pipeline.split", "split"),
    "runner.jobs": ("pipeline.runner", "runner"),
    "runner.nice": ("pipeline.runner", "runner"),
}

CONSUMER_TOUCHES: dict[str, str] = {
    "description": "description",
    **{
        path: f"feat.{path.removeprefix('features.')}"
        for path in (
            "features.samprate",
            "features.ncep",
            "features.nfilt",
            "features.nfft",
            "features.lowerf",
            "features.upperf",
            "features.alpha",
            "features.dither",
            "features.remove_dc",
            "features.remove_noise",
            "features.frate",
            "features.wlen",
            "features.lifter",
            "features.transform",
            "features.agc",
            "features.cmn",
            "features.cmninit",
            "features.varnorm",
            "features.feat_type",
        )
    },
    **{
        path: f"train.{path.removeprefix('training.')}"
        for path in (
            "training.n_state",
            "training.n_senones",
            "training.a_beam",
            "training.b_beam",
            "training.max_skip_fraction",
            "training.retry_beam_factor",
            "training.failed_alignment",
            "training.bw_checkpoint_iterations",
            "training.arctic_a0302_zero_codebook_band",
            "training.accept_arctic_a0587_known_skip",
            "training.tree_state_weights",
            "training.tree_rotate_state_weights",
            "training.tree_directional_questions",
            "training.tree_ssplitmax",
            "training.tree_ssplitthr",
            "training.tree_csplitmax",
            "training.tree_csplitthr",
            "training.tree_mwfloor",
            "training.tree_intermediate_dumps",
            "training.question_npermute",
            "training.question_quests_per_state",
            "training.question_niter",
            "training.multipron_training",
            "training.optional_final_silence",
            "training.untied_inventory",
            "training.exclusion_schedule",
            "training.ci.max_iterations",
            "training.ci.min_iterations",
            "training.ci.convergence_ratio",
            "training.untied.max_iterations",
            "training.untied.min_iterations",
            "training.untied.convergence_ratio",
            "training.tied.max_iterations",
            "training.tied.min_iterations",
            "training.tied.convergence_ratio",
        )
    },
    "split.train_ratio": "split.train_ratio",
    "split.test_count": "split.test_count",
    "split.seed": "split.seed",
    "runner.jobs": "runner.jobs",
    "runner.nice": "runner.nice",
}


def validate_consumer_coverage(field_paths: Iterable[str]) -> None:
    """Fail unless every schema field is explicitly registered and touch-proven."""
    fields = set(field_paths)
    registered = set(CONSUMERS)
    proven = set(CONSUMER_TOUCHES)
    if fields != registered or fields != proven:
        raise ValueError(
            "configuration consumer coverage mismatch: "
            f"unregistered={sorted(fields - registered)}, "
            f"unproven={sorted(fields - proven)}, "
            f"stale={sorted((registered | proven) - fields)}"
        )


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
    key = str(path.resolve())
    if key not in _LEGACY_WARNED_PATHS:
        _LEGACY_WARNED_PATHS.add(key)
        warnings.warn(message, FutureWarning, stacklevel=3)
    return message


_LEGACY_WARNED_PATHS: set[str] = set()


def _builtin_profiles() -> dict[str, dict[str, Any]]:
    # Imported lazily to avoid a models/context cycle during migration.
    from pstrain.lib.pipeline.context import DEFAULT_CONFIGS

    return copy.deepcopy(DEFAULT_CONFIGS)


def _profile_documents(
    project_dir: Path,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, str],
    dict[str, dict[str, tuple[str, str]]],
    list[str],
]:
    profiles = _builtin_profiles()
    origins = dict.fromkeys(profiles, "built-in")
    provenance = {
        name: {path: ("built-in", "built-in") for path, _ in _leaves(body)}
        for name, body in profiles.items()
    }
    messages: list[str] = []
    path = project_dir / "etc" / "configs.yaml"
    if not path.exists():
        return profiles, origins, provenance, messages
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
            provenance[str(name)] = {leaf: ("legacy", str(path)) for leaf, _ in _leaves(body)}
        return profiles, origins, provenance, messages

    try:
        document = ProfilesDocument.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(f"invalid configuration layer {path}: {exc}") from exc
    definitions = document.profiles
    resolving: set[str] = set()

    def materialize(name: str) -> tuple[dict[str, Any], dict[str, tuple[str, str]]]:
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
            return body, {leaf: ("project-profile", str(path)) for leaf, _ in _leaves(body)}
        if name in resolving:
            raise ValueError(f"profile inheritance cycle involving {name!r} in {path}")
        parent = definition.extends
        if parent not in definitions and parent not in profiles:
            raise ValueError(f"profile {name!r} extends unknown profile {parent!r}")
        resolving.add(name)
        if parent in definitions:
            base, sources = materialize(parent)
        else:
            base, sources = profiles[parent], provenance[parent]
        resolving.remove(name)
        sources = dict(sources)
        sources.update({leaf: ("project-profile", str(path)) for leaf, _ in _leaves(body)})
        return _deep_merge(base, body), sources

    for name in definitions:
        profiles[name], provenance[name] = materialize(name)
        origins[name] = str(path)
    return profiles, origins, provenance, messages


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


def _overlay(
    path: Path, project_dir: Path, kind: str, *, legacy_effective: bool = True
) -> tuple[dict[str, Any], list[str], str]:
    if not path.exists():
        return {}, [], kind
    raw = _load_yaml(path)
    if "config_version" not in raw:
        warning = [_warn_legacy(path, project_dir)]
        if not legacy_effective:
            # etc/config.yaml was not consumed before C2. Preserve that
            # behavior: announce it, but do not alter or abort a build.
            return {}, warning, "legacy"
        return _legacy_inactive_overlay(raw, path), warning, "legacy"
    try:
        doc = OverlayDocument.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(f"invalid configuration layer {path}: {exc}") from exc
    return doc.model_dump(exclude_none=True, exclude={"config_version", "profile"}), [], kind


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
    profiles, origins, profile_provenance, messages = _profile_documents(project)
    if profile_name not in profiles:
        raise ValueError(
            f"unknown config {profile_name!r}; available: {', '.join(sorted(profiles))}"
        )

    default = Profile().model_dump(mode="python")
    candidates: dict[str, list[Candidate]] = {}
    for path, value in _leaves(default):
        candidates.setdefault(path, []).append(Candidate("schema-default", "Profile", path, value))

    selected = profiles[profile_name]
    selected_sources = profile_provenance[profile_name]
    builtin_layer: dict[str, Any] = {}
    project_profile_layer: dict[str, Any] = {}
    for field_path, value in _leaves(selected):
        kind, _source = selected_sources[field_path]
        target = builtin_layer if kind == "built-in" else project_profile_layer
        cursor = target
        parts = field_path.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = copy.deepcopy(value)
    layers: list[tuple[str, str, dict[str, Any]]] = []
    if builtin_layer:
        layers.append(("built-in", "built-in", builtin_layer))
    user_path = user_config_path or Path(
        os.environ.get("PSTRAIN_USER_CONFIG", Path.home() / ".pstrain" / "config.yaml")
    )
    user, warns, user_kind = _overlay(user_path, project, "user")
    messages.extend(warns)
    layers.append((user_kind, str(user_path), user))
    if project_profile_layer:
        profile_kind = next(
            kind for kind, _source in selected_sources.values() if kind != "built-in"
        )
        layers.append((profile_kind, origins[profile_name], project_profile_layer))
    project_path = project / "etc" / "config.yaml"
    project_overlay, warns, project_kind = _overlay(
        project_path, project, "project", legacy_effective=False
    )
    messages.extend(warns)
    layers.append((project_kind, str(project_path), project_overlay))
    experiment_path = project / "experiments" / experiment / "config.yaml"
    experiment_overlay, warns, experiment_kind = _overlay(experiment_path, project, "experiment")
    messages.extend(warns)
    layers.append((experiment_kind, str(experiment_path), experiment_overlay))
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
    profiles, origins, _, _ = _profile_documents(project_dir.resolve())
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
