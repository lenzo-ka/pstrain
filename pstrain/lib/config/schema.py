"""Configuration schema introspection.

Provides introspection capabilities for the config system:
- List all parameters with descriptions
- Get schema for docs generation
- Parameter lookup by path
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, get_args, get_origin

from pydantic import BaseModel
from pydantic.fields import FieldInfo

from pstrain.lib.config.models import Profile


@dataclass
class ParameterInfo:
    """Information about a configuration parameter."""

    key: str
    type: str
    default: Any
    description: str
    required: bool


def get_schema() -> dict[str, Any]:
    """Get the canonical profile JSON schema.

    Returns:
        JSON Schema dict (for docs generation, validation, etc.)
    """
    return Profile.model_json_schema()


def list_parameters(prefix: str = "") -> list[ParameterInfo]:
    """List all configuration parameters with descriptions.

    Args:
        prefix: Optional prefix to filter parameters (e.g., "audio", "features")

    Returns:
        List of ParameterInfo objects
    """
    params: list[ParameterInfo] = []
    _collect_params(Profile, "", params)

    if prefix:
        params = [p for p in params if p.key.startswith(prefix)]

    return sorted(params, key=lambda p: p.key)


def _collect_params(
    model: type[BaseModel],
    prefix: str,
    params: list[ParameterInfo],
) -> None:
    """Recursively collect parameters from a Pydantic model."""
    for field_name, field_info in model.model_fields.items():
        key = f"{prefix}.{field_name}" if prefix else field_name

        # Skip private fields
        if field_name.startswith("_"):
            continue

        # Get type annotation
        annotation = field_info.annotation
        type_str = _format_type(annotation)

        # Check if it's a nested model
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            _collect_params(annotation, key, params)
        else:
            params.append(
                ParameterInfo(
                    key=key,
                    type=type_str,
                    default=_format_default(field_info),
                    description=field_info.description or "",
                    required=field_info.is_required(),
                )
            )


def _format_type(annotation: Any) -> str:
    """Format type annotation for display."""
    if annotation is None:
        return "None"
    if get_origin(annotation) is Literal:
        return " | ".join(str(value) for value in get_args(annotation))
    if hasattr(annotation, "__name__"):
        return str(annotation.__name__)
    if hasattr(annotation, "__origin__"):
        # Handle generics like list[int], Optional[str]
        origin = annotation.__origin__
        args = getattr(annotation, "__args__", ())
        if origin is list:
            return f"list[{_format_type(args[0]) if args else 'Any'}]"
        if origin is dict:
            return "dict"
        # Union types (Optional is Union[X, None])
        if hasattr(origin, "__name__"):
            return str(origin.__name__)
    return str(annotation).replace("typing.", "")


def _format_default(field_info: FieldInfo) -> Any:
    """Format default value for display."""
    if field_info.default_factory is not None:
        return field_info.get_default(call_default_factory=True)
    if field_info.default is not None:
        return field_info.default
    return None


def get_parameter(key: str) -> ParameterInfo | None:
    """Get info about a specific parameter.

    Args:
        key: Dot-separated parameter path (e.g., "audio.sample_rate")

    Returns:
        ParameterInfo or None if not found
    """
    params = list_parameters()
    for p in params:
        if p.key == key:
            return p
    return None


def generate_markdown_docs() -> str:
    """Generate Markdown documentation for all config parameters.

    Returns:
        Markdown string suitable for docs
    """
    lines = ["# Configuration Reference\n"]
    lines.append("All configuration parameters for pstrain.\n")

    params = list_parameters()

    # Group by top-level section
    sections: dict[str, list[ParameterInfo]] = {}
    for p in params:
        section = p.key.split(".")[0]
        if section not in sections:
            sections[section] = []
        sections[section].append(p)

    for section, section_params in sections.items():
        lines.append(f"\n## {section}\n")

        for p in section_params:
            lines.append(f"### `{p.key}`\n")
            lines.append(f"- **Type:** `{p.type}`")
            lines.append(f"- **Default:** `{p.default}`")
            if p.description:
                lines.append(f"- **Description:** {p.description}")
            lines.append("")

    return "\n".join(lines)


def generate_rst_docs() -> str:
    """Generate reStructuredText documentation for all config parameters.

    Returns:
        RST string suitable for Sphinx docs
    """
    lines = ["Configuration Reference", "=" * 23, ""]
    lines.append("All configuration parameters for pstrain.\n")

    params = list_parameters()

    # Group by top-level section
    sections: dict[str, list[ParameterInfo]] = {}
    for p in params:
        section = p.key.split(".")[0]
        if section not in sections:
            sections[section] = []
        sections[section].append(p)

    for section, section_params in sections.items():
        lines.append(f"\n{section}")
        lines.append("-" * len(section))
        lines.append("")

        for p in section_params:
            lines.append(f"``{p.key}``")
            lines.append(f"   :Type: ``{p.type}``")
            default = repr(p.default) if isinstance(p.default, str) else str(p.default)
            lines.append(f"   :Default: ``{default}``")
            if p.description:
                lines.append(f"   :Description: {p.description}")
            lines.append("")

    return "\n".join(lines)
