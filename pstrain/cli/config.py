"""Canonical configuration CLI."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import yaml

from pstrain.api.config import (
    CURRENT_CONFIG_VERSION,
    generate_markdown_docs,
    generate_rst_docs,
    get_schema,
    list_parameters,
    list_profiles,
    migrate_project,
    resolve_config,
)
from pstrain.cli.base import add_json_argument, ensure_global_option_defaults


def _selectors(parser: Any, *, key: bool = False, key_required: bool = False) -> None:
    if key:
        parser.add_argument(
            "key",
            nargs=None if key_required else "?",
            help="Canonical dotted field path",
        )
    parser.add_argument("--project-dir", default=".")
    parser.add_argument("--experiment", default="default")
    parser.add_argument("-c", "--config", dest="profile_name", default="default")
    parser.add_argument("-j", "--jobs", type=int, default=None)
    add_json_argument(parser, suppress_defaults=True)


def _set_handler(
    parser: Any, func: Any, command: str, *, supports_json_output: bool = False
) -> None:
    parser.set_defaults(
        func=func,
        json_command=f"config {command}",
        supports_json_output=supports_json_output,
    )


def _format_json(args: Any, data: Any, *, default: Any = None) -> str:
    indent = args.json_indent if args.json_indent > 0 else None
    return json.dumps(data, indent=indent, ensure_ascii=args.json_ascii, default=default)


def _error(args: Any, message: str, *, exit_code: int = 1) -> int:
    if args.json:
        print(_format_json(args, {"status": "error", "message": message}))
    else:
        print(f"Error: {message}", file=sys.stderr)
    return exit_code


def register_config_command(subparsers: Any) -> None:
    ensure_global_option_defaults(subparsers)
    parser = subparsers.add_parser("config", help="Resolve and inspect configuration")
    commands = parser.add_subparsers(dest="config_command")

    explain = commands.add_parser("explain", help="Explain resolved values and provenance")
    _selectors(explain, key=True)
    _set_handler(explain, cmd_config_explain, "explain", supports_json_output=True)

    profiles = commands.add_parser("profiles", help="List available named profiles")
    profiles.add_argument("--project-dir", default=".")
    add_json_argument(profiles, suppress_defaults=True)
    _set_handler(profiles, cmd_config_profiles, "profiles", supports_json_output=True)

    show = commands.add_parser("show", help="Show a resolved profile")
    _selectors(show)
    show.add_argument("--resolved", action="store_true", default=True)
    show.add_argument("--sources", action="store_true")
    _set_handler(show, cmd_config_show, "show", supports_json_output=True)

    get = commands.add_parser("get", help="Get one resolved value")
    _selectors(get, key=True, key_required=True)
    _set_handler(get, cmd_config_get, "get", supports_json_output=True)

    schema = commands.add_parser("schema", help="Export the canonical JSON Schema")
    schema.add_argument("--format", choices=["json", "markdown", "rst"], default="json")
    schema.add_argument("--output")
    add_json_argument(schema, suppress_defaults=True)
    _set_handler(schema, cmd_config_schema, "schema", supports_json_output=True)

    listing = commands.add_parser("list", help="List canonical semantic fields")
    listing.add_argument("--section")
    add_json_argument(listing, suppress_defaults=True)
    _set_handler(listing, cmd_config_list, "list", supports_json_output=True)

    migrate = commands.add_parser("migrate", help="Migrate legacy files to version 1")
    migrate.add_argument("--project-dir", default=".")
    migrate.add_argument("--check", action="store_true")
    _set_handler(migrate, cmd_config_migrate, "migrate")

    path = commands.add_parser("path", help="Show configuration layer paths")
    path.add_argument("--project-dir", default=".")
    _set_handler(path, cmd_config_path, "path")


def _resolve(args: Any) -> Any:
    overrides = {"runner": {"jobs": args.jobs}} if getattr(args, "jobs", None) else None
    return resolve_config(
        Path(args.project_dir),
        profile_name=args.profile_name,
        experiment=args.experiment,
        cli_overrides=overrides,
    )


def cmd_config_explain(args: Any) -> int:
    try:
        resolved = _resolve(args)
    except ValueError as exc:
        return _error(args, str(exc))
    fields = resolved.fields
    if args.key:
        if args.key not in fields:
            return _error(args, f"Unknown config key: {args.key}")
        fields = {args.key: fields[args.key]}
    output = {
        key: {
            "value": item.value,
            "type": item.canonical_type,
            "source_kind": item.winner.source_kind,
            "source": item.winner.source,
            "field_path": item.field_path,
            "overridden": [candidate.__dict__ for candidate in item.overridden],
            "default": item.default,
            "constraints": item.constraints,
            "consumer": item.consumer,
            "provenance_scope": item.provenance_scope,
            "reason": f"{item.winner.source_kind} is the highest-precedence candidate",
        }
        for key, item in fields.items()
    }
    if args.json:
        print(_format_json(args, output, default=str))
    else:
        for key, item in output.items():
            print(f"{key} = {item['value']!r} ({item['type']})")
            print(f"  source: {item['source_kind']} {item['source']} [{item['field_path']}]")
            print(f"  consumer: {item['consumer']}; provenance: {item['provenance_scope']}")
            print(f"  default: {item['default']!r}; {item['reason']}")
            for candidate in item["overridden"]:
                print(f"  overridden: {candidate['source_kind']} {candidate['value']!r}")
    return 0


def cmd_config_profiles(args: Any) -> int:
    try:
        rows = list_profiles(Path(args.project_dir))
    except ValueError as exc:
        return _error(args, str(exc))
    if args.json:
        print(_format_json(args, rows))
    else:
        for row in rows:
            shadow = " (shadows built-in)" if row["shadows_builtin"] else ""
            print(
                f"{row['name']}: {row['description']} [{row['origin']}, v{row['schema_version']}]{shadow}"
            )
    return 0


def cmd_config_show(args: Any) -> int:
    try:
        resolved = _resolve(args)
    except ValueError as exc:
        return _error(args, str(exc))
    data: dict[str, Any] = resolved.as_dict()
    if args.sources:
        data["_sources"] = {key: value.winner.__dict__ for key, value in resolved.fields.items()}
    print(_format_json(args, data) if args.json else yaml.safe_dump(data, sort_keys=False))
    return 0


def cmd_config_get(args: Any) -> int:
    try:
        resolved = _resolve(args)
        item = resolved.fields[args.key]
    except (ValueError, KeyError) as exc:
        return _error(args, f"unknown or invalid config key {args.key!r}: {exc}")
    print(_format_json(args, item.value) if args.json else f"{args.key} = {item.value}")
    return 0


def cmd_config_schema(args: Any) -> int:
    if args.json and args.format != "json":
        return _error(args, "--json cannot be combined with a non-JSON --format")
    output = (
        _format_json(args, get_schema())
        if args.format == "json"
        else generate_markdown_docs()
        if args.format == "markdown"
        else generate_rst_docs()
    )
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
    if args.json or not args.output:
        print(output)
    return 0


def cmd_config_list(args: Any) -> int:
    params = list_parameters(args.section or "")
    rows = [item.__dict__ for item in params]
    if args.json:
        print(_format_json(args, rows, default=str))
    else:
        for item in params:
            print(f"{item.key}: {item.type} = {item.default!r} — {item.description}")
    return 0 if params else 1


def cmd_config_migrate(args: Any) -> int:
    try:
        path, rendered, backup = migrate_project(Path(args.project_dir), check=args.check)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    if args.check:
        print(f"Would migrate {path} to config_version {CURRENT_CONFIG_VERSION}:")
        print(rendered, end="")
    else:
        print(f"Migrated {path} to config_version {CURRENT_CONFIG_VERSION}")
        if backup:
            print(f"Backup: {backup}")
    return 0


def cmd_config_path(args: Any) -> int:
    project = Path(args.project_dir).resolve()
    print(f"User: {Path.home() / '.pstrain' / 'config.yaml'}")
    print(f"Profiles: {project / 'etc' / 'configs.yaml'}")
    print(f"Project: {project / 'etc' / 'config.yaml'}")
    print(f"Experiments: {project / 'experiments' / '<name>' / 'config.yaml'}")
    return 0
