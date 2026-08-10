"""CLI subcommand for configuration management."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from pstrain.lib.config import ConfigManager, PstrainConfig, get_user_config
from pstrain.lib.config.schema import (
    generate_markdown_docs,
    get_parameter,
    get_schema,
    list_parameters,
)


def register_config_command(subparsers: Any) -> None:
    """Register config command with argument parser."""
    parser = subparsers.add_parser(
        "config",
        help="Configuration management and introspection",
        description="View, edit, and explore pstrain configuration",
    )

    config_subparsers = parser.add_subparsers(dest="config_command", help="Config commands")

    # config show
    show_parser = config_subparsers.add_parser(
        "show",
        help="Show current configuration",
        description="Display the current merged configuration",
    )
    show_parser.add_argument(
        "--project-dir",
        type=str,
        default=".",
        help="Project directory (default: current directory)",
    )
    show_parser.add_argument(
        "--experiment",
        type=str,
        help="Experiment name (optional)",
    )
    show_parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    show_parser.set_defaults(func=cmd_config_show)

    # config get
    get_parser = config_subparsers.add_parser(
        "get",
        help="Get a configuration value",
        description="Get a specific configuration value by path",
    )
    get_parser.add_argument(
        "key",
        type=str,
        help="Configuration key (e.g., audio.sample_rate)",
    )
    get_parser.add_argument(
        "--project-dir",
        type=str,
        default=".",
        help="Project directory (default: current directory)",
    )
    get_parser.add_argument(
        "--experiment",
        type=str,
        help="Experiment name (optional)",
    )
    get_parser.set_defaults(func=cmd_config_get)

    # config set
    set_parser = config_subparsers.add_parser(
        "set",
        help="Set a configuration value",
        description="Set a configuration value in the project config",
    )
    set_parser.add_argument(
        "key",
        type=str,
        help="Configuration key (e.g., audio.sample_rate)",
    )
    set_parser.add_argument(
        "value",
        type=str,
        help="Value to set",
    )
    set_parser.add_argument(
        "--project-dir",
        type=str,
        default=".",
        help="Project directory (default: current directory)",
    )
    set_parser.add_argument(
        "--experiment",
        type=str,
        help="Experiment to update (updates project config if not specified)",
    )
    set_parser.set_defaults(func=cmd_config_set)

    # config list
    list_parser = config_subparsers.add_parser(
        "list",
        help="List all configuration parameters",
        description="List all available configuration parameters with descriptions",
    )
    list_parser.add_argument(
        "--section",
        type=str,
        help="Filter by section (e.g., audio, features, training)",
    )
    list_parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    list_parser.set_defaults(func=cmd_config_list)

    # config schema
    schema_parser = config_subparsers.add_parser(
        "schema",
        help="Export configuration schema",
        description="Export JSON Schema for configuration validation and docs",
    )
    schema_parser.add_argument(
        "--format",
        choices=["json", "markdown", "rst"],
        default="json",
        help="Output format (default: json)",
    )
    schema_parser.add_argument(
        "--output",
        type=str,
        help="Output file (default: stdout)",
    )
    schema_parser.set_defaults(func=cmd_config_schema)

    # config init
    init_parser = config_subparsers.add_parser(
        "init",
        help="Initialize configuration files",
        description="Create default configuration files",
    )
    init_parser.add_argument(
        "--project-dir",
        type=str,
        default=".",
        help="Project directory (default: current directory)",
    )
    init_parser.add_argument(
        "--user",
        action="store_true",
        help="Initialize user config (~/.pstrain/config.yaml)",
    )
    init_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing config files",
    )
    init_parser.set_defaults(func=cmd_config_init)

    # config path
    path_parser = config_subparsers.add_parser(
        "path",
        help="Show configuration file paths",
        description="Display paths to configuration files",
    )
    path_parser.add_argument(
        "--project-dir",
        type=str,
        default=".",
        help="Project directory (default: current directory)",
    )
    path_parser.set_defaults(func=cmd_config_path)


def cmd_config_show(args: Any) -> int:
    """Show current configuration."""
    project_dir = Path(args.project_dir).resolve()

    try:
        config = ConfigManager.load_full_config(project_dir, args.experiment)
    except Exception as e:
        print(f"Error loading config: {e}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(config.to_dict(), indent=2, default=str))
    else:
        _print_config_yaml(config.to_dict())

    return 0


def cmd_config_get(args: Any) -> int:
    """Get a configuration value."""
    project_dir = Path(args.project_dir).resolve()

    try:
        config = ConfigManager.load_full_config(project_dir, args.experiment)
    except Exception as e:
        print(f"Error loading config: {e}", file=sys.stderr)
        return 1

    # Navigate to the key
    value = _get_nested_value(config.to_dict(), args.key)
    if value is None:
        # Check if parameter exists in schema
        param = get_parameter(args.key)
        if param is None:
            print(f"Unknown config key: {args.key}", file=sys.stderr)
            return 1
        print(f"{args.key} = (not set, default: {param.default})")
    else:
        print(f"{args.key} = {value}")

    return 0


def cmd_config_set(args: Any) -> int:
    """Set a configuration value."""
    project_dir = Path(args.project_dir).resolve()

    # Validate key exists
    param = get_parameter(args.key)
    if param is None:
        print(f"Unknown config key: {args.key}", file=sys.stderr)
        print("Use 'pstrain config list' to see available keys.", file=sys.stderr)
        return 1

    # Parse value based on type
    try:
        value = _parse_value(args.value, param.type)
    except ValueError as e:
        print(f"Invalid value: {e}", file=sys.stderr)
        return 1

    # Load current config
    if args.experiment:
        config_path = ConfigManager.get_experiment_config_path(project_dir, args.experiment)
    else:
        config_path = ConfigManager.get_project_config_path(project_dir)

    if config_path.exists():
        import yaml

        with config_path.open(encoding="utf-8") as f:
            config_dict = yaml.safe_load(f) or {}
    else:
        config_dict = {}

    # Set value
    _set_nested_value(config_dict, args.key, value)

    # Save
    import yaml

    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    print(f"Set {args.key} = {value} in {config_path}")
    return 0


def cmd_config_list(args: Any) -> int:
    """List all configuration parameters."""
    prefix = args.section or ""
    params = list_parameters(prefix)

    if not params:
        if prefix:
            print(f"No parameters found with prefix: {prefix}", file=sys.stderr)
        else:
            print("No parameters found", file=sys.stderr)
        return 1

    if args.json:
        print(
            json.dumps(
                [
                    {
                        "key": p.key,
                        "type": p.type,
                        "default": p.default,
                        "description": p.description,
                    }
                    for p in params
                ],
                indent=2,
                default=str,
            )
        )
    else:
        for p in params:
            print(f"{p.key}")
            print(f"  Type: {p.type}")
            print(f"  Default: {p.default}")
            if p.description:
                print(f"  {p.description}")
            print()

    return 0


def cmd_config_schema(args: Any) -> int:
    """Export configuration schema."""
    if args.format == "json":
        output = json.dumps(get_schema(), indent=2)
    elif args.format == "markdown":
        output = generate_markdown_docs()
    elif args.format == "rst":
        from pstrain.lib.config.schema import generate_rst_docs

        output = generate_rst_docs()
    else:
        print(f"Unknown format: {args.format}", file=sys.stderr)
        return 1

    if args.output:
        Path(args.output).write_text(output)
        print(f"Schema written to {args.output}")
    else:
        print(output)

    return 0


def cmd_config_init(args: Any) -> int:
    """Initialize configuration files."""
    if args.user:
        # Initialize user config
        user_config = get_user_config()
        config_path = user_config.get_config_file()

        if config_path.exists() and not args.force:
            print(f"User config already exists: {config_path}", file=sys.stderr)
            print("Use --force to overwrite.", file=sys.stderr)
            return 1

        user_config.save()
        print(f"Created user config: {config_path}")
    else:
        # Initialize project config
        project_dir = Path(args.project_dir).resolve()
        config_path = ConfigManager.get_project_config_path(project_dir)

        if config_path.exists() and not args.force:
            print(f"Project config already exists: {config_path}", file=sys.stderr)
            print("Use --force to overwrite.", file=sys.stderr)
            return 1

        config = PstrainConfig(name=project_dir.name)
        config.to_yaml(config_path)
        print(f"Created project config: {config_path}")

    return 0


def cmd_config_path(args: Any) -> int:
    """Show configuration file paths."""
    project_dir = Path(args.project_dir).resolve()

    user_config = get_user_config()
    user_path = user_config.get_config_file()
    project_path = ConfigManager.get_project_config_path(project_dir)

    print("Configuration file paths:")
    print(f"  User:    {user_path} {'(exists)' if user_path.exists() else '(not found)'}")
    print(f"  Project: {project_path} {'(exists)' if project_path.exists() else '(not found)'}")

    # List experiments if any
    experiments = ConfigManager.list_experiments(project_dir)
    if experiments:
        print("  Experiments:")
        for exp in experiments:
            exp_path = ConfigManager.get_experiment_config_path(project_dir, exp)
            print(f"    - {exp}: {exp_path}")

    return 0


def _print_config_yaml(config_dict: dict[str, Any], indent: int = 0) -> None:
    """Print config dict in YAML-like format."""
    import yaml

    print(yaml.dump(config_dict, default_flow_style=False, sort_keys=False))


def _get_nested_value(d: dict[str, Any], key: str) -> Any:
    """Get nested value from dict using dot notation."""
    keys = key.split(".")
    value = d
    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            return None
    return value


def _set_nested_value(d: dict[str, Any], key: str, value: Any) -> None:
    """Set nested value in dict using dot notation."""
    keys = key.split(".")
    for k in keys[:-1]:
        if k not in d:
            d[k] = {}
        d = d[k]
    d[keys[-1]] = value


def _parse_value(value_str: str, type_str: str) -> Any:
    """Parse string value to appropriate type."""
    type_lower = type_str.lower()

    if type_lower in ("int", "integer"):
        return int(value_str)
    if type_lower in ("float", "number"):
        return float(value_str)
    if type_lower in ("bool", "boolean"):
        return value_str.lower() in ("true", "yes", "1", "on")
    if type_lower.startswith("list"):
        # Simple list parsing: comma-separated
        return [v.strip() for v in value_str.split(",")]
    return value_str
