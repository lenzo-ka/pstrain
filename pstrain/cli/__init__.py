"""Command-line interface for pstrain.

The CLI provides commands for the complete acoustic model training workflow:
- ``pstrain setup`` - Initialize a new project
- ``pstrain validate-project`` - Validate project structure
- ``pstrain split`` - Split data into train/test sets
- ``pstrain features`` - Extract acoustic features
- ``pstrain flat`` - Initialize flat HMM models
- ``pstrain build`` - Build a model target (e.g. ci-1g, cd-8g)
- ``pstrain clean`` - Clean training outputs
- ``pstrain config`` - Manage configuration
- ``pstrain step`` - Run numbered training steps

All commands support ``--dry-run`` to preview actions without execution.
"""

from pstrain.cli.base import Command, CommandContext, CommandResult, ModelCommand, ProjectCommand
from pstrain.cli.cli import main

__all__ = [
    "main",
    "Command",
    "CommandContext",
    "CommandResult",
    "ModelCommand",
    "ProjectCommand",
]
