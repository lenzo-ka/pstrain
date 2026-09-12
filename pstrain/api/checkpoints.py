"""Public API for explicit checkpoint inspection and restoration."""

from pathlib import Path
from typing import Any

from pstrain.lib import checkpoints as _implementation

__all__ = ["list_checkpoints", "restore_checkpoint"]


def list_checkpoints(model_dir: Path) -> list[dict[str, Any]]:
    """Inspect retained update identities and their available evaluation evidence."""
    return _implementation.list_checkpoints(model_dir)


def restore_checkpoint(model_dir: Path, number: int, *, dry_run: bool = False) -> dict[str, Any]:
    """Explicitly restore one retained update, with a backup and stale provenance."""
    return _implementation.restore_checkpoint(model_dir, number, dry_run=dry_run)
