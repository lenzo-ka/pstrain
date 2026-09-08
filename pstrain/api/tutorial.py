"""Public API for copying the bundled tutorial notebook."""

from __future__ import annotations

import os
import tempfile
from contextlib import suppress
from importlib.resources import files
from pathlib import Path
from typing import TypedDict

TUTORIAL_FILENAME = "arctic_hmm_gmm_tutorial.ipynb"


class TutorialResult(TypedDict):
    """JSON-serializable result of a tutorial copy request."""

    status: str
    path: str


class TutorialExistsError(FileExistsError):
    """Raised when a tutorial destination may not be replaced."""

    def __init__(self, path: Path) -> None:
        self.path = path
        super().__init__(
            f"Refusing to replace {path}: destination already exists. Use --force to replace it."
        )


def _read_tutorial() -> bytes:
    """Read the packaged tutorial, or its source-checkout counterpart."""
    packaged = files("pstrain.data").joinpath("notebooks", TUTORIAL_FILENAME)
    if packaged.is_file():
        return packaged.read_bytes()

    repository_root = Path(__file__).resolve().parents[2]
    checkout = repository_root / "notebooks" / TUTORIAL_FILENAME
    if (repository_root / "pyproject.toml").is_file() and checkout.is_file():
        return checkout.read_bytes()

    raise FileNotFoundError(
        "the pstrain tutorial notebook is missing from this installation. "
        "Reinstall pstrain, or restore notebooks/arctic_hmm_gmm_tutorial.ipynb "
        "if this is a source checkout."
    )


def copy_tutorial(
    output: str | Path = TUTORIAL_FILENAME, *, force: bool = False, dry_run: bool = False
) -> TutorialResult:
    """Copy the bundled tutorial notebook to *output*.

    If *output* is an existing directory, the standard tutorial filename is
    appended. Existing files are protected unless *force* is true.
    """
    destination = Path(output)
    if destination.is_dir():
        destination /= TUTORIAL_FILENAME
    destination = destination.absolute()

    if not force and os.path.lexists(destination):
        raise TutorialExistsError(destination)

    result: TutorialResult = {
        "status": "dry-run" if dry_run else "written",
        "path": str(destination),
    }
    if dry_run:
        return result

    destination.parent.mkdir(parents=True, exist_ok=True)
    content = _read_tutorial()
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
        if force:
            temporary.replace(destination)
        else:
            try:
                os.link(temporary, destination)
            except FileExistsError as error:
                raise TutorialExistsError(destination) from error
            except OSError as error:
                raise OSError(
                    f"Cannot safely write {destination}: its filesystem does not support "
                    "atomic no-clobber publication with hard links. Choose a destination on "
                    "a filesystem that supports hard links, or use --force if replacing the "
                    "destination is acceptable."
                ) from error
            # The destination is already published. Failure to remove its other hard-link
            # name must not turn a successful, safe copy into a reported failure.
            with suppress(OSError):
                temporary.unlink()
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return result


__all__ = [
    "TUTORIAL_FILENAME",
    "TutorialExistsError",
    "TutorialResult",
    "copy_tutorial",
]
