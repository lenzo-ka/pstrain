"""Basic version test."""

from pathlib import Path
from tomllib import load

from pstrain import __version__


def test_version() -> None:
    with (Path(__file__).parents[1] / "pyproject.toml").open("rb") as pyproject:
        assert __version__ == load(pyproject)["project"]["version"]
