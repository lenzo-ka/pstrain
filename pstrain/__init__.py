"""pstrain - Acoustic model training toolkit."""

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from tomllib import load

try:
    __version__ = version("pstrain")
except PackageNotFoundError:  # An unpackaged source checkout.
    with (Path(__file__).parents[1] / "pyproject.toml").open("rb") as pyproject:
        __version__ = load(pyproject)["project"]["version"]
