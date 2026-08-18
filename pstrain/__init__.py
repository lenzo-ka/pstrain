"""pstrain - Acoustic model training toolkit."""

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from tomllib import load

try:
    __version__ = version("pstrain")
except PackageNotFoundError:  # An unpackaged source checkout.
    try:
        with (Path(__file__).parents[1] / "pyproject.toml").open("rb") as pyproject:
            __version__ = load(pyproject)["project"]["version"]
    except (FileNotFoundError, KeyError):
        __version__ = "0.0.0+unknown"
