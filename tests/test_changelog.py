"""Checks for release metadata drift in the changelog."""

import re
from pathlib import Path
from tomllib import load


def test_latest_changelog_version_matches_project_version() -> None:
    """The latest release heading stays synchronized with package metadata."""
    root = Path(__file__).parents[1]
    changelog = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    heading = re.search(r"^##\s+(?P<version>\S+)\s+-\s+(?P<label>.+?)\s*$", changelog, re.MULTILINE)

    assert heading is not None, "CHANGELOG.md has no version heading"
    with (root / "pyproject.toml").open("rb") as pyproject:
        project_version = load(pyproject)["project"]["version"]

    assert heading["version"] == project_version, (
        "topmost CHANGELOG.md version heading differs from pyproject.toml: "
        f"{heading['version']} != {project_version}"
    )
    assert "unreleased" not in heading["label"].casefold(), (
        "topmost CHANGELOG.md version heading is still marked Unreleased"
    )
