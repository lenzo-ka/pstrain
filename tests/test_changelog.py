"""Checks for release metadata drift in the changelog."""

import re
from datetime import date
from pathlib import Path
from tomllib import load


def test_latest_changelog_version_matches_project_version() -> None:
    """The latest dated release stays synchronized with package metadata."""
    root = Path(__file__).parents[1]
    changelog = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    headings = re.findall(r"^##[ \t]+(?P<heading>[^\n]+?)[ \t]*$", changelog, re.MULTILINE)

    assert headings, "CHANGELOG.md has no section heading"
    if headings[0].casefold() == "unreleased":
        assert len(headings) > 1, "CHANGELOG.md has no release below Unreleased"
        latest_release = headings[1]
    else:
        latest_release = headings[0]
    heading = re.fullmatch(
        r"(?P<version>\S+)[ \t]+-[ \t]+(?P<date>\d{4}-\d{2}-\d{2})", latest_release
    )
    assert heading is not None, "CHANGELOG.md has no dated release heading"
    date.fromisoformat(heading["date"])
    with (root / "pyproject.toml").open("rb") as pyproject:
        project_version = load(pyproject)["project"]["version"]

    assert heading["version"] == project_version, (
        "topmost CHANGELOG.md version heading differs from pyproject.toml: "
        f"{heading['version']} != {project_version}"
    )
