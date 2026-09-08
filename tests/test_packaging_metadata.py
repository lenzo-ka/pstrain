"""Checks that published package metadata matches the distribution builds."""

import re
from pathlib import Path
from tomllib import load

ROOT = Path(__file__).parents[1]
PLATFORM_CLASSIFIERS = {
    "macos": "Operating System :: MacOS",
    "ubuntu": "Operating System :: POSIX :: Linux",
    "windows": "Operating System :: Microsoft :: Windows",
}


def _wheel_platforms(workflow: str) -> set[str]:
    contents = (ROOT / ".github" / "workflows" / workflow).read_text(encoding="utf-8")
    match = re.search(r"^\s+os: \[(?P<runners>[^]]+)]$", contents, re.MULTILINE)
    assert match is not None, f"{workflow} has no inline wheel runner matrix"
    runners = (runner.strip() for runner in match["runners"].split(","))
    return {runner.split("-", maxsplit=1)[0] for runner in runners}


def test_classifiers_match_built_distributions() -> None:
    """Platform and Python classifiers describe the wheels built for release."""
    with (ROOT / "pyproject.toml").open("rb") as pyproject:
        configuration = load(pyproject)

    classifiers = set(configuration["project"]["classifiers"])
    platform_classifiers = {
        classifier for classifier in classifiers if classifier.startswith("Operating System ::")
    }
    build_platforms = _wheel_platforms("build.yml")
    release_platforms = _wheel_platforms("release.yml")

    assert build_platforms == release_platforms
    assert build_platforms == PLATFORM_CLASSIFIERS.keys()
    assert platform_classifiers == {PLATFORM_CLASSIFIERS[platform] for platform in build_platforms}

    wheel_builds = configuration["tool"]["cibuildwheel"]["build"]
    wheel_python_versions = {f"{build[2]}.{build[3:-2]}" for build in wheel_builds}
    classified_python_versions = {
        classifier.removeprefix("Programming Language :: Python :: ")
        for classifier in classifiers
        if re.fullmatch(r"Programming Language :: Python :: \d+\.\d+", classifier)
    }
    assert classified_python_versions == wheel_python_versions
