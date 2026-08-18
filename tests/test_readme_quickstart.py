import re
from pathlib import Path


def test_checkout_quickstart_fixture_paths_exist() -> None:
    """Keep fixture paths in the checkout quickstart runnable."""
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text()
    checkout = readme.split("### From a checkout", 1)[1].split("\n## ", 1)[0]
    fixture_paths = re.findall(r"tests/fixtures/[\w./-]+", checkout)

    assert len(fixture_paths) >= 5
    for relative_path in fixture_paths:
        assert (root / relative_path).exists(), f"README path does not exist: {relative_path}"
