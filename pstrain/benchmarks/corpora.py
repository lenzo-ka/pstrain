"""Pinned, shape-aware corpus archive fetching."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import shutil
import tarfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "benchmarks" / "corpora" / "archives.json"


@dataclass(frozen=True)
class ContentCheck:
    """One expected archive-member inventory."""

    pattern: str
    count: int


@dataclass(frozen=True)
class CorpusArchive:
    """One immutable archive and its declarative content checks."""

    url: str
    sha256: str
    checks: tuple[ContentCheck, ...]


def sha256(path: Path) -> str:
    """Hash a file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_corpus(name: str, manifest: Path = DEFAULT_MANIFEST) -> tuple[CorpusArchive, ...]:
    """Load one named corpus from the strict JSON manifest."""
    document: Any = json.loads(manifest.read_text(encoding="utf-8"))
    rows = document.get("corpora") if isinstance(document, dict) else None
    if not isinstance(rows, list):
        raise RuntimeError(f"invalid corpus manifest: {manifest}")
    matches = [row for row in rows if isinstance(row, dict) and row.get("name") == name]
    if len(matches) != 1:
        raise RuntimeError(f"corpus {name!r} appears {len(matches)} times in {manifest}")
    archives = matches[0].get("archives")
    if not isinstance(archives, list) or not archives:
        raise RuntimeError(f"corpus {name!r} has no archives")
    result = []
    for row in archives:
        if not isinstance(row, dict) or set(row) != {"url", "sha256", "checks"}:
            raise RuntimeError(f"invalid archive entry for corpus {name!r}")
        checks = row["checks"]
        if not isinstance(checks, list) or not checks:
            raise RuntimeError(f"archive for corpus {name!r} has no content checks")
        result.append(
            CorpusArchive(
                url=str(row["url"]),
                sha256=str(row["sha256"]),
                checks=tuple(ContentCheck(**check) for check in checks),
            )
        )
    return tuple(result)


def verify_archive(path: Path, archive: CorpusArchive) -> None:
    """Authenticate an archive and validate its declared shape."""
    actual = sha256(path)
    if actual != archive.sha256:
        raise RuntimeError(f"SHA-256 mismatch for {path}: expected {archive.sha256}, got {actual}")
    with tarfile.open(path, "r:*") as packed:
        members = [member.name for member in packed.getmembers()]
    for check in archive.checks:
        actual_count = sum(fnmatch.fnmatchcase(member, check.pattern) for member in members)
        if actual_count != check.count:
            raise RuntimeError(
                f"content mismatch for {path}: {check.pattern!r} expected "
                f"{check.count}, got {actual_count}"
            )


def fetch_archive(archive: CorpusArchive, cache: Path) -> Path:
    """Fetch atomically, or fully re-verify an existing pinned archive."""
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / Path(archive.url).name
    if destination.exists():
        verify_archive(destination, archive)
        return destination
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.unlink(missing_ok=True)
    try:
        with (
            urllib.request.urlopen(archive.url, timeout=60) as response,
            temporary.open("xb") as output,
        ):
            shutil.copyfileobj(response, output)
        verify_archive(temporary, archive)
        temporary.replace(destination)
    except (OSError, RuntimeError, urllib.error.URLError, tarfile.TarError):
        temporary.unlink(missing_ok=True)
        raise
    return destination


def main(argv: list[str] | None = None) -> None:
    """Fetch every archive for one manifest corpus."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus")
    parser.add_argument("cache", type=Path)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args(argv)
    for archive in load_corpus(args.corpus, args.manifest):
        print(fetch_archive(archive, args.cache))


if __name__ == "__main__":
    main()
