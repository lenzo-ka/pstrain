"""Pinned, shape-aware corpus archive fetching."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import shutil
import subprocess
import sys
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


def fetch_pinned_archive(url: str, digest: str, cache: Path) -> Path:
    """Fetch an archive without performing network operations in this process.

    This is the only place in pstrain's benchmark code that downloads anything,
    and it does so in a short-lived child. On macOS the first connection a
    process opens installs atfork handlers, after which that process can no
    longer spawn the native worker: the worker dies of SIGSEGV during startup,
    long after and far away from the fetch that poisoned it. Any new caller that
    needs a file from the network must come through here rather than open a
    connection of its own.

    The archive is authenticated against ``digest`` in the child, and a transfer
    that fails or does not match the pin leaves the cache untouched.
    """
    cache.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "_fetch-archive",
        url,
        digest,
        str(cache.resolve()),
    ]
    # close_fds=False with no cwd and no preexec keeps this launch
    # posix_spawn-eligible, which is what avoids the atfork crash in the
    # spawning process. Capturing the streams keeps it so -- the pipes are file
    # descriptors above 2 -- and lets the caller be told why a fetch failed
    # rather than which status a child exited with.
    try:
        subprocess.run(command, check=True, close_fds=False, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"could not fetch {url}: {_child_failure(exc)}") from exc
    return cache / Path(url).name


def _child_failure(exc: subprocess.CalledProcessError) -> str:
    """Report what the fetch helper said rather than the status it exited with."""
    stream = exc.stderr if exc.stderr else exc.stdout
    if isinstance(stream, bytes):
        stream = stream.decode("utf-8", errors="replace")
    lines = [line.strip() for line in (stream or "").splitlines() if line.strip()]
    return lines[-1] if lines else f"helper exited with status {exc.returncode}"


def _fetch_archive_in_helper(url: str, digest: str, cache: Path) -> Path:
    """HEAD-check, download, and authenticate an archive in the network helper."""
    cache.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            if response.status != 200:
                raise RuntimeError(f"HEAD {url}: HTTP {response.status}")
    except (OSError, urllib.error.URLError) as exc:
        raise RuntimeError(f"HEAD failed for {url}: {exc}") from exc
    destination = cache / Path(url).name
    if not destination.exists():
        temporary = destination.with_suffix(destination.suffix + ".part")
        temporary.unlink(missing_ok=True)
        try:
            with (
                urllib.request.urlopen(url, timeout=60) as response,
                temporary.open("xb") as output,
            ):
                shutil.copyfileobj(response, output)
            actual = sha256(temporary)
            if actual != digest:
                raise RuntimeError(f"SHA-256 mismatch for {url}: expected {digest}, got {actual}")
            temporary.replace(destination)
        except (OSError, RuntimeError, urllib.error.URLError):
            temporary.unlink(missing_ok=True)
            raise
    actual = sha256(destination)
    if actual != digest:
        raise RuntimeError(f"SHA-256 mismatch for {destination}: expected {digest}, got {actual}")
    return destination


def fetch_archive(archive: CorpusArchive, cache: Path) -> Path:
    """Fetch atomically, or fully re-verify an existing pinned archive."""
    destination = cache / Path(archive.url).name
    if destination.exists():
        verify_archive(destination, archive)
        return destination
    fetch_pinned_archive(archive.url, archive.sha256, cache)
    try:
        verify_archive(destination, archive)
    except (OSError, RuntimeError, tarfile.TarError):
        destination.unlink(missing_ok=True)
        raise
    return destination


def main(argv: list[str] | None = None) -> None:
    """Fetch every archive for one manifest corpus."""
    if argv is None:
        argv = sys.argv[1:]
    if argv[:1] == ["_fetch-archive"]:
        if len(argv) != 4:
            raise SystemExit("_fetch-archive requires a URL, a SHA-256, and a cache path")
        _fetch_archive_in_helper(argv[1], argv[2], Path(argv[3]))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus")
    parser.add_argument("cache", type=Path)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args(argv)
    for archive in load_corpus(args.corpus, args.manifest):
        print(fetch_archive(archive, args.cache))


if __name__ == "__main__":
    main()
