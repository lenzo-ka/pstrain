"""Fetch CMUdict from its Carnegie Mellon upstream repository.

The downloaded dictionary and accompanying files are Copyright Carnegie
Mellon University and distributed under the license cached beside them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, cast

from pstrain.lib.dictionary.cmudict import strip_dictionary_stress

UPSTREAM = "https://github.com/cmusphinx/cmudict"
API_ROOT = "https://api.github.com/repos/cmusphinx/cmudict"
ARCHIVE_ROOT = "https://codeload.github.com/cmusphinx/cmudict/tar.gz"
REQUIRED_FILES = ("cmudict.dict", "cmudict.phones", "cmudict.symbols", "LICENSE")
METADATA_NAME = "source.json"


@dataclass(frozen=True)
class CMUDictSource:
    """Paths and provenance for a cached upstream CMUdict checkout."""

    dictionary: Path
    phones: Path
    symbols: Path
    license: Path
    source_dictionary: Path
    requested_ref: str
    resolved_ref: str
    cache_directory: Path


def default_cmudict_cache() -> Path:
    """Return the CMUdict cache within pstrain's benchmark cache hierarchy."""
    root = Path(
        os.environ.get("PSTRAIN_BENCH_CACHE", Path.home() / ".cache" / "pstrain" / "benchmarks")
    )
    return root / "cmudict"


def fetch_cmudict(ref: str = "HEAD", cache: Path | None = None) -> CMUDictSource:
    """Fetch, convert, and cache CMUdict from its original upstream source.

    ``ref`` may be a tag, branch, or commit SHA. Network access happens only in
    a fresh child process. On macOS, an in-process HTTPS download followed by a
    native worker spawn crashes the interpreter, so the download is kept out of
    any process that later spawns workers.
    """
    cache_root = (cache or default_cmudict_cache()).expanduser().resolve()
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "_fetch",
        ref,
        str(cache_root),
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            close_fds=False,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or exc.stdout.strip() or str(exc)
        raise RuntimeError(f"Could not fetch CMUdict ref {ref!r}: {detail}") from exc
    try:
        return _source_from_metadata(Path(result.stdout.strip()))
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "CMUdict fetch helper returned invalid provenance; remove the CMUdict cache "
            "entry and retry"
        ) from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _open_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(
        url, headers={"Accept": "application/vnd.github+json", "User-Agent": "pstrain"}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return cast(dict[str, Any], json.load(response))


def _resolve_ref(ref: str) -> str:
    endpoint = "commits/HEAD" if ref == "HEAD" else f"commits/{urllib.parse.quote(ref, safe='')}"
    try:
        payload = _open_json(f"{API_ROOT}/{endpoint}")
        resolved = payload["sha"]
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise RuntimeError(
                f"CMUdict ref {ref!r} does not exist upstream; use a valid tag, branch, or commit SHA"
            ) from exc
        raise RuntimeError(
            f"unable to resolve CMUdict ref {ref!r}: HTTP {exc.code}; retry"
        ) from exc
    except (OSError, urllib.error.URLError, KeyError, TypeError) as exc:
        raise RuntimeError(
            f"unable to resolve CMUdict ref {ref!r}; check network access and retry: {exc}"
        ) from exc
    if not isinstance(resolved, str) or len(resolved) != 40:
        raise RuntimeError(f"upstream returned an invalid commit for CMUdict ref {ref!r}; retry")
    return resolved


def _download(url: str, destination: Path) -> None:
    try:
        with urllib.request.urlopen(url, timeout=60) as response, destination.open("wb") as output:
            shutil.copyfileobj(response, output)
    except (OSError, urllib.error.URLError) as exc:
        destination.unlink(missing_ok=True)
        raise RuntimeError(
            f"CMUdict download failed; check network access and retry: {exc}"
        ) from exc
    if destination.stat().st_size == 0:
        destination.unlink(missing_ok=True)
        raise RuntimeError("CMUdict download was empty; check network access and retry")


def _extract_required(archive: Path, destination: Path) -> None:
    try:
        with tarfile.open(archive, "r:gz") as bundle:
            members = {Path(member.name).name: member for member in bundle if member.isfile()}
            for name in REQUIRED_FILES:
                member = members.get(name)
                if member is None:
                    raise RuntimeError(f"downloaded CMUdict archive is missing {name}; retry")
                source: IO[bytes] | None = bundle.extractfile(member)
                if source is None:
                    raise RuntimeError(f"could not read {name} from CMUdict archive; retry")
                with source, (destination / name).open("wb") as output:
                    shutil.copyfileobj(source, output)
    except (OSError, tarfile.TarError) as exc:
        raise RuntimeError(f"downloaded CMUdict archive is corrupt; retry: {exc}") from exc


def _metadata(directory: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads((directory / METADATA_NAME).read_text(encoding="utf-8")))


def _validate_cache(directory: Path) -> dict[str, Any]:
    try:
        metadata = _metadata(directory)
        hashes = metadata["sha256"]
        for name in (*REQUIRED_FILES, "cmudict.dict.pocketsphinx"):
            path = directory / name
            if not path.is_file() or _sha256(path) != hashes[name]:
                raise ValueError(name)
        if metadata["resolved_ref"] != directory.name:
            raise ValueError("resolved_ref")
        return metadata
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"cached CMUdict data is damaged at {directory}; remove that directory and retry"
        ) from exc


def _fetch_in_helper(ref: str, cache_root: Path) -> Path:
    resolved = _resolve_ref(ref)
    destination = cache_root / resolved
    if destination.exists():
        _validate_cache(destination)
        return destination

    cache_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{resolved}.", dir=cache_root))
    archive = temporary / "source.tar.gz.part"
    try:
        _download(f"{ARCHIVE_ROOT}/{resolved}", archive)
        _extract_required(archive, temporary)
        archive.unlink()
        strip_dictionary_stress(temporary / "cmudict.dict", temporary / "cmudict.dict.pocketsphinx")
        names = (*REQUIRED_FILES, "cmudict.dict.pocketsphinx")
        metadata = {
            "upstream": UPSTREAM,
            "requested_ref": ref,
            "resolved_ref": resolved,
            "sha256": {name: _sha256(temporary / name) for name in names},
        }
        (temporary / METADATA_NAME).write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        try:
            temporary.replace(destination)
        except FileExistsError:
            _validate_cache(destination)
            shutil.rmtree(temporary)
        return destination
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _source_from_metadata(directory: Path) -> CMUDictSource:
    metadata = _validate_cache(directory)
    return CMUDictSource(
        dictionary=directory / "cmudict.dict.pocketsphinx",
        phones=directory / "cmudict.phones",
        symbols=directory / "cmudict.symbols",
        license=directory / "LICENSE",
        source_dictionary=directory / "cmudict.dict",
        requested_ref=metadata["requested_ref"],
        resolved_ref=metadata["resolved_ref"],
        cache_directory=directory,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["_fetch"])
    parser.add_argument("ref")
    parser.add_argument("cache", type=Path)
    args = parser.parse_args(argv)
    try:
        print(_fetch_in_helper(args.ref, args.cache))
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
