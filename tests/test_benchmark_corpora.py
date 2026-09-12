from __future__ import annotations

import hashlib
import io
import json
import subprocess
import tarfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from pstrain.benchmarks.corpora import (
    ContentCheck,
    CorpusArchive,
    _fetch_archive_in_helper,
    fetch_archive,
    load_corpus,
)


def test_kal_manifest_records_complete_shape() -> None:
    archives = load_corpus("cmu_us_kal_diphone")
    assert len(archives) == 1
    archive = archives[0]
    assert Path(archive.url).name == "cmu_us_kal_diphone.tar.bz2"
    assert len(archive.sha256) == 64
    assert {(check.pattern, check.count) for check in archive.checks} >= {
        ("cmu_us_kal_diphone/wav/*.wav", 1349),
        ("cmu_us_kal_diphone/lab/*.lab", 1349),
        ("cmu_us_kal_diphone/prompt-lab/*.lab", 1349),
        ("cmu_us_kal_diphone/COPYING", 1),
    }


def _fixture_archive(path: Path) -> CorpusArchive:
    with tarfile.open(path, "w:bz2") as packed:
        for name, content in (("corpus/wav/one.wav", b"wav"), ("corpus/COPYING", b"license")):
            member = tarfile.TarInfo(name)
            member.size = len(content)
            packed.addfile(member, io.BytesIO(content))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return CorpusArchive(
        "http://example.test/corpus.tar.bz2",
        digest,
        (ContentCheck("corpus/wav/*.wav", 1), ContentCheck("corpus/COPYING", 1)),
    )


class _Response(io.BytesIO):
    """Enough of an HTTP response for the fetch helper's HEAD and GET."""

    status = 200


def _fake_urlopen(body: bytes) -> Any:
    return lambda *_args, **_kwargs: _Response(body)


def _refuse_network(monkeypatch: pytest.MonkeyPatch, reason: str) -> None:
    monkeypatch.setattr(
        "pstrain.benchmarks.corpora.urllib.request.urlopen",
        lambda *_args, **_kwargs: pytest.fail(reason),
    )


def test_existing_archive_is_verified_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "corpus.tar.bz2"
    archive = _fixture_archive(destination)
    _refuse_network(monkeypatch, "existing archives must not be downloaded")
    monkeypatch.setattr(
        "pstrain.benchmarks.corpora.subprocess.run",
        lambda *_args, **_kwargs: pytest.fail("existing archives must not spawn a fetcher"),
    )
    assert fetch_archive(archive, tmp_path) == destination
    destination.write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        fetch_archive(archive, tmp_path)


def test_fresh_fetch_downloads_in_a_child_and_never_here(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A caller of fetch_archive must be able to spawn a native worker afterwards."""
    staged = tmp_path / "staged" / "corpus.tar.bz2"
    staged.parent.mkdir()
    archive = _fixture_archive(staged)
    cache = tmp_path / "cache"
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def fake_run(command: list[str], **kwargs: Any) -> None:
        calls.append((command, kwargs))
        (cache / "corpus.tar.bz2").write_bytes(staged.read_bytes())

    _refuse_network(monkeypatch, "the fetching process must never open a connection")
    monkeypatch.setattr("pstrain.benchmarks.corpora.subprocess.run", fake_run)

    assert fetch_archive(archive, cache) == cache / "corpus.tar.bz2"
    command, kwargs = calls[0]
    assert command[2:] == ["_fetch-archive", archive.url, archive.sha256, str(cache.resolve())]
    # The launch shape is the fix: posix_spawn eligibility is what keeps the
    # caller able to fork a native worker afterwards.
    assert kwargs["close_fds"] is False
    assert kwargs["check"] is True
    assert kwargs["capture_output"] is True
    assert "cwd" not in kwargs and "preexec_fn" not in kwargs


def test_fetch_failure_reports_what_the_child_said(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run(command: list[str], **kwargs: Any) -> None:
        raise subprocess.CalledProcessError(
            1, command, stderr="Traceback\nRuntimeError: HEAD failed for http://example.test\n"
        )

    monkeypatch.setattr("pstrain.benchmarks.corpora.subprocess.run", fake_run)
    archive = _fixture_archive(tmp_path / "staged.tar.bz2")
    with pytest.raises(RuntimeError, match="HEAD failed for http://example.test"):
        fetch_archive(archive, tmp_path / "cache")


def test_helper_rejects_a_download_that_misses_the_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("pstrain.benchmarks.corpora.urllib.request.urlopen", _fake_urlopen(b"body"))
    cache = tmp_path / "cache"
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        _fetch_archive_in_helper("http://example.test/corpus.tar.bz2", "0" * 64, cache)
    assert list(cache.iterdir()) == []
    digest = hashlib.sha256(b"body").hexdigest()
    assert _fetch_archive_in_helper("http://example.test/corpus.tar.bz2", digest, cache).exists()


def test_content_mismatch_is_rejected(tmp_path: Path) -> None:
    destination = tmp_path / "corpus.tar.bz2"
    archive = _fixture_archive(destination)
    with pytest.raises(RuntimeError, match="content mismatch"):
        fetch_archive(replace(archive, checks=(ContentCheck("corpus/lab/*.lab", 1),)), tmp_path)


def test_a_freshly_fetched_archive_of_the_wrong_shape_is_not_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staged = tmp_path / "staged" / "corpus.tar.bz2"
    staged.parent.mkdir()
    archive = _fixture_archive(staged)
    cache = tmp_path / "cache"
    monkeypatch.setattr(
        "pstrain.benchmarks.corpora.subprocess.run",
        lambda *_args, **_kwargs: (cache / "corpus.tar.bz2").write_bytes(staged.read_bytes()),
    )
    with pytest.raises(RuntimeError, match="content mismatch"):
        fetch_archive(replace(archive, checks=(ContentCheck("corpus/lab/*.lab", 1),)), cache)
    assert not (cache / "corpus.tar.bz2").exists()


def test_manifest_rejects_archive_schema_drift(tmp_path: Path) -> None:
    manifest = tmp_path / "archives.json"
    manifest.write_text(
        json.dumps(
            {
                "corpora": [
                    {
                        "name": "bad",
                        "archives": [{"url": "x", "sha256": "y", "checks": [], "extra": 1}],
                    }
                ]
            }
        )
    )
    with pytest.raises(RuntimeError, match="invalid archive entry"):
        load_corpus("bad", manifest)
