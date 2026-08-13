from __future__ import annotations

import hashlib
import io
import json
import tarfile
from dataclasses import replace
from pathlib import Path

import pytest

from pstrain.benchmarks.corpora import ContentCheck, CorpusArchive, fetch_archive, load_corpus


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


def test_existing_archive_is_verified_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "corpus.tar.bz2"
    archive = _fixture_archive(destination)
    monkeypatch.setattr(
        "pstrain.benchmarks.corpora.urllib.request.urlopen",
        lambda *_args, **_kwargs: pytest.fail("existing archives must not be downloaded"),
    )
    assert fetch_archive(archive, tmp_path) == destination
    destination.write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        fetch_archive(archive, tmp_path)


def test_content_mismatch_is_rejected(tmp_path: Path) -> None:
    destination = tmp_path / "corpus.tar.bz2"
    archive = _fixture_archive(destination)
    with pytest.raises(RuntimeError, match="content mismatch"):
        fetch_archive(replace(archive, checks=(ContentCheck("corpus/lab/*.lab", 1),)), tmp_path)


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
