from __future__ import annotations

import io
import json
import subprocess
import tarfile
import urllib.error
from pathlib import Path

import pytest

from pstrain.lib.dictionary.cmudict import strip_dictionary_stress
from pstrain.lib.dictionary.cmudict_source import (
    CMUDictSource,
    _fetch_in_helper,
    _source_from_metadata,
    fetch_cmudict,
)

COMMIT = "1" * 40
SOURCE = b";;; header\nzeta Z IY1 T AH0\nalpha AE1 L F AH0\nalpha(2) AE2 L F AH0\nalpha(3) AE1 L F AX0 # note\n"


def _archive(path: Path) -> None:
    files = {
        "cmudict.dict": SOURCE,
        "cmudict.phones": b"AE vowel\n",
        "cmudict.symbols": b"AE\nAE0\nAE1\nAE2\n",
        "LICENSE": b"Copyright Carnegie Mellon University\n",
    }
    with tarfile.open(path, "w:gz") as bundle:
        for name, content in files.items():
            info = tarfile.TarInfo(f"cmudict-test/{name}")
            info.size = len(content)
            bundle.addfile(info, io.BytesIO(content))


def _stub_transport(monkeypatch: pytest.MonkeyPatch, downloads: list[str]) -> None:
    monkeypatch.setattr(
        "pstrain.lib.dictionary.cmudict_source._open_json", lambda _url: {"sha": COMMIT}
    )

    def download(url: str, destination: Path) -> None:
        downloads.append(url)
        _archive(destination)

    monkeypatch.setattr("pstrain.lib.dictionary.cmudict_source._download", download)


def test_fresh_fetch_and_cache_hit_do_not_redownload(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    downloads: list[str] = []
    _stub_transport(monkeypatch, downloads)

    first = _fetch_in_helper("HEAD", tmp_path)
    second = _fetch_in_helper("HEAD", tmp_path)

    assert first == second == tmp_path / COMMIT
    assert len(downloads) == 1
    source = _source_from_metadata(first)
    assert isinstance(source, CMUDictSource)
    assert source.resolved_ref == COMMIT
    assert source.requested_ref == "HEAD"
    assert source.dictionary.read_text() == "alpha AE L F AH\nalpha(2) AE L F AX\nzeta Z IY T AH\n"
    metadata = json.loads((first / "source.json").read_text())
    assert metadata["upstream"] == "https://github.com/cmusphinx/cmudict"
    assert metadata["resolved_ref"] == COMMIT


def test_explicit_pinned_ref_is_resolved_and_recorded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    downloads: list[str] = []
    seen: list[str] = []
    monkeypatch.setattr(
        "pstrain.lib.dictionary.cmudict_source._open_json",
        lambda url: seen.append(url) or {"sha": COMMIT},
    )
    monkeypatch.setattr(
        "pstrain.lib.dictionary.cmudict_source._download",
        lambda url, path: (downloads.append(url), _archive(path)),
    )

    directory = _fetch_in_helper("0.7b", tmp_path)

    assert seen[0].endswith("/commits/0.7b")
    assert downloads[0].endswith(f"/{COMMIT}")
    assert json.loads((directory / "source.json").read_text())["requested_ref"] == "0.7b"


def test_unknown_ref_names_recovery(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    error = urllib.error.HTTPError("url", 404, "not found", {}, None)
    monkeypatch.setattr(
        "pstrain.lib.dictionary.cmudict_source._open_json",
        lambda _url: (_ for _ in ()).throw(error),
    )
    with pytest.raises(RuntimeError, match="does not exist upstream.*valid tag, branch, or commit"):
        _fetch_in_helper("missing", tmp_path)


def test_corrupt_cache_refuses_instead_of_redownloading(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    downloads: list[str] = []
    _stub_transport(monkeypatch, downloads)
    directory = _fetch_in_helper("HEAD", tmp_path)
    (directory / "cmudict.dict").write_text("damaged")

    with pytest.raises(RuntimeError, match="damaged.*remove that directory and retry"):
        _fetch_in_helper("HEAD", tmp_path)
    assert len(downloads) == 1


def test_conversion_matches_upstream_converter_sample(tmp_path: Path) -> None:
    source = tmp_path / "source.dict"
    output = tmp_path / "output.dict"
    source.write_bytes(SOURCE)

    assert strip_dictionary_stress(source, output) == (3, 8)
    # Output produced by cmusphinx/cmudict make_ps_dict.py convert_dict for this sample.
    assert output.read_text() == "alpha AE L F AH\nalpha(2) AE L F AX\nzeta Z IY T AH\n"


def test_public_fetch_uses_network_clean_child(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache_entry = tmp_path / COMMIT
    cache_entry.mkdir()
    names = ("cmudict.dict", "cmudict.phones", "cmudict.symbols", "LICENSE")
    for name in names:
        (cache_entry / name).write_text(name)
    (cache_entry / "cmudict.dict.pocketsphinx").write_text("WORD W ER D\n")
    import hashlib

    all_names = (*names, "cmudict.dict.pocketsphinx")
    metadata = {
        "upstream": "https://github.com/cmusphinx/cmudict",
        "requested_ref": "HEAD",
        "resolved_ref": COMMIT,
        "sha256": {
            name: hashlib.sha256((cache_entry / name).read_bytes()).hexdigest()
            for name in all_names
        },
    }
    (cache_entry / "source.json").write_text(json.dumps(metadata))
    calls: list[tuple[list[str], dict[str, object]]] = []

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout=f"{cache_entry}\n", stderr="")

    monkeypatch.setattr("pstrain.lib.dictionary.cmudict_source.subprocess.run", run)
    result = fetch_cmudict(cache=tmp_path)

    assert result.resolved_ref == COMMIT
    assert calls[0][0][2] == "_fetch"
    assert calls[0][1]["close_fds"] is False
    assert calls[0][1]["capture_output"] is True
