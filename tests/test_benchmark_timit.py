from __future__ import annotations

from pathlib import Path

import yaml

from pstrain.benchmarks.timit import pinned_profile, prepare_project, tree_sha256


def _write_data(data: Path) -> None:
    data.mkdir()
    (data / "train.fileids").write_text("train/dr1/fabc0/sx1\n")
    (data / "test.fileids").write_text("test/dr2/mdef0/sx2\n")
    (data / "train.transcription").write_text("train/dr1/fabc0/sx1 hello\n")
    (data / "test.transcription").write_text("test/dr2/mdef0/sx2 world\n")
    (data / "timit.reduced.dict").write_text("hello hh eh l ow\nworld w er l d\n")


def test_paired_profiles_differ_only_on_dither() -> None:
    off = pinned_profile(dither=False, seed=243)
    on = pinned_profile(dither=True, seed=243)
    assert off["features"]["seed"] == on["features"]["seed"] == 243
    off["features"]["dither"] = True
    assert off == on


def test_project_installs_canonical_external_split(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    data = tmp_path / "data"
    _write_data(data)
    project = tmp_path / "project"
    prepare_project(project, corpus, data, dither=True, seed=243)
    assert project.joinpath("audio").resolve() == corpus
    assert project.joinpath("etc/all.transcription").read_text().splitlines() == [
        "train/dr1/fabc0/sx1 hello",
        "test/dr2/mdef0/sx2 world",
    ]
    for name in ("train.fileids", "test.fileids", "train.transcription", "test.transcription"):
        assert project.joinpath("etc", name).read_bytes() == data.joinpath(name).read_bytes()
    config = yaml.safe_load(project.joinpath("etc/configs.yaml").read_text())
    assert (
        config["profiles"]["timit"]["features"] | {"dither": True, "seed": 243}
        == config["profiles"]["timit"]["features"]
    )


def test_tree_hash_covers_names_and_contents(tmp_path: Path) -> None:
    (tmp_path / "one").write_text("same")
    first = tree_sha256(tmp_path)
    (tmp_path / "one").rename(tmp_path / "two")
    assert tree_sha256(tmp_path) != first
    assert len(tree_sha256(tmp_path)) == 64
