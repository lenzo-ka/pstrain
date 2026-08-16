"""Canonical-split TIMIT dither adjudication benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from pstrain.lib.config.models import Profile
from pstrain.lib.corpus import split_is_external
from pstrain.lib.corpus.split import SPLIT_FILENAMES, VALIDATED_SPLIT

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEED = 243
DEFAULT_TARGET = "ci-1g"
CELLS = {"dither-off": False, "dither-on": True}
FILLER_DICTIONARY = "<sil> SIL\n<s> SIL\n</s> SIL\n"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_sha256(
    root: Path, *, names: set[str] | None = None, suffixes: set[str] | None = None
) -> str:
    """Hash relative names and contents of every regular file below a directory."""
    digest = hashlib.sha256()
    paths = (
        item
        for item in root.rglob("*")
        if item.is_file()
        and (names is None or item.name in names)
        and (suffixes is None or item.suffix in suffixes)
    )
    for path in sorted(paths):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def pinned_profile(*, dither: bool, seed: int) -> dict[str, Any]:
    """Materialize a complete profile with only the adjudicated axis changed."""
    profile = Profile().model_dump(mode="json")
    profile["features"]["dither"] = dither
    profile["features"]["seed"] = seed
    profile["split"]["test_count"] = 0
    return profile


def prepare_project(project: Path, corpus: Path, data: Path, *, dither: bool, seed: int) -> None:
    """Install canonical TIMIT inputs as one isolated training project."""
    config_etc = project / "etc"
    etc = project / "experiments" / "default" / "etc"
    shared = project / "shared"
    config_etc.mkdir(parents=True, exist_ok=True)
    etc.mkdir(parents=True, exist_ok=True)
    shared.mkdir(parents=True, exist_ok=True)
    train = data / "train.transcription"
    test = data / "test.transcription"
    (etc / "all.transcription").write_text(
        train.read_text(encoding="utf-8") + test.read_text(encoding="utf-8"), encoding="utf-8"
    )
    for name in SPLIT_FILENAMES:
        shutil.copyfile(data / name, etc / name)
        if (etc / name).read_bytes() != (data / name).read_bytes():
            raise RuntimeError(f"canonical split manifest failed byte verification: {name}")
    shutil.copyfile(data / "timit.reduced.dict", shared / "dictionary.dict")
    phones = {
        phone
        for line in (shared / "dictionary.dict").read_text(encoding="utf-8").splitlines()
        for phone in line.split()[1:]
    }
    (shared / "phoneset.txt").write_text(
        "\n".join(sorted(phones | {"SIL"})) + "\n", encoding="utf-8"
    )
    (shared / "filler.dict").write_text(FILLER_DICTIONARY, encoding="utf-8")
    (config_etc / "configs.yaml").write_text(
        yaml.safe_dump(
            {
                "config_version": 1,
                "profiles": {"timit": pinned_profile(dither=dither, seed=seed)},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    audio = project / "audio"
    if not audio.exists():
        audio.symlink_to(corpus, target_is_directory=True)


def _validated_split_counts(project: Path) -> dict[str, int]:
    """Read counts emitted by the pipeline's successful split validation."""
    marker = project / "experiments" / "default" / "etc" / VALIDATED_SPLIT
    validation = json.loads(marker.read_text(encoding="utf-8"))
    if validation.get("mode") != "external":
        raise RuntimeError(f"pipeline did not validate an external split: {marker}")
    try:
        return {
            "train_fileids": int(validation["n_train"]),
            "test_fileids": int(validation["n_test"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"pipeline validation has no usable split counts: {marker}") from exc


def run(corpus: Path, work_dir: Path, *, jobs: int, seed: int, target: str) -> dict[str, Any]:
    """Prepare canonical inputs, train both cells, and record artifact identities."""
    data = work_dir / "canonical-data"
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "setup_timit.py"), str(corpus), str(data)],
        check=True,
    )
    preparation = json.loads((data / "preparation.json").read_text(encoding="utf-8"))
    results: dict[str, Any] = {}
    validated_counts: dict[str, int] | None = None
    for cell, dither in CELLS.items():
        project = work_dir / cell
        prepare_project(project, corpus, data, dither=dither, seed=seed)
        split_dir = project / "experiments" / "default" / "etc"
        if not split_is_external(split_dir):
            raise RuntimeError(f"canonical external split was not detected for {cell}: {split_dir}")
        command = [
            sys.executable,
            "-m",
            "pstrain.cli.cli",
            "build",
            target,
            "--project-dir",
            str(project),
            "--config",
            "timit",
            "--force",
            "--jobs",
            str(jobs),
            "--resolved-config-output",
            str(project / "resolved-config.json"),
        ]
        with (project / "training.log").open("w", encoding="utf-8") as log:
            env = os.environ.copy()
            env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(ROOT), env.get("PYTHONPATH"))))
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, env=env)
        cell_counts = _validated_split_counts(project)
        if validated_counts is not None and cell_counts != validated_counts:
            raise RuntimeError(
                f"pipeline validated different split counts across cells: "
                f"{validated_counts} != {cell_counts}"
            )
        validated_counts = cell_counts
        features = project / "shared" / "features"
        model = project / "shared" / "models" / target / "timit"
        results[cell] = {
            "dither": dither,
            "seed": seed,
            "feature_tree_sha256": tree_sha256(features, suffixes={".mfc"}),
            "model_tree_sha256": tree_sha256(
                model,
                names={"means", "mdef", "mixture_weights", "transition_matrices", "variances"},
            ),
            "resolved_config_sha256": sha256(project / "resolved-config.json"),
        }
    output = {
        "schema_version": 1,
        "design": "paired canonical-split training; dither is the only varying profile field",
        "target": target,
        "jobs": jobs,
        "canonical_split": {
            **preparation,
            **(validated_counts or {}),
        },
        "results": results,
        "adjudication": {
            "features_changed": results["dither-off"]["feature_tree_sha256"]
            != results["dither-on"]["feature_tree_sha256"],
            "model_changed": results["dither-off"]["model_tree_sha256"]
            != results["dither-on"]["model_tree_sha256"],
        },
    }
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "results.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path, help="TIMIT root containing train/, test/, and doc/")
    parser.add_argument("--work-dir", type=Path, default=Path(".pstrain-benchmark/timit"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("-j", "--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = parser.parse_args(argv)
    try:
        result = run(
            args.corpus.resolve(),
            args.work_dir.resolve(),
            jobs=args.jobs,
            seed=args.seed,
            target=args.target,
        )
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"TIMIT dither adjudication failed: {exc}\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
