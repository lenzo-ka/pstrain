"""Turnkey CMU Arctic parity benchmark harness."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.error
import urllib.request
from contextlib import suppress
from dataclasses import asdict, dataclass, fields
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, TypeVar

import yaml

ROOT = Path(__file__).resolve().parents[2]


def resolve_data_dir(*, package_root: Path | None = None, repo_root: Path | None = None) -> Path:
    """Resolve Arctic data from an installed wheel or a source checkout."""
    installed = (package_root or ROOT) / "benchmarks" / "arctic" / "data"
    checkout = (repo_root or Path.cwd()) / "benchmarks" / "arctic" / "data"
    for candidate in (installed, checkout):
        if (candidate / "pin-train.transcription").is_file():
            return candidate
    raise RuntimeError(
        "CMU Arctic benchmark data is unavailable; checked "
        f"{installed} and repository-relative {checkout}"
    )


DATA_DIR = resolve_data_dir()
RECORD_SCHEMA_VERSION = 3
BOOTSTRAP_RESAMPLES = 100_000
BOOTSTRAP_SEED = 7
DECODER_CONDITIONS: dict[str, Any] = {
    "beam": 1e-80,
    "wbeam": 1e-40,
    "pl_window": 5,
    "lw": 10.0,
    "wip": 0.2,
    "pbeam": 1e-80,
    "lpbeam": 1e-80,
    "lponlybeam": 1e-80,
    "fwdflatbeam": 1e-80,
    "fwdflatwbeam": 1e-40,
}
T = TypeVar("T")
PINNED_RESOURCE_HASHES = {
    "lm_sha256": "2cf11ab0474a0bdd165cbee59db674b05764fdb00bf6f9824c0dccce571637b5",
    "dictionary_sha256": "24ff2852a707b63f499fd968294d5e4c02d44e0eb1ec511e40be1f380d785846",
    "filler_dictionary_sha256": "fb50883998c41a5030c2a602965935c647563321e84a86f2adabb377ec24b49c",
}
FILLER_DICTIONARY = "<sil> SIL\n<s> SIL\n</s> SIL\n"
KNOWN_SKIPS: list[dict[str, Any]] = [
    {
        "mode": "off",
        "stage": "cd-2g",
        "pass": 1,
        "utterance": "arctic_a0587",
        "mechanism": (
            "beam failure on a hard utterance after permitted retry; mirrored upstream: "
            "the preserved upstream build ignores the same utterance at CI passes 5-6"
        ),
        "recorded-in": "thresh01-off anchor",
    },
    {
        "mode": "on",
        "stage": "cd-untied",
        "passes": list(range(3, 11)),
        "utterance": "arctic_a0302",
        "mechanism": (
            "beam failure on a known-hard utterance after permitted retry in the multipron "
            "posture — the recorded on-mode remainder class (V7-era: a0302/b0486; set shifted "
            "with the reachable inventory: b0320 now trains); deep diagnosis deferred per "
            "banked Q6, tracked as an open item"
        ),
        "recorded-in": "on-mode parity anchor",
    },
    {
        "mode": "on",
        "stage": "cd-1g",
        "pass": 6,
        "utterance": "arctic_a0587",
        "mechanism": (
            "beam failure on a known-hard utterance after permitted retry in the multipron "
            "posture — the recorded on-mode remainder class (V7-era: a0302/b0486; set shifted "
            "with the reachable inventory: b0320 now trains); deep diagnosis deferred per "
            "banked Q6, tracked as an open item"
        ),
        "recorded-in": "on-mode parity anchor",
    },
]
PIPELINE_STAGE_ORDER = (
    "flat",
    "ci-1g",
    "cd-untied-init",
    "cd-untied",
    "trees",
    "prune-trees",
    "cd-1g-init",
    "cd-1g",
    "cd-2g",
    "cd-4g",
    "cd-8g",
    "cd-16g",
    "cd-32g",
)
_PIPELINE_STAGE_RANK = {stage: rank for rank, stage in enumerate(PIPELINE_STAGE_ORDER)}


def _canonical_known_skips(skips: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return known skips in stable pipeline, utterance, and pass order."""

    def key(item: dict[str, Any]) -> tuple[int, str, int]:
        passes = item.get("passes", [])
        pass_number = item.get("pass", min(passes) if passes else -1)
        return (
            _PIPELINE_STAGE_RANK.get(str(item["stage"]), len(PIPELINE_STAGE_ORDER)),
            str(item["utterance"]),
            int(pass_number),
        )

    return sorted(skips, key=key)


@dataclass(frozen=True)
class Archive:
    """One immutable corpus download."""

    voice: str
    url: str
    sha256: str
    expected_wavs: int


# WAV counts were measured from the archives authenticated by these SHA-256 pins.
ARCHIVES = (
    Archive(
        "slt",
        "http://festvox.org/cmu_arctic/packed/cmu_us_slt_arctic.tar.bz2",
        "7c173297916acf3cc7fcab2713be4c60b27312316765a90934651d367226b4ea",
        1132,
    ),
    Archive(
        "bdl",
        "http://festvox.org/cmu_arctic/packed/cmu_us_bdl_arctic.tar.bz2",
        "26b91aaf48b2799b2956792b4632c2f926cd0542f402b5452d5adecb60942904",
        1132,
    ),
    Archive(
        "rms",
        "http://festvox.org/cmu_arctic/packed/cmu_us_rms_arctic.tar.bz2",
        "c6dc11235629c58441c071a7ba8a2d067903dfefbaabc4056d87da35b72ecda4",
        1132,
    ),
    Archive(
        "clb",
        "http://festvox.org/cmu_arctic/packed/cmu_us_clb_arctic.tar.bz2",
        "3f16dc3f3b97955ea22623efb33b444341013fc660677b2e170efdcc959fa7c6",
        1132,
    ),
)

PIN_CONFIGS: dict[str, dict[str, Any]] = {
    "off": {
        "description": "BM1 multipron off pin",
        "features": {
            "samprate": 16000,
            "ncep": 13,
            "nfilt": 25,
            "nfft": 512,
            "lowerf": 130,
            "upperf": 6800,
            "alpha": 0.97,
            "lifter": 22,
            "transform": "dct",
            "agc": "none",
            "cmn": "batch",
            "varnorm": "no",
            "feat_type": "1s_c_d_dd",
        },
        "training": {
            "n_state": 3,
            "n_senones": 200,
            "a_beam": 1e-90,
            "b_beam": 1e-10,
            "max_skip_fraction": 0.05,
            "retry_beam_factor": 1e10,
            "tree_state_weights": [1.0, 0.05, 0.0],
            # Historical pin predates S1/S2; keep its recorded baseline explicit.
            "tree_rotate_state_weights": False,
            "tree_directional_questions": False,
            "tree_ssplitmax": 7,
            "tree_ssplitthr": 0.0,
            "tree_csplitmax": 2000,
            "tree_csplitthr": 0.0,
            "tree_mwfloor": 1e-8,
            "tree_intermediate_dumps": False,
            "question_npermute": 12,
            "question_quests_per_state": 20,
            "question_niter": 1,
            "multipron_training": False,
            "untied_inventory": "linear",
            "ci": {"max_iterations": 10, "min_iterations": 1, "convergence_ratio": 0.1},
            "untied": {"max_iterations": 6, "min_iterations": 1, "convergence_ratio": 0.1},
            "tied": {"max_iterations": 10, "min_iterations": 1, "convergence_ratio": 0.1},
            "exclusion_schedule": {},
        },
        "split": {"test_count": 0, "seed": 42},
    },
    "on": {
        "description": "BM1 multipron on pin",
        "features": {
            "samprate": 16000,
            "ncep": 13,
            "nfilt": 25,
            "nfft": 512,
            "lowerf": 130,
            "upperf": 6800,
            "alpha": 0.97,
            "lifter": 22,
            "transform": "dct",
            "agc": "none",
            "cmn": "batch",
            "varnorm": "no",
            "feat_type": "1s_c_d_dd",
        },
        "training": {
            "n_state": 3,
            "n_senones": 200,
            "a_beam": 1e-90,
            "b_beam": 1e-10,
            "max_skip_fraction": 0.05,
            "retry_beam_factor": 1e10,
            "tree_state_weights": [1.0, 0.05, 0.0],
            # Historical pin predates S1/S2; keep its recorded baseline explicit.
            "tree_rotate_state_weights": False,
            "tree_directional_questions": False,
            "tree_ssplitmax": 7,
            "tree_ssplitthr": 0.0,
            "tree_csplitmax": 2000,
            "tree_csplitthr": 0.0,
            "tree_mwfloor": 1e-8,
            "tree_intermediate_dumps": False,
            "question_npermute": 12,
            "question_quests_per_state": 20,
            "question_niter": 1,
            "multipron_training": True,
            "untied_inventory": "transcript-reachable",
            "exclusion_schedule": {},
            "ci": {"max_iterations": 10, "min_iterations": 1, "convergence_ratio": 0.001},
            "untied": {"max_iterations": 10, "min_iterations": 1, "convergence_ratio": 0.001},
            "tied": {"max_iterations": 10, "min_iterations": 1, "convergence_ratio": 0.001},
        },
        "split": {"test_count": 0, "seed": 42},
    },
}


def sha256(path: Path) -> str:
    """Hash a file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_trusted_child(
    command: list[str],
    *,
    stdout: Any = None,
    stderr: Any = None,
    env: dict[str, str] | None = None,
) -> None:
    """Run a trusted benchmark child using the posix_spawn-eligible shape."""
    subprocess.run(
        command,
        check=True,
        close_fds=False,
        stdout=stdout,
        stderr=stderr,
        env=env,
    )


def _tracked_modifications_hash() -> str:
    """Hash paths, states, and contents of tracked worktree modifications."""
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=no"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    digest = hashlib.sha256()
    for line in sorted(status):
        path_text = line[3:]
        if " -> " in path_text:
            path_text = path_text.split(" -> ", 1)[1]
        path = ROOT / path_text
        digest.update(line[:2].encode())
        digest.update(path_text.encode())
        digest.update(b"\0")
        if path.is_file():
            digest.update(path.read_bytes())
        else:
            digest.update(b"<deleted>")
    return digest.hexdigest()


def engine_identity(dictionary: Path | None = None) -> dict[str, str]:
    """Identify all executable and package inputs used for a benchmark run."""
    from pstrain import __version__
    from pstrain.lib.paths import get_lib_path

    identity = {"version": __version__, "python_version": sys.version}
    if dictionary is None:
        dictionary = pocketsphinx_dictionary()
    identity["decode_dictionary_sha256"] = sha256(dictionary)
    try:
        identity["pocketsphinx_version"] = version("pocketsphinx")
    except PackageNotFoundError as exc:
        raise RuntimeError("pocketsphinx package metadata is unavailable") from exc
    native = get_lib_path()
    identity["native_library_sha256"] = sha256(native) if native is not None else "absent"
    try:
        describe = subprocess.run(
            ["git", "describe", "--always", "--dirty"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        digest = hashlib.sha256()
        package = Path(__file__).resolve().parents[1]
        for path in sorted(package.rglob("*.py")):
            digest.update(path.relative_to(package).as_posix().encode())
            digest.update(path.read_bytes())
        identity["installed_package_sha256"] = digest.hexdigest()
    else:
        identity["git_describe"] = describe
        identity["tracked_modifications_sha256"] = _tracked_modifications_hash()
    return identity


def benchmark_conditions(band: str = "pin") -> dict[str, Any]:
    """Return every pinned benchmark condition that comparison authenticates."""
    from pstrain.lib.pipeline.context import FeatParams, TrainParams

    # Description and split policy are metadata outside these dataclasses. All
    # dataclass knobs are frozen; schedules are represented by their nested maps.
    frozen_fields = {
        "features": [item.name for item in fields(FeatParams)],
        "training": [item.name for item in fields(TrainParams)],
        "exemptions": {},
    }
    for mode, config in PIN_CONFIGS.items():
        for section in ("features", "training"):
            missing = set(frozen_fields[section]) - set(config[section])
            if missing:
                raise RuntimeError(f"PIN config {mode}/{section} leaves knobs unfrozen: {missing}")
    return {
        "band": "BM1" if band == "pin" else "BM1-pip-en-us-alternative",
        "pin_conditions": PIN_CONFIGS,
        "decoder": DECODER_CONDITIONS,
        "bootstrap": {
            "method": "matched-pair percentile",
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "big_speaker_stratified": True,
        },
        "cells": {"slt55": "same-speaker held-out cell", "big": "cross-speaker"},
        "known_skips": KNOWN_SKIPS,
        "frozen_dataclass_fields": frozen_fields,
    }


def _wav_manifest(destination: Path) -> list[dict[str, Any]]:
    wav_root = destination / "wav"
    return [
        {
            "path": path.relative_to(destination).as_posix(),
            "size": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(wav_root.glob("*.wav"))
    ]


def _cached_extraction_valid(
    destination: Path, archive: Archive, marker_data: Any, *, deep_verify: bool
) -> bool:
    if (
        not isinstance(marker_data, dict)
        or marker_data.get("source_archive_sha256") != archive.sha256
    ):
        return False
    recorded = marker_data.get("wav_manifest")
    if not isinstance(recorded, list) or len(recorded) != archive.expected_wavs:
        return False
    if any(
        not isinstance(row, dict)
        or not isinstance(row.get("path"), str)
        or not str(row["path"]).startswith("wav/")
        or not isinstance(row.get("size"), int)
        or not isinstance(row.get("sha256"), str)
        or len(row["sha256"]) != 64
        for row in recorded
    ):
        return False
    wav_root = destination / "wav"
    current = {
        path.relative_to(destination).as_posix(): path.stat().st_size
        for path in wav_root.glob("*.wav")
    }
    expected = {row.get("path"): row.get("size") for row in recorded if isinstance(row, dict)}
    if current != expected:
        return False
    check_rows = recorded
    if not deep_verify:
        check_rows = sorted(
            recorded,
            key=lambda row: hashlib.sha256(str(row["path"]).encode()).digest(),
        )[: min(32, len(recorded))]
    return all(sha256(destination / row["path"]) == row.get("sha256") for row in check_rows)


def extract_archive(
    archive_path: Path, archive: Archive, corpus: Path, *, deep_verify: bool = False
) -> Path:
    """Extract an authenticated archive, recovering incomplete cached extraction."""
    destination = corpus / f"cmu_us_{archive.voice}_arctic"
    marker = destination / ".pstrain-extraction.json"
    valid = False
    if marker.is_file():
        with suppress(OSError, json.JSONDecodeError):
            marker_data = json.loads(marker.read_text(encoding="utf-8"))
            valid = _cached_extraction_valid(
                destination, archive, marker_data, deep_verify=deep_verify
            )
    if not valid:
        if destination.exists():
            shutil.rmtree(destination)
        with tarfile.open(archive_path, "r:bz2") as tar:
            tar.extractall(corpus, filter="data")
        count = len(list((destination / "wav").glob("*.wav")))
        if count != archive.expected_wavs:
            shutil.rmtree(destination)
            raise RuntimeError(
                f"WAV inventory mismatch for {archive.voice}: "
                f"expected {archive.expected_wavs}, got {count}"
            )
        marker_data = {
            "source_archive_sha256": archive.sha256,
            "wav_manifest": _wav_manifest(destination),
        }
        marker.write_text(json.dumps(marker_data, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def bootstrap_ci(
    pairs: dict[str, dict[str, Any]],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> list[float]:
    """Return a 95% matched-pair WER percentile interval, in percentage points."""
    import numpy as np

    usable = {key: value for key, value in pairs.items() if "errors" in value}
    if not usable:
        raise RuntimeError("bootstrap requires at least one decoded matched pair")
    strata: dict[str, list[dict[str, Any]]] = {}
    for utterance, value in usable.items():
        speaker = utterance.split("/", 1)[0] if "/" in utterance else "slt"
        strata.setdefault(speaker, []).append(value)
    rng = np.random.default_rng(seed)
    samples = np.empty(resamples, dtype=np.float64)
    for start in range(0, resamples, 1000):
        size = min(1000, resamples - start)
        errors = np.zeros(size, dtype=np.int64)
        words = np.zeros(size, dtype=np.int64)
        for values in strata.values():
            err = np.asarray([value["errors"] for value in values], dtype=np.int64)
            ref = np.asarray([value["ref_words"] for value in values], dtype=np.int64)
            choices = rng.integers(0, len(values), size=(size, len(values)))
            errors += err[choices].sum(axis=1)
            words += ref[choices].sum(axis=1)
        samples[start : start + size] = 100.0 * errors / words
    low, high = np.percentile(samples, [2.5, 97.5])
    return [float(low), float(high)]


def paired_delta_ci(
    recorded_rows: list[list[Any]],
    actual_pairs: dict[str, dict[str, Any]],
    *,
    speaker_stratified: bool,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> list[float]:
    """Bootstrap current-minus-recorded WER from aligned utterance rows."""
    import numpy as np

    recorded = {str(row[0]): (int(row[1]), int(row[2])) for row in recorded_rows}
    actual = {
        utterance: (int(value["ref_words"]), int(value["errors"]))
        for utterance, value in actual_pairs.items()
    }
    if set(recorded) != set(actual):
        missing = sorted(set(recorded) - set(actual))
        extra = sorted(set(actual) - set(recorded))
        raise RuntimeError(f"matched-pair utterance ID mismatch: missing={missing}, extra={extra}")
    for utterance in recorded:
        if recorded[utterance][0] != actual[utterance][0]:
            raise RuntimeError(f"matched-pair reference word mismatch for {utterance}")
    strata: dict[str, list[str]] = {}
    for utterance in sorted(recorded):
        speaker = utterance.split("/", 1)[0] if speaker_stratified else "all"
        strata.setdefault(speaker, []).append(utterance)
    rng = np.random.default_rng(seed)
    samples = np.empty(resamples, dtype=np.float64)
    for start in range(0, resamples, 1000):
        size = min(1000, resamples - start)
        old_errors = np.zeros(size, dtype=np.int64)
        new_errors = np.zeros(size, dtype=np.int64)
        words = np.zeros(size, dtype=np.int64)
        for utterances in strata.values():
            old = np.asarray([recorded[key][1] for key in utterances], dtype=np.int64)
            new = np.asarray([actual[key][1] for key in utterances], dtype=np.int64)
            refs = np.asarray([recorded[key][0] for key in utterances], dtype=np.int64)
            choices = rng.integers(0, len(utterances), size=(size, len(utterances)))
            old_errors += old[choices].sum(axis=1)
            new_errors += new[choices].sum(axis=1)
            words += refs[choices].sum(axis=1)
        samples[start : start + size] = 100.0 * (new_errors - old_errors) / words
    low, high = np.percentile(samples, [2.5, 97.5])
    return [float(low), float(high)]


def fetch_archive(archive: Archive, cache: Path) -> Path:
    """Fetch an archive without performing network operations in this process."""
    destination = cache / Path(archive.url).name
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "_fetch-archive",
        json.dumps(asdict(archive), separators=(",", ":")),
        str(cache.resolve()),
    ]
    # pp8-segv-report-2026-08-11.md diagnosed a macOS 27 Network.framework atfork
    # crash: after any in-process HTTP connection, a raw fork child can segfault
    # before exec. Keep the harness network-clean and this helper posix_spawn-eligible.
    _run_trusted_child(command)
    return destination


def _fetch_archive_in_helper(archive: Archive, cache: Path) -> Path:
    """HEAD-check, download, and authenticate an archive in the network helper."""
    cache.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(archive.url, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            if response.status != 200:
                raise RuntimeError(f"HEAD {archive.url}: HTTP {response.status}")
    except (OSError, urllib.error.URLError) as exc:
        raise RuntimeError(f"HEAD failed for {archive.url}: {exc}") from exc
    destination = cache / Path(archive.url).name
    if not destination.exists():
        temporary = destination.with_suffix(destination.suffix + ".part")
        try:
            with (
                urllib.request.urlopen(archive.url, timeout=60) as response,
                temporary.open("wb") as output,
            ):
                shutil.copyfileobj(response, output)
            temporary.replace(destination)
        except (OSError, urllib.error.URLError) as exc:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"download failed for {archive.url}: {exc}") from exc
    actual = sha256(destination)
    if actual != archive.sha256:
        raise RuntimeError(
            f"SHA-256 mismatch for {destination}: expected {archive.sha256}, got {actual}"
        )
    return destination


def load_transcripts(path: Path) -> dict[str, str]:
    """Read normalized ``fileid text`` or Sphinx transcript lines."""
    result: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("<s>") and "(" in line:
            start = line.rfind("(")
            result[line[start + 1 : -1]] = line[3:start].removesuffix("</s> ").strip()
        else:
            utterance, text = line.split(maxsplit=1)
            result[utterance] = text
    return result


def training_corpus_identity() -> dict[str, Any]:
    """Return and validate the immutable pin-training corpus identity."""
    transcription = DATA_DIR / "pin-train.transcription"
    fileids_path = DATA_DIR / "pin-train.fileids"
    transcripts = load_transcripts(transcription)
    fileids = [line.strip() for line in fileids_path.read_text().splitlines() if line.strip()]
    if fileids != list(transcripts):
        raise RuntimeError("pin training transcript and fileids differ")
    return {
        "utterances": len(fileids),
        "transcription": {"name": transcription.name, "sha256": sha256(transcription)},
        "fileids": {"name": fileids_path.name, "sha256": sha256(fileids_path)},
    }


def pocketsphinx_dictionary() -> Path:
    """Resolve the dictionary shipped by the required pip package."""
    try:
        from pocketsphinx import get_model_path
    except ImportError as exc:
        raise RuntimeError(
            "BM1 requires the pip requirement 'pocketsphinx' (install pstrain[test])"
        ) from exc
    path = Path(get_model_path()) / "en-us" / "cmudict-en-us.dict"
    if not path.is_file():
        raise RuntimeError(f"pocketsphinx pip package dictionary is unavailable: {path}")
    return path


def band_resources(band: str) -> tuple[Path, Path]:
    """Return the decode dictionary and canonical LM for a named band."""
    if band == "pin":
        dictionary = DATA_DIR / "cmu_arctic_slt.dict"
    elif band == "pip-en-us":
        dictionary = pocketsphinx_dictionary()
    else:
        raise ValueError(f"unknown benchmark band: {band}")
    lm = DATA_DIR / "training-unigram.lm"
    return dictionary, lm


def authenticate_pin_resources(dictionary: Path, lm: Path, filler: bytes) -> None:
    """Fail before training if any canonical PP3c resource has drifted."""
    actual = {
        "lm_sha256": sha256(lm),
        "dictionary_sha256": sha256(dictionary),
        "filler_dictionary_sha256": hashlib.sha256(filler).hexdigest(),
    }
    if actual != PINNED_RESOURCE_HASHES:
        raise RuntimeError(
            f"PIN resource SHA-256 mismatch: expected={PINNED_RESOURCE_HASHES}, actual={actual}"
        )


def write_project(project: Path, corpus: Path, dictionary: Path) -> None:
    """Materialize shared benchmark inputs and explicit pin configs."""
    (project / "etc").mkdir(parents=True, exist_ok=True)
    (project / "shared").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(DATA_DIR / "pin-train.transcription", project / "etc" / "all.transcription")
    shutil.copyfile(dictionary, project / "shared" / "dictionary.dict")
    phones = {
        phone.rstrip("0123456789")
        for line in dictionary.read_text(encoding="utf-8", errors="replace").splitlines()
        if line and not line.startswith(";")
        for phone in line.split()[1:]
    }
    (project / "shared" / "phoneset.txt").write_text(
        "\n".join(sorted(phones | {"SIL"})) + "\n", encoding="utf-8"
    )
    (project / "shared" / "filler.dict").write_text(FILLER_DICTIONARY, encoding="utf-8")
    (project / "etc" / "configs.yaml").write_text(
        yaml.safe_dump(PIN_CONFIGS, sort_keys=False), encoding="utf-8"
    )
    audio = project / "audio"
    if not audio.exists():
        audio.symlink_to(corpus / "cmu_us_slt_arctic" / "wav", target_is_directory=True)


def audit_monotonicity(project: Path) -> list[dict[str, Any]]:
    """Gate training telemetry and return manifest-listed terminal skips."""
    telemetry = list(project.glob("shared/models/**/bw_telemetry.json"))
    if not telemetry:
        raise RuntimeError(f"no BW telemetry found under {project}")
    failures: list[str] = []
    known_skips: list[dict[str, Any]] = []
    mode = project.name
    for path in telemetry:
        document = json.loads(path.read_text(encoding="utf-8"))
        relative = path.relative_to(project / "shared" / "models")
        stage = relative.parts[0]
        rows = document["passes"]
        for row in rows:
            delta = row.get("signed_convergence_delta")
            if delta is not None and float(delta) < 0:
                failures.append(f"{path}: pass {row.get('pass')}: {delta}")
            accounting = row.get("accounting")
            if not isinstance(accounting, dict):
                failures.append(f"{path}: pass {row.get('pass')}: missing structured accounting")
                continue
            reasons = accounting.get("skip_reasons")
            if not isinstance(reasons, dict) or sum(
                int(value) for value in reasons.values()
            ) != int(accounting["skipped_utts"]):
                failures.append(f"{path}: pass {row.get('pass')}: invalid skip accounting")
                continue
            terminal = accounting.get("terminal_skips", [])
            if not isinstance(terminal, list):
                failures.append(f"{path}: pass {row.get('pass')}: invalid terminal skip detail")
                continue
            expected_terminal = int(accounting["skipped_utts"]) - int(
                reasons["excluded_by_schedule"]
            )
            if len(terminal) != expected_terminal:
                failures.append(
                    f"{path}: pass {row.get('pass')}: terminal skip detail disagrees "
                    f"with accounting: {reasons}"
                )
                continue
            for skip in terminal:
                utterance = skip.get("utterance") if isinstance(skip, dict) else None
                reason = skip.get("reason") if isinstance(skip, dict) else None
                match = next(
                    (
                        item
                        for item in KNOWN_SKIPS
                        if item["mode"] == mode
                        and item["stage"] == stage
                        and item["utterance"] == utterance
                        and (
                            item.get("pass") == row.get("pass")
                            or row.get("pass") in item.get("passes", [])
                        )
                    ),
                    None,
                )
                if match is None or reason != "alignment_failure":
                    failures.append(
                        f"{path}: pass {row.get('pass')}: unlisted terminal skip: "
                        f"utterance={utterance!r}, reason={reason!r}"
                    )
                else:
                    if match not in known_skips:
                        known_skips.append(match)
    if failures:
        raise RuntimeError("training telemetry gate failed:\n" + "\n".join(failures))
    return _canonical_known_skips(known_skips)


def _decode_partition(
    model: Path,
    dictionary: Path,
    filler: Path,
    lm: Path,
    utterances: list[tuple[str, Path]],
) -> list[tuple[str, Any]]:
    """Decode one contiguous corpus partition with one reusable decoder."""
    from pstrain.lib.testing.decoder import Decoder

    decoder = Decoder(
        model,
        dictionary,
        filler,
        lm,
        beam=1e-80,
        wbeam=1e-40,
        pl_window=5,
        lw=10.0,
        wip=0.2,
        pbeam=1e-80,
        lpbeam=1e-80,
        lponlybeam=1e-80,
        fwdflatbeam=1e-80,
        fwdflatwbeam=1e-40,
    )
    return [(utterance, decoder.decode_file(audio_file)) for utterance, audio_file in utterances]


def _corpus_partitions(items: list[T], count: int) -> list[list[T]]:
    """Split items into the contiguous ranges used by SphinxTrain decode."""
    if count < 1:
        raise ValueError("decode partitions must be at least 1")
    return [
        items[len(items) * part // count : len(items) * (part + 1) // count]
        for part in range(count)
    ]


def score_model(
    model: Path,
    audio_roots: dict[str, Path],
    refs: dict[str, str],
    dictionary: Path,
    lm: Path,
    decode_partitions: int = 1,
) -> dict[str, Any]:
    """Decode, matched-pair score, and count reference OOVs."""
    from pstrain.lib.testing.wer import aggregate_wer, calculate_wer

    filler = model.parents[2] / "filler.dict"
    utterances = []
    for utterance in refs:
        if "/" in utterance:
            voice, local_id = utterance.split("/", 1)
        else:
            voice, local_id = "slt", utterance
        utterances.append((utterance, audio_roots[voice] / f"{local_id}.wav"))
    partitions = [part for part in _corpus_partitions(utterances, decode_partitions) if part]
    if decode_partitions == 1:
        decoded_partitions = [_decode_partition(model, dictionary, filler, lm, partitions[0])]
    else:
        context = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=len(partitions), mp_context=context
        ) as executor:
            decoded_partitions = list(
                executor.map(
                    _decode_partition,
                    [model] * len(partitions),
                    [dictionary] * len(partitions),
                    [filler] * len(partitions),
                    [lm] * len(partitions),
                    partitions,
                )
            )
    results = []
    pairs: dict[str, dict[str, Any]] = {}
    decoded = 0
    for utterance, result in (item for part in decoded_partitions for item in part):
        text = refs[utterance]
        if not result.success:
            pairs[utterance] = {"reference": text, "error": result.error}
            continue
        score = calculate_wer(text, result.hypothesis)
        results.append(score)
        decoded += 1
        pairs[utterance] = {
            "reference": text,
            "hypothesis": result.hypothesis,
            "errors": score.errors,
            "ref_words": score.ref_words,
        }

    total = aggregate_wer(results)
    vocabulary = {
        line.split()[0].lower().split("(", 1)[0]
        for line in dictionary.read_text(encoding="utf-8", errors="replace").splitlines()
        if line and not line.startswith(";")
    }
    oov = sum(word.lower() not in vocabulary for text in refs.values() for word in text.split())
    payload = total.to_dict()
    payload.update(
        {"utterances": len(refs), "decoded": decoded, "oov_tokens": oov, "matched_pairs": pairs}
    )
    if decoded != len(refs):
        raise RuntimeError(f"zero-failed-alignment gate: decoded {decoded}/{len(refs)} utterances")
    return payload


def _require_equal(label: str, actual: Any, recorded: Any) -> None:
    if actual != recorded:
        raise RuntimeError(f"benchmark {label} mismatch: recorded={recorded!r}, actual={actual!r}")


def _validate_cell(cell: Any, label: str, *, recorded: bool) -> None:
    required = {
        "wer",
        "errors",
        "ref_words",
        "utterances",
        "decoded",
        "oov_tokens",
        "known_skips",
    }
    if recorded:
        required.add("utterance_rows")
    if not isinstance(cell, dict) or not required <= set(cell):
        missing = sorted(required - set(cell) if isinstance(cell, dict) else required)
        raise RuntimeError(f"benchmark record has invalid result cell {label}: missing {missing}")
    integer_fields = ("errors", "ref_words", "utterances", "decoded", "oov_tokens")
    if any(not isinstance(cell[key], int) or cell[key] < 0 for key in integer_fields):
        raise RuntimeError(f"benchmark record has invalid counts in result cell: {label}")
    if cell["ref_words"] <= 0 or cell["decoded"] > cell["utterances"]:
        raise RuntimeError(f"benchmark record has impossible counts in result cell: {label}")
    if not isinstance(cell["known_skips"], list) or any(
        item not in KNOWN_SKIPS for item in cell["known_skips"]
    ):
        raise RuntimeError(f"benchmark record has invalid known skips in result cell: {label}")
    expected_wer = cell["errors"] / cell["ref_words"]
    actual_wer = float(cell["wer"]) / 100 if recorded else float(cell["wer"])
    if abs(actual_wer - expected_wer) > 1e-12:
        raise RuntimeError(f"benchmark WER is inconsistent with errors/words: {label}")
    if recorded:
        rows = cell["utterance_rows"]
        if (
            not isinstance(rows, list)
            or len(rows) != cell["utterances"]
            or any(
                not isinstance(row, list)
                or len(row) != 3
                or not isinstance(row[0], str)
                or not row[0]
                or not isinstance(row[1], int)
                or row[1] <= 0
                or not isinstance(row[2], int)
                or row[2] < 0
                for row in rows
            )
        ):
            raise RuntimeError(f"benchmark record has invalid utterance rows: {label}")
        ids = [row[0] for row in rows]
        if len(ids) != len(set(ids)):
            raise RuntimeError(f"benchmark record has duplicate utterance rows: {label}")
        if (
            sum(row[1] for row in rows) != cell["ref_words"]
            or sum(row[2] for row in rows) != cell["errors"]
        ):
            raise RuntimeError(f"benchmark record rows disagree with aggregates: {label}")
    else:
        pairs = cell.get("matched_pairs")
        if not isinstance(pairs, dict) or len(pairs) != cell["utterances"]:
            raise RuntimeError(f"benchmark actual result has invalid matched pairs: {label}")
        try:
            pair_words = sum(int(value["ref_words"]) for value in pairs.values())
            pair_errors = sum(int(value["errors"]) for value in pairs.values())
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"benchmark actual result has invalid matched pairs: {label}"
            ) from exc
        if pair_words != cell["ref_words"] or pair_errors != cell["errors"]:
            raise RuntimeError(f"benchmark actual pairs disagree with aggregates: {label}")


def validate_record(record: dict[str, Any]) -> None:
    """Validate the complete, versioned benchmark-record schema."""
    if record.get("schema_version") != RECORD_SCHEMA_VERSION:
        raise RuntimeError(f"unsupported benchmark record schema: {record.get('schema_version')!r}")
    for field in ("engine", "conditions", "resources", "results", "off_big_floor_plus_1"):
        if field not in record:
            raise RuntimeError(f"benchmark record missing required field: {field}")
    for field in ("engine", "conditions", "resources", "results"):
        if not isinstance(record[field], dict):
            raise RuntimeError(f"benchmark record field must be an object: {field}")
    if not isinstance(record["off_big_floor_plus_1"], bool):
        raise RuntimeError("benchmark record off_big_floor_plus_1 must be boolean")
    bootstrap = record["conditions"].get("bootstrap")
    if (
        not isinstance(bootstrap, dict)
        or not isinstance(bootstrap.get("resamples"), int)
        or bootstrap["resamples"] <= 0
        or not isinstance(bootstrap.get("seed"), int)
    ):
        raise RuntimeError("benchmark record has invalid bootstrap conditions")
    for mode in ("off", "on"):
        if not isinstance(record["results"].get(mode), dict):
            raise RuntimeError(f"benchmark record missing result mode: {mode}")
        for dataset in ("slt55", "big"):
            if dataset not in record["results"][mode]:
                raise RuntimeError(f"benchmark record missing result cell: {mode}/{dataset}")
            _validate_cell(record["results"][mode][dataset], f"{mode}/{dataset}", recorded=True)


def compare_results(
    actual: dict[str, Any], record: dict[str, Any], *, allow_engine_drift: bool = False
) -> list[dict[str, Any]]:
    """Authenticate inputs and bootstrap aligned current-minus-recorded WER."""
    validate_record(record)
    _require_equal("resources", actual.get("resources"), record.get("resources"))
    _require_equal("conditions", actual.get("conditions"), record.get("conditions"))
    if not allow_engine_drift:
        _require_equal("engine identity", actual.get("engine"), record.get("engine"))
    rows: list[dict[str, Any]] = []
    for mode in ("off", "on"):
        for dataset in ("slt55", "big"):
            actual_cell = actual["results"][mode][dataset]
            record_cell = record["results"][mode][dataset]
            _validate_cell(actual_cell, f"actual {mode}/{dataset}", recorded=False)
            for field in ("utterances", "decoded", "oov_tokens", "ref_words"):
                _require_equal(f"{mode}/{dataset} {field}", actual_cell[field], record_cell[field])
            _require_equal(
                f"{mode}/{dataset} known_skips",
                actual_cell["known_skips"],
                record_cell["known_skips"],
            )
            observed = float(actual_cell["wer"]) * 100
            expected = float(record_cell["wer"])
            delta = observed - expected
            ci = paired_delta_ci(
                record_cell["utterance_rows"],
                actual_cell["matched_pairs"],
                speaker_stratified=dataset == "big",
                resamples=int(record["conditions"]["bootstrap"]["resamples"]),
                seed=int(record["conditions"]["bootstrap"]["seed"]),
            )
            bar = (
                1.0
                if mode == "off" and dataset == "big" and record.get("off_big_floor_plus_1", False)
                else 0.0
            )
            passed = ci[1] <= bar
            rows.append(
                {
                    "mode": mode,
                    "dataset": dataset,
                    "actual": observed,
                    "recorded": expected,
                    "delta": delta,
                    "paired_delta_ci_95": ci,
                    "bar": bar,
                    "pass": passed,
                }
            )
    return rows


def make_record(output: dict[str, Any]) -> dict[str, Any]:
    """Convert a completed run payload into the stable serialized record schema."""
    record_results = {
        mode: {
            dataset: {
                **{key: value for key, value in cell.items() if key != "matched_pairs"},
                "known_skips": _canonical_known_skips(cell["known_skips"]),
                "wer": float(cell["wer"]) * 100,
                "utterance_rows": [
                    [utterance, value["ref_words"], value["errors"]]
                    for utterance, value in sorted(cell["matched_pairs"].items())
                ],
            }
            for dataset, cell in cells.items()
        }
        for mode, cells in output["results"].items()
    }
    record = {
        "schema_version": RECORD_SCHEMA_VERSION,
        "engine": output["engine"],
        "conditions": output["conditions"],
        "resources": output["resources"],
        "results": record_results,
        "off_big_floor_plus_1": True,
    }
    validate_record(record)
    return record


def run(
    work_dir: Path,
    record_path: Path | None,
    jobs: int | None,
    decode_partitions: int = 1,
    *,
    emit_record: Path | None = None,
    allow_engine_drift: bool = False,
    deep_verify: bool = False,
    band: str = "pin",
) -> dict[str, Any]:
    """Run BM1 from downloads through comparison."""
    if decode_partitions < 1:
        raise RuntimeError("decode partitions must be at least 1")
    cache = Path(
        os.environ.get("PSTRAIN_BENCH_CACHE", Path.home() / ".cache" / "pstrain" / "benchmarks")
    )
    archives = {item.voice: fetch_archive(item, cache) for item in ARCHIVES}
    corpus = work_dir / "corpora"
    corpus.mkdir(parents=True, exist_ok=True)
    for archive in ARCHIVES:
        extract_archive(archives[archive.voice], archive, corpus, deep_verify=deep_verify)
    dictionary, lm = band_resources(band)
    if band == "pin":
        authenticate_pin_resources(dictionary, lm, FILLER_DICTIONARY.encode())
    manifest: dict[str, Any] = {
        "archives": {
            item.voice: {**asdict(item), "actual_sha256": sha256(archives[item.voice])}
            for item in ARCHIVES
        },
        "transcripts": {path.name: sha256(path) for path in DATA_DIR.glob("*.transcription")},
        "training_corpus": training_corpus_identity(),
        "dictionary_sha256": sha256(dictionary),
        "lm_sha256": sha256(lm),
        "filler_dictionary_sha256": hashlib.sha256(FILLER_DICTIONARY.encode()).hexdigest(),
        "band": band,
    }
    (work_dir / "resource-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    actual: dict[str, Any] = {}
    audio_roots = {
        voice: corpus / f"cmu_us_{voice}_arctic" / "wav" for voice in ("slt", "bdl", "rms", "clb")
    }
    for mode in ("off", "on"):
        project = work_dir / mode
        write_project(project, corpus, dictionary)
        command = [
            sys.executable,
            "-m",
            "pstrain.cli.cli",
            "build",
            "cd-8g",
            "--project-dir",
            str(project),
            "--config",
            mode,
            "--force",
        ]
        if jobs is not None:
            command.extend(["--jobs", str(jobs)])
        training_log = project / "training.log"
        with training_log.open("w", encoding="utf-8") as log:
            env = os.environ.copy()
            env["PYTHONPATH"] = os.pathsep.join(
                filter(None, (str(ROOT), env.get("PYTHONPATH", "")))
            )
            _run_trusted_child(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
            )
        known_skips = audit_monotonicity(project)
        model = project / "shared" / "models" / "cd-8g" / mode
        slt = load_transcripts(DATA_DIR / "slt55.transcription")
        big = load_transcripts(DATA_DIR / "big.transcription")
        actual[mode] = {
            "slt55": score_model(model, audio_roots, slt, dictionary, lm, decode_partitions),
            "big": score_model(model, audio_roots, big, dictionary, lm, decode_partitions),
        }
        for cell in actual[mode].values():
            cell["known_skips"] = known_skips
            cell["bootstrap_ci_95"] = bootstrap_ci(cell["matched_pairs"])
    output: dict[str, Any] = {
        "schema_version": RECORD_SCHEMA_VERSION,
        "results": actual,
        "resources": manifest,
        "engine": engine_identity(dictionary),
        "conditions": benchmark_conditions(band),
        "resource_manifest": str(work_dir / "resource-manifest.json"),
    }
    if emit_record is not None:
        benchmark_record = make_record(output)
        emit_record.parent.mkdir(parents=True, exist_ok=True)
        emit_record.write_text(
            json.dumps(benchmark_record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        output["emitted_record"] = str(emit_record)
    if record_path is not None:
        record = json.loads(record_path.read_text(encoding="utf-8"))
        output["comparison"] = compare_results(
            output, record, allow_engine_drift=allow_engine_drift
        )
        if not all(row["pass"] for row in output["comparison"]):
            raise RuntimeError("benchmark comparison failed")
    (work_dir / "results.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if argv[:1] == ["_fetch-archive"]:
        if len(argv) != 3:
            raise SystemExit("_fetch-archive requires an archive JSON object and cache path")
        _fetch_archive_in_helper(Archive(**json.loads(argv[1])), Path(argv[2]))
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=Path(".pstrain-benchmark/arctic"))
    parser.add_argument("--record", type=Path, help="committed docs/benchmarks JSON record")
    parser.add_argument("--emit-record", type=Path, help="write a complete PIN benchmark record")
    parser.add_argument("--allow-engine-drift", action="store_true")
    parser.add_argument("--deep-verify", action="store_true", help="rehash every cached corpus WAV")
    parser.add_argument(
        "--band",
        choices=("pin", "pip-en-us"),
        default="pin",
        help="decode resource band (default: ratified PP3c pin)",
    )
    parser.add_argument(
        "--no-compare", action="store_true", help="run before the PIN record exists"
    )
    parser.add_argument("-j", "--jobs", type=int)
    parser.add_argument(
        "--decode-partitions",
        type=int,
        default=1,
        help="contiguous decode partitions (default: 1, preserving serial behavior)",
    )
    args = parser.parse_args(argv)
    if args.no_compare and args.record:
        parser.error("--record and --no-compare are mutually exclusive")
    if args.emit_record and args.record:
        parser.error("--record and --emit-record are mutually exclusive")
    if args.decode_partitions < 1:
        parser.error("--decode-partitions must be at least 1")
    if not args.no_compare and args.record is None and args.emit_record is None:
        parser.error("--record is required unless --no-compare or --emit-record is used")
    try:
        result = run(
            args.work_dir.resolve(),
            args.record,
            args.jobs,
            args.decode_partitions,
            emit_record=args.emit_record,
            allow_engine_drift=args.allow_engine_drift,
            deep_verify=args.deep_verify,
            band=args.band,
        )
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"BM1 failed: {exc}\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
