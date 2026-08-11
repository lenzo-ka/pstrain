"""Turnkey CMU Arctic parity benchmark harness."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.error
import urllib.request
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "benchmarks" / "arctic" / "data"
RECORD_SCHEMA_VERSION = 1
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


@dataclass(frozen=True)
class Archive:
    """One immutable corpus download."""

    voice: str
    url: str
    sha256: str
    expected_wavs: int


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
        1131,
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
            "tree_ssplitmax": 7,
            "tree_ssplitthr": 0.0,
            "tree_csplitmax": 2000,
            "tree_csplitthr": 0.0,
            "tree_mwfloor": 1e-8,
            "question_npermute": 12,
            "question_quests_per_state": 20,
            "question_niter": 1,
            "multipron_training": False,
            "untied_inventory": "linear",
            "ci": {"max_iterations": 10, "min_iterations": 1, "convergence_ratio": 0.1},
            "untied": {"max_iterations": 6, "min_iterations": 1, "convergence_ratio": 0.1},
            "tied": {"max_iterations": 10, "min_iterations": 1, "convergence_ratio": 0.1},
            "exclusion_schedule": {
                "ci-1g": {5: ["arctic_a0587"], 6: ["arctic_a0587"]},
                "cd-untied": {"*": ["arctic_a0587"]},
                "cd-1g": {
                    1: ["arctic_a0587"],
                    2: ["arctic_a0587"],
                    3: ["arctic_a0587"],
                    4: ["arctic_a0448", "arctic_a0587"],
                },
                "cd-2g": {
                    1: ["arctic_a0448", "arctic_a0587"],
                    2: ["arctic_a0448", "arctic_a0587"],
                },
            },
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
            "tree_ssplitmax": 7,
            "tree_ssplitthr": 0.0,
            "tree_csplitmax": 2000,
            "tree_csplitthr": 0.0,
            "tree_mwfloor": 1e-8,
            "question_npermute": 12,
            "question_quests_per_state": 20,
            "question_niter": 1,
            "multipron_training": True,
            "untied_inventory": "transcript-reachable",
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


def engine_identity() -> dict[str, str]:
    """Identify the Python engine source used for a benchmark run."""
    from pstrain import __version__

    identity = {"version": __version__}
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        digest = hashlib.sha256()
        package = Path(__file__).resolve().parents[1]
        for path in sorted(package.rglob("*.py")):
            digest.update(path.relative_to(package).as_posix().encode())
            digest.update(path.read_bytes())
        identity["installed_package_sha256"] = digest.hexdigest()
    else:
        identity["git_commit"] = commit
    return identity


def benchmark_conditions() -> dict[str, Any]:
    """Return every pinned benchmark condition that comparison authenticates."""
    return {
        "band": "BM1",
        "pin_conditions": PIN_CONFIGS,
        "decoder": DECODER_CONDITIONS,
        "bootstrap": {
            "method": "matched-pair percentile",
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "big_speaker_stratified": True,
        },
        "cells": {"slt55": "same-speaker resubstitution cell", "big": "cross-speaker"},
    }


def extract_archive(archive_path: Path, archive: Archive, corpus: Path) -> Path:
    """Extract an authenticated archive, recovering incomplete cached extraction."""
    destination = corpus / f"cmu_us_{archive.voice}_arctic"
    marker = destination / ".pstrain-extraction.json"
    expected_marker = {"source_archive_sha256": archive.sha256}
    valid = False
    if marker.is_file():
        with suppress(OSError, json.JSONDecodeError):
            valid = json.loads(marker.read_text(encoding="utf-8")) == expected_marker
    if valid:
        valid = len(list((destination / "wav").glob("*.wav"))) == archive.expected_wavs
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
        marker.write_text(json.dumps(expected_marker, sort_keys=True) + "\n", encoding="utf-8")
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


def fetch_archive(archive: Archive, cache: Path) -> Path:
    """HEAD-check, download, and authenticate an archive."""
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


def write_project(project: Path, corpus: Path, dictionary: Path) -> None:
    """Materialize shared benchmark inputs and explicit pin configs."""
    (project / "etc").mkdir(parents=True, exist_ok=True)
    (project / "shared").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(DATA_DIR / "train.transcription", project / "etc" / "all.transcription")
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
    (project / "shared" / "filler.dict").write_text(
        "<sil> SIL\n<s> SIL\n</s> SIL\n", encoding="utf-8"
    )
    (project / "etc" / "configs.yaml").write_text(
        yaml.safe_dump(PIN_CONFIGS, sort_keys=False), encoding="utf-8"
    )
    audio = project / "audio"
    if not audio.exists():
        audio.symlink_to(corpus / "cmu_us_slt_arctic" / "wav", target_is_directory=True)


def audit_monotonicity(project: Path) -> None:
    """Fail when any training pass reports a negative likelihood delta."""
    telemetry = list(project.glob("shared/models/**/bw_telemetry.json"))
    if not telemetry:
        raise RuntimeError(f"no BW telemetry found under {project}")
    failures: list[str] = []
    for path in telemetry:
        document = json.loads(path.read_text(encoding="utf-8"))
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
            unexpected = int(accounting["skipped_utts"]) - int(reasons["excluded_by_schedule"])
            if unexpected:
                failures.append(
                    f"{path}: pass {row.get('pass')}: {unexpected} unexpected skip(s): {reasons}"
                )
    if failures:
        raise RuntimeError("training telemetry gate failed:\n" + "\n".join(failures))


def score_model(
    model: Path, audio_roots: dict[str, Path], refs: dict[str, str], dictionary: Path, lm: Path
) -> dict[str, Any]:
    """Decode, matched-pair score, and count reference OOVs."""
    from pstrain.lib.testing.decoder import Decoder
    from pstrain.lib.testing.wer import aggregate_wer, calculate_wer

    filler = model.parents[2] / "filler.dict"
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
    results = []
    pairs: dict[str, dict[str, Any]] = {}
    decoded = 0
    for utterance, text in refs.items():
        if "/" in utterance:
            voice, local_id = utterance.split("/", 1)
        else:
            voice, local_id = "slt", utterance
        result = decoder.decode_file(audio_roots[voice] / f"{local_id}.wav")
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


def validate_record(record: dict[str, Any]) -> None:
    """Validate the complete, versioned benchmark-record schema."""
    if record.get("schema_version") != RECORD_SCHEMA_VERSION:
        raise RuntimeError(f"unsupported benchmark record schema: {record.get('schema_version')!r}")
    for field in ("engine", "conditions", "resources", "results", "off_big_floor_plus_1"):
        if field not in record:
            raise RuntimeError(f"benchmark record missing required field: {field}")
    for mode in ("off", "on"):
        for dataset in ("slt55", "big"):
            cell = record["results"][mode][dataset]
            if "wer" not in cell or len(cell.get("bootstrap_ci_95", ())) != 2:
                raise RuntimeError(f"benchmark record has invalid result cell: {mode}/{dataset}")


def compare_results(
    actual: dict[str, Any], record: dict[str, Any], *, allow_engine_drift: bool = False
) -> list[dict[str, Any]]:
    """Authenticate run inputs and compare WER cells using record CIs."""
    validate_record(record)
    _require_equal("resources", actual.get("resources"), record.get("resources"))
    _require_equal("conditions", actual.get("conditions"), record.get("conditions"))
    if not allow_engine_drift:
        _require_equal("engine identity", actual.get("engine"), record.get("engine"))
    rows: list[dict[str, Any]] = []
    for mode in ("off", "on"):
        for dataset in ("slt55", "big"):
            observed = float(actual["results"][mode][dataset]["wer"]) * 100
            expected = float(record["results"][mode][dataset]["wer"])
            ci = record["results"][mode][dataset]["bootstrap_ci_95"]
            tolerance = max(abs(expected - float(ci[0])), abs(float(ci[1]) - expected))
            delta = observed - expected
            floor_clause = (
                mode == "off" and dataset == "big" and record.get("off_big_floor_plus_1", False)
            )
            passed = abs(delta) <= tolerance or (floor_clause and delta <= 1.0)
            rows.append(
                {
                    "mode": mode,
                    "dataset": dataset,
                    "actual": observed,
                    "recorded": expected,
                    "delta": delta,
                    "tolerance": tolerance,
                    "pass": passed,
                }
            )
    return rows


def run(
    work_dir: Path,
    record_path: Path | None,
    jobs: int | None,
    *,
    emit_record: Path | None = None,
    allow_engine_drift: bool = False,
) -> dict[str, Any]:
    """Run BM1 from downloads through comparison."""
    cache = Path(
        os.environ.get("PSTRAIN_BENCH_CACHE", Path.home() / ".cache" / "pstrain" / "benchmarks")
    )
    archives = {item.voice: fetch_archive(item, cache) for item in ARCHIVES}
    corpus = work_dir / "corpora"
    corpus.mkdir(parents=True, exist_ok=True)
    for archive in ARCHIVES:
        extract_archive(archives[archive.voice], archive, corpus)
    dictionary = pocketsphinx_dictionary()
    from pstrain.lib.lm import build_lm

    train = load_transcripts(DATA_DIR / "train.transcription")
    lm = work_dir / "training-unigram.lm"
    build_lm(train, lm, max_order=1)
    manifest: dict[str, Any] = {
        "archives": {
            item.voice: {**asdict(item), "actual_sha256": sha256(archives[item.voice])}
            for item in ARCHIVES
        },
        "transcripts": {path.name: sha256(path) for path in DATA_DIR.glob("*.transcription")},
        "dictionary_sha256": sha256(dictionary),
        "lm_sha256": sha256(lm),
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
            subprocess.run(command, cwd=ROOT, check=True, stdout=log, stderr=subprocess.STDOUT)
        audit_monotonicity(project)
        model = project / "shared" / "models" / "cd-8g" / mode
        slt = load_transcripts(DATA_DIR / "slt55.transcription")
        big = load_transcripts(DATA_DIR / "big.transcription")
        actual[mode] = {
            "slt55": score_model(model, audio_roots, slt, dictionary, lm),
            "big": score_model(model, audio_roots, big, dictionary, lm),
        }
        for cell in actual[mode].values():
            cell["bootstrap_ci_95"] = bootstrap_ci(cell["matched_pairs"])
    output: dict[str, Any] = {
        "schema_version": RECORD_SCHEMA_VERSION,
        "results": actual,
        "resources": manifest,
        "engine": engine_identity(),
        "conditions": benchmark_conditions(),
        "resource_manifest": str(work_dir / "resource-manifest.json"),
    }
    if emit_record is not None:
        record_results = {
            mode: {
                dataset: {
                    **{key: value for key, value in cell.items() if key != "matched_pairs"},
                    "wer": float(cell["wer"]) * 100,
                }
                for dataset, cell in cells.items()
            }
            for mode, cells in actual.items()
        }
        benchmark_record = {
            "schema_version": RECORD_SCHEMA_VERSION,
            "engine": output["engine"],
            "conditions": output["conditions"],
            "resources": manifest,
            "results": record_results,
            "off_big_floor_plus_1": True,
        }
        validate_record(benchmark_record)
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=Path(".pstrain-benchmark/arctic"))
    parser.add_argument("--record", type=Path, help="committed docs/benchmarks JSON record")
    parser.add_argument("--emit-record", type=Path, help="write a complete PIN benchmark record")
    parser.add_argument("--allow-engine-drift", action="store_true")
    parser.add_argument(
        "--no-compare", action="store_true", help="run before the PIN record exists"
    )
    parser.add_argument("-j", "--jobs", type=int)
    args = parser.parse_args(argv)
    if args.no_compare and args.record:
        parser.error("--record and --no-compare are mutually exclusive")
    if args.emit_record and args.record:
        parser.error("--record and --emit-record are mutually exclusive")
    if not args.no_compare and args.record is None and args.emit_record is None:
        parser.error("--record is required unless --no-compare or --emit-record is used")
    try:
        result = run(
            args.work_dir.resolve(),
            args.record,
            args.jobs,
            emit_record=args.emit_record,
            allow_engine_drift=args.allow_engine_drift,
        )
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"BM1 failed: {exc}\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
