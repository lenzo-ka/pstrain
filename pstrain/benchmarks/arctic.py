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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "benchmarks" / "arctic" / "data"


@dataclass(frozen=True)
class Archive:
    """One immutable corpus download."""

    voice: str
    url: str
    sha256: str


ARCHIVES = (
    Archive(
        "slt",
        "http://festvox.org/cmu_arctic/packed/cmu_us_slt_arctic.tar.bz2",
        "7c173297916acf3cc7fcab2713be4c60b27312316765a90934651d367226b4ea",
    ),
    Archive(
        "bdl",
        "http://festvox.org/cmu_arctic/packed/cmu_us_bdl_arctic.tar.bz2",
        "26b91aaf48b2799b2956792b4632c2f926cd0542f402b5452d5adecb60942904",
    ),
    Archive(
        "rms",
        "http://festvox.org/cmu_arctic/packed/cmu_us_rms_arctic.tar.bz2",
        "c6dc11235629c58441c071a7ba8a2d067903dfefbaabc4056d87da35b72ecda4",
    ),
    Archive(
        "clb",
        "http://festvox.org/cmu_arctic/packed/cmu_us_clb_arctic.tar.bz2",
        "3f16dc3f3b97955ea22623efb33b444341013fc660677b2e170efdcc959fa7c6",
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
            if accounting is not None:
                excluded = accounting["skip_reasons"]["excluded_by_schedule"]
                unexpected = accounting["skipped_utts"] - excluded
                if unexpected:
                    failures.append(
                        f"{path}: pass {row.get('pass')}: {unexpected} failed alignment(s)"
                    )
    if failures:
        raise RuntimeError("negative inter-pass likelihood delta(s):\n" + "\n".join(failures))


def score_model(
    model: Path, audio_roots: dict[str, Path], refs: dict[str, str], dictionary: Path, lm: Path
) -> dict[str, Any]:
    """Decode, matched-pair score, and count reference OOVs."""
    from pstrain.lib.testing.decoder import Decoder
    from pstrain.lib.testing.wer import aggregate_wer, calculate_wer

    filler = model.parents[2] / "filler.dict"
    decoder = Decoder(model, dictionary, filler, lm)
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


def compare_results(actual: dict[str, Any], record: dict[str, Any]) -> list[dict[str, Any]]:
    """Compare WER cells in percentage points using record tolerances."""
    rows: list[dict[str, Any]] = []
    for mode in ("off", "on"):
        for dataset in ("slt55", "big"):
            observed = float(actual[mode][dataset]["wer"]) * 100
            expected = float(record["results"][mode][dataset]["wer"])
            tolerance = float(record["tolerances"][mode][dataset])
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


def run(work_dir: Path, record_path: Path | None, jobs: int | None) -> dict[str, Any]:
    """Run BM1 from downloads through comparison."""
    cache = Path(
        os.environ.get("PSTRAIN_BENCH_CACHE", Path.home() / ".cache" / "pstrain" / "benchmarks")
    )
    archives = {item.voice: fetch_archive(item, cache) for item in ARCHIVES}
    corpus = work_dir / "corpora"
    corpus.mkdir(parents=True, exist_ok=True)
    for archive in ARCHIVES:
        destination = corpus / f"cmu_us_{archive.voice}_arctic"
        if not destination.exists():
            with tarfile.open(archives[archive.voice], "r:bz2") as tar:
                tar.extractall(corpus, filter="data")
    dictionary = pocketsphinx_dictionary()
    from pstrain.lib.lm import build_lm

    train = load_transcripts(DATA_DIR / "train.transcription")
    lm = work_dir / "training-unigram.lm"
    build_lm(train, lm, max_order=1)
    manifest = {
        "archives": {item.voice: asdict(item) for item in ARCHIVES},
        "resources": {path.name: sha256(path) for path in DATA_DIR.glob("*.transcription")},
        "dictionary": {"path": str(dictionary), "sha256": sha256(dictionary)},
        "lm": {"path": str(lm), "sha256": sha256(lm)},
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
        failed_alignments = training_log.read_text(encoding="utf-8").count("Failed to process:")
        if failed_alignments:
            raise RuntimeError(
                f"zero-failed-alignment gate: {failed_alignments} failure(s) in {training_log}"
            )
        audit_monotonicity(project)
        model = project / "shared" / "models" / "cd-8g" / mode
        slt = load_transcripts(DATA_DIR / "slt55.transcription")
        big = load_transcripts(DATA_DIR / "big.transcription")
        actual[mode] = {
            "slt55": score_model(model, audio_roots, slt, dictionary, lm),
            "big": score_model(model, audio_roots, big, dictionary, lm),
        }
    output: dict[str, Any] = {
        "results": actual,
        "resource_manifest": str(work_dir / "resource-manifest.json"),
    }
    if record_path is not None:
        record = json.loads(record_path.read_text(encoding="utf-8"))
        output["comparison"] = compare_results(actual, record)
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
    parser.add_argument(
        "--no-compare", action="store_true", help="run before the PIN record exists"
    )
    parser.add_argument("-j", "--jobs", type=int)
    args = parser.parse_args(argv)
    if args.no_compare and args.record:
        parser.error("--record and --no-compare are mutually exclusive")
    if not args.no_compare and args.record is None:
        parser.error("--record is required unless --no-compare is used")
    try:
        result = run(args.work_dir.resolve(), args.record, args.jobs)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"BM1 failed: {exc}\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
