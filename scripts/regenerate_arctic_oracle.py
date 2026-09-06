#!/usr/bin/env python3
"""Re-decode the preserved Arctic oracle models and rebuild the oracle sidecar.

The oracle arm exists to be compared with the pstrain arm. That comparison is
only a comparison when both arms consume the same decode resources, so the live
``on`` cells are decoded again whenever the pinned dictionary or language model
changes. The retired ``off`` cells are historical measurements taken with the
earlier resources; they are copied verbatim and their producing identity is
retained in ``historical_provenance`` rather than inherited from this run.

Two claims about that re-decode are measured here rather than asserted. The
completion of the stock front-end record is decoded both ways and the arms must
agree row for row, and the preserved model is decoded again under the historical
dictionary so the difference from the historical rows is a retained result. Both
controls leave result and row digests in the sidecar, and ``--check``
recomputes them from the serialized rows.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pstrain.benchmarks.arctic import (  # noqa: E402
    ARCHIVES,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    DATA_DIR,
    FILLER_DICTIONARY,
    band_resources,
    engine_identity,
    extract_archive,
    load_transcripts,
    model_directory_identity,
    paired_delta_ci,
    score_model,
    sha256,
)

DEFAULT_RECORD = ROOT / "evidence/arctic-pin/record.json"
DEFAULT_SIDECAR = ROOT / "evidence/arctic-pin/oracle-sidecar.json"
DEFAULT_WORK_DIR = Path.home() / ".cache" / "pstrain" / "oracle"
SIDECAR_SCHEMA_VERSION = 2
LIVE_MODE = "on"
RETIRED_MODE = "off"
DATASETS = ("slt55", "big")
VOICES = ("slt", "bdl", "rms", "clb")
DICTIONARY_PATH = "benchmarks/arctic/data/cmu_arctic_slt.dict"
ENGINE_SUMMARY_FIELDS = (
    "version",
    "git_describe",
    "native_library_sha256",
    "pocketsphinx_version",
)
# PocketSphinx's own built-in front-end defaults, from
# csrc/include/sphinxbase/fe.h and csrc/include/sphinxbase/feat.h. The preserved
# stock models omit these fields, so PocketSphinx supplied exactly these values
# when the oracle was first decoded. Writing them explicitly is therefore
# expected to change nothing, and regeneration decodes the preserved model both
# ways and refuses to emit a sidecar unless the rows agree.
POCKETSPHINX_FRONT_END_DEFAULTS = {
    "-alpha": "0.97",
    "-ceplen": "13",
    "-cmninit": "40,3,-1",
    "-dither": "no",
    "-frate": "100",
    "-ncep": "13",
    "-nfft": "512",
    "-remove_dc": "no",
    "-remove_noise": "yes",
    "-round_filters": "yes",
    "-samprate": "16000",
    "-seed": "-1",
    "-unit_area": "yes",
    "-wlen": "0.025625",
}


def model_identity(source: Path) -> dict[str, Any]:
    """Hash a preserved model directory exactly as the original sidecar did."""
    files = {}
    digest = hashlib.sha256()
    for path in sorted(source.iterdir()):
        if not path.is_file():
            continue
        value = sha256(path)
        files[path.name] = {"bytes": path.stat().st_size, "sha256": value}
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(value))
    return {
        "preserved_root": str(source.parents[1]),
        "model_path": str(source),
        "model_directory_manifest_sha256": digest.hexdigest(),
        "files": files,
    }


def compact(cell: dict[str, Any]) -> dict[str, Any]:
    """Reduce a scored cell to serialized rows, as the original sidecar did."""
    result = {
        **{key: value for key, value in cell.items() if key != "matched_pairs"},
        "utterance_rows": [
            [utterance, value["ref_words"], value["errors"]]
            for utterance, value in sorted(cell["matched_pairs"].items())
        ],
    }
    result["wer"] = float(cell["wer"]) * 100
    return result


def canonical_sha256(payload: Any) -> str:
    """Digest a serialized structure the way the sidecar stores it."""
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def rows_digest(rows: Any) -> str:
    """Digest serialized utterance rows, normalizing JSON list or tuple form."""
    return canonical_sha256([[str(row[0]), int(row[1]), int(row[2])] for row in rows])


def cell_digest(cell: dict[str, Any]) -> dict[str, Any]:
    """Summarize one decoded cell by its counts and its result and row digests."""
    return {
        "errors": int(cell["errors"]),
        "ref_words": int(cell["ref_words"]),
        "results_sha256": canonical_sha256(cell),
        "utterance_rows_sha256": rows_digest(cell["utterance_rows"]),
        "wer": float(cell["wer"]),
    }


def engine_summary(engine: dict[str, Any]) -> dict[str, str]:
    return {field: str(engine[field]) for field in ENGINE_SUMMARY_FIELDS}


def historical_provenance(previous: dict[str, Any]) -> dict[str, Any]:
    """Return the retired arm's own producing identity, never the current run's."""
    retained = previous.get("historical_provenance")
    if isinstance(retained, dict):
        return retained
    return {
        "cells": [f"{RETIRED_MODE}/{dataset}" for dataset in sorted(DATASETS)],
        "conditions": previous["conditions"],
        "created_utc": previous["created_utc"],
        "decode_wall_seconds": previous["decode_wall_seconds"],
        "engine": previous["engine"],
        "note": (
            "The retired off-mode oracle cells were decoded by this run with these "
            "resources. They are preserved verbatim and are not attributable to the "
            "engine, dictionary, or language model recorded at the top level."
        ),
        "resources": previous["resources"],
    }


def historical_on_decode(previous: dict[str, Any]) -> dict[str, Any]:
    """Return the retained rows of the decode the current oracle arm replaced.

    They are carried forward once retained. On the first regeneration the input
    sidecar is still the historical artifact, so its own live rows are that
    decode. They are kept in full, not only as a digest, so the recorded
    difference stays checkable from this file alone.
    """
    control = previous.get("historical_dictionary_control")
    if isinstance(control, dict) and "historical_decode" in control:
        retained: dict[str, Any] = control["historical_decode"]
        return retained
    return {
        dataset: {
            "created_utc": previous["created_utc"],
            "errors": int(previous["results"][LIVE_MODE][dataset]["errors"]),
            "ref_words": int(previous["results"][LIVE_MODE][dataset]["ref_words"]),
            "utterance_rows": [
                [str(row[0]), int(row[1]), int(row[2])]
                for row in previous["results"][LIVE_MODE][dataset]["utterance_rows"]
            ],
            "utterance_rows_sha256": rows_digest(
                previous["results"][LIVE_MODE][dataset]["utterance_rows"]
            ),
            "wer": float(previous["results"][LIVE_MODE][dataset]["wer"]),
        }
        for dataset in DATASETS
    }


def _verify_front_end_control(sidecar: dict[str, Any]) -> None:
    """Recompute the front-end A/B digests from the rows the sidecar carries."""
    reconstruction = sidecar["front_end_reconstruction"]
    if reconstruction["added_fields"] != POCKETSPHINX_FRONT_END_DEFAULTS:
        raise SystemExit(
            "the front-end defaults this script writes are not the ones the sidecar was "
            "decoded with; re-run the regeneration"
        )
    if reconstruction["staged_model_identity"]["replaced_files"] != ["feat.params"]:
        raise SystemExit(
            "the staged oracle model replaced more than the front-end record: "
            f"{reconstruction['staged_model_identity']['replaced_files']!r}"
        )
    control = reconstruction["neutrality_control"]
    resources = sidecar["resources"]
    if control["dictionary_sha256"] != resources["dictionary_sha256"]:
        raise SystemExit(
            "front-end control was decoded with a dictionary the live cells did not use: "
            f"{control['dictionary_sha256']}"
        )
    completed = control["arms"]["completed"]
    preserved = control["arms"]["preserved"]
    for dataset in DATASETS:
        emitted = sidecar["results"][LIVE_MODE][dataset]
        recomputed = {
            "results_sha256": canonical_sha256(emitted),
            "utterance_rows_sha256": rows_digest(emitted["utterance_rows"]),
        }
        stated = {field: completed[dataset][field] for field in recomputed}
        if stated != recomputed:
            raise SystemExit(
                f"front-end control does not describe the emitted {LIVE_MODE}/{dataset} "
                f"cell; stated={stated}, actual={recomputed}"
            )
        if completed[dataset] != preserved[dataset]:
            raise SystemExit(
                f"front-end completion is not neutral for {LIVE_MODE}/{dataset}: "
                f"completed={completed[dataset]}, preserved={preserved[dataset]}"
            )
        if int(emitted["errors"]) != completed[dataset]["errors"]:
            raise SystemExit(
                f"front-end control error count disagrees with the emitted "
                f"{LIVE_MODE}/{dataset} cell"
            )
    if control["results_identical"] is not True:
        raise SystemExit("front-end control reports a non-neutral completion")


def _require_consistent_wer(summary: dict[str, Any], label: str) -> None:
    """Refuse a summary whose stated WER is not its own errors over its own words."""
    ref_words = int(summary["ref_words"])
    if ref_words <= 0:
        raise SystemExit(f"{label} states a non-positive reference-word count")
    expected = 100.0 * int(summary["errors"]) / ref_words
    if abs(float(summary["wer"]) - expected) > 1e-9:
        raise SystemExit(
            f"{label} states a WER of {summary['wer']}, but its own counts give {expected}"
        )


def _verify_historical_control(sidecar: dict[str, Any], record: dict[str, Any]) -> None:
    """Check the retained historical-dictionary control against its own claim."""
    control = sidecar["historical_dictionary_control"]
    expected = record["historical_provenance"]["resources"]["dictionary_sha256"]
    if control["dictionary_sha256"] != expected:
        raise SystemExit(
            "historical-dictionary control does not use the dictionary the retired cells "
            f"were decoded with: control={control['dictionary_sha256']}, record={expected}"
        )
    for dataset in DATASETS:
        historical = control["historical_decode"][dataset]
        recomputed = rows_digest(historical["utterance_rows"])
        if historical["utterance_rows_sha256"] != recomputed:
            raise SystemExit(
                f"retained historical {LIVE_MODE}/{dataset} rows do not hash to the digest "
                f"the control states: stated={historical['utterance_rows_sha256']}, "
                f"actual={recomputed}"
            )
        counted = sum(int(row[2]) for row in historical["utterance_rows"])
        if counted != int(historical["errors"]):
            raise SystemExit(
                f"retained historical {LIVE_MODE}/{dataset} rows sum to {counted} errors, "
                f"not the {historical['errors']} the control states"
            )
        spoken = sum(int(row[1]) for row in historical["utterance_rows"])
        if spoken != int(historical["ref_words"]):
            raise SystemExit(
                f"retained historical {LIVE_MODE}/{dataset} rows span {spoken} reference "
                f"words, not the {historical['ref_words']} the control states"
            )
        _require_consistent_wer(historical, f"retained historical {LIVE_MODE}/{dataset}")
    reproduced = all(
        control["current_engine"][dataset]["utterance_rows_sha256"]
        == control["historical_decode"][dataset]["utterance_rows_sha256"]
        for dataset in DATASETS
    )
    if control["rows_reproduce_historical"] is not reproduced:
        raise SystemExit(
            "historical-dictionary control misstates whether it reproduced the historical "
            f"rows: stated={control['rows_reproduce_historical']}, actual={reproduced}"
        )
    moved = any(
        control["current_engine"][dataset]["utterance_rows_sha256"]
        != rows_digest(sidecar["results"][LIVE_MODE][dataset]["utterance_rows"])
        for dataset in DATASETS
    )
    if control["dictionary_change_moved_rows"] is not moved:
        raise SystemExit(
            "historical-dictionary control misstates whether the dictionary change moved "
            f"rows: stated={control['dictionary_change_moved_rows']}, actual={moved}"
        )
    for dataset in DATASETS:
        current = control["current_engine"][dataset]
        if moved:
            # The rows are the control's own and are not carried here, so only the
            # summary's internal arithmetic can be checked against itself.
            _require_consistent_wer(current, f"current-engine {LIVE_MODE}/{dataset}")
            continue
        # The control decoded to the same rows as the live cells, so every field a
        # reader would quote is derivable from the emitted cell. Deriving it means
        # an edited count or result digest cannot pass unnoticed.
        expected = cell_digest(sidecar["results"][LIVE_MODE][dataset])
        if current != expected:
            raise SystemExit(
                f"current-engine {LIVE_MODE}/{dataset} summary disagrees with the emitted "
                f"cell it decoded to: stated={current}, actual={expected}"
            )


def verify_sidecar(sidecar: dict[str, Any], record: dict[str, Any]) -> None:
    """Refuse a sidecar that misstates its resource axis or either control."""
    if sidecar.get("schema_version") != SIDECAR_SCHEMA_VERSION:
        raise SystemExit(f"unsupported oracle sidecar schema: {sidecar.get('schema_version')!r}")
    resources = sidecar["resources"]
    expected = {
        "dictionary": resources["dictionary_sha256"] == record["resources"]["dictionary_sha256"],
        "lm": resources["lm_sha256"] == record["resources"]["lm_sha256"],
    }
    if resources["record_resources_match"] != expected:
        raise SystemExit(
            "oracle sidecar record_resources_match disagrees with the recorded digests: "
            f"stated={resources['record_resources_match']}, actual={expected}"
        )
    if not all(expected.values()):
        raise SystemExit(
            "oracle sidecar live cells were decoded with resources the record does not use"
        )
    retained = sidecar["historical_provenance"]
    if sorted(retained["cells"]) != sorted(f"{RETIRED_MODE}/{dataset}" for dataset in DATASETS):
        raise SystemExit(f"unexpected retired oracle cells: {retained['cells']!r}")
    historical_record = record["historical_provenance"]["resources"]["dictionary_sha256"]
    historical_dictionary = retained["resources"]["dictionary_sha256"]
    if (
        historical_dictionary != resources["dictionary_sha256"]
        and historical_dictionary != historical_record
    ):
        raise SystemExit(
            "retired oracle cells and retired record cells disagree on the decode dictionary: "
            f"oracle={historical_dictionary}, record={historical_record}"
        )
    _verify_front_end_control(sidecar)
    _verify_historical_control(sidecar, record)


def prepare_corpora(work_dir: Path, cache: Path) -> dict[str, Path]:
    """Extract the authenticated corpus archives already present in the cache."""
    corpus = work_dir / "corpora"
    corpus.mkdir(parents=True, exist_ok=True)
    for archive in ARCHIVES:
        if archive.voice not in VOICES:
            continue
        archive_path = cache / Path(archive.url).name
        if not archive_path.is_file():
            raise SystemExit(f"corpus archive is not cached locally: {archive_path}")
        actual = sha256(archive_path)
        if actual != archive.sha256:
            raise SystemExit(
                f"cached corpus archive SHA-256 mismatch for {archive.voice}: "
                f"expected {archive.sha256}, got {actual}"
            )
        extract_archive(archive_path, archive, corpus)
    return {voice: corpus / f"cmu_us_{voice}_arctic" / "wav" for voice in VOICES}


def completed_feat_params(source: Path) -> str:
    """Return the stock front-end record completed with PocketSphinx's defaults."""
    preserved = dict(
        line.split(maxsplit=1)
        for line in (source / "feat.params").read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    completed = {**POCKETSPHINX_FRONT_END_DEFAULTS, **preserved}
    return "".join(f"{name} {value}\n" for name, value in sorted(completed.items()))


def stage_model(work_dir: Path, source: Path, arm: str, *, complete: bool) -> Path:
    """Stage preserved model bytes into a freshly created directory.

    Staging always starts from an empty directory. Copying over a reused one
    would let an artifact from an earlier run, such as a stale ``sendump``,
    survive into the model the decoder actually consumes.
    """
    project = work_dir / "stage" / arm
    if project.exists():
        shutil.rmtree(project)
    staged = project / "shared" / "models" / source.name
    staged.mkdir(parents=True)
    for path in sorted(source.iterdir()):
        if path.is_file():
            (staged / path.name).write_bytes(path.read_bytes())
    if complete:
        (staged / "feat.params").write_text(completed_feat_params(source), encoding="utf-8")
    (project / "filler.dict").write_text(FILLER_DICTIONARY, encoding="utf-8")
    return staged


def staged_identity(staged: Path, source: Path, *, replaced: tuple[str, ...]) -> dict[str, Any]:
    """Identify the staged directory and bound its difference from the source."""
    identity = model_directory_identity(staged)
    source_files = {row["path"]: row["sha256"] for row in model_directory_identity(source)["files"]}
    staged_files = {row["path"]: row["sha256"] for row in identity["files"]}
    if set(staged_files) != set(source_files):
        raise SystemExit(
            "staged oracle model does not carry the authenticated file set: "
            f"added={sorted(set(staged_files) - set(source_files))}, "
            f"missing={sorted(set(source_files) - set(staged_files))}"
        )
    differing = sorted(
        path for path, digest in staged_files.items() if source_files[path] != digest
    )
    if differing != sorted(replaced):
        raise SystemExit(
            f"staged oracle model differs from the authenticated source in {differing!r}, "
            f"but only {sorted(replaced)!r} may be replaced"
        )
    return {**identity, "replaced_files": sorted(replaced), "staged_path": str(staged)}


@contextlib.contextmanager
def preserved_front_end_record() -> Iterator[None]:
    """Decode with the stock nine-field record the historical oracle actually used.

    The complete-model contract would refuse it. This control exists precisely
    to measure what that record decodes to, so the gate is lifted for the
    control arm only and restored immediately afterwards.
    """
    import pstrain.lib.testing.decoder as decoder_module

    original = decoder_module.require_complete_model
    decoder_module.require_complete_model = lambda model_dir: Path(model_dir) / "feat.params"
    try:
        yield
    finally:
        decoder_module.require_complete_model = original


def historical_dictionary(record: dict[str, Any], work_dir: Path, given: Path | None) -> Path:
    """Resolve and authenticate the dictionary the retired cells were decoded with."""
    expected = record["historical_provenance"]["resources"]["dictionary_sha256"]
    if given is not None:
        path = given
    else:
        commit = record["historical_provenance"]["source_record"]["commit"]
        path = work_dir / "historical.dict"
        try:
            blob = subprocess.run(
                ["git", "show", f"{commit}:{DICTIONARY_PATH}"],
                cwd=ROOT,
                check=True,
                capture_output=True,
            ).stdout
        except (OSError, subprocess.CalledProcessError) as exc:
            raise SystemExit(
                f"cannot recover the historical dictionary from {commit}:{DICTIONARY_PATH}; "
                "pass --historical-dictionary"
            ) from exc
        path.write_bytes(blob)
    actual = sha256(path)
    if actual != expected:
        raise SystemExit(
            f"historical dictionary SHA-256 mismatch: expected {expected}, got {actual}"
        )
    return path


def decode_cells(
    staged: Path,
    audio_roots: dict[str, Path],
    refs: dict[str, dict[str, str]],
    dictionary: Path,
    lm: Path,
) -> dict[str, dict[str, Any]]:
    return {
        dataset: compact(score_model(staged, audio_roots, refs[dataset], dictionary, lm))
        for dataset in DATASETS
    }


def regenerate(
    record_path: Path,
    sidecar_path: Path,
    work_dir: Path,
    cache: Path,
    given_historical_dictionary: Path | None = None,
) -> str:
    started = time.time()
    record = json.loads(record_path.read_text(encoding="utf-8"))
    previous = json.loads(sidecar_path.read_text(encoding="utf-8"))
    retained = historical_provenance(previous)
    dictionary, lm = band_resources("pin")
    preserved = {
        mode: Path(previous["preserved_models"][mode]["model_path"])
        for mode in (RETIRED_MODE, LIVE_MODE)
    }
    models = {mode: model_identity(source) for mode, source in preserved.items()}
    for mode, identity in models.items():
        recorded = previous["preserved_models"][mode]["model_directory_manifest_sha256"]
        if identity["model_directory_manifest_sha256"] != recorded:
            raise SystemExit(
                f"preserved {mode} oracle model bytes have changed: "
                f"expected {recorded}, got {identity['model_directory_manifest_sha256']}"
            )
    source = preserved[LIVE_MODE]
    audio_roots = prepare_corpora(work_dir, cache)
    refs = {
        dataset: load_transcripts(DATA_DIR / f"{dataset}.transcription") for dataset in DATASETS
    }
    engine = engine_identity(dictionary)
    earlier_dictionary = historical_dictionary(record, work_dir, given_historical_dictionary)

    completed_stage = stage_model(work_dir, source, "completed", complete=True)
    completed_identity = staged_identity(completed_stage, source, replaced=("feat.params",))
    preserved_stage = stage_model(work_dir, source, "preserved", complete=False)
    staged_identity(preserved_stage, source, replaced=())

    live = decode_cells(completed_stage, audio_roots, refs, dictionary, lm)
    with preserved_front_end_record():
        control = decode_cells(preserved_stage, audio_roots, refs, dictionary, lm)
    for dataset in DATASETS:
        if live[dataset] != control[dataset]:
            raise SystemExit(
                "completing the stock front-end record changed the decode for "
                f"{LIVE_MODE}/{dataset}; the added defaults are not the values "
                "PocketSphinx already supplied"
            )
    earlier = decode_cells(completed_stage, audio_roots, refs, earlier_dictionary, lm)

    reconstruction = {
        "added_fields": dict(sorted(POCKETSPHINX_FRONT_END_DEFAULTS.items())),
        "neutrality_control": {
            "arms": {
                "completed": {dataset: cell_digest(live[dataset]) for dataset in DATASETS},
                "preserved": {dataset: cell_digest(control[dataset]) for dataset in DATASETS},
            },
            "dictionary_sha256": sha256(dictionary),
            "engine": engine_summary(engine),
            "note": (
                "Regeneration decodes the preserved model with the original nine-field "
                "record and with the completed record and refuses to emit unless every "
                "per-utterance row agrees."
            ),
            "results_identical": True,
        },
        "preserved_feat_params_sha256": sha256(source / "feat.params"),
        "source": "PocketSphinx built-in defaults in csrc/include/sphinxbase/fe.h and feat.h",
        "staged_feat_params_sha256": sha256(completed_stage / "feat.params"),
        "staged_model_identity": completed_identity,
    }
    historical_decode = historical_on_decode(previous)
    historical_control = {
        "current_engine": {dataset: cell_digest(earlier[dataset]) for dataset in DATASETS},
        "dictionary_change_moved_rows": any(
            rows_digest(earlier[dataset]["utterance_rows"])
            != rows_digest(live[dataset]["utterance_rows"])
            for dataset in DATASETS
        ),
        "dictionary_sha256": sha256(earlier_dictionary),
        "dictionary_source": {
            "commit": record["historical_provenance"]["source_record"]["commit"],
            "path": DICTIONARY_PATH,
        },
        "engine": engine_summary(engine),
        "historical_decode": historical_decode,
        "historical_engine": engine_summary(retained["engine"]),
        "note": (
            "The preserved model decoded again under the dictionary the retired cells "
            "used, on the current engine, against the retained rows of the decode that "
            "produced them. It is the evidence for the recorded difference from those "
            "rows, which is engine sensitivity rather than a staging or resource "
            "artifact. It does not measure how far the pstrain arm moved over the same "
            "interval, and it does not attribute any earlier reported difference "
            "entirely to the engine."
        ),
        "rows_reproduce_historical": all(
            rows_digest(earlier[dataset]["utterance_rows"])
            == historical_decode[dataset]["utterance_rows_sha256"]
            for dataset in DATASETS
        ),
    }

    results = {RETIRED_MODE: previous["results"][RETIRED_MODE], LIVE_MODE: live}
    verdicts = [
        verdict for verdict in previous["pstrain_vs_oracle"] if verdict["mode"] == RETIRED_MODE
    ]
    for dataset in DATASETS:
        cell = live[dataset]
        recorded_cell = record["results"][LIVE_MODE][dataset]
        recorded_pairs = {
            str(row[0]): {"ref_words": int(row[1]), "errors": int(row[2])}
            for row in recorded_cell["utterance_rows"]
        }
        interval = paired_delta_ci(
            cell["utterance_rows"],
            recorded_pairs,
            speaker_stratified=dataset == "big",
            resamples=BOOTSTRAP_RESAMPLES,
            seed=BOOTSTRAP_SEED,
        )
        verdicts.append(
            {
                "dataset": dataset,
                "mode": LIVE_MODE,
                "oracle_wer": cell["wer"],
                "paired_ci_95_pp": interval,
                "pass": interval[1] <= 0.0,
                "pstrain_minus_oracle_pp": float(recorded_cell["wer"]) - cell["wer"],
                "pstrain_wer": float(recorded_cell["wer"]),
                "upper_bound_bar_pp": 0.0,
            }
        )
    dictionary_sha256 = sha256(dictionary)
    lm_sha256 = sha256(lm)
    sidecar = {
        "basis": {
            "live_cells": sorted(f"{LIVE_MODE}/{dataset}" for dataset in DATASETS),
            "retired_cells": sorted(f"{RETIRED_MODE}/{dataset}" for dataset in DATASETS),
        },
        "conditions": {
            "big_speaker_stratified": True,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "paired_delta_path": "pstrain.benchmarks.arctic.paired_delta_ci",
            "platform": platform.platform(),
            "score_path": "pstrain.benchmarks.arctic.score_model",
        },
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "decode_wall_seconds": time.time() - started,
        "engine": engine,
        "front_end_reconstruction": reconstruction,
        "historical_dictionary_control": historical_control,
        "historical_provenance": retained,
        "preserved_models": models,
        "pstrain_vs_oracle": verdicts,
        "resources": {
            "band": "pin",
            "dictionary_path": dictionary.relative_to(ROOT).as_posix(),
            "dictionary_sha256": dictionary_sha256,
            "lm_path": lm.relative_to(ROOT).as_posix(),
            "lm_sha256": lm_sha256,
            "record_resources_match": {
                "dictionary": dictionary_sha256 == record["resources"]["dictionary_sha256"],
                "lm": lm_sha256 == record["resources"]["lm_sha256"],
            },
        },
        "results": results,
        "schema_version": SIDECAR_SCHEMA_VERSION,
    }
    verify_sidecar(sidecar, record)
    return json.dumps(sidecar, indent=2, sort_keys=True) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, default=DEFAULT_RECORD)
    parser.add_argument("--sidecar", type=Path, default=DEFAULT_SIDECAR)
    parser.add_argument("--output", type=Path, default=DEFAULT_SIDECAR)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--historical-dictionary", type=Path, default=None)
    parser.add_argument(
        "--cache", type=Path, default=Path.home() / ".cache" / "pstrain" / "benchmarks"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="authenticate the checked-in sidecar's resource axis and controls without decoding",
    )
    args = parser.parse_args()
    if args.check:
        verify_sidecar(
            json.loads(args.sidecar.read_text(encoding="utf-8")),
            json.loads(args.record.read_text(encoding="utf-8")),
        )
        print("oracle sidecar is resource-matched to the record and its controls recompute")
        return
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        regenerate(
            args.record,
            args.sidecar,
            args.work_dir,
            args.cache,
            args.historical_dictionary,
        ),
        encoding="utf-8",
    )
    print(f"regenerated {args.output}")


if __name__ == "__main__":
    main()
