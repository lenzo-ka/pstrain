"""Shared machinery for the committed numerical-correctness harness."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

from pstrain.lib import _pstrainc
from pstrain.lib.bw import BWConfig
from pstrain.lib.features import read_sphinx_mfc
from pstrain.lib.pipeline import PipelineContext
from pstrain.lib.pipeline.tasks import build_pipeline
from pstrain.lib.setup import setup_project
from pstrain.lib.steps.train import TrainingResult, run_bw_training

FIXTURE = Path(__file__).parent / "fixtures" / "mini_arctic"
GOLDEN = Path(__file__).parent / "golden" / "numeric_bw.json"
GOLDEN_FILEIDS = ("arctic_a0001", "arctic_a0002", "arctic_a0003")
SEED = 42
PORTABLE_TOLERANCE = {"rtol": 1e-6, "atol": 1e-4}
STRICT_TOLERANCE = {"rtol": 1e-12, "atol": 1e-8}
FEATURE_TOLERANCE = {"rtol": 1e-4, "atol": 1e-3}


def sha256(path: Path) -> str:
    """Return a byte-level checksum."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def create_project(project_dir: Path, target: str = "flat") -> PipelineContext:
    """Create the fixed mini corpus project and build through ``target``."""
    setup_project(
        project_dir,
        transcription_path=FIXTURE / "transcription.txt",
        audio_path=FIXTURE / "wav",
        dictionary_path=FIXTURE / "dictionary.dict",
        phoneset_path=FIXTURE / "phoneset.txt",
        filler_dict_path=FIXTURE / "filler.dict",
    )
    ctx = PipelineContext.from_config(project_dir)
    ctx = replace(
        ctx,
        train=replace(ctx.train, a_beam=1e-200),
        split=replace(ctx.split, seed=SEED),
    )
    assert ctx.split.seed == SEED
    if build_pipeline(ctx).run(target, jobs=1) != 0:
        raise RuntimeError(f"numeric fixture pipeline failed at {target}")
    return ctx


def write_golden_subset(ctx: PipelineContext) -> tuple[Path, Path]:
    """Write the deterministic three-utterance BW input lists."""
    fileids = ctx.etc_dir / "numeric.fileids"
    transcription = ctx.etc_dir / "numeric.transcription"
    source = {
        line.split(maxsplit=1)[0]: line
        for line in (FIXTURE / "transcription.txt").read_text().splitlines()
    }
    fileids.write_text("".join(f"{fileid}\n" for fileid in GOLDEN_FILEIDS))
    transcription.write_text("".join(f"{source[fileid]}\n" for fileid in GOLDEN_FILEIDS))
    return fileids, transcription


def train_golden(ctx: PipelineContext, output: Path) -> TrainingResult:
    """Run the fixed three-pass CI BW trajectory."""
    fileids, transcription = write_golden_subset(ctx)
    return run_bw_training(
        model_dir=ctx.model_dir("flat"),
        output_dir=output,
        features_dir=ctx.features_dir,
        train_fileids=fileids,
        transcription=transcription,
        dictionary=ctx.shared_dir / "dictionary.dict",
        filler_dict=ctx.filler_dict,
        n_iter=3,
        min_iterations=4,
        config=BWConfig(a_beam=1e-200),
    )


def golden_payload(ctx: PipelineContext, result: TrainingResult) -> dict[str, Any]:
    """Serialize only stable numerical inputs and outputs."""
    feature = ctx.features_dir / f"{GOLDEN_FILEIDS[0]}.mfc"
    values = read_sphinx_mfc(feature)
    return {
        "schema": 2,
        "seed": SEED,
        "portable_tolerance": PORTABLE_TOLERANCE,
        "strict_tolerance": STRICT_TOLERANCE,
        "feature_tolerance": FEATURE_TOLERANCE,
        "feature": {
            "fileid": GOLDEN_FILEIDS[0],
            "frames": int(values.shape[0]),
            "values": int(values.size),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
            "mean": float(values.mean()),
            "stddev": float(values.std()),
            "l2_norm": float(np.linalg.norm(values)),
            "sha256": sha256(feature),
        },
        "trajectory": [asdict(item) for item in result.trajectory],
    }


def strict_golden_enabled() -> bool:
    """Return whether same-machine bitwise golden checks were requested."""
    return os.environ.get("PSTRAIN_GOLDEN_STRICT") == "1"


def read_model_arrays(model_dir: Path) -> dict[str, np.ndarray[Any, Any]]:
    """Load all floating-point model parameters."""
    return {
        "means": _pstrainc.read_gau(str(model_dir / "means"))[0],
        "variances": _pstrainc.read_gau(str(model_dir / "variances"))[0],
        "mixture_weights": _pstrainc.read_mixw(str(model_dir / "mixture_weights"))[0],
        "transition_matrices": _pstrainc.read_tmat(str(model_dir / "transition_matrices"))[0],
    }


def write_golden(path: Path, project_dir: Path) -> None:
    """Regenerate the checked-in same-machine numerical oracle."""
    ctx = create_project(project_dir)
    result = train_golden(ctx, project_dir / "numeric-model")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(golden_payload(ctx, result), indent=2) + "\n")
