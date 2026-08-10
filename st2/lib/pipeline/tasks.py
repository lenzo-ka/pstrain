"""Task definitions for the ST2 training pipeline.

This module is the single source of truth for the pipeline DAG. It mirrors
what the previous `Snakefile` did: one builder function per task, plus a
top-level `build_pipeline(ctx)` factory that registers everything for a
given `PipelineContext`.

Two kinds of tasks:

* **Linear chain** (flat -> ci-1g -> ci-2g -> ... -> cd-32g, plus trees and
  packaging): each is one `Task` whose `fn` is a closure over the context.
  These run in-process.

* **Fan-out** (feature extraction): one `Task` per fileid, all sharing
  `parallel_group="features"`. The `fn` is a `functools.partial` over a
  module-level worker so it pickles cleanly for `ProcessPoolExecutor`.

Targets exposed to `st2 build`:

    flat, ci-1g, ci-2g, ci-4g, ci-8g,
    cd-untied, cd-1g, cd-2g, cd-4g, cd-8g, cd-16g, cd-32g,
    features, lm, test-ci-8g, test-cd-8g,
    package-ci-8g, package-cd-8g, package-cd-32g
"""

from __future__ import annotations

import functools
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from st2.lib.pipeline.context import PipelineContext
from st2.lib.pipeline.feat_params import write_feat_params
from st2.lib.pipeline.runner import Pipeline, Task

# Top-level worker functions for ProcessPoolExecutor (must be picklable).


def _extract_features_worker(
    audio_path: Path,
    output_path: Path,
    params: dict[str, Any],
) -> None:
    """Module-level entry point used by the feature-extraction fan-out.

    Defined at module top level so `functools.partial(...)` instances are
    picklable for `ProcessPoolExecutor`.
    """
    from st2.lib.features import extract_features

    output_path.parent.mkdir(parents=True, exist_ok=True)
    extract_features(audio_path, output_path, **params)


def _build_tree_worker(
    untied_dir: Path,
    questions_path: Path,
    output_path: Path,
    phone: str,
    state: int,
) -> None:
    """Module-level entry point used by the tree-building fan-out."""
    from st2.lib.steps.cd_pipeline import build_tree_one

    output_path.parent.mkdir(parents=True, exist_ok=True)
    build_tree_one(
        untied_model_dir=untied_dir,
        questions_path=questions_path,
        output_path=output_path,
        phone=phone,
        state=state,
        continuous=True,
    )


# Linear-chain task callables. Each closes over `ctx` plus its specific
# parameters; the wrapping `Task.fn` takes no args.


def _make_feat_params_task(ctx: PipelineContext) -> Task:
    """Write `feat.params` after all feature files exist. By depending on
    every `.mfc`, this task acts as the sentinel for the "features"
    target: building it plans the full fan-out."""
    out = ctx.features_dir / "feat.params"

    def run() -> None:
        write_feat_params(out, ctx.feat)

    return Task(
        name="feat_params",
        fn=run,
        inputs=_all_feature_files(ctx),
        outputs=(out,),
        description="Write feature extraction parameters (after all MFCCs exist)",
    )


def _make_extract_tasks(ctx: PipelineContext) -> list[Task]:
    """One task per fileid; parallel-safe.

    Uses `audio_fileids()` (corpus-wide, recursively derived from `audio/`) so
    extraction can be planned before the train/test split has run.
    """
    params: dict[str, Any] = {
        "samprate": ctx.feat.samprate,
        "ncep": ctx.feat.ncep,
        "nfilt": ctx.feat.nfilt,
        "nfft": ctx.feat.nfft,
        "lowerf": ctx.feat.lowerf,
        "upperf": ctx.feat.upperf,
        "lifter": ctx.feat.lifter,
        "alpha": ctx.feat.alpha,
    }
    fileids = ctx.audio_fileids()
    if not fileids:
        pattern = "**/*.wav"
        raise ValueError(
            f"No audio files found in {ctx.audio_dir} matching recursive glob {pattern!r}"
        )

    out: list[Task] = []
    for fileid in fileids:
        audio = ctx.audio_dir / f"{fileid}.wav"
        mfc = ctx.features_dir / f"{fileid}.mfc"
        out.append(
            Task(
                name=f"extract:{fileid}",
                fn=functools.partial(_extract_features_worker, audio, mfc, params),
                inputs=(audio,),
                outputs=(mfc,),
                parallel_group="features",
            )
        )
    return out


def _all_feature_files(ctx: PipelineContext) -> tuple[Path, ...]:
    """Feature files for every audio file in the corpus."""
    return tuple(ctx.features_dir / f"{fid}.mfc" for fid in ctx.audio_fileids())


def _make_split_task(ctx: PipelineContext) -> Task:
    """Partition `etc/all.transcription` into train/test fileids and
    transcriptions, written under the experiment's etc/ directory."""
    src = ctx.all_transcription
    etc = ctx.etc_dir
    outputs = (
        etc / "train.fileids",
        etc / "test.fileids",
        etc / "train.transcription",
        etc / "test.transcription",
    )

    def run() -> None:
        from st2.lib.corpus import train_test_split

        result = train_test_split(
            src,
            etc,
            train_ratio=ctx.split.train_ratio,
            test_count=ctx.split.test_count,
            seed=ctx.split.seed,
        )
        total = result.n_train + result.n_test
        print(
            f"   split: {result.n_train} train / {result.n_test} test "
            f"({result.n_train / total * 100:.1f}% / {result.n_test / total * 100:.1f}%)"
        )

    return Task(
        name="split",
        fn=run,
        inputs=(src,),
        outputs=outputs,
        description="Partition all.transcription into train/test fileids + transcripts",
    )


def _make_flat_task(ctx: PipelineContext) -> Task:
    phoneset = ctx.shared_dir / "phoneset.txt"
    train_fileids = ctx.etc_dir / "train.fileids"
    feature_files = _all_feature_files(ctx)
    out_dir = ctx.model_dir("flat")

    def run() -> None:
        from st2.lib.flat import init_flat_model

        out_dir.mkdir(parents=True, exist_ok=True)
        with open(phoneset) as f:
            phones = [line.strip() for line in f if line.strip()]
        init_flat_model(
            phones,
            out_dir,
            n_density=1,
            n_state=ctx.train.n_state,
            ctl_path=train_fileids,
            cep_dir=ctx.features_dir,
            cep_ext=".mfc",
            feat_type=ctx.feat.feat_type,
            ceplen=ctx.feat.ncep,
        )
        write_feat_params(out_dir / "feat.params", ctx.feat)

    return Task(
        name="flat",
        fn=run,
        inputs=(phoneset, train_fileids, *feature_files),
        outputs=tuple(ctx.model_files("flat")),
        description="Initialize flat (uniform) acoustic model",
    )


def _make_bw_train_task(
    ctx: PipelineContext,
    *,
    name: str,
    src_model: str,
    out_model: str,
    description: str,
    extra_inputs: Iterable[Path] = (),
    copy_mdef_from_src: bool = False,
) -> Task:
    """Build a Baum-Welch training task.

    `src_model`/`out_model` are model directory names (e.g. "flat", "ci-1g").
    """
    train_fileids = ctx.etc_dir / "train.fileids"
    transcription = ctx.etc_dir / "train.transcription"
    dictionary = ctx.shared_dir / "dictionary.dict"
    feature_files = _all_feature_files(ctx)
    src_dir = ctx.model_dir(src_model)
    out_dir = ctx.model_dir(out_model)

    inputs: tuple[Path, ...] = (
        *ctx.model_files(src_model),
        train_fileids,
        transcription,
        dictionary,
        *feature_files,
        *extra_inputs,
    )

    def run() -> None:
        from st2.lib.steps.train import run_bw_training

        result = run_bw_training(
            model_dir=src_dir,
            output_dir=out_dir,
            features_dir=ctx.features_dir,
            train_fileids=train_fileids,
            transcription=transcription,
            dictionary=dictionary,
            filler_dict=ctx.filler_dict,
            n_iter=ctx.train.max_iterations,
            multipron=ctx.train.multipron_training,
        )
        if copy_mdef_from_src:
            shutil.copy(src_dir / "mdef", out_dir / "mdef")
        write_feat_params(out_dir / "feat.params", ctx.feat)
        print(f"   {name}: {result.iterations} iter(s), converged={result.converged}")

    return Task(
        name=name,
        fn=run,
        inputs=inputs,
        outputs=tuple(ctx.model_files(out_model)),
        description=description,
    )


def _make_split_and_train_task(
    ctx: PipelineContext,
    *,
    name: str,
    src_model: str,
    out_model: str,
    description: str,
) -> Task:
    """Build a 'split Gaussians then train' task (e.g. ci-1g -> ci-2g)."""
    train_fileids = ctx.etc_dir / "train.fileids"
    transcription = ctx.etc_dir / "train.transcription"
    dictionary = ctx.shared_dir / "dictionary.dict"
    feature_files = _all_feature_files(ctx)
    src_dir = ctx.model_dir(src_model)
    split_dir = ctx.model_dir(f"{out_model}-split")
    out_dir = ctx.model_dir(out_model)

    def run() -> None:
        from st2.lib.steps.split import run_split
        from st2.lib.steps.train import run_bw_training

        run_split(input_model_dir=src_dir, output_model_dir=split_dir)
        result = run_bw_training(
            model_dir=split_dir,
            output_dir=out_dir,
            features_dir=ctx.features_dir,
            train_fileids=train_fileids,
            transcription=transcription,
            dictionary=dictionary,
            filler_dict=ctx.filler_dict,
            n_iter=ctx.train.max_iterations,
            multipron=ctx.train.multipron_training,
        )
        write_feat_params(out_dir / "feat.params", ctx.feat)
        print(f"   {name}: split + {result.iterations} iter(s), converged={result.converged}")

    return Task(
        name=name,
        fn=run,
        inputs=(
            *ctx.model_files(src_model),
            train_fileids,
            transcription,
            dictionary,
            *feature_files,
        ),
        outputs=tuple(ctx.model_files(out_model)),
        description=description,
    )


def _make_cd_untied_init_task(ctx: PipelineContext) -> Task:
    phoneset = ctx.shared_dir / "phoneset.txt"
    dictionary = ctx.shared_dir / "dictionary.dict"
    transcription = ctx.etc_dir / "train.transcription"
    ci_src = ctx.model_dir("ci-1g")
    out_dir = ctx.model_dir("cd-untied-init")
    untied_mdef = out_dir / "mdef"

    def run() -> None:
        from st2.lib.mdef import generate_untied_mdef
        from st2.lib.steps.cd_pipeline import run_init_cd_untied

        out_dir.mkdir(parents=True, exist_ok=True)
        generate_untied_mdef(
            phone_list=phoneset,
            dict_path=dictionary,
            transcript=transcription,
            output=untied_mdef,
            filler_dict=ctx.filler_dict,
            n_state=ctx.train.n_state,
        )
        run_init_cd_untied(
            ci_model_dir=ci_src,
            untied_mdef=untied_mdef,
            output_dir=out_dir,
        )
        write_feat_params(out_dir / "feat.params", ctx.feat)

    return Task(
        name="cd-untied-init",
        fn=run,
        inputs=(*ctx.model_files("ci-1g"), phoneset, dictionary, transcription),
        outputs=tuple(ctx.model_files("cd-untied-init")),
        description="Initialize CD untied model from CI-1g",
    )


def _make_cd_untied_task(ctx: PipelineContext) -> Task:
    return _make_bw_train_task(
        ctx,
        name="cd-untied",
        src_model="cd-untied-init",
        out_model="cd-untied",
        description="Train CD untied (per-triphone) model",
        copy_mdef_from_src=True,
    )


def _make_questions_task(ctx: PipelineContext) -> Task:
    out_path = ctx.trees_dir / "questions"

    def run() -> None:
        from st2.lib.steps.cd_pipeline import run_make_questions

        run_make_questions(
            ci_model_dir=ctx.model_dir("ci-1g"),
            output_path=out_path,
            continuous=True,
        )

    return Task(
        name="questions",
        fn=run,
        inputs=tuple(ctx.model_files("ci-1g")),
        outputs=(out_path,),
        description="Generate phonetic questions for decision tree clustering",
    )


def _make_tree_tasks(ctx: PipelineContext) -> list[Task]:
    """One task per (phone, state) tree, plus a sentinel join task.

    Reads `phoneset.txt` at plan time to enumerate the trees. If the
    phoneset is missing (project not yet set up), only the sentinel
    is returned and the downstream `prune-trees` task will be unable
    to plan — the failure mode matches every other "missing setup file"
    case in the runner.
    """
    phoneset = ctx.shared_dir / "phoneset.txt"
    questions = ctx.trees_dir / "questions"
    out_dir = ctx.trees_dir / "unpruned"
    untied_dir = ctx.model_dir("cd-untied")
    untied_inputs = tuple(ctx.model_files("cd-untied"))
    sentinel = out_dir / ".built"

    if phoneset.exists():
        from st2.lib.steps.cd_pipeline import filter_tree_phones

        phones = filter_tree_phones(phoneset)
    else:
        phones = []

    tasks: list[Task] = []
    tree_outputs: list[Path] = []
    for phone in phones:
        for state in range(ctx.train.n_state):
            tree_path = out_dir / f"{phone}-{state}.dtree"
            tree_outputs.append(tree_path)
            tasks.append(
                Task(
                    name=f"tree:{phone}-{state}",
                    fn=functools.partial(
                        _build_tree_worker,
                        untied_dir,
                        questions,
                        tree_path,
                        phone,
                        state,
                    ),
                    inputs=(*untied_inputs, questions),
                    outputs=(tree_path,),
                    parallel_group="trees",
                )
            )

    def write_sentinel() -> None:
        sentinel.parent.mkdir(parents=True, exist_ok=True)
        sentinel.touch()

    tasks.append(
        Task(
            name="trees",
            fn=write_sentinel,
            inputs=(*untied_inputs, phoneset, questions, *tree_outputs),
            outputs=(sentinel,),
            description=f"Sentinel for {len(tree_outputs)} per-(phone,state) trees",
        )
    )
    return tasks


def _make_prune_trees_task(ctx: PipelineContext) -> Task:
    questions = ctx.trees_dir / "questions"
    unpruned_sentinel = ctx.trees_dir / "unpruned" / ".built"
    out_dir = ctx.trees_dir / "pruned"
    sentinel = out_dir / ".pruned"

    def run() -> None:
        from st2.lib.steps.cd_pipeline import run_prune_trees

        run_prune_trees(
            untied_mdef=ctx.model_dir("cd-untied") / "mdef",
            questions_path=questions,
            input_tree_dir=ctx.trees_dir / "unpruned",
            output_tree_dir=out_dir,
            n_senones=ctx.train.n_senones,
        )
        sentinel.touch()

    return Task(
        name="prune-trees",
        fn=run,
        inputs=(*ctx.model_files("cd-untied"), questions, unpruned_sentinel),
        outputs=(sentinel,),
        description=f"Prune decision trees to {ctx.train.n_senones} senones",
    )


def _make_alltriphones_mdef_task(ctx: PipelineContext) -> Task:
    phoneset = ctx.shared_dir / "phoneset.txt"
    dictionary = ctx.shared_dir / "dictionary.dict"
    out_path = ctx.architecture_dir / "alltriphones.mdef"

    def run() -> None:
        from st2.lib.mdef import generate_alltriphones_mdef

        out_path.parent.mkdir(parents=True, exist_ok=True)
        generate_alltriphones_mdef(
            phone_list=phoneset,
            dict_path=dictionary,
            output=out_path,
            filler_dict=ctx.filler_dict,
            n_state=ctx.train.n_state,
        )

    return Task(
        name="alltriphones-mdef",
        fn=run,
        inputs=(phoneset, dictionary),
        outputs=(out_path,),
        description="Generate alltriphones mdef from dictionary",
    )


def _make_cd_1g_init_task(ctx: PipelineContext) -> Task:
    alltri = ctx.architecture_dir / "alltriphones.mdef"
    pruned_sentinel = ctx.trees_dir / "pruned" / ".pruned"
    questions = ctx.trees_dir / "questions"
    out_dir = ctx.model_dir("cd-1g-init")
    tied_mdef = out_dir / "mdef"

    def run() -> None:
        from st2.lib.steps.cd_pipeline import run_init_cd_tied, run_tiestate

        out_dir.mkdir(parents=True, exist_ok=True)
        run_tiestate(
            untied_mdef=alltri,
            tree_dir=ctx.trees_dir / "pruned",
            questions_path=questions,
            output_mdef=tied_mdef,
        )
        run_init_cd_tied(
            ci_model_dir=ctx.model_dir("ci-1g"),
            tied_mdef=tied_mdef,
            output_dir=out_dir,
        )
        write_feat_params(out_dir / "feat.params", ctx.feat)

    return Task(
        name="cd-1g-init",
        fn=run,
        inputs=(
            *ctx.model_files("cd-untied"),
            *ctx.model_files("ci-1g"),
            pruned_sentinel,
            questions,
            alltri,
        ),
        outputs=tuple(ctx.model_files("cd-1g-init")),
        description="Tie states and initialize tied CD-1g from CI-1g",
    )


def _make_cd_1g_train_task(ctx: PipelineContext) -> Task:
    return _make_bw_train_task(
        ctx,
        name="cd-1g",
        src_model="cd-1g-init",
        out_model="cd-1g",
        description="Train tied CD-1g model",
        copy_mdef_from_src=True,
    )


def _make_package_task(
    ctx: PipelineContext,
    *,
    src_model: str,
) -> Task:
    name = f"package-{src_model}"
    pkg_name = f"{src_model}-{ctx.config_name}"
    pkg_dir = ctx.dist_dir / pkg_name
    dictionary = ctx.shared_dir / "dictionary.dict"

    outputs: tuple[Path, ...] = (
        pkg_dir / "acoustic" / "feat.params",
        pkg_dir / "acoustic" / "mdef",
        pkg_dir / "acoustic" / "means",
        pkg_dir / "acoustic" / "variances",
        pkg_dir / "acoustic" / "mixture_weights",
        pkg_dir / "acoustic" / "transition_matrices",
        pkg_dir / "acoustic" / "noisedict",
        pkg_dir / "README.txt",
    )

    def run() -> None:
        from st2.lib.steps.package import package_model

        package_model(
            model_dir=ctx.model_dir(src_model),
            output_dir=ctx.dist_dir,
            model_name=pkg_name,
            dictionary_path=dictionary,
            filler_dict_path=ctx.filler_dict,
        )

    return Task(
        name=name,
        fn=run,
        inputs=(*ctx.model_files(src_model), dictionary),
        outputs=outputs,
        description=f"Package {src_model} for distribution",
    )


def _make_lm_task(ctx: PipelineContext) -> Task:
    transcription = ctx.etc_dir / "train.transcription"
    out_path = ctx.lm_dir / "train.arpa"

    def run() -> None:
        from st2.lib.steps.lm import run_build_lm

        run_build_lm(
            train_transcripts=transcription,
            output_path=out_path,
            max_order=3,
            smoothing="auto",
        )

    return Task(
        name="lm",
        fn=run,
        inputs=(transcription,),
        outputs=(out_path,),
        description="Build 3-gram language model from training transcripts",
    )


def _make_test_task(ctx: PipelineContext, *, model: str) -> Task:
    name = f"test-{model}"
    transcription = ctx.etc_dir / "test.transcription"
    dictionary = ctx.shared_dir / "dictionary.dict"
    lm_path = ctx.lm_dir / "train.arpa"
    report_path = ctx.reports_dir / f"{model}_test.json"

    def run() -> None:
        from st2.lib.testing import create_report, load_transcripts, test_model

        report_path.parent.mkdir(parents=True, exist_ok=True)
        result = test_model(
            model_dir=ctx.model_dir(model),
            test_audio_dir=ctx.audio_dir,
            test_transcripts=load_transcripts(transcription),
            dict_file=dictionary,
            filler_dict=ctx.filler_dict,
            lm=lm_path,
            verbose=True,
            compute_cer=True,
        )
        report = create_report(
            result,
            corpus_name=ctx.project_dir.name,
            test_set_name="test",
        )
        report.save_json(report_path)
        print(f"   {name}: WER={result.wer:.2%}")

    return Task(
        name=name,
        fn=run,
        inputs=(*ctx.model_files(model), lm_path, transcription, dictionary),
        outputs=(report_path,),
        description=f"Test {model} and write WER report",
    )


# Target metadata (replaces targets.yaml).


@dataclass(frozen=True)
class TargetSpec:
    name: str
    kind: str  # "ci", "cd", "trees", "lm", "test", "package", "features"
    description: str
    n_density: int = 0
    n_senones: int = 0


TARGETS: list[TargetSpec] = [
    TargetSpec("split", "split", "Partition the corpus into train/test sets"),
    TargetSpec("flat", "ci", "Initial flat (uniform) model", n_density=1),
    TargetSpec("ci-1g", "ci", "CI model with 1 Gaussian per state", n_density=1),
    TargetSpec("ci-2g", "ci", "CI model with 2 Gaussians per state", n_density=2),
    TargetSpec("ci-4g", "ci", "CI model with 4 Gaussians per state", n_density=4),
    TargetSpec("ci-8g", "ci", "CI model with 8 Gaussians per state", n_density=8),
    TargetSpec("cd-untied", "cd", "CD untied (per-triphone) model", n_density=1),
    TargetSpec("cd-1g", "cd", "CD tied model with 1 Gaussian", n_density=1, n_senones=200),
    TargetSpec("cd-2g", "cd", "CD tied model with 2 Gaussians", n_density=2, n_senones=200),
    TargetSpec("cd-4g", "cd", "CD tied model with 4 Gaussians", n_density=4, n_senones=200),
    TargetSpec(
        "cd-8g", "cd", "CD tied model with 8 Gaussians (default)", n_density=8, n_senones=200
    ),
    TargetSpec("cd-16g", "cd", "CD tied model with 16 Gaussians", n_density=16, n_senones=200),
    TargetSpec("cd-32g", "cd", "CD tied model with 32 Gaussians", n_density=32, n_senones=200),
    TargetSpec("features", "features", "Extract MFCC features for all audio"),
    TargetSpec("lm", "lm", "3-gram language model from training transcripts"),
    TargetSpec("test-ci-8g", "test", "Decode test set with ci-8g and write WER report"),
    TargetSpec("test-cd-8g", "test", "Decode test set with cd-8g and write WER report"),
    TargetSpec("package-ci-8g", "package", "Package ci-8g for distribution"),
    TargetSpec("package-cd-8g", "package", "Package cd-8g for distribution"),
    TargetSpec("package-cd-32g", "package", "Package cd-32g for distribution"),
]


def build_pipeline(ctx: PipelineContext) -> Pipeline:
    """Build a pipeline with all ST2 training tasks registered for `ctx`.

    Feature-extraction tasks are expanded to one task per audio file found
    recursively under the audio directory. An empty or missing audio
    directory fails immediately during pipeline construction.
    """
    pl = Pipeline()

    pl.add(_make_split_task(ctx))
    pl.add(_make_feat_params_task(ctx))
    pl.add_all(_make_extract_tasks(ctx))

    pl.add(_make_flat_task(ctx))
    pl.add(
        _make_bw_train_task(
            ctx,
            name="ci-1g",
            src_model="flat",
            out_model="ci-1g",
            description="Train CI-1g (1 Gaussian per state)",
        )
    )
    for src, dst in [("ci-1g", "ci-2g"), ("ci-2g", "ci-4g"), ("ci-4g", "ci-8g")]:
        pl.add(
            _make_split_and_train_task(
                ctx,
                name=dst,
                src_model=src,
                out_model=dst,
                description=f"Split + train {dst}",
            )
        )

    pl.add(_make_cd_untied_init_task(ctx))
    pl.add(_make_cd_untied_task(ctx))
    pl.add(_make_questions_task(ctx))
    pl.add_all(_make_tree_tasks(ctx))
    pl.add(_make_prune_trees_task(ctx))
    pl.add(_make_alltriphones_mdef_task(ctx))
    pl.add(_make_cd_1g_init_task(ctx))
    pl.add(_make_cd_1g_train_task(ctx))

    for src, dst in [
        ("cd-1g", "cd-2g"),
        ("cd-2g", "cd-4g"),
        ("cd-4g", "cd-8g"),
        ("cd-8g", "cd-16g"),
        ("cd-16g", "cd-32g"),
    ]:
        pl.add(
            _make_split_and_train_task(
                ctx,
                name=dst,
                src_model=src,
                out_model=dst,
                description=f"Split + train {dst}",
            )
        )

    pl.add(_make_lm_task(ctx))
    pl.add(_make_test_task(ctx, model="ci-8g"))
    pl.add(_make_test_task(ctx, model="cd-8g"))

    pl.add(_make_package_task(ctx, src_model="ci-8g"))
    pl.add(_make_package_task(ctx, src_model="cd-8g"))
    pl.add(_make_package_task(ctx, src_model="cd-32g"))

    # Register named targets so users can type `st2 build cd-8g`.
    for spec in TARGETS:
        if spec.name == "features":
            # "features" is a fan-out; the sentinel is feat.params.
            pl.register_target(spec.name, ctx.features_dir / "feat.params")
        elif spec.kind == "split":
            pl.register_target(spec.name, ctx.etc_dir / "train.fileids")
        elif spec.kind == "ci" or spec.kind == "cd":
            pl.register_target(spec.name, ctx.model_dir(spec.name) / "mdef")
        elif spec.kind == "lm":
            pl.register_target(spec.name, ctx.lm_dir / "train.arpa")
        elif spec.kind == "test":
            model = spec.name.removeprefix("test-")
            pl.register_target(spec.name, ctx.reports_dir / f"{model}_test.json")
        elif spec.kind == "package":
            src = spec.name.removeprefix("package-")
            pkg_name = f"{src}-{ctx.config_name}"
            pl.register_target(spec.name, ctx.dist_dir / pkg_name / "acoustic" / "mdef")

    return pl
