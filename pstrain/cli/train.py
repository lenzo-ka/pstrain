"""One-command project setup and acoustic-model training."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

from pstrain.api import validate_project
from pstrain.cli.base import Command, CommandContext, CommandResult
from pstrain.lib.one_command import (
    PROMPT_FORMATS,
    PromptFormatError,
    identity_difference,
    input_identity,
    installed_corpus_identity,
    validate_inputs,
    write_training_transcription,
    write_validation_reports,
)
from pstrain.lib.pipeline import PipelineContext, UnknownTargetError
from pstrain.lib.pipeline.context import DEFAULT_CONFIGS
from pstrain.lib.pipeline.tasks import TARGETS, build_pipeline
from pstrain.lib.setup import setup_project


def _refuse_project_symlinks(project_dir: Path) -> None:
    """Replacement never adopts or traverses links from the old project."""
    for directory, dirnames, filenames in os.walk(project_dir, followlinks=False):
        base = Path(directory)
        for name in [*dirnames, *filenames]:
            path = base / name
            if path.is_symlink():
                raise ValueError(f"Refusing to replace project containing symlink: {path}")


def _install_project_transactionally(
    project_dir: Path,
    *,
    transcription_path: Path,
    audio_path: Path,
    dictionary_path: Path,
    phoneset_path: Path | None,
    filler_path: Path | None,
    prompts_path: Path,
    source_identity: dict[str, object],
    link_audio: bool,
    replacing: bool,
) -> None:
    """Build a complete replacement beside the destination, then rename it into place."""
    project_dir.parent.mkdir(parents=True, exist_ok=True)
    if replacing:
        _refuse_project_symlinks(project_dir)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{project_dir.name}.replacement-", dir=project_dir.parent)
    )
    backup: Path | None = None
    installed = False
    try:
        setup_project(
            project_dir=staging,
            transcription_path=transcription_path,
            audio_path=audio_path,
            dictionary_path=dictionary_path,
            phoneset_path=phoneset_path,
            filler_dict_path=filler_path,
            link_audio=link_audio,
        )
        shutil.copyfile(prompts_path, staging / "etc" / "prompts.source")
        installed_identity = installed_corpus_identity(
            staging, audio_ownership="link" if link_audio else "copy"
        )
        manifest = {"version": 2, "source": source_identity, "installed": installed_identity}
        (staging / "etc" / "input-identity.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        if replacing:
            backup = Path(
                tempfile.mkdtemp(prefix=f".{project_dir.name}.previous-", dir=project_dir.parent)
            )
            backup.rmdir()
            project_dir.rename(backup)
        staging.rename(project_dir)
        installed = True
        if backup is not None:
            shutil.rmtree(backup)
            backup = None
    except Exception:
        if backup is not None and backup.exists() and not project_dir.exists():
            backup.rename(project_dir)
            backup = None
        raise
    finally:
        if staging.exists() and not installed:
            shutil.rmtree(staging)


class TrainCommand(Command):
    """Set up, validate, and train a project through shared library APIs."""

    name = "train"
    help = "Set up a project and train a model with one command"
    description = "Validate corpus inputs, create or resume a project, and build a model target"
    needs_project_dir = False
    supports_json_output = True

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("project_dir", type=str, help="New or resumable project directory")
        parser.add_argument("--audio", required=True, type=str, help="Directory of WAV files")
        parser.add_argument("--prompts", required=True, type=str, help="Prompt mapping file")
        parser.add_argument(
            "--dictionary", required=True, type=str, help="Pronunciation dictionary"
        )
        parser.add_argument("--target", default="ci-1g", help="Training target (default: ci-1g)")
        parser.add_argument(
            "--profile", default="default", help="Canonical profile (default: default)"
        )
        parser.add_argument("--experiment", default="default", help="Experiment name")
        parser.add_argument("--phoneset", type=str, help="Phoneset file (default: extract)")
        parser.add_argument("--filler-dict", type=str, help="Filler dictionary file")
        parser.add_argument("--prompt-format", choices=PROMPT_FORMATS, default="auto")
        parser.add_argument("--link-audio", action="store_true", help="Link instead of copy audio")
        parser.add_argument("--resume", action="store_true", help="Resume a compatible project")
        parser.add_argument(
            "--replace-inputs",
            action="store_true",
            help="Explicitly replace installed inputs in an existing project",
        )
        parser.add_argument("--force", action="store_true", help="Force reachable build tasks")
        parser.add_argument("-j", "--jobs", type=int, default=None, help="Parallel workers")
        parser.add_argument("--normalize-with", metavar="POLICY", help="Named normalization policy")
        parser.add_argument("--normalization-report", type=str, help="Normalization report path")

    def execute(self, ctx: CommandContext) -> CommandResult:
        started = time.monotonic()
        project_dir = Path(ctx.args.project_dir).resolve()
        audio_dir = Path(ctx.args.audio).resolve()
        prompts_path = Path(ctx.args.prompts).resolve()
        dictionary_path = Path(ctx.args.dictionary).resolve()
        phoneset_path = Path(ctx.args.phoneset).resolve() if ctx.args.phoneset else None
        filler_path = Path(ctx.args.filler_dict).resolve() if ctx.args.filler_dict else None

        if ctx.args.resume and ctx.args.replace_inputs:
            return self._failure(
                ctx, "destination_mode_conflict", "Use only one of --resume or --replace-inputs"
            )
        if ctx.args.normalize_with:
            return self._failure(
                ctx,
                "unknown_normalization_policy",
                "No normalization policies are installed; prompts are used exactly as supplied",
            )
        sources = [
            ("Audio", audio_dir, "directory"),
            ("Prompts", prompts_path, "file"),
            ("Dictionary", dictionary_path, "file"),
            ("Phoneset", phoneset_path, "file"),
            ("Filler dictionary", filler_path, "file"),
        ]
        for label, path, kind in sources:
            if path is None:
                continue
            valid = path.is_dir() if kind == "directory" else path.is_file()
            if not valid:
                return self._failure(ctx, "missing_input", f"{label} must be a {kind}: {path}")
        if project_dir.exists() and not (ctx.args.resume or ctx.args.replace_inputs):
            return self._failure(
                ctx,
                "destination_exists",
                f"Destination exists: {project_dir}. Use --resume for identical inputs or "
                "--replace-inputs to replace them explicitly.",
            )
        if not project_dir.exists() and (ctx.args.resume or ctx.args.replace_inputs):
            return self._failure(
                ctx, "destination_missing", f"Destination does not exist: {project_dir}"
            )
        if ctx.args.target not in {spec.name for spec in TARGETS}:
            return self._failure(
                ctx,
                "unknown_target",
                f"Unknown target {ctx.args.target!r}; use 'pstrain build --list' for choices",
            )
        if not ctx.args.resume and ctx.args.profile not in DEFAULT_CONFIGS:
            choices = ", ".join(sorted(DEFAULT_CONFIGS))
            return self._failure(
                ctx,
                "unknown_profile",
                f"Unknown profile {ctx.args.profile!r}; available profiles: {choices}",
            )

        try:
            report, prompts = validate_inputs(
                audio_dir,
                prompts_path,
                dictionary_path,
                ctx.args.prompt_format,
                phoneset_path,
                filler_path,
            )
        except (PromptFormatError, UnicodeDecodeError, ValueError) as exc:
            return self._failure(ctx, "invalid_input", str(exc))

        source_identity = input_identity(
            audio_dir,
            prompts_path,
            dictionary_path,
            filler_path,
            phoneset_path,
            ctx.args.link_audio,
        )
        manifest_path = project_dir / "etc" / "input-identity.json"
        if ctx.args.resume:
            try:
                current = project_dir
                for part in manifest_path.relative_to(project_dir).parts:
                    current /= part
                    if current.is_symlink():
                        raise ValueError(f"Unexpected symlink in installed corpus: {current}")
            except ValueError as exc:
                return self._failure(ctx, "incompatible_resume", f"Cannot resume: {exc}")
            if not manifest_path.is_file():
                return self._failure(
                    ctx,
                    "incompatible_resume",
                    f"Cannot resume: input identity is missing at {manifest_path}. "
                    "Use --replace-inputs to adopt the requested inputs.",
                )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("version") != 2:
                return self._failure(
                    ctx,
                    "incompatible_resume",
                    "Cannot resume: installed corpus identity uses an unsupported manifest. "
                    "Use --replace-inputs to replace them explicitly.",
                )
            source_difference = identity_difference(manifest.get("source"), source_identity)
            if source_difference:
                return self._failure(
                    ctx,
                    "incompatible_resume",
                    f"Cannot resume: requested source arguments differ ({source_difference}). "
                    "Use --replace-inputs to replace them explicitly.",
                )
            expected_installed = manifest.get("installed")
            ownership = (
                expected_installed.get("audio_ownership")
                if isinstance(expected_installed, dict)
                else None
            )
            try:
                actual_installed = installed_corpus_identity(
                    project_dir, audio_ownership=str(ownership)
                )
            except (OSError, ValueError) as exc:
                return self._failure(ctx, "incompatible_resume", f"Cannot resume: {exc}")
            installed_difference = identity_difference(expected_installed, actual_installed)
            if installed_difference:
                return self._failure(
                    ctx,
                    "incompatible_resume",
                    f"Cannot resume: installed corpus diverged ({installed_difference}). "
                    "Use --replace-inputs to replace it explicitly.",
                )

        if not report.valid:
            if not ctx.dry_run:
                validation_path, oov_path = write_validation_reports(project_dir, report)
            else:
                validation_path = project_dir / "reports" / "input-validation.json"
                oov_path = project_dir / "reports" / "oov.txt"
            cause = report.errors[0]
            if report.oov:
                cause = f"OOV validation blocked training: {len(report.oov)} unique token(s)"
            return self._failure(
                ctx,
                "input_validation_failed",
                f"{cause}. Full validation report: {validation_path}. OOV report: {oov_path}",
                report.to_dict(),
            )

        if ctx.dry_run:
            with tempfile.TemporaryDirectory(prefix="pstrain-train-plan-") as temporary:
                plan_project = project_dir if ctx.args.resume else Path(temporary) / "project"
                if not ctx.args.resume:
                    canonical = Path(temporary) / "all.transcription"
                    write_training_transcription(canonical, prompts)
                    setup_project(
                        project_dir=plan_project,
                        transcription_path=canonical,
                        audio_path=audio_dir,
                        dictionary_path=dictionary_path,
                        phoneset_path=phoneset_path,
                        filler_dict_path=filler_path,
                        link_audio=True,
                    )
                try:
                    plan_ctx = PipelineContext.from_config(
                        plan_project,
                        experiment=ctx.args.experiment,
                        config_name=ctx.args.profile,
                        cli_overrides={"runner": {"jobs": ctx.args.jobs}}
                        if ctx.args.jobs
                        else None,
                    )
                    plan = build_pipeline(plan_ctx)
                    plan.run(
                        ctx.args.target,
                        dry_run=True,
                        force=ctx.args.force,
                        jobs=ctx.args.jobs,
                    )
                except ValueError as exc:
                    return self._failure(ctx, "unknown_profile", str(exc))
            payload = {
                "status": "dry-run",
                "project": str(project_dir),
                "setup": "resume"
                if ctx.args.resume
                else "replace"
                if ctx.args.replace_inputs
                else "create",
                "prompt_format": report.prompt_format,
                "profile": ctx.args.profile,
                "experiment": ctx.args.experiment,
                "target": ctx.args.target,
                "jobs": ctx.args.jobs,
                "input_validation": report.to_dict(),
            }
            self._emit(ctx, payload)
            return CommandResult.ok()

        with tempfile.TemporaryDirectory(prefix="pstrain-train-") as temporary:
            canonical = Path(temporary) / "all.transcription"
            write_training_transcription(canonical, prompts)
            if not ctx.args.resume:
                try:
                    _install_project_transactionally(
                        project_dir,
                        transcription_path=canonical,
                        audio_path=audio_dir,
                        dictionary_path=dictionary_path,
                        phoneset_path=phoneset_path,
                        filler_path=filler_path,
                        prompts_path=prompts_path,
                        source_identity=source_identity,
                        link_audio=ctx.args.link_audio,
                        replacing=ctx.args.replace_inputs,
                    )
                except (OSError, ValueError) as exc:
                    return self._failure(
                        ctx, "project_setup_failed", f"Project setup failed: {exc}"
                    )
        validation_path, oov_path = write_validation_reports(project_dir, report)

        try:
            pipeline_ctx = PipelineContext.from_config(
                project_dir,
                experiment=ctx.args.experiment,
                config_name=ctx.args.profile,
                cli_overrides={"runner": {"jobs": ctx.args.jobs}} if ctx.args.jobs else None,
            )
        except ValueError as exc:
            return self._failure(ctx, "unknown_profile", str(exc))
        project_report = validate_project(project_dir, experiment=ctx.args.experiment)
        if not project_report.is_valid:
            return self._failure(
                ctx,
                "project_validation_failed",
                f"Project validation failed with {len(project_report.errors)} error(s). "
                f"Input report: {validation_path}",
                project_report.to_dict(),
            )
        try:
            pipeline = build_pipeline(pipeline_ctx)
            rc = pipeline.run(
                ctx.args.target,
                force=ctx.args.force,
                jobs=ctx.args.jobs,
                verbose=ctx.verbose,
            )
        except UnknownTargetError as exc:
            return self._failure(ctx, "unknown_target", str(exc))
        except Exception as exc:
            return self._failure(
                ctx,
                "training_failed",
                f"Training failed: {exc}. Resume with: pstrain train {project_dir} "
                f"--audio {audio_dir} --prompts {prompts_path} --dictionary {dictionary_path} --resume",
            )
        if rc:
            return self._failure(ctx, "training_failed", f"Pipeline exited with code {rc}")

        payload = {
            "status": "trained",
            "model": str(pipeline_ctx.model_dir(ctx.args.target)),
            "profile": ctx.args.profile,
            "target": ctx.args.target,
            "experiment": ctx.args.experiment,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "reports": {"validation": str(validation_path), "oov": str(oov_path)},
            "resume_command": f"pstrain train {project_dir} --audio {audio_dir} --prompts "
            f"{prompts_path} --dictionary {dictionary_path} --resume",
            "test_command": f"pstrain test {ctx.args.target} --project-dir {project_dir} --no-lm",
            "config_command": f"pstrain config show --project-dir {project_dir} --profile {ctx.args.profile}",
        }
        self._emit(ctx, payload)
        return CommandResult.ok()

    @staticmethod
    def _emit(ctx: CommandContext, payload: dict[str, object]) -> None:
        if ctx.json_output:
            print(ctx.format_json(payload))
            return
        for key, value in payload.items():
            if isinstance(value, (dict, list)):
                continue
            print(f"{key.replace('_', ' ').title()}: {value}")

    @staticmethod
    def _failure(
        ctx: CommandContext,
        code: str,
        message: str,
        details: dict[str, object] | None = None,
    ) -> CommandResult:
        if ctx.json_output:
            print(
                ctx.format_json(
                    {"status": "error", "code": code, "message": message, "details": details or {}}
                )
            )
            return CommandResult.fail("")
        return CommandResult.fail(message)


train_command = TrainCommand()
