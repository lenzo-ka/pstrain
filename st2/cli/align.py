"""Align command: run forced alignment with an existing model.

Pattern is the same as :mod:`st2.cli.test` -- resolve project paths,
load transcripts, then drive :func:`st2.lib.alignment.align_corpus`
with a single long-lived :class:`Aligner`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from st2.cli.base import Command, CommandContext, CommandResult


class AlignCommand(Command):
    """Forced-align a corpus against a trained model."""

    name = "align"
    help = "Align audio to transcripts with a trained model"
    description = (
        "Run forced alignment on every utterance in the test (or train) "
        "transcription file and write per-utterance segmentations to disk."
    )
    needs_project_dir = False

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "model",
            type=str,
            help=(
                "Model to align with (e.g. 'cd-8g', 'ci-1g') or absolute "
                "path to a model directory containing mdef + means + ..."
            ),
        )
        parser.add_argument(
            "--project-dir",
            "-p",
            type=str,
            help="Project directory (default: current directory)",
        )
        parser.add_argument(
            "--config",
            "-c",
            type=str,
            default="default",
            help="Config / experiment name (default: 'default')",
        )
        parser.add_argument(
            "--transcripts",
            type=str,
            help=(
                "Path to a Sphinx-format transcription file. Default: "
                "experiments/<config>/etc/test.transcription"
            ),
        )
        parser.add_argument(
            "--audio-dir",
            type=str,
            help="Override the audio directory (default: project audio/ or shared/wav/)",
        )
        parser.add_argument(
            "--audio-ext",
            type=str,
            default=".wav",
            help="Audio file extension (default: .wav)",
        )
        parser.add_argument(
            "--dict",
            type=str,
            help="Pronunciation dictionary (overrides shared/dictionary.dict)",
        )
        parser.add_argument(
            "--filler-dict",
            type=str,
            help="Filler dictionary (default: shared/filler.dict if present)",
        )
        parser.add_argument(
            "--output-dir",
            "-o",
            type=str,
            help="Directory to write per-utterance .TextGrid files (default: skip)",
        )
        parser.add_argument(
            "--ctm",
            type=str,
            help="Write a single CTM file at this path with word-level segments",
        )
        parser.add_argument(
            "--no-phones",
            action="store_true",
            help="Skip phone-level segmentation (~2x faster, word-only output)",
        )
        parser.add_argument(
            "--beam",
            type=float,
            default=1e-64,
            help="Viterbi pruning beam (default: 1e-64, matches sphinx3_align)",
        )

    def execute(self, ctx: CommandContext) -> CommandResult:
        from st2.lib.alignment import align_corpus, load_transcripts, save_ctm, save_textgrid

        project_dir = Path(ctx.args.project_dir).resolve() if ctx.args.project_dir else Path.cwd()
        if not project_dir.exists():
            return CommandResult.fail(f"Project directory does not exist: {project_dir}")

        config_name = ctx.args.config
        model_arg = ctx.args.model

        if "/" in model_arg or Path(model_arg).is_absolute():
            model_dir = Path(model_arg)
        else:
            model_dir = project_dir / "shared" / "models" / model_arg / config_name
        if not model_dir.exists():
            return CommandResult.fail(f"Model directory not found: {model_dir}")

        dict_file = (
            Path(ctx.args.dict) if ctx.args.dict else project_dir / "shared" / "dictionary.dict"
        )
        if not dict_file.exists():
            return CommandResult.fail(f"Dictionary not found: {dict_file}")

        if ctx.args.filler_dict:
            filler_dict: Path | None = Path(ctx.args.filler_dict)
        else:
            default_fdict = project_dir / "shared" / "filler.dict"
            filler_dict = default_fdict if default_fdict.exists() else None

        if ctx.args.transcripts:
            transcript_file = Path(ctx.args.transcripts)
        else:
            transcript_file = (
                project_dir / "experiments" / config_name / "etc" / "test.transcription"
            )
        if not transcript_file.exists():
            return CommandResult.fail(f"Transcript file not found: {transcript_file}")

        transcripts = load_transcripts(transcript_file)
        if not transcripts:
            return CommandResult.fail("No transcripts loaded")

        audio_dir: Path | None
        if ctx.args.audio_dir:
            audio_dir = Path(ctx.args.audio_dir)
        else:
            audio_dir = None
            for candidate in ("audio", "shared/wav", "wav"):
                candidate_path = project_dir / candidate
                if candidate_path.exists():
                    audio_dir = candidate_path
                    break
        if audio_dir is None:
            return CommandResult.fail("Audio directory not found. Tried: audio/, shared/wav/, wav/")

        include_phones = not ctx.args.no_phones

        ctx.log_action("Align", str(model_dir))
        ctx.log(f"  Utterances: {len(transcripts)}")
        ctx.log(f"  Audio dir:  {audio_dir}")
        ctx.log(f"  Dictionary: {dict_file}")
        ctx.log(f"  Phones:     {'yes' if include_phones else 'no'}")
        if ctx.dry_run:
            ctx.log("# Would align and write segmentations")
            return CommandResult.ok("Dry run complete")

        ctx.log("Aligning...")
        job = align_corpus(
            transcripts=transcripts,
            audio_dir=audio_dir,
            model_dir=model_dir,
            dict_path=dict_file,
            filler_dict=filler_dict,
            audio_ext=ctx.args.audio_ext,
            include_phones=include_phones,
        )

        ctx.log(
            f"Aligned {job.n_aligned}/{job.n_utterances} "
            f"({job.success_rate * 100:.1f}%); {job.n_failed} failed"
        )

        if ctx.args.output_dir and job.results:
            out_dir = Path(ctx.args.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            for utt_id, result in job.results.items():
                save_textgrid(result, out_dir / f"{utt_id}.TextGrid")
            ctx.log(f"TextGrids saved to: {out_dir}")

        if ctx.args.ctm and job.results:
            ctm_path = Path(ctx.args.ctm)
            ctm_path.parent.mkdir(parents=True, exist_ok=True)
            with ctm_path.open("w") as fh:
                for utt_id in transcripts:
                    if utt_id in job.results:
                        save_ctm(job.results[utt_id], ctm_path.parent / f"{utt_id}.ctm")
                        with (ctm_path.parent / f"{utt_id}.ctm").open() as src:
                            fh.write(src.read())
                        (ctm_path.parent / f"{utt_id}.ctm").unlink()
            ctx.log(f"CTM saved to: {ctm_path}")

        if job.errors:
            for utt_id, msg in list(job.errors.items())[:5]:
                ctx.log(f"  failure: {utt_id}: {msg}")
            if len(job.errors) > 5:
                ctx.log(f"  ... and {len(job.errors) - 5} more")

        return CommandResult.ok(f"Aligned {job.n_aligned}/{job.n_utterances}")


align_command = AlignCommand()
