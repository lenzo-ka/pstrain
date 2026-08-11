"""Test command for evaluating trained models."""

from __future__ import annotations

import argparse
from pathlib import Path

from pstrain.cli.base import Command, CommandContext, CommandResult


class TestCommand(Command):
    """Test trained models and generate WER report."""

    name = "test"
    help = "Test a trained model"
    description = "Decode test utterances and calculate WER with all jiwer metrics"
    needs_project_dir = False

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add command-specific arguments."""
        parser.add_argument(
            "model",
            type=str,
            help="Model to test (e.g., 'ci-8g', 'cd-8g') or path to model directory",
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
            help="Config/experiment name (default: 'default')",
        )
        parser.add_argument(
            "--lm",
            type=str,
            help="Language model file (ARPA format). If not specified, builds from training transcripts.",
        )
        parser.add_argument(
            "--no-lm",
            action="store_true",
            help="Don't use any language model (uniform probabilities)",
        )
        parser.add_argument(
            "--lm-order",
            type=int,
            default=3,
            help="N-gram order for auto-built LM (default: 3)",
        )
        parser.add_argument(
            "--dict",
            type=str,
            help="Dictionary file (overrides project dictionary)",
        )
        parser.add_argument(
            "--verbose",
            "-v",
            action="store_true",
            help="Include per-utterance results in report",
        )
        parser.add_argument(
            "--cer",
            action="store_true",
            help="Also compute Character Error Rate",
        )
        parser.add_argument(
            "--json",
            action="store_true",
            help="Output JSON to stdout instead of summary",
        )
        parser.add_argument(
            "--output",
            "-o",
            type=str,
            help="Write report to file (JSON for .json, text otherwise)",
        )

    def execute(self, ctx: CommandContext) -> CommandResult:
        """Execute test command."""
        from pstrain.lib.testing import (
            check_pocketsphinx,
            create_report,
            load_transcripts,
            test_model,
        )

        # Check PocketSphinx availability
        available, msg = check_pocketsphinx()
        if not available:
            return CommandResult.fail(f"PocketSphinx not available: {msg}")

        # Resolve project directory
        project_dir = Path(ctx.args.project_dir).resolve() if ctx.args.project_dir else Path.cwd()

        if not project_dir.exists():
            return CommandResult.fail(f"Project directory does not exist: {project_dir}")

        config_name = ctx.args.config
        model_arg = ctx.args.model

        # Resolve model directory
        if "/" in model_arg or Path(model_arg).is_absolute():
            # Direct path to model
            model_dir = Path(model_arg)
        else:
            # Model name like "ci-8g" or "cd-8g"
            model_dir = project_dir / "shared" / "models" / model_arg / config_name

        if not model_dir.exists():
            return CommandResult.fail(f"Model directory not found: {model_dir}")

        # Resolve dictionary
        if ctx.args.dict:
            dict_file = Path(ctx.args.dict)
        else:
            dict_file = project_dir / "shared" / "dictionary.dict"

        if not dict_file.exists():
            return CommandResult.fail(f"Dictionary not found: {dict_file}")

        # Resolve filler dictionary
        _filler = project_dir / "shared" / "filler.dict"
        filler_dict: Path | None = _filler if _filler.exists() else None

        # Resolve language model
        lm_path = None
        if ctx.args.lm:
            lm_path = Path(ctx.args.lm)
            if not lm_path.exists():
                return CommandResult.fail(f"Language model not found: {lm_path}")

        # Load test transcripts
        exp_dir = project_dir / "experiments" / config_name
        test_transcript_file = exp_dir / "etc" / "test.decoder.transcription"
        if not test_transcript_file.exists():
            # Compatibility for projects created before typed split artifacts.
            test_transcript_file = exp_dir / "etc" / "test.transcription"

        if not test_transcript_file.exists():
            return CommandResult.fail(f"Test transcripts not found: {test_transcript_file}")

        test_transcripts = load_transcripts(test_transcript_file)
        if not test_transcripts:
            return CommandResult.fail("No test transcripts loaded")

        # Resolve audio directory (try multiple locations)
        audio_dir = None
        for candidate in ["audio", "shared/wav", "wav"]:
            candidate_dir = project_dir / candidate
            if candidate_dir.exists():
                audio_dir = candidate_dir
                break

        if audio_dir is None:
            return CommandResult.fail("Audio directory not found. Tried: audio/, shared/wav/, wav/")

        ctx.log_action("Test", str(model_dir))
        ctx.log(f"  Test utterances: {len(test_transcripts)}")
        ctx.log(f"  Dictionary: {dict_file}")
        if lm_path:
            ctx.log(f"  Language model: {lm_path}")
        else:
            ctx.log("  Language model: None (using uniform)")

        if ctx.dry_run:
            ctx.log("# Would run decoding and calculate WER")
            return CommandResult.ok("Dry run complete")

        # Run test
        try:
            result = test_model(
                model_dir=model_dir,
                test_audio_dir=audio_dir,
                test_transcripts=test_transcripts,
                dict_file=dict_file,
                filler_dict=filler_dict,
                lm=lm_path,
                verbose=ctx.args.verbose,
                compute_cer=ctx.args.cer,
            )
        except ImportError as e:
            return CommandResult.fail(f"Import error: {e}")
        except RuntimeError as e:
            return CommandResult.fail(f"Test failed: {e}")
        except FileNotFoundError as e:
            return CommandResult.fail(f"File not found: {e}")

        # Create report
        report = create_report(
            result=result,
            corpus_name=project_dir.name,
            test_set_name="test",
        )

        # Output
        if ctx.args.output:
            output_path = Path(ctx.args.output)
            if output_path.suffix == ".json":
                report.save_json(output_path)
                ctx.log(f"JSON report saved: {output_path}")
            else:
                report.save_text(output_path, show_per_utterance=ctx.args.verbose)
                ctx.log(f"Text report saved: {output_path}")

        if ctx.args.json:
            ctx.log(report.to_json())
        else:
            ctx.log(report.format_text(show_per_utterance=ctx.args.verbose))

        # Return success with WER summary
        return CommandResult.ok(
            f"WER: {result.wer:.2%} ({result.n_decoded}/{result.n_utterances} decoded)"
        )


# Singleton instance for registration
test_command = TestCommand()
