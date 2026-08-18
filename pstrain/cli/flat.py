"""CLI command for flat model initialization."""

from __future__ import annotations

import argparse
from pathlib import Path

from pstrain.cli.base import CommandContext, CommandResult, ModelCommand


class FlatCommand(ModelCommand):
    """Initialize flat (uniform) HMM models."""

    name = "flat"
    help = "Initialize flat (uniform) HMM models"
    description = "Create initial flat HMM models for Baum-Welch training"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add flat-specific arguments."""
        self.add_model_arguments(parser)

        parser.add_argument(
            "--dictionary",
            type=str,
            help="Dictionary file (default: {project_dir}/shared/dictionary.dict)",
        )
        parser.add_argument(
            "--phoneset",
            type=str,
            help="Phoneset file (default: {project_dir}/shared/phoneset.txt)",
        )
        parser.add_argument(
            "--features-dir",
            type=str,
            help="Feature directory (default: {project_dir}/shared/features/{feature_set_id})",
        )
        parser.add_argument(
            "--output-dir",
            type=str,
            help="Output directory (default: {experiment_dir}/models/{model_type}/{config}/model/flat)",
        )
        parser.add_argument(
            "--feat-params",
            type=str,
            help="Feature parameters file (default: {features_dir}/feat.params)",
        )
        parser.add_argument(
            "--n-state",
            type=int,
            default=3,
            help="Number of emitting states per phone (default: 3)",
        )
        parser.add_argument(
            "--n-density",
            type=int,
            default=1,
            help="Number of Gaussians per state (default: 1)",
        )
        parser.add_argument(
            "--n-feat",
            type=int,
            default=39,
            help="Feature dimension (default: 39 = 13 ceps * 3)",
        )

    def execute(self, ctx: CommandContext) -> CommandResult:
        """Initialize flat model using pure Python."""
        from pstrain.api import init_flat_model

        model = self.get_model(ctx)
        self.get_config(ctx)  # Validate the selected canonical profile.

        # Resolve paths
        project_dir = ctx.project_dir
        experiment_dir = project_dir / "experiments" / ctx.experiment
        flat_dir = (
            Path(ctx.args.output_dir)
            if ctx.args.output_dir
            else experiment_dir / "models" / model.model_type / ctx.config_name / "model" / "flat"
        )
        phoneset_path = (
            Path(ctx.args.phoneset)
            if ctx.args.phoneset
            else project_dir / "shared" / "phoneset.txt"
        )
        features_dir = (
            Path(ctx.args.features_dir)
            if ctx.args.features_dir
            else project_dir / "shared" / "features" / ctx.config_name
        )
        feat_params = (
            Path(ctx.args.feat_params) if ctx.args.feat_params else features_dir / "feat.params"
        )

        # Read phoneset
        if not phoneset_path.exists():
            return CommandResult.fail(f"Phoneset not found: {phoneset_path}")
        phones = phoneset_path.read_text().strip().split("\n")
        phones = [p.strip() for p in phones if p.strip()]

        ctx.log_action("Initialize flat model", str(flat_dir))
        ctx.log(f"  Model type: {model.display_name}")
        ctx.log(f"  Config: {ctx.config_name}")
        ctx.log(f"  Phones: {len(phones)}")
        ctx.log(f"  States: {ctx.args.n_state}, Densities: {ctx.args.n_density}")
        ctx.log(f"  Features: {ctx.args.n_feat} dims")

        if ctx.dry_run:
            ctx.emit_blank()
            ctx.comment(f"mkdir -p {flat_dir}")
            ctx.comment("# Create mdef, means, variances, mixture_weights, transition_matrices")
            ctx.comment(f"cp {feat_params} {flat_dir}/feat.params")
            return CommandResult.ok("Dry run complete")

        # Check feat.params exists
        if not feat_params.exists():
            return CommandResult.fail(f"Feature params not found: {feat_params}")

        # Get control file for computing global mean/variance
        _ctl = experiment_dir / "etc" / "train.fileids"
        ctl_path: Path | None
        cep_dir: Path | None
        if not _ctl.exists():
            ctx.log("  WARNING: No train.fileids found, using placeholder means/variances")
            ctl_path = None
            cep_dir = None
        else:
            ctl_path = _ctl
            cep_dir = features_dir

        # Create flat model (compute global mean/var from features if available)
        files = init_flat_model(
            phones=phones,
            output_dir=flat_dir,
            n_state=ctx.args.n_state,
            n_density=ctx.args.n_density,
            ctl_path=ctl_path,
            cep_dir=cep_dir,
            cep_ext=".mfc",
            feat_type="1s_c_d_dd",
            ceplen=13,
        )

        # Copy feat.params
        import shutil

        shutil.copy(feat_params, flat_dir / "feat.params")

        ctx.log("")
        ctx.log("Created files:")
        for name, path in files.items():
            ctx.log(f"  {name}: {path}")
        ctx.log(f"  feat.params: {flat_dir / 'feat.params'}")

        return CommandResult.ok(f"Flat model initialized in {flat_dir}")


# Singleton instance for registration
flat_command = FlatCommand()
