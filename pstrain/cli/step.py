"""CLI subcommand for running training steps.

Steps can be run as subcommands: pstrain step ci_hmm [args]
"""

from __future__ import annotations

from typing import Any

from pstrain.api import run_step_cd_hmm_untied, run_step_ci_hmm


def register_step_command(subparsers: Any) -> None:
    """Register step command with argument parser."""
    parser = subparsers.add_parser(
        "step",
        help="Run a training step",
        description="Run a training step (e.g., ci_hmm, cd_hmm_untied)",
    )

    step_subparsers = parser.add_subparsers(dest="step_name", help="Step name")

    # CI HMM
    ci_hmm_parser = step_subparsers.add_parser(
        "ci_hmm",
        help="CI HMM training (context-independent models)",
        description="Train context-independent HMM models using Baum-Welch",
    )
    ci_hmm_parser.add_argument(
        "--project-dir",
        type=str,
        default=".",
        help="Project directory (default: current directory)",
    )
    ci_hmm_parser.add_argument(
        "--experiment",
        type=str,
        default="default",
        help="Experiment name (default: default)",
    )
    ci_hmm_parser.add_argument(
        "--config",
        type=str,
        default="baseline",
        help="Model configuration name (default: baseline)",
    )
    ci_hmm_parser.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        help="Maximum training iterations (default: 10)",
    )
    ci_hmm_parser.add_argument(
        "--min-iterations",
        type=int,
        default=3,
        help="Minimum iterations before checking convergence (default: 3)",
    )
    ci_hmm_parser.add_argument(
        "--convergence-threshold",
        type=float,
        default=0.001,
        help="Convergence threshold (default: 0.001)",
    )
    ci_hmm_parser.set_defaults(func=cmd_ci_hmm)

    # CD HMM untied
    cd_hmm_untied_parser = step_subparsers.add_parser(
        "cd_hmm_untied",
        help="CD HMM untied training (context-dependent untied models)",
        description="Train context-dependent untied HMM models",
    )
    cd_hmm_untied_parser.add_argument(
        "--project-dir",
        type=str,
        default=".",
        help="Project directory (default: current directory)",
    )
    cd_hmm_untied_parser.add_argument(
        "--experiment",
        type=str,
        default="default",
        help="Experiment name (default: default)",
    )
    cd_hmm_untied_parser.add_argument(
        "--config",
        type=str,
        default="baseline",
        help="Model configuration name (default: baseline)",
    )
    cd_hmm_untied_parser.set_defaults(func=cmd_cd_hmm_untied)


def cmd_ci_hmm(args: Any) -> int:
    """Execute CI HMM training."""
    return run_step_ci_hmm(
        args.project_dir,
        args.experiment,
        args.config,
        max_iterations=args.max_iterations,
        min_iterations=args.min_iterations,
        convergence_threshold=args.convergence_threshold,
    )


def cmd_cd_hmm_untied(args: Any) -> int:
    """Execute CD HMM untied training."""
    return run_step_cd_hmm_untied(
        args.project_dir,
        args.experiment,
        args.config,
    )
