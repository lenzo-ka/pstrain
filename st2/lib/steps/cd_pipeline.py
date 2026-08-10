"""Context-Dependent (CD) training pipeline steps.

Implements the CD training pipeline:
1. Generate untied triphone mdef from transcripts
2. Initialize CD untied model from CI model
3. Build decision trees for state tying
4. Tie states and create CD model
5. Train CD model with Gaussian splitting
"""

from __future__ import annotations

import contextlib
import logging
import shutil
from pathlib import Path

import numpy as np

from st2.lib import _st2c
from st2.lib.dtree import build_tree, make_quests, prune_tree, tie_states
from st2.lib.mdef import generate_untied_mdef
from st2.lib.validate import validate_files_exist

logger = logging.getLogger(__name__)

__all__ = [
    "run_untied_mdef",
    "run_init_cd_untied",
    "run_make_questions",
    "run_build_trees",
    "run_tiestate",
    "run_init_cd_tied",
]


def run_untied_mdef(
    phone_list: Path,
    dictionary: Path,
    transcription: Path,
    output_mdef: Path,
    filler_dict: Path | None = None,
    n_state: int = 3,
) -> dict[str, int]:
    """Generate untied triphone mdef from transcripts.

    Creates a model definition file containing all triphones observed
    in the training transcripts.

    Args:
        phone_list: Path to phone list file (one phone per line)
        dictionary: Path to pronunciation dictionary
        transcription: Path to transcription file
        output_mdef: Output mdef file path
        filler_dict: Optional filler dictionary
        n_state: Number of emitting states per phone

    Returns:
        Dict with mdef statistics (n_ci, n_tri, n_tied_state)
    """
    phone_list = Path(phone_list)
    dictionary = Path(dictionary)
    transcription = Path(transcription)
    output_mdef = Path(output_mdef)

    # Validate inputs
    validate_files_exist([phone_list, dictionary, transcription], context="untied mdef generation")

    output_mdef.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Generating untied mdef from transcripts...")
    generate_untied_mdef(
        phone_list=phone_list,
        dict_path=dictionary,
        transcript=transcription,
        output=output_mdef,
        filler_dict=filler_dict,
        n_state=n_state,
    )

    # Parse mdef to get stats
    stats = _parse_mdef_stats(output_mdef)
    logger.info(
        "Created untied mdef: %d CI phones, %d triphones, %d tied states",
        stats["n_ci"],
        stats["n_tri"],
        stats["n_tied_state"],
    )

    return stats


def _parse_mdef_stats(mdef_path: Path) -> dict[str, int]:
    """Parse mdef file to extract statistics."""
    n_ci = 0
    n_tri = 0
    n_tied_state = 0

    with mdef_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith("#"):
                # Check header comments for n_tied_state
                if "n_tied_state" in line:
                    parts = line.split()
                    for i, p in enumerate(parts):
                        if p == "n_tied_state" and i + 1 < len(parts):
                            with contextlib.suppress(ValueError):
                                n_tied_state = int(parts[i + 1])
                            break
                continue

            # Non-comment, non-empty line
            parts = line.split()
            if len(parts) >= 2 and parts[0].isdigit():
                # This is a phone entry
                # Base phone only (no - and +) is CI
                if "-" not in parts[0] and "+" not in parts[0]:
                    n_ci += 1
                else:
                    n_tri += 1

    return {"n_ci": n_ci, "n_tri": n_tri, "n_tied_state": n_tied_state}


def run_init_cd_untied(
    ci_model_dir: Path,
    untied_mdef: Path,
    output_dir: Path,
) -> dict[str, Path]:
    """Initialize CD untied model parameters from CI model.

    Creates initial means, variances, mixture weights, and transition matrices
    for the untied CD model by copying CI parameters.

    Args:
        ci_model_dir: Directory containing trained CI model
        untied_mdef: Path to untied triphone mdef
        output_dir: Output directory for CD model

    Returns:
        Dict mapping model file types to paths
    """
    ci_model_dir = Path(ci_model_dir)
    untied_mdef = Path(untied_mdef)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy mdef if not already in place
    dest_mdef = output_dir / "mdef"
    if untied_mdef.resolve() != dest_mdef.resolve():
        shutil.copy(untied_mdef, dest_mdef)

    # Read CI model parameters
    ci_means, n_ci_mgau, n_feat, n_density, veclen = _st2c.read_gau(str(ci_model_dir / "means"))
    ci_vars, _, _, _, _ = _st2c.read_gau(str(ci_model_dir / "variances"))
    ci_mixw_raw, n_mixw, _, _ = _st2c.read_mixw(str(ci_model_dir / "mixture_weights"))
    ci_mixw = ci_mixw_raw.reshape(n_mixw, n_feat, n_density)

    # Parse untied mdef to get number of tied states and mapping
    n_cd_tied, tied_to_ci = _parse_untied_mdef_for_init(untied_mdef)

    logger.info("Initializing %d CD tied states from CI model", n_cd_tied)

    # Initialize CD parameters from CI
    n_veclen = ci_means.shape[-1]
    cd_means = np.zeros((n_cd_tied, n_feat, n_density, n_veclen), dtype=np.float32)
    cd_vars = np.ones((n_cd_tied, n_feat, n_density, n_veclen), dtype=np.float32)
    cd_mixw = np.full((n_cd_tied, n_feat, n_density), 1.0 / n_density, dtype=np.float32)

    # Copy CI parameters to CD tied states based on base phone mapping
    for cd_ts, ci_ts in tied_to_ci.items():
        if ci_ts < ci_means.shape[0]:
            cd_means[cd_ts] = ci_means[ci_ts]
            cd_vars[cd_ts] = ci_vars[ci_ts]
            cd_mixw[cd_ts] = ci_mixw[ci_ts]

    # Write CD model
    _st2c.write_gau(str(output_dir / "means"), cd_means)
    _st2c.write_gau(str(output_dir / "variances"), cd_vars)
    _st2c.write_mixw(str(output_dir / "mixture_weights"), cd_mixw)

    # Copy transition matrices (unchanged)
    shutil.copy(ci_model_dir / "transition_matrices", output_dir / "transition_matrices")

    return {
        "mdef": output_dir / "mdef",
        "means": output_dir / "means",
        "variances": output_dir / "variances",
        "mixture_weights": output_dir / "mixture_weights",
        "transition_matrices": output_dir / "transition_matrices",
    }


def _parse_untied_mdef_for_init(mdef_path: Path) -> tuple[int, dict[int, int]]:
    """Parse untied mdef to get count and mapping for initialization.

    Returns:
        (n_tied_states, mapping from CD tied state -> CI tied state)
    """
    n_tied_state = 0

    # First pass: get header info
    with mdef_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) == 2:
                if parts[1] == "n_tied_state":
                    n_tied_state = int(parts[0])
                elif parts[1] == "n_state_pm":
                    # Not in header, but if it were
                    pass
            # Check for comment with n_state_pm
            if line.startswith("#") and "n_state_pm" in line:
                # Try to parse
                pass

    # Determine n_state from CI entries
    # Mdef format: base lft rt pos attrib tmat s0 s1 s2 N
    # CI phones have "-" for lft and rt

    # Build mapping: for each phone entry, map tied states to their index
    # For CD initialization, we map each triphone's tied states to the
    # corresponding CI phone's tied states

    # Build phone name -> tmat index mapping
    phone_to_tmat: dict[str, int] = {}

    # Build mapping
    tied_to_ci: dict[int, int] = {}

    with mdef_path.open() as f:
        in_data = False
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                if "base" in line and "lft" in line:
                    in_data = True
                continue
            if not in_data:
                # Header lines
                parts = line.split()
                if len(parts) == 1 and parts[0] in ("0.3", "0.4", "0.5"):
                    continue  # Version
                continue

            # Data line
            parts = line.split()
            if len(parts) < 7:
                continue

            base = parts[0]
            lft = parts[1]
            rt = parts[2]
            # pos = parts[3]  # word position
            # attrib = parts[4]
            try:
                tmat = int(parts[5])
            except ValueError:
                continue

            # Get tied states (after tmat, before N)
            tied_states = []
            for p in parts[6:]:
                if p == "N":
                    break
                try:
                    tied_states.append(int(p))
                except ValueError:
                    break

            # CI phone (no context)
            if lft == "-" and rt == "-":
                phone_to_tmat[base] = tmat
                # CI tied states map to themselves
                for ts in tied_states:
                    tied_to_ci[ts] = ts
            else:
                # Triphone - map to CI phone's tied states
                # Find the CI phone's tmat
                if base in phone_to_tmat:
                    ci_tmat = phone_to_tmat[base]
                    n_state_here = len(tied_states)
                    for i, ts in enumerate(tied_states):
                        # Map to corresponding CI state
                        ci_ts = ci_tmat * n_state_here + i
                        tied_to_ci[ts] = ci_ts

    return (n_tied_state, tied_to_ci)


def run_make_questions(
    ci_model_dir: Path,
    output_path: Path,
    continuous: bool = True,
) -> Path:
    """Generate phonetic questions for decision tree building.

    Clusters CI phone distributions to create phonetic question sets.

    Args:
        ci_model_dir: Directory containing trained CI model
        output_path: Output questions file path
        continuous: Use continuous HMM mode

    Returns:
        Path to generated questions file
    """
    ci_model_dir = Path(ci_model_dir)
    output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Generating phonetic questions...")
    make_quests(
        mdef_path=ci_model_dir / "mdef",
        mixw_path=ci_model_dir / "mixture_weights",
        output_path=output_path,
        mean_path=ci_model_dir / "means" if continuous else None,
        var_path=ci_model_dir / "variances" if continuous else None,
        continuous=continuous,
    )

    logger.info("Generated questions: %s", output_path)
    return output_path


# Phones excluded from decision-tree clustering (silence + filler).
# Anything starting with "+" is also filtered.
TREE_SKIP_PHONES = frozenset({"SIL", "+BREATH+", "+COUGH+", "+NOISE+", "+SMACK+", "+UH+", "+UM+"})


def filter_tree_phones(phone_list: Path) -> list[str]:
    """Phones eligible for decision-tree building.

    Drops silence and filler phones. The pipeline runner reads this list
    at plan time to fan out per-(phone, state) tree-build tasks.
    """
    with phone_list.open() as f:
        phones = [line.strip() for line in f if line.strip()]
    return [p for p in phones if p not in TREE_SKIP_PHONES and not p.startswith("+")]


def build_tree_one(
    untied_model_dir: Path,
    questions_path: Path,
    output_path: Path,
    phone: str,
    state: int,
    continuous: bool = True,
) -> None:
    """Build a single decision tree file for one (phone, state).

    On C-level failure, writes a trivial tree placeholder rather than
    raising; this matches the prior `run_build_trees` behavior so a
    couple of pathological phones don't bring the whole pipeline down.
    """
    try:
        build_tree(
            mdef_path=untied_model_dir / "mdef",
            mixw_path=untied_model_dir / "mixture_weights",
            pset_path=questions_path,
            output_path=output_path,
            phone=phone,
            state=state,
            mean_path=untied_model_dir / "means" if continuous else None,
            var_path=untied_model_dir / "variances" if continuous else None,
            continuous=continuous,
        )
    except RuntimeError as exc:
        logger.warning("Failed to build tree for %s state %d: %s", phone, state, exc)
        output_path.write_text(f"# Trivial tree for {phone} state {state}\n")


def run_build_trees(
    untied_model_dir: Path,
    questions_path: Path,
    output_dir: Path,
    phone_list: Path,
    n_state: int = 3,
    continuous: bool = True,
) -> Path:
    """Build decision trees for all phones, one per phone per state.

    Sequential implementation; the pipeline runner provides a
    parallelized per-(phone, state) fan-out version. Use this directly
    only for non-pipeline callers.
    """
    untied_model_dir = Path(untied_model_dir)
    questions_path = Path(questions_path)
    output_dir = Path(output_dir)
    phone_list = Path(phone_list)

    output_dir.mkdir(parents=True, exist_ok=True)
    phones = filter_tree_phones(phone_list)
    logger.info("Building trees for %d phones x %d states...", len(phones), n_state)

    for phone in phones:
        for state in range(n_state):
            tree_file = output_dir / f"{phone}-{state}.dtree"
            build_tree_one(
                untied_model_dir=untied_model_dir,
                questions_path=questions_path,
                output_path=tree_file,
                phone=phone,
                state=state,
                continuous=continuous,
            )

    logger.info("Built %d trees in %s", len(phones) * n_state, output_dir)
    return output_dir


def run_prune_trees(
    untied_mdef: Path,
    questions_path: Path,
    input_tree_dir: Path,
    output_tree_dir: Path,
    n_senones: int,
) -> Path:
    """Prune decision trees to target number of senones.

    Args:
        untied_mdef: Untied CD model definition file
        questions_path: Questions file
        input_tree_dir: Input trees directory
        output_tree_dir: Output (pruned) trees directory
        n_senones: Target number of senones

    Returns:
        Path to pruned trees directory
    """
    untied_mdef = Path(untied_mdef)
    questions_path = Path(questions_path)
    input_tree_dir = Path(input_tree_dir)
    output_tree_dir = Path(output_tree_dir)

    output_tree_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Pruning trees to %d senones...", n_senones)

    prune_tree(
        mdef_path=untied_mdef,
        pset_path=questions_path,
        input_tree_dir=input_tree_dir,
        output_tree_dir=output_tree_dir,
        n_seno_target=n_senones,
        allphones=False,
    )

    logger.info("Pruned trees in %s", output_tree_dir)
    return output_tree_dir


def run_tiestate(
    untied_mdef: Path,
    tree_dir: Path,
    questions_path: Path,
    output_mdef: Path,
) -> Path:
    """Tie states using decision trees.

    Creates a tied mdef by clustering triphone states using decision trees.

    Args:
        untied_mdef: Input untied mdef
        tree_dir: Directory containing decision tree files
        questions_path: Questions file path
        output_mdef: Output tied mdef path

    Returns:
        Path to tied mdef
    """
    untied_mdef = Path(untied_mdef)
    tree_dir = Path(tree_dir)
    questions_path = Path(questions_path)
    output_mdef = Path(output_mdef)

    output_mdef.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Tying states using decision trees...")
    tie_states(
        input_mdef_path=untied_mdef,
        output_mdef_path=output_mdef,
        tree_dir=tree_dir,
        pset_path=questions_path,
        allphones=False,
    )

    # Get stats
    stats = _parse_mdef_stats(output_mdef)
    logger.info(
        "Created tied mdef: %d senones",
        stats["n_tied_state"],
    )

    return output_mdef


def run_init_cd_tied(
    ci_model_dir: Path,
    tied_mdef: Path,
    output_dir: Path,
    continuous: bool = True,
) -> dict[str, Path]:
    """Initialize tied CD model from CI model.

    Maps CI parameters to tied CD states based on the tied mdef.

    Args:
        ci_model_dir: Directory containing trained CI model (e.g., ci-8g)
        tied_mdef: Tied model definition file (output of tiestate)
        output_dir: Output directory for tied model
        continuous: Use continuous HMM mode

    Returns:
        Dict mapping model file types to paths
    """
    from st2.lib.dtree import init_mixw

    ci_model_dir = Path(ci_model_dir)
    tied_mdef = Path(tied_mdef)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing tied CD model from CI...")

    init_mixw(
        src_mdef_path=ci_model_dir / "mdef",
        src_mixw_path=ci_model_dir / "mixture_weights",
        src_mean_path=ci_model_dir / "means",
        src_var_path=ci_model_dir / "variances",
        src_tmat_path=ci_model_dir / "transition_matrices",
        dest_mdef_path=tied_mdef,
        dest_mixw_path=output_dir / "mixture_weights",
        dest_mean_path=output_dir / "means",
        dest_var_path=output_dir / "variances",
        dest_tmat_path=output_dir / "transition_matrices",
        continuous=continuous,
    )

    # Copy tied mdef
    dest_mdef = output_dir / "mdef"
    if tied_mdef.resolve() != dest_mdef.resolve():
        shutil.copy(tied_mdef, dest_mdef)

    logger.info("Initialized tied CD model in %s", output_dir)

    return {
        "mdef": output_dir / "mdef",
        "means": output_dir / "means",
        "variances": output_dir / "variances",
        "mixture_weights": output_dir / "mixture_weights",
        "transition_matrices": output_dir / "transition_matrices",
    }


def _parse_state_tying(untied_mdef: Path, tied_mdef: Path) -> dict[int, int]:
    """Parse state tying mapping from mdef files.

    Returns mapping from untied tied state IDs to tied tied state IDs.
    """
    # Simple identity mapping for now - in reality would need to parse
    # both mdefs and match phones/states
    mapping: dict[int, int] = {}

    # Get number of tied states in each mdef
    untied_n = _count_tied_states(untied_mdef)
    tied_n = _count_tied_states(tied_mdef)

    # For now, simple modulo mapping (real impl would use tree outputs)
    for i in range(untied_n):
        mapping[i] = i % tied_n if tied_n > 0 else 0

    return mapping


def _count_tied_states(mdef_path: Path) -> int:
    """Count number of tied states in mdef."""
    max_ts = -1
    with mdef_path.open() as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            parts = line.split()
            if len(parts) >= 5 and parts[0] not in ("0.3", "0.4", "0.5"):
                try:
                    tied_states = [int(p) for p in parts[4:] if p != "N"]
                    if tied_states:
                        max_ts = max(max_ts, max(tied_states))
                except ValueError:
                    continue
    return max_ts + 1
