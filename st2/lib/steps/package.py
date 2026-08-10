"""Model packaging for distribution.

Creates distributable model packages compatible with PocketSphinx,
Sphinx3, and other Sphinx-based decoders.
"""

from __future__ import annotations

import logging
import shutil
from datetime import datetime
from pathlib import Path

from st2.lib.model import MODEL_FILES_REQUIRED
from st2.lib.pipeline.context import FeatParams
from st2.lib.pipeline.feat_params import write_feat_params

logger = logging.getLogger(__name__)

__all__ = ["package_model", "create_feat_params", "create_noisedict"]

def create_feat_params(
    output_path: Path,
    feat_params: FeatParams,
) -> Path:
    """Create ``feat.params`` from the profile that trained the model."""
    output_path = write_feat_params(output_path, feat_params)
    logger.info("Created feat.params: %s", output_path)
    return output_path


def create_noisedict(
    output_path: Path,
    filler_dict_path: Path | None = None,
) -> Path:
    """Create noisedict file for Sphinx decoders.

    This is the filler dictionary used during decoding.

    Args:
        output_path: Output file path
        filler_dict_path: Source filler dictionary (optional)

    Returns:
        Path to created file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if filler_dict_path and Path(filler_dict_path).exists():
        # Copy existing filler dict
        shutil.copy(filler_dict_path, output_path)
    else:
        # Create minimal noisedict (matches st2/data/filler.dict)
        with open(output_path, "w") as f:
            f.write("<sil> SIL\n")
            f.write("<s> SIL\n")
            f.write("</s> SIL\n")

    logger.info("Created noisedict: %s", output_path)
    return output_path


def package_model(
    model_dir: Path,
    output_dir: Path,
    feat_params: FeatParams,
    model_name: str | None = None,
    dictionary_path: Path | None = None,
    filler_dict_path: Path | None = None,
    include_dict: bool = True,
) -> dict[str, Path]:
    """Package a trained model for distribution.

    Creates a complete, self-contained model directory that can be
    used directly with PocketSphinx and other Sphinx decoders.

    Args:
        model_dir: Source model directory
        output_dir: Output directory for packaged model
        feat_params: Feature profile used to train the model
        model_name: Name for the model (used in output path)
        dictionary_path: Path to pronunciation dictionary
        filler_dict_path: Path to filler dictionary
        include_dict: Whether to include dictionary in package

    Returns:
        Dict mapping file types to output paths

    Example output structure::

        dist/models/my-model/
        ├── acoustic/
        │   ├── feat.params
        │   ├── mdef
        │   ├── means
        │   ├── variances
        │   ├── mixture_weights
        │   ├── transition_matrices
        │   └── noisedict
        ├── dict/
        │   ├── cmudict.dict
        │   └── filler.dict
        └── README.txt
    """
    model_dir = Path(model_dir)
    output_dir = Path(output_dir)

    if model_name:
        package_dir = output_dir / model_name
    else:
        package_dir = output_dir

    # Create directory structure
    acoustic_dir = package_dir / "acoustic"
    acoustic_dir.mkdir(parents=True, exist_ok=True)

    result: dict[str, Path] = {}

    # Copy acoustic model files
    for fname in MODEL_FILES_REQUIRED:
        src = model_dir / fname
        dst = acoustic_dir / fname
        if src.exists():
            shutil.copy(src, dst)
            result[fname] = dst
            logger.debug("Copied %s -> %s", src, dst)
        else:
            logger.warning("Model file not found: %s", src)

    feat_path = create_feat_params(acoustic_dir / "feat.params", feat_params)
    result["feat_params"] = feat_path

    # Create noisedict
    noisedict_path = create_noisedict(
        acoustic_dir / "noisedict",
        filler_dict_path,
    )
    result["noisedict"] = noisedict_path

    # Copy dictionary files if requested
    if include_dict:
        dict_dir = package_dir / "dict"
        dict_dir.mkdir(parents=True, exist_ok=True)

        if dictionary_path and Path(dictionary_path).exists():
            dict_dst = dict_dir / "cmudict.dict"
            shutil.copy(dictionary_path, dict_dst)
            result["dictionary"] = dict_dst
            logger.debug("Copied dictionary: %s", dict_dst)

        if filler_dict_path and Path(filler_dict_path).exists():
            filler_dst = dict_dir / "filler.dict"
            shutil.copy(filler_dict_path, filler_dst)
            result["filler_dict"] = filler_dst
            logger.debug("Copied filler dict: %s", filler_dst)

    # Create README
    readme_path = package_dir / "README.txt"
    _create_readme(readme_path, model_name, feat_params)
    result["readme"] = readme_path

    logger.info("Packaged model to: %s", package_dir)
    return result


def _create_readme(
    output_path: Path,
    model_name: str | None,
    feat_params: FeatParams,
) -> None:
    """Create README file for the model package."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    content = f"""ST2 Acoustic Model Package
==========================

Model: {model_name or "unnamed"}
Created: {now}
Generator: st2 (SphinxTrain 2)

Directory Structure
-------------------
acoustic/       - Acoustic model files for Sphinx decoders
  feat.params   - Feature extraction parameters
  mdef          - Model definition (phones, states, triphones)
  means         - Gaussian means
  variances     - Gaussian variances
  mixture_weights - Mixture weights
  transition_matrices - HMM state transition probabilities
  noisedict     - Filler/noise dictionary for decoding

dict/           - Dictionary files
  cmudict.dict  - Pronunciation dictionary
  filler.dict   - Filler word dictionary

Usage with PocketSphinx
-----------------------
Python:
    from pocketsphinx import Decoder

    config = Decoder.default_config()
    config.set_string('-hmm', '/path/to/{model_name or "model"}/acoustic')
    config.set_string('-dict', '/path/to/{model_name or "model"}/dict/cmudict.dict')
    decoder = Decoder(config)

Command line:
    pocketsphinx -hmm {model_name or "model"}/acoustic \\
                 -dict {model_name or "model"}/dict/cmudict.dict \\
                 -infile audio.wav

Feature Parameters
------------------
Sample rate: {feat_params.samprate} Hz
Mel filters: {feat_params.nfilt}
FFT size: {feat_params.nfft}
Frequency range: {feat_params.lowerf}-{feat_params.upperf} Hz
Cepstral coefficients: {feat_params.ncep}
Feature type: {feat_params.feat_type}

License
-------
See the project repository for license information.
"""
    with open(output_path, "w") as f:
        f.write(content)
