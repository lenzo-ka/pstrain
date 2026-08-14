"""Deliberately incomplete consumer used to demonstrate the model-load boundary gap."""

from pathlib import Path


def build_decoder(model_dir: Path, dictionary: Path) -> object:
    """Construct PocketSphinx directly, bypassing pstrain's model validation."""
    from pocketsphinx import Decoder

    return Decoder(hmm=str(model_dir), dict=str(dictionary))
