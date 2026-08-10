"""Serialization of training feature parameters for Sphinx decoders."""

from __future__ import annotations

from pathlib import Path

from st2.lib.pipeline.context import FeatParams


def _validate_honored_values(feat: FeatParams) -> None:
    """Reject settings that the native training engine hardcodes."""
    honored = {
        "transform": "dct",
        "feat_type": "1s_c_d_dd",
        "agc": "none",
        "varnorm": "no",
    }
    for field, actual in honored.items():
        requested = getattr(feat, field)
        if requested != actual:
            raise ValueError(
                f"{field}={requested!r} cannot be serialized truthfully: "
                f"the training engine hardcodes {field}={actual!r}"
            )


def feat_params_lines(feat: FeatParams) -> list[str]:
    """Return a complete Sphinx ``feat.params`` for *feat*."""
    _validate_honored_values(feat)
    return [
        f"-samprate {feat.samprate}\n",
        f"-ncep {feat.ncep}\n",
        f"-nfilt {feat.nfilt}\n",
        f"-nfft {feat.nfft}\n",
        f"-lowerf {feat.lowerf}\n",
        f"-upperf {feat.upperf}\n",
        f"-alpha {feat.alpha}\n",
        f"-feat {feat.feat_type}\n",
        f"-transform {feat.transform}\n",
        f"-lifter {feat.lifter}\n",
        f"-agc {feat.agc}\n",
        f"-cmn {feat.cmn}\n",
        f"-varnorm {feat.varnorm}\n",
        # Invariants of ST2's extraction path: unit_area/round_filters are the
        # sphinxbase fe defaults it never overrides, and st2_fe_create pins
        # remove_dc/remove_noise in its synthetic command line
        # (csrc/libs/libst2/st2_fe.c). Writing them explicitly prevents
        # decoder-version defaults drifting from the training front end.
        "-unit_area yes\n",
        "-round_filters yes\n",
        "-remove_dc no\n",
        "-remove_noise yes\n",
    ]


def write_feat_params(path: Path, feat: FeatParams) -> Path:
    """Write the canonical Sphinx ``feat.params`` representation."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(feat_params_lines(feat)))
    return path
