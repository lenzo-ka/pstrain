"""Serialization of training feature parameters for Sphinx decoders."""

from __future__ import annotations

from pathlib import Path

from pstrain.lib.pipeline.context import FeatParams

# One ordered declaration drives both serialization and the waveform-extractor
# projection. ``extractor_name=None`` classifies a downstream transform field or
# a second native spelling of an already-projected value.
_FIELD_SPECS: tuple[tuple[str, str | None, str, str | None], ...] = (
    ("-samprate", "samprate", "number", "samprate"),
    ("-ncep", "ncep", "number", "ncep"),
    ("-ceplen", "ncep", "number", None),
    ("-nfilt", "nfilt", "number", "nfilt"),
    ("-nfft", "nfft", "number", "nfft"),
    ("-lowerf", "lowerf", "number", "lowerf"),
    ("-upperf", "upperf", "number", "upperf"),
    ("-alpha", "alpha", "number", "alpha"),
    ("-dither", "dither", "boolean", "dither"),
    ("-seed", "seed", "number", "seed"),
    ("-remove_dc", "remove_dc", "boolean", "remove_dc"),
    ("-remove_noise", "remove_noise", "boolean", "remove_noise"),
    ("-frate", "frate", "number", "frate"),
    ("-wlen", "wlen", "number", "wlen"),
    ("-feat", "feat_type", "number", None),
    ("-transform", "transform", "number", "transform"),
    ("-lifter", "lifter", "number", "lifter"),
    ("-agc", "agc", "number", None),
    ("-cmn", "cmn", "number", None),
    ("-cmninit", "cmninit", "number", None),
    ("-varnorm", "varnorm", "number", None),
    ("-unit_area", None, "yes", None),
    ("-round_filters", None, "yes", None),
)
EXTRACTOR_FIELDS = tuple(spec[3] for spec in _FIELD_SPECS if spec[3] is not None)


FeatureValue = int | float | bool | str


def feature_extractor_config(feat: FeatParams) -> dict[str, FeatureValue]:
    """Project the authoritative training record into waveform extraction args."""
    return {
        extractor_name: getattr(feat, attribute)
        for _, attribute, _, extractor_name in _FIELD_SPECS
        if extractor_name is not None and attribute is not None
    }


def _native_bool(value: str) -> bool:
    return value[0] in "ytYT1"


def feature_extractor_config_from_record(record: dict[str, str]) -> dict[str, FeatureValue]:
    """Project a validated Sphinx record into waveform extraction args."""
    defaults = FeatParams()
    return {
        extractor_name: _record_value(record[flag], getattr(defaults, attribute))
        for flag, attribute, formatting, extractor_name in _FIELD_SPECS
        if extractor_name is not None and attribute is not None
    }


def _record_value(value: str, exemplar: FeatureValue) -> FeatureValue:
    """Convert a validated token to the declared field's Python type."""
    if isinstance(exemplar, bool):
        return _native_bool(value)
    if isinstance(exemplar, int):
        return int(value)
    if isinstance(exemplar, float):
        return float(value)
    return value


def _validate_honored_values(feat: FeatParams) -> None:
    """Reject settings that the native training engine hardcodes."""
    honored = {
        "feat_type": "1s_c_d_dd",
        "agc": "none",
        "cmn": "batch",
        "cmninit": "40,3,-1",
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
    lines = []
    for flag, attribute, formatting, _ in _FIELD_SPECS:
        value = "yes" if attribute is None else getattr(feat, attribute)
        if formatting == "boolean":
            value = "yes" if value else "no"
        lines.append(f"{flag} {value}\n")
    return lines


def write_feat_params(path: Path, feat: FeatParams) -> Path:
    """Write the canonical Sphinx ``feat.params`` representation."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(feat_params_lines(feat)))
    return path
