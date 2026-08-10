"""Export alignment results to various formats.

Supported formats:
- Praat TextGrid (word + phone tiers)
- CTM (NIST Conversation Time Marked, one row per token)
- Sphinx segmentation files (wdseg/phseg style)
"""

from __future__ import annotations

from pathlib import Path

from pstrain.lib.alignment.core import AlignedSegment, AlignmentResult


def to_textgrid(
    result: AlignmentResult,
    frame_shift: float = 0.01,
    include_states: bool = False,
) -> str:
    """Render an alignment as a Praat TextGrid (long format).

    Produces a TextGrid with a "words" tier and a "phones" tier (and optionally
    a "states" tier). Empty regions before/after segments and between adjacent
    non-touching segments are emitted as empty intervals so the tier covers the
    full utterance duration.

    Args:
        result: Alignment to render.
        frame_shift: Seconds per frame (default 0.01 = 10ms, Sphinx standard).
        include_states: If True, include a state-level tier.

    Returns:
        TextGrid file contents as a UTF-8 string.
    """
    xmax = result.duration_time(frame_shift)

    tiers: list[tuple[str, list[AlignedSegment]]] = [
        ("words", result.words),
        ("phones", result.phones),
    ]
    if include_states:
        tiers.append(("states", result.states))

    lines: list[str] = []
    lines.append('File type = "ooTextFile"')
    lines.append('Object class = "TextGrid"')
    lines.append("")
    lines.append("xmin = 0")
    lines.append(f"xmax = {xmax:.4f}")
    lines.append("tiers? <exists>")
    lines.append(f"size = {len(tiers)}")
    lines.append("item []:")

    for tier_idx, (tier_name, segments) in enumerate(tiers, start=1):
        intervals = _segments_to_intervals(segments, xmax, frame_shift)
        lines.append(f"    item [{tier_idx}]:")
        lines.append('        class = "IntervalTier"')
        lines.append(f'        name = "{tier_name}"')
        lines.append("        xmin = 0")
        lines.append(f"        xmax = {xmax:.4f}")
        lines.append(f"        intervals: size = {len(intervals)}")
        for i, (start, end, label) in enumerate(intervals, start=1):
            lines.append(f"        intervals [{i}]:")
            lines.append(f"            xmin = {start:.4f}")
            lines.append(f"            xmax = {end:.4f}")
            lines.append(f'            text = "{_escape_textgrid(label)}"')

    return "\n".join(lines) + "\n"


def to_ctm(
    result: AlignmentResult,
    channel: str = "A",
    frame_shift: float = 0.01,
    level: str = "words",
) -> str:
    """Render an alignment as a CTM (Conversation Time Marked) file.

    CTM is a simple whitespace-delimited format used by NIST scoring tools:

        <file> <channel> <start_sec> <duration_sec> <token> [confidence]

    Args:
        result: Alignment to render.
        channel: Channel identifier (default "A").
        frame_shift: Seconds per frame.
        level: Which segment level to write ("words" or "phones").

    Returns:
        CTM file contents as a string (one row per segment).
    """
    if level == "words":
        segments = result.words
    elif level == "phones":
        segments = result.phones
    else:
        raise ValueError(f"Unsupported CTM level: {level!r} (use 'words' or 'phones')")

    file_id = result.utterance_id or "utt"

    rows: list[str] = []
    for seg in segments:
        start = seg.start_time(frame_shift)
        duration = seg.duration_time(frame_shift)
        rows.append(f"{file_id} {channel} {start:.3f} {duration:.3f} {seg.name}")
    return "\n".join(rows) + ("\n" if rows else "")


def to_sphinx_segments(
    result: AlignmentResult,
    level: str = "words",
) -> str:
    """Render an alignment in Sphinx wdseg/phseg text format.

    Mirrors the human-readable output of ``sphinx3_align``:

        SFrm EFrm SegScore Word
        ...
        Total score: <total>

    Args:
        result: Alignment to render.
        level: Which segment level to write ("words" or "phones").

    Returns:
        Text suitable for writing to a ``.wdseg`` or ``.phseg`` file.
    """
    if level == "words":
        segments = result.words
    elif level == "phones":
        segments = result.phones
    else:
        raise ValueError(f"Unsupported sphinx segment level: {level!r} (use 'words' or 'phones')")

    lines: list[str] = ["\tSFrm\tEFrm\tSegScore\tWord"]
    for seg in segments:
        lines.append(f"\t{seg.start_frame}\t{seg.end_frame}\t{seg.score}\t{seg.name}")
    lines.append(f" Total score: {result.total_score}")
    return "\n".join(lines) + "\n"


def save_textgrid(
    result: AlignmentResult,
    path: Path,
    frame_shift: float = 0.01,
    include_states: bool = False,
) -> None:
    """Write ``result`` as a Praat TextGrid file at ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        to_textgrid(result, frame_shift=frame_shift, include_states=include_states),
        encoding="utf-8",
    )


def save_ctm(
    result: AlignmentResult,
    path: Path,
    channel: str = "A",
    frame_shift: float = 0.01,
    level: str = "words",
) -> None:
    """Write ``result`` as a CTM file at ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        to_ctm(result, channel=channel, frame_shift=frame_shift, level=level),
        encoding="utf-8",
    )


def _segments_to_intervals(
    segments: list[AlignedSegment],
    xmax: float,
    frame_shift: float,
) -> list[tuple[float, float, str]]:
    """Convert a list of segments into Praat-style intervals covering [0, xmax].

    Gaps between segments (or before/after the first/last segment) are emitted
    as empty-label intervals. Adjacent segments whose times exactly match are
    not separated by a synthetic gap.
    """
    intervals: list[tuple[float, float, str]] = []
    cursor = 0.0

    for seg in segments:
        start = seg.start_time(frame_shift)
        end = seg.end_time(frame_shift)
        if start > cursor + 1e-9:
            intervals.append((cursor, start, ""))
        intervals.append((start, end, seg.name))
        cursor = end

    if cursor < xmax - 1e-9:
        intervals.append((cursor, xmax, ""))

    # If there were no segments at all, emit a single empty interval so the
    # tier is still well-formed.
    if not intervals:
        intervals.append((0.0, xmax, ""))

    return intervals


def _escape_textgrid(label: str) -> str:
    """Escape a label for the Praat TextGrid long format."""
    return label.replace('"', '""')
