"""Validation and input preparation for the one-command training workflow."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import unicodedata
import wave
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath

from pstrain.lib.dictionary import Dictionary
from pstrain.lib.phoneset import Phoneset

PROMPT_FORMATS = ("auto", "leading-id", "sphinx", "tsv", "csv")
_SPHINX_RE = re.compile(r"^<s>\s+(.*?)\s+</s>\s+\(([^)]+)\)\s*$")


class PromptFormatError(ValueError):
    """A prompt file is malformed or its format cannot be detected safely."""


@dataclass(frozen=True)
class Prompt:
    fileid: str
    text: str
    line: int


@dataclass
class InputReport:
    prompt_format: str = ""
    prompt_count: int = 0
    audio_count: int = 0
    dictionary_words: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    duplicate_ids: list[str] = field(default_factory=list)
    missing_audio: list[str] = field(default_factory=list)
    extra_audio: list[str] = field(default_factory=list)
    oov: dict[str, dict[str, object]] = field(default_factory=dict)
    wav_properties: dict[str, object] = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, object]:
        return {**asdict(self), "valid": self.valid}


def _nonempty_lines(path: Path) -> list[tuple[int, str]]:
    return [
        (number, raw.strip())
        for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if raw.strip()
    ]


def detect_prompt_format(path: Path) -> str:
    """Detect only formats having an unambiguous lexical signature."""
    lines = _nonempty_lines(path)
    if not lines:
        raise PromptFormatError(f"Prompt file is empty: {path}")
    values = [line for _, line in lines]
    sphinx = [_SPHINX_RE.fullmatch(line) is not None for line in values]
    if any(sphinx):
        if not all(sphinx):
            raise PromptFormatError("Mixed Sphinx and non-Sphinx prompt lines; use one format")
        return "sphinx"
    if any("\t" in line for line in values):
        if not all(line.count("\t") == 1 for line in values):
            raise PromptFormatError("Ambiguous TSV prompts; pass --prompt-format explicitly")
        return "tsv"
    if any("," in line for line in values):
        if all(line.startswith('"') for line in values):
            rows = [next(csv.reader([line], delimiter=",", strict=True)) for line in values]
            if all(len(row) == 2 for row in rows):
                return "csv"
        raise PromptFormatError(
            "Comma-containing prompts are ambiguous; pass --prompt-format csv or leading-id"
        )
    return "leading-id"


def parse_prompts(path: Path, prompt_format: str = "auto") -> tuple[str, list[Prompt]]:
    """Parse prompts without normalizing or otherwise changing their text."""
    selected = detect_prompt_format(path) if prompt_format == "auto" else prompt_format
    if selected not in PROMPT_FORMATS[1:]:
        raise PromptFormatError(f"Unknown prompt format {selected!r}")
    lines = _nonempty_lines(path)
    prompts: list[Prompt] = []
    for number, line in lines:
        fileid = text = ""
        if selected == "sphinx":
            match = _SPHINX_RE.fullmatch(line)
            if match is None:
                raise PromptFormatError(f"Line {number}: expected <s> WORDS </s> (fileid)")
            text, fileid = match.groups()
        elif selected in {"tsv", "csv"}:
            delimiter = "\t" if selected == "tsv" else ","
            row = next(csv.reader([line], delimiter=delimiter, strict=True))
            if len(row) != 2:
                raise PromptFormatError(
                    f"Line {number}: expected exactly two {selected.upper()} fields"
                )
            fileid, text = (value.strip() for value in row)
        else:
            parts = line.split(None, 1)
            if len(parts) != 2:
                raise PromptFormatError(f"Line {number}: expected 'fileid WORDS'")
            fileid, text = parts
        prompts.append(Prompt(fileid=fileid, text=text, line=number))
    return selected, prompts


def _unsafe_fileid(fileid: str) -> bool:
    path = PurePosixPath(fileid)
    return (
        not fileid
        or "\\" in fileid
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or fileid.endswith(".wav")
    )


def validate_inputs(
    audio_dir: Path,
    prompts_path: Path,
    dictionary_path: Path,
    prompt_format: str,
    phoneset_path: Path | None = None,
    filler_path: Path | None = None,
) -> tuple[InputReport, list[Prompt]]:
    """Fully validate corpus inputs before project setup or training."""
    report = InputReport()
    selected, prompts = parse_prompts(prompts_path, prompt_format)
    report.prompt_format = selected
    report.prompt_count = len(prompts)
    ids = [prompt.fileid for prompt in prompts]
    counts = Counter(ids)
    report.duplicate_ids = sorted(fileid for fileid, count in counts.items() if count > 1)
    if report.duplicate_ids:
        report.errors.append(f"Duplicate prompt IDs: {len(report.duplicate_ids)}")
    unsafe = sorted({fileid for fileid in ids if _unsafe_fileid(fileid)})
    if unsafe:
        report.errors.append(f"Unsafe prompt IDs: {len(unsafe)} (e.g. {unsafe[0]!r})")
    empty = [prompt.line for prompt in prompts if not prompt.text.strip()]
    if empty:
        report.errors.append(f"Empty prompts: {len(empty)} (e.g. line {empty[0]})")

    wavs = sorted(audio_dir.rglob("*.wav"))
    audio_ids = {path.relative_to(audio_dir).with_suffix("").as_posix() for path in wavs}
    report.audio_count = len(wavs)
    report.missing_audio = sorted(set(ids) - audio_ids)
    report.extra_audio = sorted(audio_ids - set(ids))
    if report.missing_audio:
        report.errors.append(
            f"Prompt IDs with no WAV: {len(report.missing_audio)} "
            f"(e.g. {report.missing_audio[0]}.wav)"
        )
    if report.extra_audio:
        report.errors.append(
            f"WAVs with no prompt: {len(report.extra_audio)} (e.g. {report.extra_audio[0]}.wav)"
        )

    properties: Counter[tuple[int, int, int]] = Counter()
    for wav_path in wavs:
        try:
            with wave.open(str(wav_path), "rb") as wav_file:
                props = (wav_file.getframerate(), wav_file.getnchannels(), wav_file.getsampwidth())
                properties[props] += 1
                if wav_file.getcomptype() != "NONE":
                    report.errors.append(f"Compressed WAV is unsupported: {wav_path}")
        except (EOFError, wave.Error) as exc:
            report.errors.append(f"Unreadable WAV {wav_path}: {exc}")
    report.wav_properties = {
        f"{rate}Hz/{channels}ch/{width * 8}bit": count
        for (rate, channels, width), count in properties.items()
    }
    unsupported = [props for props in properties if props[0] not in {8000, 16000} or props[1] != 1]
    if unsupported:
        report.errors.append("Unsupported WAV properties; expected mono 8 kHz or 16 kHz PCM")
    if len(properties) > 1:
        report.errors.append(f"Inconsistent WAV properties: {len(properties)} distinct formats")

    dictionary = Dictionary.from_file(dictionary_path)
    filler = Dictionary.from_file(filler_path) if filler_path else None
    if phoneset_path:
        phoneset = Phoneset.from_file(phoneset_path)
        for label, lexicon in (("dictionary", dictionary), ("filler dictionary", filler)):
            if lexicon is None:
                continue
            valid_phones, missing_phones = phoneset.validate_dictionary(lexicon)
            if not valid_phones:
                report.errors.append(
                    f"Phones outside phoneset in {label}: {len(missing_phones)} "
                    f"(e.g. {sorted(missing_phones)[0]})"
                )
    words = dictionary.words()
    report.dictionary_words = len(words)
    occurrences: Counter[str] = Counter()
    examples: dict[str, list[str]] = defaultdict(list)
    for prompt in prompts:
        for token in prompt.text.split():
            if token not in words:
                occurrences[token] += 1
                if len(examples[token]) < 5:
                    examples[token].append(prompt.fileid)
    report.oov = {
        token: {"count": occurrences[token], "examples": examples[token]}
        for token in sorted(occurrences)
    }
    if report.oov:
        report.errors.append(
            f"Out-of-vocabulary tokens: {len(report.oov)} unique, "
            f"{sum(occurrences.values())} occurrences"
        )
        folded = {word.casefold() for word in words}
        if any(token.casefold() in folded for token in report.oov):
            report.warnings.append("Some OOV tokens differ from dictionary entries only by case")
        if any(token.strip(".,!?;:\"'") in words for token in report.oov):
            report.warnings.append("Some OOV tokens have punctuation attached")
        if any(unicodedata.normalize("NFC", token) in words for token in report.oov):
            report.warnings.append("Some OOV tokens differ only by Unicode normalization")
    return report, prompts


def write_validation_reports(project_dir: Path, report: InputReport) -> tuple[Path, Path]:
    reports = project_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    validation_path = reports / "input-validation.json"
    oov_path = reports / "oov.txt"
    validation_path.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
    lines = ["token\tcount\texample_utterances"]
    for token, detail in report.oov.items():
        examples = detail["examples"]
        if not isinstance(examples, list) or not all(isinstance(item, str) for item in examples):
            raise TypeError("OOV examples must be a list of utterance IDs")
        lines.append(f"{token}\t{detail['count']}\t{','.join(examples)}")
    oov_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return validation_path, oov_path


def write_training_transcription(path: Path, prompts: list[Prompt]) -> None:
    path.write_text(
        "".join(f"{prompt.fileid} {prompt.text}\n" for prompt in prompts), encoding="utf-8"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_identity(
    audio_dir: Path,
    prompts_path: Path,
    dictionary_path: Path,
    filler_path: Path | None,
    phoneset_path: Path | None,
    link_audio: bool = False,
) -> dict[str, object]:
    audio = {
        path.relative_to(audio_dir).as_posix(): sha256_file(path)
        for path in sorted(audio_dir.rglob("*.wav"))
    }
    return {
        "audio": audio,
        "prompts_sha256": sha256_file(prompts_path),
        "dictionary_sha256": sha256_file(dictionary_path),
        "filler_sha256": sha256_file(filler_path) if filler_path else None,
        "phoneset_sha256": sha256_file(phoneset_path) if phoneset_path else None,
        "normalization": None,
        "audio_ownership": "link" if link_audio else "copy",
    }
