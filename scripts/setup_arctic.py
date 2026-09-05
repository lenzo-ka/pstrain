#!/usr/bin/env python3
"""Set up CMU Arctic SLT corpus for pstrain training.

This script:
1. Copies wav files to audio/
2. Creates transcription files in Sphinx format
3. Creates train/test fileids (90/10 split)
4. Creates pronunciation dictionary from CMUdict
5. Creates phoneset from dictionary
"""

import re
import shutil
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
ARCTIC_DIR = PROJECT_DIR / "data" / "cmu_us_slt_arctic"
CMUDICT_PATH = PROJECT_DIR / "data" / "cmudict.dict"
AUDIO_DIR = PROJECT_DIR / "audio"
SHARED_DIR = PROJECT_DIR / "shared"
EXPERIMENT_DIR = PROJECT_DIR / "experiments" / "default"


def parse_arctic_transcripts(txt_done_data: Path) -> dict[str, str]:
    """Parse CMU Arctic txt.done.data file.

    Format: ( utterance_id "transcription text" )
    """
    transcripts = {}
    with txt_done_data.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            match = re.match(r'\(\s*(\S+)\s+"(.*)"\s*\)', line)
            if match:
                utt_id = match.group(1)
                text = match.group(2)
                transcripts[utt_id] = text
    return transcripts


def normalize_text(text: str) -> str:
    """Normalize transcript text for ASR training.

    - Convert to uppercase
    - Remove punctuation except apostrophes in contractions
    - Handle special cases
    """
    # First handle contractions - keep them as single words
    text = text.upper()

    # Remove most punctuation but keep apostrophes in words
    text = re.sub(r"[^\w\s']", " ", text)

    # Clean up apostrophes at word boundaries
    text = re.sub(r"\s+'", " ", text)
    text = re.sub(r"'\s+", " ", text)
    text = re.sub(r"^'", "", text)
    text = re.sub(r"'$", "", text)

    # Collapse whitespace
    return " ".join(text.split())



def load_cmudict(dict_path: Path) -> dict[str, list[list[str]]]:
    """Load CMU Pronouncing Dictionary with all variants.

    Returns dict mapping base_word -> list of pronunciations (each is list of phones)
    """
    entries: dict[str, list[list[str]]] = {}
    with dict_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(";;;"):
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            word = parts[0].upper()

            # Parse variant: WORD(2) -> WORD
            base_word = re.sub(r"\(\d+\)$", "", word)

            # Remove stress markers from phones
            phones = [re.sub(r"\d", "", p) for p in parts[1:]]

            # Add to list of pronunciations for this word
            if base_word not in entries:
                entries[base_word] = []
            # Avoid duplicate pronunciations
            if phones not in entries[base_word]:
                entries[base_word].append(phones)

    return entries


def create_dictionary(
    transcripts: dict[str, str],
    cmudict: dict[str, list[list[str]]],
    output_path: Path,
) -> tuple[set[str], set[str]]:
    """Create pronunciation dictionary from transcripts.

    Includes all pronunciation variants from CMUDict.

    Returns (words_found, words_missing)
    """
    # Get all unique words from transcripts
    all_words = set()
    for text in transcripts.values():
        normalized = normalize_text(text)
        all_words.update(normalized.split())

    found = set()
    missing = set()

    with output_path.open("w", encoding="utf-8") as f:
        for word in sorted(all_words):
            if word in cmudict:
                pronunciations = cmudict[word]
                # Write first pronunciation without variant number
                f.write(f"{word} {' '.join(pronunciations[0])}\n")
                # Write additional variants with (2), (3), etc.
                for i, pron in enumerate(pronunciations[1:], start=2):
                    f.write(f"{word}({i}) {' '.join(pron)}\n")
                found.add(word)
            else:
                missing.add(word)

    return found, missing


def write_missing_vocabulary(
    transcripts: dict[str, str], missing: set[str], output_path: Path
) -> int:
    """Write every missing word and the utterances containing it.

    Returns the number of affected utterances. Those utterances remain in the
    generated train/test data; this manifest makes that existing behavior
    visible to the person running setup.
    """
    utterances_by_word = {word: [] for word in missing}
    for utt_id, text in transcripts.items():
        for word in set(normalize_text(text).split()) & missing:
            utterances_by_word[word].append(utt_id)

    with output_path.open("w", encoding="utf-8") as f:
        for word in sorted(utterances_by_word):
            utterance_ids = sorted(utterances_by_word[word])
            f.write("\t".join((word, *utterance_ids)) + "\n")

    return len({utt_id for ids in utterances_by_word.values() for utt_id in ids})


def create_filler_dict(output_path: Path) -> None:
    """Create filler dictionary.

    Matches SphinxTrain's minimal filler dict:
    - <s> and </s> are sentence markers
    - <sil> is optional silence
    """
    with output_path.open("w", encoding="utf-8") as f:
        f.write("<sil> SIL\n")
        f.write("<s> SIL\n")
        f.write("</s> SIL\n")


def create_phoneset(dictionary_path: Path, output_path: Path) -> set[str]:
    """Extract unique phones from dictionary and create phoneset file."""
    phones = set()
    with dictionary_path.open(encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                phones.update(parts[1:])

    # Add silence
    phones.add("SIL")

    sorted_phones = sorted(phones)
    with output_path.open("w", encoding="utf-8") as f:
        for phone in sorted_phones:
            f.write(f"{phone}\n")

    return phones


def main() -> None:
    print("Setting up CMU Arctic SLT corpus for pstrain...")

    # Check prerequisites
    if not ARCTIC_DIR.exists():
        print(f"ERROR: Arctic data not found at {ARCTIC_DIR}")
        print("Download from http://festvox.org/cmu_arctic/")
        return

    if not CMUDICT_PATH.exists():
        print(f"ERROR: CMUdict not found at {CMUDICT_PATH}")
        return

    # Create directories
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    SHARED_DIR.mkdir(parents=True, exist_ok=True)
    (EXPERIMENT_DIR / "etc").mkdir(parents=True, exist_ok=True)

    # 1. Parse transcripts
    print("Parsing transcripts...")
    txt_done_data = ARCTIC_DIR / "etc" / "txt.done.data"
    transcripts = parse_arctic_transcripts(txt_done_data)
    print(f"  Found {len(transcripts)} utterances")

    # 2. Copy wav files
    print("Copying wav files...")
    wav_dir = ARCTIC_DIR / "wav"
    copied = 0
    for utt_id in transcripts:
        src = wav_dir / f"{utt_id}.wav"
        dst = AUDIO_DIR / f"{utt_id}.wav"
        if src.exists() and not dst.exists():
            shutil.copy(src, dst)
            copied += 1
    print(f"  Copied {copied} new wav files")

    # 3. Load CMUdict and create pronunciation dictionary
    print("Loading CMU dictionary...")
    cmudict = load_cmudict(CMUDICT_PATH)
    print(f"  Loaded {len(cmudict)} entries")

    print("Creating pronunciation dictionary...")
    dict_path = SHARED_DIR / "dictionary.dict"
    found, missing = create_dictionary(transcripts, cmudict, dict_path)
    print(f"  Words found: {len(found)}")
    missing_path = EXPERIMENT_DIR / "etc" / "missing_vocabulary.txt"
    affected_utterances = write_missing_vocabulary(transcripts, missing, missing_path)
    print(
        f"missing_vocabulary\twords={len(missing)}\t"
        f"utterances_kept={affected_utterances}\tpath={missing_path}"
    )

    # 4. Create filler dictionary
    print("Creating filler dictionary...")
    filler_path = SHARED_DIR / "filler.dict"
    create_filler_dict(filler_path)

    # 5. Create phoneset
    print("Creating phoneset...")
    phoneset_path = SHARED_DIR / "phoneset.txt"
    phones = create_phoneset(dict_path, phoneset_path)
    print(f"  {len(phones)} phones")

    # 6. Create train/test split (90/10)
    print("Creating train/test split...")
    utt_ids = sorted(transcripts.keys())
    test_ids = set(utt_ids[::10])  # Every 10th utterance (10% for test)
    train_ids = [uid for uid in utt_ids if uid not in test_ids]

    print(f"  Train: {len(train_ids)}, Test: {len(test_ids)}")

    # Write fileids
    with (EXPERIMENT_DIR / "etc" / "train.fileids").open("w") as f:
        for uid in train_ids:
            f.write(f"{uid}\n")

    with (EXPERIMENT_DIR / "etc" / "test.fileids").open("w") as f:
        for uid in sorted(test_ids):
            f.write(f"{uid}\n")

    # Write transcriptions in Sphinx format
    with (EXPERIMENT_DIR / "etc" / "train.transcription").open("w") as f:
        for uid in train_ids:
            text = normalize_text(transcripts[uid])
            f.write(f"<s> {text} </s> ({uid})\n")

    with (EXPERIMENT_DIR / "etc" / "test.transcription").open("w") as f:
        for uid in sorted(test_ids):
            text = normalize_text(transcripts[uid])
            f.write(f"<s> {text} </s> ({uid})\n")

    print("\nSetup complete!")
    print(f"  Audio:         {AUDIO_DIR}")
    print(f"  Dictionary:    {dict_path}")
    print(f"  Phoneset:      {phoneset_path}")
    print(f"  Train fileids: {EXPERIMENT_DIR / 'etc' / 'train.fileids'}")
    print(f"  Test fileids:  {EXPERIMENT_DIR / 'etc' / 'test.fileids'}")


if __name__ == "__main__":
    main()
