# Testing with CMU Arctic Corpus

**NOTE: This document is for development/testing purposes only. Do NOT commit scripts or downloaded data to the repository.**

## Overview

CMU Arctic is a standard speech corpus for testing acoustic model training. It provides clean, read speech from multiple speakers with high-quality transcriptions.

**Source:** http://festvox.org/cmu_arctic/cmu_arctic/packed/

## CMU Arctic Structure

### Available Speakers
- `slt` - Female speaker (114MB)
- `bdl` - Male speaker (90MB)
- `clb` - Female speaker (124MB)
- `awb` - Male speaker (123MB)
- `rms` - Male speaker (111MB)
- `ksp` - Male speaker (114MB)
- `jmk` - Male speaker (87MB)

### Packed Archive Format
- Files: `cmu_us_{speaker}_arctic-0.95-release.tar.bz2`
- Contains: Audio files, transcriptions, prompts, etc.

## Process for Testing

### Step 1: Download CMU Arctic Data

**Manual download (for testing):**
```bash
# Download a speaker (e.g., slt)
cd /tmp  # or wherever you want test data
wget http://festvox.org/cmu_arctic/cmu_arctic/packed/cmu_us_slt_arctic-0.95-release.tar.bz2
tar -xjf cmu_us_slt_arctic-0.95-release.tar.bz2
```

**What's in the archive:**
- `wav/` - Audio files (16kHz WAV)
- `etc/` - Configuration and transcription files
- `prompt/` - Prompt files
- Other metadata

### Step 2: Extract Required Files

From the CMU Arctic archive and external sources, we need:

1. **Transcription file** (`etc/txt.done.data`)
   - Source: `cmu_us_{speaker}_arctic/etc/txt.done.data`
   - Format: `( arctic_a0001 "transcript text here" )`
   - **Action:** Coerce into normalized transcripts
   - Target: `project/etc/all.transcription`
   - Output format: `<fileid> <word1> <word2> ...` or `<s> words </s> (fileid)`

2. **Audio files** (`wav/` directory)
   - Source: `cmu_us_{speaker}_arctic/wav/*.wav`
   - Target: `project/audio/{fileid}.wav`
   - Fileid is base name (e.g., `arctic_a0001`)
   - Must match fileids in transcription file

3. **Pronunciation dictionary** (`shared/dictionary.dict`)
   - Source: CMUDict from GitHub
   - URL: https://github.com/cmusphinx/cmudict (or raw file)
   - Must contain all words used in normalized transcripts
   - Format: `<word> <phone1> <phone2> ...`

4. **Phoneset** (`shared/phoneset.txt`)
   - Extract from CMUDict automatically
   - Or use phoneset file from CMUDict repository

### Step 3: Download CMUDict

**CMUDict from GitHub:**
```bash
# Download CMUDict
wget https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.dict
# Or clone the repository
git clone https://github.com/cmusphinx/cmudict.git
```

**CMUDict format:**
- One entry per line: `<WORD> <phone1> <phone2> ...`
- Stress markers: `AH0`, `AH1`, `AH2` (primary, secondary, no stress)
- May need to remove stress markers for acoustic training: `AH0` → `AH`

**Phoneset:**
- Can be extracted from CMUDict automatically
- Or use phoneset file from CMUDict repository if available

### Step 4: Convert CMU Arctic Format

**Input:** `etc/txt.done.data` from CMU Arctic
- Format: `( arctic_a0001 "transcript text here" )`
- One line per utterance

**Conversion process:**
1. Parse `txt.done.data` to extract fileid and text
2. **Normalize transcript:**
   - Lowercase
   - Remove punctuation (keep apostrophes/hyphens for contractions/compounds)
   - Clean up multiple spaces
   - Tokenize into words
3. Create transcription file: `<fileid> <word1> <word2> ...`
   - Or Sphinx format: `<s> words </s> (fileid)`
4. Ensure audio files are named correctly: `{fileid}.wav`

### Step 5: Set Up Project

```bash
pstrain setup test_arctic \
  --transcription /path/to/converted/all.transcription \
  --audio /path/to/cmu_arctic/wav \
  --dictionary /path/to/cmudict.dict \
  --phoneset /path/to/phoneset.txt  # or extract from dictionary
```

**Required inputs:**
- `etc/txt.done.data` → converted to normalized `all.transcription`
- `wav/*.wav` files → copied/linked to `audio/{fileid}.wav`
- CMUDict from GitHub → `shared/dictionary.dict`
- Phoneset (extracted from CMUDict) → `shared/phoneset.txt`

## Implementation Notes

### For Development/Testing Only

**Do NOT commit:**
- Downloaded CMU Arctic archives
- Extracted CMU Arctic data
- Conversion scripts (keep local only)
- Test project directories

**Can commit:**
- Documentation of the process (this file)
- Design decisions about format conversion
- Integration with project setup

### Conversion Process (Local Script)

**Example conversion script (local only, don't commit):**
```python
#!/usr/bin/env python3
"""Convert CMU Arctic txt.done.data to normalized transcription format.

LOCAL USE ONLY - DO NOT COMMIT

Requirements:
- Parse etc/txt.done.data from CMU Arctic
- Normalize transcripts (lowercase, remove punctuation, tokenize)
- Output: fileid + normalized words
"""

import re
from pathlib import Path

def normalize_transcript(text: str) -> str:
    """Normalize transcript for acoustic training.

    - Lowercase
    - Remove punctuation (keep apostrophes/hyphens)
    - Clean up multiple spaces
    """
    text = text.lower()
    # Remove punctuation but keep apostrophes and hyphens
    text = re.sub(r'[.,!?;:"()]', "", text)
    # Clean up multiple spaces
    text = " ".join(text.split())
    return text

def convert_arctic_transcripts(txt_done_data: Path, output: Path) -> None:
    """Convert CMU Arctic txt.done.data to normalized transcription format.

    Input format: ( arctic_a0001 "transcript text here" )
    Output format: arctic_a0001 word1 word2 word3
    """
    with open(txt_done_data, encoding="utf-8") as f_in, \
         open(output, "w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue

            # Format: ( arctic_a0001 "text here" )
            match = re.match(r'\(\s*(\S+)\s+"([^"]+)"\s*\)', line)
            if match:
                fileid = match.group(1)
                text = match.group(2)

                # Normalize transcript
                normalized = normalize_transcript(text)

                # Write as: fileid word1 word2 ...
                f_out.write(f"{fileid} {normalized}\n")
```

### Integration with Project Setup

The project setup process should:
1. Accept CMU Arctic data (or any corpus) as input
2. Validate fileid matching between transcription and audio
3. Create proper project structure
4. Extract phoneset from dictionary

**No special CMU Arctic handling needed** - just standard project setup with:
- Transcription file (converted from CMU Arctic format)
- Audio files (renamed to match fileids)
- Dictionary (CMUDict or filtered)
- Phoneset (from dictionary)

## Testing Workflow

1. **Download CMU Arctic** (manual, local)
   - Download: `cmu_us_{speaker}_arctic-0.95-release.tar.bz2`
   - Extract: `etc/txt.done.data` and `wav/*.wav`

2. **Download CMUDict** (from GitHub)
   - Source: https://github.com/cmusphinx/cmudict
   - File: `cmudict.dict` (or raw URL)

3. **Convert format** (local script, don't commit)
   - Parse `etc/txt.done.data`
   - Normalize transcripts (lowercase, remove punctuation)
   - Create `all.transcription` with fileid + normalized words

4. **Set up project** using `pstrain setup`
   - Transcription: converted `all.transcription`
   - Audio: `wav/*.wav` files
   - Dictionary: CMUDict
   - Phoneset: extract from CMUDict

5. **Run CI training** workflow
6. **Validate** model training works

## Required Files Summary

**From CMU Arctic archive:**
- `etc/txt.done.data` → Convert to normalized transcripts
- `wav/*.wav` → Audio files (named by fileid)

**From GitHub (CMUDict):**
- `cmudict.dict` → Pronunciation dictionary
- Phoneset (extract from dictionary or use phoneset file)

## Notes

- CMU Arctic is ~100-125MB per speaker
- Good for testing: clean speech, good transcriptions
- Can test with subset (e.g., first 100 utterances)
- Standard benchmark corpus for speech recognition
