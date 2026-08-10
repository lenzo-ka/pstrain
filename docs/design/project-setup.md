# Project Setup Design

This document outlines how to set up a project structure in pstrain.

## Overview

Project setup creates the initial directory structure and prepares the project for training. This is the first step before any training can begin.

## Project Structure

Design for sharing resources across experiments:

```
project/
├── etc/
│   └── config.yaml          # Project configuration
├── audio/                    # Audio files (WAV format) - shared across experiments
├── shared/                   # Shared resources across ALL experiments
│   ├── dictionary.dict      # Main pronunciation dictionary
│   ├── filler.dict          # Filler word dictionary
│   └── features/            # Extracted features (shared across experiments)
│       └── {feature_set_id}/  # Feature set identified by audio+feature config
│           ├── {fileid1}.mfc # Feature files (one per audio file)
│           ├── {fileid2}.mfc
│           ├── ...
│           ├── feat.params   # Feature parameters
│           ├── global_mean    # Global statistics
│           └── global_variance
└── experiments/             # Training experiments (isolated)
    └── {experiment_name}/   # Individual experiment
        ├── models/          # Trained models (experiment-specific)
        │   ├── ci/          # Context-Independent models
        │   │   └── {config}/  # Model configuration (e.g., "baseline", "1g", "lda")
        │   │       └── model/
        │   │           ├── flat/  # Flat initialization models
        │   │           └── hmm/   # Trained CI model files
        │   └── cd/          # Context-Dependent models (future)
        │       └── {config}/
        │           └── model/
        │               └── hmm/   # Trained CD model files
        ├── logs/            # Training logs
        └── work/            # Working directory for training
```

### Key Design Principles

1. **Shared Resources at Project Level**
   - `audio/` - All experiments use the same audio files
   - `shared/dictionary.dict` - All experiments use the same dictionary
   - `shared/features/{feature_set_id}/` - Features shared by experiments with same config

2. **Feature Set Identification**
   - Feature directory name (`feature_set_id`) is deterministic based on:
     - Audio configuration (sample rate, format)
     - Feature extraction configuration (num_ceps, filters, etc.)
   - Experiments with identical audio+feature config share the same feature directory
   - Different feature configs get different directories (e.g., `mfcc_13`, `mfcc_39`, `lda_29`)

3. **Experiment Isolation**
   - Each experiment has its own `models/`, `logs/`, and `work/` directories
   - Experiments can reference shared features without copying them
   - Multiple experiments can run in parallel using the same features

4. **Space Efficiency**
   - Features are computed once per unique (audio, feature) configuration
   - No duplication of features across experiments
   - Experiments only store their unique outputs (models, logs)

## Setup Requirements

### Required Inputs

1. **Configuration file** (`etc/config.yaml`)
   - Project metadata (name, description)
   - Audio settings (sample rate, format)
   - Feature extraction settings
   - Training parameters
   - Dictionary and phoneset paths

2. **Fileid list with transcripts** (`etc/all.transcription` or similar)
   - Format: One line per utterance: `<fileid> <word1> <word2> ...`
   - Example: `arctic_a0001 hello world`
   - Fileid is the base name (without extension) of the audio file
   - Used to map audio files to their transcriptions

3. **Audio files** (in `audio/` directory)
   - Named by fileid: `{fileid}.wav` (e.g., `arctic_a0001.wav`)
   - Must match fileids in transcription file
   - Format: WAV (16kHz recommended for speech)
   - One audio file per fileid

4. **Pronunciation dictionary** (`shared/dictionary.dict`)
   - Format: One entry per line: `<word> <phone1> <phone2> ...`
   - Example: `hello HH AH L OW`
   - Must contain all words used in transcripts
   - UTF-8 encoding

5. **Phoneset** (`shared/phoneset.txt` or extracted from dictionary)
   - List of all valid phones used in dictionary
   - Format: One phone per line (comments with #)
   - Can be extracted from dictionary automatically
   - Must include special phones (e.g., SIL for silence)

### Optional Inputs
- **Filler dictionary** (`shared/filler.dict`) - Filler words (SIL, noise, etc.)
- **Test split** - Separate test transcription file (or split from all.transcription)

### Outputs
- **Project structure** - All directories created
- **Configuration file** - `etc/config.yaml` (created or validated)
- **Dictionary files** - `shared/dictionary.dict`, `shared/filler.dict`
- **Phoneset file** - `shared/phoneset.txt` (extracted from dictionary)
- **Transcription file** - `etc/all.transcription` (validated)
- **Directory structure** - `audio/`, `shared/`, `experiments/`

## Setup Process

### Step 1: Create Directory Structure
- Create `etc/` directory
- Create `audio/` directory
- Create `shared/` directory
- Create `experiments/` directory
- Create `shared/features/` directory

### Step 2: Create/Validate Configuration File
- Create `etc/config.yaml` with:
  - Project metadata (name, description)
  - Audio settings (sample rate, format)
  - Feature extraction settings
  - Training parameters
  - Dictionary and phoneset paths
- Or validate existing config file

### Step 3: Prepare Dictionary and Phoneset
- Copy dictionary file to `shared/dictionary.dict`
- Validate dictionary format (UTF-8, one word per line)
- Extract phoneset from dictionary to `shared/phoneset.txt`
- Create filler dictionary to `shared/filler.dict` (if not provided)
- Validate all words in transcripts are in dictionary

### Step 4: Prepare Transcription File
- Create or validate `etc/all.transcription` file
- Format: `<fileid> <word1> <word2> ...` (one line per utterance)
- Validate format (UTF-8, proper structure)
- Extract list of fileids from transcription file

### Step 5: Prepare Audio Files
- Copy or link audio files to `audio/` directory
- Name files by fileid: `{fileid}.wav` (must match transcription fileids)
- Validate audio files:
  - All fileids in transcription have corresponding audio files
  - Audio files are readable WAV format
  - Audio format matches config (sample rate, etc.)

### Step 6: Validate Setup
- Check all required directories exist
- Validate dictionary format and phoneset
- Validate transcription file format
- Check all fileids have corresponding audio files
- Verify all words in transcripts are in dictionary
- Verify configuration is valid

## CLI Command

```bash
pstrain setup [project_dir] [options]
```

**Project Directory:**
- If `project_dir` is provided: Initialize project in that directory (create if needed)
- If `project_dir` is omitted: Initialize project in current directory (`.`)

**Options:**
- `--config <path>` - Path to config file (or create default)
- `--transcription <path>` - Path to transcription file (fileid + transcripts)
- `--audio <path>` - Path to audio files directory (files named by fileid)
- `--dictionary <path>` - Path to pronunciation dictionary file
- `--phoneset <path>` - Path to phoneset file (or extract from dictionary)
- `--filler-dict <path>` - Path to filler dictionary (optional)
- `--force` - Force recreation if project exists

**Examples:**

Initialize in current directory:
```bash
cd my_project
pstrain setup \
  --transcription /path/to/all.transcription \
  --audio /path/to/audio \
  --dictionary /path/to/dictionary.dict
```

Initialize in specified directory:
```bash
pstrain setup my_project \
  --transcription /path/to/all.transcription \
  --audio /path/to/audio \
  --dictionary /path/to/dictionary.dict
```

Initialize with all options:
```bash
pstrain setup my_project \
  --config /path/to/config.yaml \
  --transcription /path/to/all.transcription \
  --audio /path/to/audio \
  --dictionary /path/to/dictionary.dict \
  --phoneset /path/to/phoneset.txt
```

**Transcription file format** (`etc/all.transcription`):
```
arctic_a0001 hello world
arctic_a0002 this is a test
arctic_a0003 good morning
```

**Audio file naming**:
- Fileid from transcription: `arctic_a0001`
- Audio file: `audio/arctic_a0001.wav`
- Must match exactly (case-sensitive)

## Implementation for pstrain

### Phase 0: Project Setup (NEW - First Priority)

#### TODO 0.1: Setup CLI Command
- [ ] Create `pstrain/cli/setup.py` module
- [ ] Add `pstrain setup` command to main CLI
- [ ] Parse command-line arguments:
  - [ ] `project_dir` (optional positional) - If provided, use that directory; if omitted, use current directory
  - [ ] `--config`, `--transcription`, `--audio`, `--dictionary`, `--phoneset`, `--filler-dict`
  - [ ] `--force` flag
- [ ] Resolve project directory:
  - [ ] If `project_dir` provided: create if needed, use absolute path
  - [ ] If omitted: use `Path.cwd()` (current directory)
- [ ] Validate inputs
- [ ] Call setup function

#### TODO 0.2: Setup Function
- [ ] Create `pstrain/lib/setup.py` module
- [ ] Implement `setup_project(project_dir: Path, ...)` function
- [ ] Handle project directory:
  - [ ] Accept `project_dir` as Path (absolute or relative)
  - [ ] Create directory if it doesn't exist (when `project_dir` is provided)
  - [ ] If directory exists and not empty, check `--force` flag
- [ ] Create directory structure
- [ ] Create default config file
- [ ] Copy/prepare dictionary
- [ ] Copy/link audio files
- [ ] Prepare transcripts
- [ ] Validate setup
- [ ] Return setup status and paths

#### TODO 0.3: Configuration Management
- [ ] Create `pstrain/lib/config.py` module
- [ ] Implement config loading/saving
- [ ] Create default config template
- [ ] Validate config schema
- [ ] Support config overrides

#### TODO 0.4: Dictionary and Phoneset Handling
- [ ] Create `pstrain/lib/dictionary.py` module
- [ ] Implement dictionary loading/validation
- [ ] Support filler dictionary
- [ ] Validate dictionary format (UTF-8, proper structure)
- [ ] Create `pstrain/lib/phoneset.py` module
- [ ] Implement phoneset loading/validation
- [ ] Extract phoneset from dictionary
- [ ] Validate dictionary phones against phoneset
- [ ] Support phoneset file format (one phone per line, # comments)

#### TODO 0.5: Transcription File Handling
- [ ] Create `pstrain/lib/transcription.py` module
- [ ] Implement transcription file parsing
- [ ] Format: `<fileid> <word1> <word2> ...` (one line per utterance)
- [ ] Extract fileid list from transcription
- [ ] Validate transcription format (UTF-8, proper structure)
- [ ] Validate all words in transcripts are in dictionary

#### TODO 0.6: Feature Set Identification
- [ ] Create `pstrain/lib/features.py` module
- [ ] Implement `get_feature_set_id(audio_config, feature_config)` function
- [ ] Generate deterministic identifier from audio + feature parameters
- [ ] Use hash or descriptive name (e.g., `mfcc_13`, `lda_29`)
- [ ] Ensure same config always produces same feature_set_id
- [ ] Document feature directory naming convention

#### TODO 0.7: Project Validation
- [ ] Create `pstrain/lib/validate.py` module
- [ ] Implement project structure validation
- [ ] Check required files exist:
  - [ ] Configuration file
  - [ ] Transcription file
  - [ ] Dictionary file
  - [ ] Phoneset file (or extract from dictionary)
- [ ] Validate file formats
- [ ] Validate fileid matching:
  - [ ] All fileids in transcription have corresponding audio files
  - [ ] All audio files have corresponding transcriptions
- [ ] Validate dictionary coverage:
  - [ ] All words in transcripts are in dictionary
  - [ ] All phones in dictionary are in phoneset
- [ ] Check audio files are readable WAV format
- [ ] Validate feature directory structure (if features exist)

**Deliverables:**
- `pstrain/cli/setup.py` - Setup CLI command
- `pstrain/lib/setup.py` - Setup implementation
- `pstrain/lib/config.py` - Configuration management
- `pstrain/lib/dictionary.py` - Dictionary handling
- `pstrain/lib/phoneset.py` - Phoneset handling
- `pstrain/lib/transcription.py` - Transcription file handling
- `pstrain/lib/features.py` - Feature set identification
- `pstrain/lib/validate.py` - Project validation
- `tests/test_setup.py` - Unit tests
- Documentation

## Configuration File Format

```yaml
# etc/config.yaml
project:
  name: "my_project"
  description: "Acoustic model training project"
  version: "1.0.0"

audio:
  sample_rate: 16000
  format: "wav"
  directory: "audio"

features:
  num_ceps: 13
  num_filters: 40
  lower_freq: 133.33334
  upper_freq: 6855.4976
  preemphasis: 0.97
  transform: "dct"
  lifter: 22
  agc: "max"
  cmn: "batch"
  varnorm: false
  feature_type: "1s_c_d_dd"

dictionary:
  main_dict: "shared/dictionary.dict"
  filler_dict: "shared/filler.dict"
  phoneset: "shared/phoneset.txt"  # Can be extracted from dictionary

corpus:
  transcription_file: "etc/all.transcription"  # Fileid + transcripts
  audio_dir: "audio"  # Audio files named by fileid: {fileid}.wav

features:
  # Feature directory is automatically determined by audio + feature config
  # Path: shared/features/{feature_set_id}/
  # feature_set_id is hash/identifier based on audio and feature parameters
  # Multiple experiments with same config share the same feature directory

training:
  ci:
    n_iterations: 10
    convergence_threshold: 0.001
    min_iterations: 3
    abeam: 1e-100
    bbeam: 1e-100
    varfloor: 0.0001
    mixw_floor: 0.00001
    topn: 1

experiments:
  directory: "experiments"
  default_name: "baseline"
```

## Validation

After setup, validate:
- [ ] All required directories exist
- [ ] Configuration file is valid YAML
- [ ] Transcription file exists and is valid format
- [ ] Dictionary file exists and is readable
- [ ] Phoneset file exists (or can be extracted from dictionary)
- [ ] All fileids in transcription have corresponding audio files
- [ ] All audio files are readable WAV format
- [ ] All words in transcripts are in dictionary
- [ ] All phones in dictionary are in phoneset
- [ ] Project structure matches expected layout

## File Format Details

### Transcription File Format (`etc/all.transcription`)

**Format 1: Simple (fileid + words)**
```
<fileid1> <word1> <word2> <word3>
<fileid2> <word1> <word2>
<fileid3> <word1> <word2> <word3> <word4>
```

**Format 2: Sphinx format (with sentence markers)**
```
<s> <word1> <word2> <word3> </s> (<fileid1>)
<s> <word1> <word2> </s> (<fileid2>)
```

- One line per utterance
- First token is fileid (Format 1) or fileid in parentheses at end (Format 2)
- Remaining tokens are words
- UTF-8 encoding
- Words must match dictionary entries
- Fileid is base name without extension (e.g., `arctic_a0001` → audio file `arctic_a0001.wav`)

### Dictionary Format (`shared/dictionary.dict`)
```
<word1> <phone1> <phone2> <phone3>
<word2> <phone1> <phone2>
```
- One entry per line
- First token is word
- Remaining tokens are phones
- UTF-8 encoding
- Case-sensitive

### Phoneset Format (`shared/phoneset.txt`)
```
# Phoneset for acoustic model
AA
AE
AH
SIL
```
- One phone per line
- Comments start with #
- UTF-8 encoding
- Case-sensitive
- Can be extracted from dictionary automatically

### Audio File Naming
- Fileid from transcription: `arctic_a0001`
- Audio file: `audio/arctic_a0001.wav`
- Must match exactly (case-sensitive, no extension in fileid)

## Integration with Workflows

Once project is set up:
- Workflows can reference project structure
- Config file provides paths to resources
- Shared resources (dictionary, features) are accessible
- Experiments can be created in `experiments/` directory

## Next Steps After Setup

1. **Feature extraction** - Extract features from audio
2. **Data splitting** - Split into train/test sets
3. **Flat initialization** - Initialize flat models
4. **CI training** - Train CI models

## Design Notes

1. **Shared resources** - Dictionary, audio, and features are shared across experiments
2. **Feature set identification** - Features are stored in `shared/features/{feature_set_id}/` where `feature_set_id` is deterministic based on audio+feature config
3. **Space efficiency** - Features computed once per unique config, shared by all experiments using that config
4. **Experiment isolation** - Each experiment has its own models and logs
5. **Config-driven** - All paths come from config file
6. **Validation** - Setup validates project structure before proceeding
7. **Idempotent** - Setup can be run multiple times safely (skips if exists)

### Feature Sharing Design

**Problem:** Multiple experiments often use the same audio and feature extraction settings, but features are expensive to compute.

**Solution:** Features are stored in `shared/features/{feature_set_id}/` where `feature_set_id` is a deterministic identifier based on:
- Audio configuration (sample rate, format)
- Feature extraction configuration (num_ceps, filters, transforms, etc.)

**Benefits:**
- Features computed once per unique (audio, feature) configuration
- All experiments with matching config automatically share features
- No manual coordination needed - feature directory is determined by config
- Experiments can run in parallel using the same features
- Space efficient - no duplication of large feature files

**Example:**
- Experiment `baseline` uses `mfcc_13` features → `shared/features/mfcc_13/`
- Experiment `lda_v1` uses `lda_29` features → `shared/features/lda_29/`
- Experiment `baseline_v2` also uses `mfcc_13` → reuses `shared/features/mfcc_13/` (no recomputation)

## Testing with CMU Arctic

For testing purposes, CMU Arctic corpus can be used:
- **Source:** http://festvox.org/cmu_arctic/cmu_arctic/packed/
- **Format:** Packed archives (tar.bz2) with audio and transcripts
- **Process:** Download, extract, convert format, set up project
- **Note:** See `docs/design/testing-with-cmu-arctic.md` for details (local testing only, do not commit data/scripts)
