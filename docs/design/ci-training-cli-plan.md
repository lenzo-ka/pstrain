# CI Training CLI Commands Plan

This document outlines the CLI commands needed to build CI models step by step.

## Overview

This plan defines the CLI commands for building Context-Independent (CI) acoustic models. Each command corresponds to a stage in the CI training pipeline.

## Dependency Chain

```
setup → features → flat → ci
         ↓
       split ────┘
```

**Parallel execution:**
- `features` and `split` can run in parallel (both only need audio/corpus)
- `flat` needs: `features`, `setup` (dictionary, phoneset)
- `ci` needs: `flat`, `features`, `dictionary`, `split` (training transcriptions)

## CLI Commands

### 1. `pstrain features` - Extract Features

**Purpose:** Extract acoustic features from audio files using `sphinx_fe`.

**Command:**
```bash
pstrain features [--project-dir DIR] [--experiment NAME] [--config PATH] [options]
```

**Key Options:**
- `--project-dir DIR` - Project directory (default: current directory)
- `--experiment NAME` - Experiment name (default: "baseline")
- `--config NAME` - Model configuration name (not used for features, but for consistency)
- `--config PATH` - Feature config file (overrides project config)
- `--audio-dir DIR` - Audio directory (default: `{project_dir}/audio`)
- `--output-dir DIR` - Output feature directory (default: `{project_dir}/shared/features/{feature_set_id}`)
- `--feature-set-id ID` - Feature set ID (auto-generated if not provided)
- `--control-file FILE` - Control file with list of fileids (auto-generated if not provided)
- `--batch` - Process all files in batch mode
- `--fileid FILEID` - Process single file (for testing)

**Underlying Program:** `sphinx_fe`

**Key `sphinx_fe` Arguments:**
- `-c FILE` - Control file (list of fileids)
- `-di DIR` - Input directory (audio files)
- `-ei EXT` - Input extension (default: `.wav`)
- `-do DIR` - Output directory (feature files)
- `-eo EXT` - Output extension (default: `.mfc`)
- `-argfile FILE` - Feature parameter file (e.g., `feat.params`)
- `-npart N` - Number of parallel parts
- `-part I` - Part index (for parallelization)

**Outputs:**
- `{output_dir}/{fileid}.mfc` - Feature files (one per audio file)
- `{output_dir}/feat.params` - Feature parameters (written by sphinx_fe)
- `{output_dir}/feat.defs` - Feature definition (if needed)
- `{output_dir}/global_mean` - Global feature mean (if computed)
- `{output_dir}/global_variance` - Global feature variance (if computed)

**Example:**
```bash
# Extract features for all audio files
pstrain features --project-dir my_project

# Extract features for specific experiment
pstrain features --project-dir my_project --experiment baseline

# Extract features with custom config
pstrain features --project-dir my_project --config custom_feat.yaml
```

---

### 2. `pstrain split` - Split Data into Train/Test

**Purpose:** Split corpus into training and test sets.


**Command:**
```bash
pstrain split [--project-dir DIR] [--experiment NAME] [options]
```

**Key Options:**
- `--project-dir DIR` - Project directory (default: current directory)
- `--experiment NAME` - Experiment name (default: "baseline")
- `--transcription-file FILE` - Input transcription file (default: `{project_dir}/etc/all.transcription`)
- `--train-file FILE` - Output training transcription file (default: `{experiment_dir}/etc/train.transcription`)
- `--test-file FILE` - Output test transcription file (default: `{experiment_dir}/etc/test.transcription`)
- `--train-ratio FLOAT` - Training set ratio (default: 0.8, i.e., 80% train, 20% test)
- `--seed INT` - Random seed for reproducibility (default: 42)
- `--control-file FILE` - Output control file (list of fileids for training, default: `{experiment_dir}/etc/train.fileids`)

**Outputs:**
- `{experiment_dir}/etc/train.transcription` - Training transcriptions
- `{experiment_dir}/etc/test.transcription` - Test transcriptions
- `{experiment_dir}/etc/train.fileids` - Training fileid list (control file)
- `{experiment_dir}/etc/test.fileids` - Test fileid list (optional)

**Example:**
```bash
# Split data (80/20 train/test)
pstrain split --project-dir my_project

# Split with custom ratio
pstrain split --project-dir my_project --train-ratio 0.9

# Split with specific seed
pstrain split --project-dir my_project --seed 123
```

**Note:** This is a Python-only command (no C program). It reads the transcription file and splits it deterministically.

---

### 3. `pstrain flat` - Initialize Flat Models

**Purpose:** Create initial flat (uniform) HMM models using `mk_flat`.

**Command:**
```bash
pstrain flat [--project-dir DIR] [--experiment NAME] [options]
```

**Key Options:**
- `--project-dir DIR` - Project directory (default: current directory)
- `--experiment NAME` - Experiment name (default: "baseline")
- `--dictionary FILE` - Dictionary file (default: `{project_dir}/shared/dictionary.dict`)
- `--phoneset FILE` - Phoneset file (default: `{project_dir}/shared/phoneset.txt`)
- `--features-dir DIR` - Feature directory (default: `{project_dir}/shared/features/{feature_set_id}`)
- `--output-dir DIR` - Output model directory (default: `{experiment_dir}/models/ci/{config}/model/flat`)
- `--feat-params FILE` - Feature parameters file (default: `{features_dir}/feat.params`)
- `--topo FILE` - HMM topology file (optional, uses default if not provided)
- `--n-state INT` - Number of states per phone (default: 3)
- `--n-density INT` - Number of Gaussians per state (default: 1)

**Underlying Program:** `mk_flat`

**Key `mk_flat` Arguments:**
- `-moddeffn FILE` - Model definition output file (e.g., `mdef`)
- `-tmatfn FILE` - Transition matrix output file
- `-mixwfn FILE` - Mixture weight output file
- `-meanfn FILE` - Mean output file
- `-varfn FILE` - Variance output file
- `-topo FILE` - HMM topology file
- `-featdefs FILE` - Feature definition file (`feat.defs` or `feat.params`)
- `-nstate INT` - Number of states per phone
- `-ndensity INT` - Number of Gaussians per state

**Outputs:**
- `{output_dir}/mdef` - Model definition file
- `{output_dir}/transition_matrices` - Initial transition probabilities
- `{output_dir}/means` - Initial Gaussian means
- `{output_dir}/variances` - Initial Gaussian variances
- `{output_dir}/mixture_weights` - Initial mixture weights

**Example:**
```bash
# Initialize flat models
pstrain flat --project-dir my_project --experiment baseline

# Initialize with custom topology
pstrain flat --project-dir my_project --topo custom_topo.txt
```

---

### 4. `pstrain ci` - Train CI Models

**Purpose:** Train CI models using Baum-Welch algorithm (`bw` program) with iterations.

**Command:**
```bash
pstrain ci [--project-dir DIR] [--experiment NAME] [options]
```

**Key Options:**
- `--project-dir DIR` - Project directory (default: current directory)
- `--experiment NAME` - Experiment name (default: "baseline")
- `--flat-dir DIR` - Flat model directory (default: `{experiment_dir}/models/ci/{config}/model/flat`)
- `--from NAME` - Start training from another model configuration instead of flat (e.g., `--from 1g` to start from 1g model)
- `--features-dir DIR` - Feature directory (default: `{project_dir}/shared/features/{feature_set_id}`)
- `--dictionary FILE` - Dictionary file (default: `{project_dir}/shared/dictionary.dict`)
- `--control-file FILE` - Training control file (default: `{experiment_dir}/etc/train.fileids`)
- `--transcription-file FILE` - Training transcription file (default: `{experiment_dir}/etc/train.transcription`)
- `--output-dir DIR` - Output model directory (default: `{experiment_dir}/models/ci/{config}/model/hmm`)
- `--max-iterations INT` - Maximum training iterations (default: 10)
- `--min-iterations INT` - Minimum iterations before checking convergence (default: 3)
- `--convergence-threshold FLOAT` - Relative log likelihood improvement threshold (default: 0.001)
- `--abeam FLOAT` - Alpha beam width (default: 1e-100)
- `--bbeam FLOAT` - Beta beam width (default: 1e-100)
- `--topn INT` - Top N Gaussians for CI (default: 1)
- `--varfloor FLOAT` - Variance floor (default: 0.0001)
- `--mixw-floor FLOAT` - Mixture weight floor (default: 0.00001)
- `--2passvar BOOL` - Use 2-pass variance (default: `no` for first iteration, `yes` for subsequent)
- `--save-alignments` - Save phone alignments (`phseg` files)

**Starting Point:**
- If `--from` is provided: Start from that model's `hmm/` directory
- Otherwise: Start from flat models (default behavior)

**Underlying Program:** `bw` (Baum-Welch)

**Key `bw` Arguments:**
- `-moddeffn FILE` - Model definition file (input)
- `-ts2cbfn FILE` - Tied state to codebook mapping (optional)
- `-featdefs FILE` - Feature definition file (`feat.params` or `feat.defs`)
- `-meanfn FILE` - Mean file (input/output)
- `-varfn FILE` - Variance file (input/output)
- `-mixwfn FILE` - Mixture weight file (input/output)
- `-tmatfn FILE` - Transition matrix file (input/output)
- `-accumdir DIR` - Accumulator directory (working directory)
- `-ctlfn FILE` - Control file (list of fileids)
- `-part INT` - Part index (for parallelization)
- `-npart INT` - Number of parts (for parallelization)
- `-dictfn FILE` - Dictionary file
- `-fdictfn FILE` - Filler dictionary file
- `-lsnfn FILE` - Transcription file
- `-abeam FLOAT` - Alpha beam width
- `-bbeam FLOAT` - Beta beam width
- `-topn INT` - Top N Gaussians
- `-varfloor FLOAT` - Variance floor
- `-mixwfloor FLOAT` - Mixture weight floor
- `-2passvar BOOL` - Use 2-pass variance (`yes` or `no`)

**Iteration Loop:**
1. Run `bw` for iteration N (reads models from iteration N-1, writes to iteration N)
2. Check log likelihood from `bw` output
3. Compare with previous iteration
4. Check convergence: `(LL_current - LL_previous) / |LL_previous| < threshold`
5. If converged or max iterations reached, stop
6. Otherwise, continue to next iteration

**Outputs:**
- `{output_dir}/mdef` - Model definition (copied from flat)
- `{output_dir}/means` - Trained Gaussian means
- `{output_dir}/variances` - Trained Gaussian variances
- `{output_dir}/mixture_weights` - Trained mixture weights
- `{output_dir}/transition_matrices` - Trained transition probabilities
- `{output_dir}/feat.params` - Feature parameters (copied from features)
- `{output_dir}/iter{N}/` - Per-iteration models (optional, for debugging)
- `{output_dir}/phseg/` - Phone alignments (if `--save-alignments`)

**Example:**
```bash
# Train CI models with default parameters (starts from flat)
pstrain ci --project-dir my_project --experiment baseline --config baseline

# Train with custom iteration settings (different config, starts from flat)
pstrain ci --project-dir my_project --experiment baseline --config high_iter --max-iterations 15 --convergence-threshold 0.0005

# Train with alignments saved (another config, starts from flat)
pstrain ci --project-dir my_project --experiment baseline --config with_alignments --save-alignments

# Train with different feature set (different config, starts from flat)
pstrain ci --project-dir my_project --experiment baseline --config lda_features --features-dir ../shared/features/lda_29cep

# Gaussian splitting workflow: Start from 1G, then split to 2G, 4G, etc.
# Step 1: Train 1G model (from flat)
pstrain flat --config 1g
pstrain ci --config 1g --topn 1

# Step 2: Split to 2G (starts from 1G model)
pstrain ci --config 2g --from 1g --topn 2

# Step 3: Split to 4G (starts from 2G model)
pstrain ci --config 4g --from 2g --topn 4

# Step 4: Split to 8G (starts from 4G model)
pstrain ci --config 8g --from 4g --topn 8
```

---

## Complete Workflow Example

```bash
# 1. Setup project (already done)
cd my_project

# 2. Extract features (can run in parallel with split)
pstrain features

# 3. Split data (can run in parallel with features)
pstrain split

# 4. Initialize flat models (needs features and dictionary)
pstrain flat --config baseline

# 5. Train CI models (needs flat, features, dictionary, split)
pstrain ci --config baseline

# Gaussian splitting workflow:
# Step 1: Train 1G from flat
pstrain flat --config 1g
pstrain ci --config 1g --topn 1

# Step 2: Split to 2G (starts from 1G model, not flat)
pstrain ci --config 2g --from 1g --topn 2

# Step 3: Split to 4G (starts from 2G model)
pstrain ci --config 4g --from 2g --topn 4

# Step 4: Split to 8G (starts from 4G model)
pstrain ci --config 8g --from 4g --topn 8
```

**Or as a single command (future):**
```bash
pstrain train-ci --project-dir my_project --experiment baseline --config baseline
```

## CI Model Configurations

Different CI model configurations can be trained within the same experiment:

- **`baseline`** - Default CI configuration (1 Gaussian, standard parameters)
- **`1g`** - Single Gaussian per state
- **`2g`** - Two Gaussians per state (typically split from 1g)
- **`4g`** - Four Gaussians per state (typically split from 2g)
- **`8g`** - Eight Gaussians per state (typically split from 4g)
- **`lda`** - With LDA transform features
- **`high_iter`** - More training iterations
- **`with_alignments`** - Save phone alignments
- **`custom`** - Custom configuration name

Each configuration gets its own directory:
- `models/ci/{ci_config}/model/flat` - Flat models for this config (if initialized from flat)
- `models/ci/{ci_config}/model/hmm` - Trained CI models for this config

**Starting Points:**
- **From Flat**: Default behavior - starts from flat initialization
- **From Another Model**: Use `--from NAME` to start from an existing model (e.g., for Gaussian splitting)

**Gaussian Splitting Workflow:**
1. Train 1G model from flat: `pstrain flat --config 1g` → `pstrain ci --config 1g`
2. Split to 2G from 1G: `pstrain ci --config 2g --from 1g`
3. Split to 4G from 2G: `pstrain ci --config 4g --from 2g`
4. Continue as needed...

This allows:
- Multiple CI models in the same experiment
- Easy comparison of different configurations
- Gaussian splitting workflows (1G → 2G → 4G → 8G)
- Reuse of shared resources (features, dictionary, split)

---

## Implementation Plan

### Phase 1: Individual Commands (Stub)

1. **`pstrain features`**
   - [ ] Create `pstrain/cli/features.py`
   - [ ] Parse command-line arguments
   - [ ] Generate control file from audio directory
   - [ ] Call `sphinx_fe` wrapper (to be implemented)
   - [ ] Handle feature set ID generation
   - [ ] Support batch and single-file modes

2. **`pstrain split`**
   - [ ] Create `pstrain/cli/split.py`
   - [ ] Parse command-line arguments
   - [ ] Read transcription file
   - [ ] Split deterministically (with seed)
   - [ ] Write train/test transcriptions
   - [ ] Generate control files

3. **`pstrain flat`**
   - [ ] Create `pstrain/cli/flat.py`
   - [ ] Parse command-line arguments
   - [ ] Validate inputs (dictionary, phoneset, features)
   - [ ] Call `mk_flat` wrapper (to be implemented)
   - [ ] Verify outputs

4. **`pstrain ci`**
   - [ ] Create `pstrain/cli/ci.py`
   - [ ] Parse command-line arguments
   - [ ] Implement iteration loop
   - [ ] Call `bw` wrapper (to be implemented) for each iteration
   - [ ] Parse log likelihood from `bw` output
   - [ ] Check convergence
   - [ ] Copy final models to output directory
   - [ ] Track progress (iterations, LL, timing)

### Phase 2: Library Functions

1. **Feature Extraction**
   - [ ] `pstrain/lib/features.py` - `extract_features()` function
   - [ ] Wrapper for `sphinx_fe` C program
   - [ ] Control file generation
   - [ ] Feature set ID computation

2. **Data Splitting**
   - [ ] `pstrain/lib/split.py` - `split_data()` function
   - [ ] Deterministic splitting with seed
   - [ ] Control file generation

3. **Flat Initialization**
   - [ ] `pstrain/lib/flat.py` - `init_flat()` function
   - [ ] Wrapper for `mk_flat` C program
   - [ ] Model definition generation

4. **CI Training**
   - [ ] `pstrain/lib/ci.py` - `train_ci()` function
   - [ ] Wrapper for `bw` C program
   - [ ] Iteration loop logic
   - [ ] Convergence checking
   - [ ] Progress tracking

### Phase 3: Integration

1. **Workflow Command**
   - [ ] `pstrain train-ci` - Run complete CI training pipeline
   - [ ] Handles dependencies automatically
   - [ ] Supports parallel execution where possible

2. **Progress Tracking**
   - [ ] Track each stage in build tracker
   - [ ] Log outputs, timings, parameters
   - [ ] Support resume/restart

3. **Error Handling**
   - [ ] Validate prerequisites before each stage
   - [ ] Clear error messages
   - [ ] Support for partial failures

---

## Notes

1. **Feature Set ID**: Deterministic ID based on audio files + feature config. Allows sharing features across experiments.

2. **Control Files**: List of fileids (one per line) used by `sphinx_fe` and `bw` to process files in batch.

3. **Iteration Management**: CI training runs multiple iterations. Each iteration:
   - Reads models from previous iteration (or flat for iteration 0)
   - Runs Baum-Welch training
   - Writes updated models
   - Checks convergence

4. **Parallelization**:
   - `sphinx_fe` supports `-npart` and `-part` for parallel feature extraction
   - `bw` supports `-npart` and `-part` for parallel Baum-Welch training
   - CLI commands should support `--jobs N` to enable parallelization

5. **Progress Tracking**: Each command should:
   - Log start/end times
   - Track outputs generated
   - Record parameters used
   - Support resume from last successful stage

---

## Next Steps

1. Stub out CLI commands (Phase 1)
2. Implement library functions (Phase 2)
3. Test with CMU Arctic data
4. Integrate into workflow system (Phase 3)
