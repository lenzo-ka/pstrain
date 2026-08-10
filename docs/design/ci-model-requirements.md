# CI Model Building Requirements

This document outlines what's needed to build Context-Independent (CI) acoustic models in pstrain.

## Overview

CI models are monophone (context-independent) HMM-GMM acoustic models trained using Baum-Welch (EM algorithm). They are the first step in acoustic model training, before context-dependent (CD) models.

## Prerequisites (Dependencies)

### 1. Flat Models
- **Stage:** `flat` (flat initialization)
- **What it is:** Initial HMM models with uniform distributions
- **Output:** Flat model directory with:
  - `mdef` - Model definition file
  - `transition_matrices` - Initial transition probabilities
  - `means` - Initial Gaussian means
  - `variances` - Initial Gaussian variances
  - `mixture_weights` - Initial mixture weights
- **Program:** `mk_flat` (SphinxTrain)

### 2. Features
- **Stage:** `features` (feature extraction)
- **What it is:** Extracted acoustic features from audio files
- **Input:** Audio files, `feat.params` (config file for sphinx_fe)
- **Output:** Feature directory with:
  - `.mfc` files - Mel-frequency cepstral coefficients (one per audio file)
  - `feat.defs` - Feature definition file (may be needed for some programs)
- **Program:** `sphinx_fe` (SphinxTrain)
- **Note:** `feat.params` is an INPUT (config file), not an output. It should be created before feature extraction (during setup or as a separate config step).
- **Note:** Features are shared across experiments
- **Note:** `feat.defs` may be needed for `mk_flat` or other initialization steps

### 3. Dictionary
- **Stage:** `setup` (project setup)
- **What it is:** Pronunciation dictionary
- **Output:** Dictionary file (`.dict` format)
- **Contains:** Word → phone sequence mappings
- **Note:** Dictionary is created during project setup

### 4. Training Transcriptions
- **Stage:** `split` (data splitting)
- **What it is:** Training set transcriptions (word-level text)
- **Output:** Transcript file(s) with word-level transcriptions
- **Format:** One transcription per utterance ID: `<fileid> <word1> <word2> ...`
- **Note:** Split stage divides data into train/test sets

## CI Training Process

### Step 1: Flat Initialization
- **Program:** `mk_flat`
- **Inputs:**
  - Dictionary (phone inventory)
  - Feature parameters (`feat.params` or `feat.defs`)
  - Feature directory (for feature statistics)
- **Outputs:**
  - Flat model directory (iteration 0)
  - Model definition (`mdef`)
  - Initial Gaussian parameters
- **Note:** May require `feat.defs` file (feature definition) in addition to or instead of `feat.params`

### Step 2: Baum-Welch Training
- **Program:** `bw` (Baum-Welch)
- **Inputs:**
  - Flat models (iteration 0) or previous iteration models
  - Features (`.mfc` files)
  - Training transcriptions
  - Dictionary
  - Control file (list of utterance IDs)
- **Outputs:**
  - Updated model parameters (means, variances, mixture weights, transitions)
  - Log likelihood (for convergence checking)
  - Optional: Phone alignments (`phseg` files)
- **Iterations:**
  - Runs multiple iterations (typically 3-10)
  - Each iteration refines model parameters
  - Checks convergence (relative log likelihood improvement)
  - Stops when converged or max iterations reached

### Step 3: Normalization (Optional)
- **Program:** `norm`
- **What it is:** Normalizes feature vectors
- **When:** Between iterations or after training
- **Note:** This is optional for CI training but available if needed

### Step 4: Gaussian Splitting (Optional)
- **Program:** `mixw_interp` or custom splitting, then `bw` retraining
- **What it is:** Increases number of Gaussians per state
- **Schedule:** Start with 1 Gaussian, split to 2, 4, 8, 16, etc.
- **Process:**
  1. Train with initial Gaussians (e.g., 1G) from flat
  2. Split Gaussians (e.g., 1G → 2G) using `mixw_interp` or similar
  3. Retrain with more Gaussians using `bw` (starts from split model, not flat)
  4. Repeat until target count reached
- **pstrain Implementation:** Use `--from-ci-config` to start CI training from a previous CI model instead of flat

## CI Training Parameters

### Required Parameters
- `max_iterations` - Maximum training iterations (default: 10)
- `convergence_threshold` - Relative LL improvement threshold (default: 0.001)
- `min_iterations` - Minimum iterations before checking convergence (default: 3)
- `abeam` - Alpha beam width (default: 1e-100)
- `bbeam` - Beta beam width (default: 1e-100)
- `topn` - Top N Gaussians for CI (default: 1, not 4 like CD)

### Optional Parameters
- `use_splitting` - Enable Gaussian splitting schedule
- `use_lda` - Use LDA transformation (requires LDA file from LDA stage)
- `save_alignments` - Save phone alignments during training

## Workflow Dependencies

```
setup (dictionary, phone_file)
  ↓
features (audio) ──┐
  ↓                │
split (corpus) ────┼──→ flat (features, dictionary, phone_file)
                   │         ↓
                   └─────────┴──→ ci (flat, features, dictionary, split)
```

**Dependency chain:**
1. **setup** - Creates project structure, dictionary, phone_file
2. **features** - Extracts features from audio (depends on audio files, can run in parallel with setup)
3. **split** - Splits data into train/test (depends on corpus/audio files, NOT features - can run in parallel with features)
4. **flat** - Initializes flat models (depends on: features, dictionary, phone_file)
5. **ci** - Trains CI models (depends on: flat, features, dictionary, and split for training transcriptions)

**Key points:**
- `features` and `split` can run in parallel (both only need audio/corpus files)
- `flat` needs both `features` AND `setup` (dictionary, phone_file)
- `ci` needs `flat`, `features`, `dictionary`, AND `split` (for training transcriptions)

## Implementation for pstrain

### Phase 1: Prerequisites
- [ ] **Setup stage** - Project initialization, dictionary creation
- [ ] **Feature extraction** - `sphinx_fe` wrapper
- [ ] **Data splitting** - Train/test split
- [ ] **Flat initialization** - `mk_flat` wrapper

### Phase 2: CI Training
- [ ] **CI training stage** - `bw` wrapper with iteration loop
- [ ] **Convergence checking** - Log likelihood comparison
- [ ] **Model copying** - Copy final model to output directory
- [ ] **Progress tracking** - Track iterations, log likelihood

### Phase 3: Advanced Features
- [ ] **Gaussian splitting** - Support for Gaussian increment schedule
- [ ] **LDA integration** - Optional LDA transform support
- [ ] **Alignment output** - Optional phone alignment saving

## Key Programs Needed

1. **`mk_flat`** - Flat model initialization
2. **`bw`** - Baum-Welch training (main CI training)
3. **`norm`** - Feature normalization (optional)
4. **`sphinx_fe`** - Feature extraction (prerequisite)
5. **`mixw_interp`** - Gaussian splitting (optional)

## Output Products

After CI training completes:
- `models/ci/{ci_config}/model/hmm/` directory containing:
  - `mdef` - Model definition
  - `means` - Gaussian means
  - `variances` - Gaussian variances
  - `mixture_weights` - Mixture weights
  - `transition_matrices` - Transition probabilities
  - `feat.params` - Feature parameters (copied from features stage)
  - `sendump` - Optional sendump file for faster loading

## Implementation Notes

1. **CI uses `topn=1`** (not 4 like CD models)
2. **First iteration uses `-2passvar no`**, subsequent iterations use `-2passvar yes`
3. **Features are shared** across experiments (stored in `shared/features/`)
4. **Models are experiment-specific** (stored in `experiments/<exp>/models/ci/hmm/`)
5. **Progress tracking** includes:
   - Current iteration
   - Log likelihood per iteration
   - Iteration timings
   - Convergence status

## Next Steps

After CI models are built:
- **CD training** - Context-dependent models (depends on CI)
- **Testing** - Evaluate CI model performance
- **LDA training** - Linear Discriminant Analysis (optional, can use CI models)
