# Training Pipeline Plan

## Implementation Status Grid

| Step | Pipeline Task | Step Function | Lib Function | CFFI Binding | Status |
|------|----------------|---------------|--------------|--------------|--------|
| **Features** | `step_extract_features` | ✅ | `features.extract_features()` | ✅ `pstrain_fe_create` | ✅ Done |
| **Flat** | `step_flat` | ✅ | `flat.init_flat_model()` | ✅ `pstrain_flat_*`, `pstrain_init_gau` | ✅ Done |
| **CI Train** | `step_ci_1g` | ✅ | `bw.BWTrainer` | ✅ `pstrain_bw_*` | ✅ Done |
| **CI Split** | `step_ci_2g/4g/8g` | ✅ | `split.split_gaussians()` | ✅ `pstrain_inc_comp` | ✅ Done |
| **CD Untied** | `step_cd_untied` | ✅ | `mdef.create_untied_mdef()` | ✅ `pstrain_mdef_gen_untied` | ✅ Done |
| **Make Quests** | `step_questions` | ✅ | `dtree.make_quests()` | ✅ `pstrain_make_quests` | ✅ Done |
| **Trees** | `step_trees` | ✅ | `dtree.build_tree()` | ✅ `pstrain_build_tree` | ✅ Done |
| **Prune** | `step_prune_trees` | ✅ | `dtree.prune_tree()` | ✅ `pstrain_prune_tree` | ✅ Done |
| **Tiestate** | `step_cd_1g` | ✅ | `dtree.tie_states()` | ✅ `pstrain_tie_states` | ✅ Done |
| **Init CD** | `step_cd_1g` | ✅ | `dtree.init_mixw()` | ✅ `pstrain_init_mixw` | ✅ Done |
| **CD Train** | `step_cd_2g/4g/8g` | ✅ | `bw.BWTrainer` | ✅ `pstrain_bw_*` | ✅ Done |
| **CD Split** | `step_cd_2g/4g/.../32g` | ✅ | `split.split_gaussians()` | ✅ `pstrain_inc_comp` | ✅ Done |
| **Package** | `package_ci_8g/cd_8g/cd_32g` | ✅ | `package.package_model()` | N/A | ✅ Done |

### Legend
- ✅ Done - Fully implemented and tested
- ⏳ Wire up - CFFI binding exists, need step function to orchestrate
- ❌ Missing - Not yet implemented

## Key Insight: CFFI Bindings Are Complete

All core C functions are already wrapped via CFFI:

```
pstrain/lib/
├── bw.py           ✅ BWTrainer class: pstrain_bw_init, pstrain_bw_process_utt,
│                      pstrain_bw_normalize, pstrain_bw_save, pstrain_bw_get_stats
├── split.py        ✅ split_gaussians(): pstrain_inc_comp
│                   ✅ kmeans(), kmeans_init(): pstrain_kmeans, pstrain_kmeans_init
├── dtree.py        ✅ build_tree(): pstrain_build_tree
│                   ✅ tie_states(): pstrain_tie_states
│                   ✅ make_quests(): pstrain_make_quests
│                   ✅ prune_tree(): pstrain_prune_tree
├── mdef.py         ✅ create_ci_mdef(): pstrain_mdef_gen_ci
│                   ✅ create_untied_mdef(): pstrain_mdef_gen_untied
│                   ✅ create_alltriphones_mdef(): pstrain_mdef_gen_alltriphones
├── flat.py         ✅ init_flat_model(): pstrain_flat_tmat, pstrain_flat_mixw,
│                      pstrain_init_gau, pstrain_norm_gau
├── features.py     ✅ FeatureExtractor: pstrain_fe_create
├── agg_seg.py      ✅ agg_seg(): pstrain_agg_seg
├── commands.py     ✅ core command paths: param_cnt, map_adapt, kdtree, delint
└── _cffi/io.py     ✅ read_gau, write_gau, read_mixw, write_mixw,
                       read_tmat, write_tmat, read_dnom, write_dnom
```

## What's Missing: Step Orchestration

The CFFI bindings provide low-level operations. We need higher-level "step" functions
that orchestrate these into complete training steps:

### Example: CI Training Step

The CFFI binding exists (`BWTrainer`), but we need a step function:

```python
# pstrain/lib/steps/ci_hmm.py - needs implementation

def run_ci_training(
    model_dir: Path,
    output_dir: Path,
    features_dir: Path,
    train_fileids: Path,
    transcription: Path,
    dictionary: Path,
    n_iter: int = 10,
    convergence_ratio: float = 0.001,
) -> TrainingResult:
    """Run CI HMM training using BWTrainer.

    This orchestrates the existing CFFI bindings:
    1. Load model from model_dir
    2. For each iteration:
       a. Create BWTrainer with current model
       b. For each utterance in train_fileids:
          - Load features from features_dir
          - Look up transcript from transcription
          - Call trainer.process_utterance()
       c. Call trainer.normalize()
       d. Call trainer.save() to output_dir
       e. Check convergence
    3. Return final stats
    """
    from pstrain.lib.bw import BWTrainer, BWConfig

    trainer = BWTrainer(
        mdef_path=model_dir / "mdef",
        means_path=model_dir / "means",
        vars_path=model_dir / "variances",
        mixw_path=model_dir / "mixture_weights",
        tmat_path=model_dir / "transition_matrices",
    )

    # Set dictionary for text-based processing
    trainer.set_dict(dictionary, filler_dict)

    # Process each utterance
    for fileid in read_fileids(train_fileids):
        features = load_features(features_dir / f"{fileid}.mfc")
        transcript = get_transcript(transcription, fileid)
        trainer.process_utterance_text(features, transcript)

    # Normalize and save
    trainer.normalize()
    trainer.save(output_dir / "means", ...)

    return trainer.get_stats()
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER INTERFACE                                  │
│  pstrain build cd-8g --config wideband                                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PIPELINE RUNNER                                    │
│  pstrain.lib.pipeline.runner.Pipeline: resolves DAG from file inputs/outputs    │
│  pstrain.lib.pipeline.tasks.TARGETS:   target definitions (ci-1g, cd-8g, ...)   │
│  etc/configs.yaml:                 named configs (wideband, telephone, ...) │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           STEP FUNCTIONS                                     │
│  pstrain.lib.steps.ci_hmm.run_ci_training()                                     │
│  pstrain.lib.steps.cd_hmm.run_cd_training()                                     │
│  pstrain.lib.steps.trees.build_trees()                                          │
│  Orchestrate multiple CFFI calls into complete training steps               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           LIBRARY FUNCTIONS (CFFI)                           │
│  pstrain.lib.bw.BWTrainer    pstrain.lib.split.split_gaussians()                    │
│  pstrain.lib.dtree.*         pstrain.lib.mdef.*    pstrain.lib.flat.*                   │
│  All already wrapped via CFFI - no shell-outs needed                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           C LIBRARY (libpstrainc)                                │
│  pstrain_bw_*, pstrain_inc_comp, pstrain_build_tree, pstrain_tie_states, pstrain_make_quests   │
│  pstrain_prune_tree, pstrain_mdef_gen_*, pstrain_flat_*, pstrain_init_gau, pstrain_norm_gau    │
└─────────────────────────────────────────────────────────────────────────────┘
```

## What's Complete ✅

### CFFI Bindings (All Done)
- [x] Feature extraction (`pstrain_fe_create`)
- [x] Baum-Welch training (`pstrain_bw_*`)
- [x] Gaussian splitting (`pstrain_inc_comp`)
- [x] Flat model init (`pstrain_flat_*`, `pstrain_init_gau`, `pstrain_norm_gau`)
- [x] Mdef generation (`pstrain_mdef_gen_*`)
- [x] Decision trees (`pstrain_build_tree`, `pstrain_make_quests`, `pstrain_prune_tree`)
- [x] State tying (`pstrain_tie_states`)
- [x] Model I/O (`read_gau`, `write_gau`, `read_mixw`, etc.)
- [x] K-means (`pstrain_kmeans`, `pstrain_kmeans_init`)
- [x] Parameter counting (core `param_cnt` command)
- [x] Segment aggregation (`pstrain_agg_seg`)
- [x] MAP adaptation (core `map_adapt` command)
- [x] KD-tree building (core `kdtree` command)
- [x] Deleted interpolation (core `delint` command)

### Orchestration Layer
- [x] Pipeline runner with all tasks registered (`pstrain.lib.pipeline`)
- [x] Dependency resolution (runner topo-sorts from inputs/outputs)
- [x] Named configs (`etc/configs.yaml`: wideband, telephone, etc.)
- [x] Target definitions (`targets.yaml`: ci-1g through cd-32g)
- [x] Shared output paths (same config → same outputs)
- [x] CLI (`pstrain build cd-8g --config wideband`)
- [x] Dry-run support (`pstrain --dry-run build cd-8g`)

### Working Steps
- [x] `step_extract_features` - Feature extraction via CFFI
- [x] `step_flat` - Flat model initialization via CFFI

## What Needs Implementation ⏳

### Complete Training Pipeline (execution order)

```
User Setup (manual):
  0. pstrain split                    # Create train/test fileids (optional)

Feature Extraction:
  1. step_extract_features        # ✅ DONE - MFCC extraction

Flat Model:
  2. step_flat                    # ✅ DONE - Initial uniform model

CI Training:
  3. step_ci_1g                   # ⏳ BW training on flat model
  4. step_ci_2g                   # ⏳ Split 1→2 Gaussians, BW
  5. step_ci_4g                   # ⏳ Split 2→4 Gaussians, BW
  6. step_ci_8g                   # ⏳ Split 4→8 Gaussians, BW

CD Untied:
  7. step_cd_untied               # ⏳ Generate triphone mdef, BW

Decision Trees:
  8. step_make_quests             # ⏳ Generate phonetic questions
  9. step_build_trees             # ⏳ Build decision trees
 10. step_prune_trees             # ⏳ Prune to target senones

State Tying:
 11. step_tie_states              # ⏳ Create tied mdef

CD Training:
 12. step_cd_1g                   # ⏳ BW on tied model
 13. step_cd_2g                   # ⏳ Split 1→2 Gaussians, BW
 14. step_cd_4g                   # ⏳ Split 2→4 Gaussians, BW
 15. step_cd_8g                   # ⏳ Split 4→8 Gaussians, BW
 16. step_cd_16g                  # ⏳ Split 8→16 Gaussians, BW
 17. step_cd_32g                  # ⏳ Split 16→32 Gaussians, BW
```

### Step Functions to Implement (total order)

Many pipeline tasks share the same step function:

| # | Step Function | File | CFFI Wrapper | Used By |
|---|---------------|------|--------------|---------|
| 1 | `run_bw_training()` | `steps/train.py` | `BWTrainer` | ci-1g→8g, cd-untied, cd-1g→32g |
| 2 | `run_gaussian_split()` | `steps/split.py` | `split_gaussians()` | ci-2g→8g, cd-2g→32g |
| 3 | `run_cd_untied_setup()` | `steps/cd_untied.py` | `create_untied_mdef()` | cd-untied |
| 4 | `run_make_quests()` | `steps/trees.py` | `make_quests()` | step 8 |
| 5 | `run_build_trees()` | `steps/trees.py` | `build_tree()` | step 9 |
| 6 | `run_prune_trees()` | `steps/trees.py` | `prune_tree()` | step 10 |
| 7 | `run_tie_states()` | `steps/tiestate.py` | `tie_states()` | step 11 |

**Implementation order:** 1 → 2 → 3 → 4 → 5 → 6 → 7

After implementing #1 and #2: `pstrain build ci-8g` works
After implementing #3-#7: `pstrain build cd-8g` works

**Existing CFFI wrappers (all done):**
- `pstrain.lib.bw.BWTrainer` - BW training
- `pstrain.lib.split.split_gaussians()` - Gaussian splitting
- `pstrain.lib.mdef.create_untied_mdef()` - Triphone mdef generation
- `pstrain.lib.dtree.make_quests()` - Question generation
- `pstrain.lib.dtree.build_tree()` - Tree building
- `pstrain.lib.dtree.prune_tree()` - Tree pruning
- `pstrain.lib.dtree.tie_states()` - State tying

**Each step function is ~50-100 lines of orchestration code.**

---

### Step 0: Train/Test Split — ✅ DONE

Library: `pstrain.lib.corpus.train_test_split`. Pipeline: registered as the
`split` task in `pstrain/lib/pipeline/tasks.py` (target name `split`).

* Input: `etc/all.transcription`.
* Outputs: `experiments/{experiment}/etc/{train,test}.{fileids,transcription}`.
* Default split: 95% train, seed 42. Override via a `split:` block in
  `etc/configs.yaml`.
* A complete pre-existing set of the four output files is instead treated as
  an authoritative external split. It is validated for exact ordered pairing,
  disjoint and complete corpus membership, transcript equality, and audio
  presence, then preserved byte-for-byte.

The CLI (`pstrain split`) and the pipeline task both call the same library
function. The pipeline's feature-extraction fan-out is keyed on
`audio_fileids()` (corpus-wide, derived from `audio/*.wav`), so the
extract tasks and the split task are **parallel branches** in the
DAG that join at the model tasks. A content-hash sidecar distinguishes
untouched automatically generated files from supplied or subsequently edited
files. The pipeline runner's completion-marker suffix hashes the task name,
not file contents; split files are therefore explicit inputs to the validation
task and its sentinel is an explicit input to downstream consumers, so an edit
invalidates the protected work by mtime.

---

### Step 1: BW Training (`steps/train.py`) — FIRST PRIORITY

Generic BW training that works for both CI and CD models:

```python
# pstrain/lib/steps/train.py (~80 lines)

def run_bw_training(
    model_dir: Path,
    output_dir: Path,
    features_dir: Path,
    train_ctl: Path,
    transcription: Path,
    dictionary: Path,
    filler_dict: Path,
    n_iter: int = 10,
    convergence_ratio: float = 0.001,
) -> dict:
    """Run BW training iterations.

    Uses:
      - pstrain.lib.bw.BWTrainer (already implemented)
      - pstrain.lib.transcription.read_transcription()
      - pstrain.lib._cffi.io.read_mfc() or similar

    Returns:
      {"iterations": n, "final_likelihood": x, "converged": bool}
    """
    from pstrain.lib.bw import BWTrainer

    prev_likelihood = float("-inf")

    for i in range(n_iter):
        trainer = BWTrainer(
            mdef_path=model_dir / "mdef",
            means_path=model_dir / "means",
            vars_path=model_dir / "variances",
            mixw_path=model_dir / "mixture_weights",
            tmat_path=model_dir / "transition_matrices",
        )
        trainer.set_dict(str(dictionary), str(filler_dict))

        # Process all utterances
        for fileid in read_ctl(train_ctl):
            features = load_features(features_dir / f"{fileid}.mfc")
            transcript = lookup_transcript(transcription, fileid)
            trainer.process_utterance_text(features, transcript)

        trainer.normalize()
        trainer.save(output_dir / "means", output_dir / "variances",
                     output_dir / "mixture_weights", output_dir / "transition_matrices")

        stats = trainer.get_stats()
        if check_convergence(stats.avg_log_prob, prev_likelihood, convergence_ratio):
            return {"iterations": i + 1, "converged": True, ...}

        prev_likelihood = stats.avg_log_prob
        model_dir = output_dir  # next iteration reads from output

    return {"iterations": n_iter, "converged": False, ...}
```

**Enables:** `ci-1g` target, and all subsequent training

Forward pruning can occasionally remove the transcript's final state. The live
training driver treats that native status as recoverable: it retries the
utterance exactly once with a forward beam widened by `retry_beam_factor`
(default `1e10`, so `1e-90` becomes `1e-100`), restores the stage beam, and
only then includes a second failure in the normal skip count and limit. Other
Baum-Welch failures are never retried.

**BASIS / deliberate deviation:** upstream SphinxTrain has no retry and silently
skips an utterance when forward pruning loses the final state. The wider beam
changes that utterance's lattice and posteriors for the iteration, but recovers
training evidence that upstream discards. Pstrain therefore permits exactly one
bounded retry, and accumulator counts come only from a successful pass; the
failed pass contributes no counts.

---

### Step 2: Gaussian Split (`steps/split.py`)

Split Gaussians to double density (works for CI and CD):

```python
# pstrain/lib/steps/split.py (~40 lines)

def run_split(
    input_model_dir: Path,
    output_model_dir: Path,
) -> None:
    """Double Gaussian density.

    Uses:
      - pstrain.lib.split.split_gaussians() (already implemented)

    Also copies mdef, tmat (unchanged) to output.
    """
    from pstrain.lib.split import split_gaussians
    import shutil

    output_model_dir.mkdir(parents=True, exist_ok=True)

    # Split means, variances, mixture weights
    split_gaussians(
        in_mean_path=input_model_dir / "means",
        in_var_path=input_model_dir / "variances",
        in_mixw_path=input_model_dir / "mixture_weights",
        out_mean_path=output_model_dir / "means",
        out_var_path=output_model_dir / "variances",
        out_mixw_path=output_model_dir / "mixture_weights",
    )

    # Copy unchanged files
    shutil.copy(input_model_dir / "mdef", output_model_dir / "mdef")
    shutil.copy(input_model_dir / "transition_matrices",
                output_model_dir / "transition_matrices")
```

**Enables:** `ci-2g`, `ci-4g`, `ci-8g` (and later `cd-2g`, etc.)

---

### Step 3: CD Untied (`steps/cd_untied.py`)

Generate triphone mdef and train untied CD model:

```python
# pstrain/lib/steps/cd_untied.py (~60 lines)

def run_cd_untied(
    ci_model_dir: Path,
    output_dir: Path,
    features_dir: Path,
    train_ctl: Path,
    transcription: Path,
    dictionary: Path,
    filler_dict: Path,
    phoneset: Path,
) -> None:
    """Create and train untied CD model.

    Uses:
      - pstrain.lib.mdef.create_untied_mdef() (already implemented)
      - run_bw_training() from step 1
    """
    from pstrain.lib.mdef import create_untied_mdef

    # 1. Generate untied triphone mdef from transcripts
    create_untied_mdef(
        phone_list=phoneset,
        dict_path=dictionary,
        filler_dict_path=filler_dict,
        transcript_path=transcription,
        output_path=output_dir / "mdef",
    )

    # 2. Initialize parameters from CI model (expanded for triphones)
    init_cd_params_from_ci(ci_model_dir, output_dir)

    # 3. Train with BW
    run_bw_training(output_dir, output_dir, features_dir, ...)
```

**Enables:** `cd-untied` intermediate model

---

### Step 4: Build Trees (`steps/trees.py`)

Generate questions, build trees, prune to target senones:

```python
# pstrain/lib/steps/trees.py (~80 lines)

def run_build_trees(
    model_dir: Path,
    output_dir: Path,
    phoneset: Path,
    n_senones: int,
) -> None:
    """Build and prune decision trees.

    Uses:
      - pstrain.lib.dtree.make_quests() (already implemented)
      - pstrain.lib.dtree.build_tree() (already implemented)
      - pstrain.lib.dtree.prune_tree() (already implemented)
    """
    from pstrain.lib.dtree import make_quests, build_tree, prune_tree

    # 1. Generate phonetic questions
    make_quests(
        mdef_path=model_dir / "mdef",
        mixw_path=model_dir / "mixture_weights",
        mean_path=model_dir / "means",
        var_path=model_dir / "variances",
        output_path=output_dir / "questions",
    )

    # 2. Build tree for each phone and state
    phones = read_phoneset(phoneset)
    for phone in phones:
        for state in range(n_states):
            build_tree(
                mdef_path=model_dir / "mdef",
                mixw_path=model_dir / "mixture_weights",
                pset_path=output_dir / "questions",
                output_path=output_dir / f"{phone}.{state}.tree",
                phone=phone,
                state=state,
            )

    # 3. Prune to target senones
    prune_tree(
        mdef_path=model_dir / "mdef",
        pset_path=output_dir / "questions",
        input_tree_dir=output_dir,
        output_tree_dir=output_dir / "pruned",
        n_seno_target=n_senones,
    )
```

**Enables:** Decision trees for state clustering

---

### Step 5: Tie States (`steps/tiestate.py`)

Apply trees to create tied mdef and initialize tied model:

```python
# pstrain/lib/steps/tiestate.py (~50 lines)

def run_tie_states(
    untied_model_dir: Path,
    trees_dir: Path,
    output_dir: Path,
    phoneset: Path,
) -> None:
    """Apply decision trees to create tied model.

    Uses:
      - pstrain.lib.dtree.tie_states() (already implemented)
    """
    from pstrain.lib.dtree import tie_states

    # 1. Create tied mdef by applying trees
    tie_states(
        input_mdef_path=untied_model_dir / "mdef",
        output_mdef_path=output_dir / "mdef",
        tree_dir=trees_dir / "pruned",
        pset_path=trees_dir / "questions",
        allphones=True,
    )

    # 2. Initialize tied model parameters from untied
    init_tied_params(untied_model_dir, output_dir)
```

**Enables:** `cd-1g` (tied CD model), then training + splitting for `cd-2g`...`cd-32g`

## Implementation Order

```
Phase 1: CI Pipeline (Already have CFFI, need step functions)
├── Implement steps/ci_hmm.py - orchestrate BWTrainer
├── Implement steps/ci_split.py - orchestrate split_gaussians
├── Wire tasks in pstrain/lib/pipeline/tasks.py to call step functions
├── Test: pstrain build ci-8g produces working model
└── Validate: decode with PocketSphinx

Phase 2: CD Pipeline (Already have CFFI, need step functions)
├── Implement steps/cd_untied.py - triphone mdef + BW
├── Implement steps/trees.py - questions + trees + prune
├── Implement steps/tiestate.py - apply trees
├── Test: pstrain build cd-8g works end-to-end
└── Validate: cd-8g achieves target WER
```

## File Locations

```
pstrain/
├── lib/
│   ├── bw.py           ✅ BWTrainer CFFI wrapper
│   ├── split.py        ✅ split_gaussians() CFFI wrapper
│   ├── dtree.py        ✅ build_tree(), tie_states(), make_quests() CFFI
│   ├── mdef.py         ✅ create_ci_mdef(), create_untied_mdef() CFFI
│   ├── flat.py         ✅ init_flat_model() CFFI
│   ├── features.py     ✅ FeatureExtractor CFFI
│   ├── _cffi/          ✅ All CFFI bindings
│   └── steps/
│       ├── ci_hmm.py   ⏳ Orchestrate CI training
│       ├── ci_split.py ⏳ Orchestrate CI split
│       ├── cd_untied.py⏳ Orchestrate CD untied
│       ├── trees.py    ⏳ Orchestrate tree building
│       └── tiestate.py ⏳ Orchestrate state tying
├── lib/pipeline/
│   ├── runner.py       ✅ Task, Pipeline, staleness, executor
│   ├── context.py      ✅ PipelineContext + config loading
│   └── tasks.py        ✅ All tasks + TARGETS registry
├── etc/
│   └── configs.yaml    ✅ Named configs
└── cli/
    └── build.py        ✅ Build command (drives the runner)
```

## Success Criteria

- [ ] `pstrain build ci-1g` trains a working CI model
- [ ] `pstrain build ci-8g` trains CI with Gaussian splitting
- [ ] `pstrain build cd-8g` trains full CD pipeline
- [ ] Models decode with PocketSphinx
- [ ] cd-8g on CMU Arctic: WER < 10%
- [ ] Training time: < 30 min for cd-8g on Arctic
- [ ] No shell-outs - everything via CFFI
