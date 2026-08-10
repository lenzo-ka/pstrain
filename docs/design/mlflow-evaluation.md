# MLflow Evaluation for Traditional ML Training

## Context

This project trains **traditional acoustic models** (HMM/GMM-based), not neural networks. The training pipeline involves:
- Feature extraction (`sphinx_fe`)
- Normalization (`norm`)
- Gaussian initialization (`init_gau`)
- Baum-Welch training (`bw`) with iterations
- Various configuration parameters (beam widths, floors, reestimation flags)

## What MLflow Provides

### 1. Experiment Tracking
- **Parameters**: Log hyperparameters, config values
- **Metrics**: Log training metrics (loss, accuracy, WER, etc.)
- **Artifacts**: Store model files, logs, plots
- **Code versioning**: Track git commit, code state
- **Run comparison**: Compare different runs side-by-side

### 2. Model Registry
- **Model versioning**: Track model versions
- **Model staging**: Dev → Staging → Production
- **Model metadata**: Tags, descriptions, aliases

### 3. Model Packaging
- **Standardized format**: MLflow model format
- **Dependencies**: Track Python/environment dependencies
- **Deployment**: Deploy to various platforms

### 4. Hyperparameter Tuning Integration
- **Optuna/Hyperopt**: Integration with tuning libraries
- **Parallel runs**: Run multiple experiments in parallel

## What We Actually Need

### For Traditional Acoustic Model Training:

1. **Parameter Tracking**
   - Beam widths (`-abeam`, `-bbeam`)
   - Floors (`-mwfloor`, `-tpfloor`, `-varfloor`)
   - Iteration counts
   - File paths (input/output)
   - Reestimation flags
   - **Assessment**: These are more like "configuration" than "hyperparameters"

2. **Metrics Tracking**
   - Training metrics: Likely none during training (Baum-Welch doesn't have loss)
   - Evaluation metrics: WER (Word Error Rate) - but this is evaluated separately, not during training
   - **Assessment**: Metrics come from separate evaluation step, not training

3. **Model Artifacts**
   - Model files: `.mdef`, `.gau`, `.mixw`, `.tmat`, etc.
   - Feature files: `.mfc`, `.norm`
   - **Assessment**: We already track these in build tracker (file outputs)

4. **Reproducibility**
   - Code version: Git commit
   - Config version: Config file checksums
   - Data version: Input file checksums
   - **Assessment**: We can track this in build tracker

5. **Run Comparison**
   - Compare different training runs
   - See what changed between runs
   - **Assessment**: Useful, but can build simple comparison in build tracker

## MLflow vs Build Tracker

### What Build Tracker Already Provides:
- ✅ Build/run tracking (SQLite database)
- ✅ Step status and metadata
- ✅ Log storage
- ✅ File artifact tracking (output files)
- ✅ Config tracking (can add checksums)
- ✅ Timestamps and duration

### What MLflow Adds:
- ✅ **UI for browsing experiments** (web interface)
- ✅ **Automatic parameter/metric logging** (convenient API)
- ✅ **Model registry** (if we need model versioning/staging)
- ✅ **Model packaging** (if we need standardized deployment)
- ⚠️ **Hyperparameter tuning integration** (probably not needed)
- ⚠️ **Neural network model support** (not applicable)

## Evaluation: Do We Need MLflow?

### Arguments FOR MLflow:

1. **Nice UI for experiment browsing**
   - Web interface to browse runs
   - Compare runs visually
   - Search/filter runs

2. **Convenient API for logging**
   - `mlflow.log_param()`, `mlflow.log_metric()`, `mlflow.log_artifact()`
   - Less code than building our own

3. **Standard format**
   - If we want to share models with others
   - If we want to integrate with other tools

4. **Model registry** (if needed)
   - Track model versions
   - Stage models (dev → prod)

### Arguments AGAINST MLflow:

1. **Overkill for traditional ML**
   - Designed for neural networks (PyTorch, TensorFlow)
   - We're doing HMM/GMM training (different paradigm)

2. **Additional dependency**
   - Another thing to install and maintain
   - Requires MLflow server (or local files)

3. **Metrics don't fit our use case**
   - Training doesn't produce metrics (Baum-Welch iterations)
   - Evaluation metrics come from separate step
   - MLflow expects metrics during training

4. **Build tracker can handle it**
   - We already have build tracking
   - Can add parameter/metric logging to build tracker
   - Can build simple UI if needed

5. **Not really "experiments"**
   - More like "builds" or "training runs"
   - Less experimentation, more production training
   - Parameters are configuration, not hyperparameters

## Recommendation

### Option 1: Skip MLflow (Recommended)

**Rationale:**
- Traditional ML training doesn't fit MLflow's model
- Build tracker can handle our needs
- Less complexity, fewer dependencies
- Can add features to build tracker as needed

**What to build instead:**
- Parameter logging in build tracker (store config in `builds` table)
- Simple comparison queries (SQL queries)
- Simple UI or CLI for browsing runs (optional)
- Model artifact tracking (already have file outputs)

**When to reconsider:**
- If we need a web UI for browsing experiments
- If we need model registry/staging
- If we switch to neural networks

### Option 2: Use MLflow (If UI/Registry Needed)

**Rationale:**
- Want web UI for browsing experiments
- Need model registry/staging
- Want standard format for sharing models

**What to use:**
- MLflow Tracking API for logging
- MLflow UI for browsing
- MLflow Model Registry (if needed)
- Skip MLflow Projects (use the pstrain pipeline runner instead)

**Integration:**
- Log parameters/metrics from pipeline tasks
- Store model artifacts in MLflow
- Use build tracker for orchestration, MLflow for experiment tracking

### Option 3: Hybrid Approach

**Use MLflow only for:**
- Model registry (if needed)
- Web UI for browsing (if needed)

**Use build tracker for:**
- Orchestration tracking
- Step execution
- File dependencies
- Logs

## Comparison Table

| Feature | Build Tracker | MLflow | Needed? |
|---------|---------------|--------|---------|
| **Run tracking** | ✅ SQLite | ✅ Server/DB | ✅ Yes |
| **Parameter logging** | ⚠️ Can add | ✅ Built-in | ✅ Yes |
| **Metric logging** | ⚠️ Can add | ✅ Built-in | ⚠️ Maybe (evaluation only) |
| **Artifact storage** | ✅ File paths | ✅ Built-in | ✅ Yes |
| **Code versioning** | ⚠️ Can add | ✅ Built-in | ✅ Yes |
| **Run comparison** | ⚠️ SQL queries | ✅ UI | ⚠️ Nice to have |
| **Web UI** | ❌ No | ✅ Yes | ⚠️ Nice to have |
| **Model registry** | ❌ No | ✅ Yes | ⚠️ Maybe |
| **Hyperparameter tuning** | ❌ No | ✅ Yes | ❌ No |
| **Neural network support** | ❌ No | ✅ Yes | ❌ No |

## Decision Framework

### Choose Build Tracker Only if:
- ✅ You want simplicity (fewer dependencies)
- ✅ You don't need a web UI
- ✅ You don't need model registry
- ✅ You're comfortable building simple comparison tools
- ✅ You want full control over tracking

### Choose MLflow if:
- ✅ You want a web UI for browsing experiments
- ✅ You need model registry/staging
- ✅ You want standard format for sharing models
- ✅ You're willing to maintain MLflow server
- ✅ You want convenient logging API

### Choose Hybrid if:
- ✅ You want build tracker for orchestration
- ✅ You want MLflow for experiment browsing/registry
- ✅ You're okay with maintaining both

## My Recommendation

**Start with Build Tracker Only**

**Rationale:**
1. Traditional ML training doesn't fit MLflow's model well
2. Build tracker can handle our needs
3. Simpler setup (no MLflow server)
4. Can add MLflow later if needed (migration is possible)

**What to build:**
1. **Parameter logging in build tracker**
   - Store config parameters in `builds` table (JSON column)
   - Store step parameters in `steps` table

2. **Simple comparison tool**
   - CLI command: `pstrain compare-builds <build1> <build2>`
   - SQL queries to compare parameters/metrics

3. **Simple UI (optional, later)**
   - Can build simple web UI if needed
   - Or use SQLite browser for now

4. **Model artifact tracking**
   - Already have file outputs tracked
   - Can add model metadata if needed

**Revisit MLflow if:**
- Web UI becomes critical
- Model registry becomes needed
- We switch to neural networks
- Team requests it

## Implementation Plan

### Phase 1: Build Tracker Enhancement (Instead of MLflow)

**Tasks:**
- [ ] Add `config` JSON column to `builds` table
- [ ] Add `parameters` JSON column to `steps` table
- [ ] Add `metrics` JSON column to `steps` table (for evaluation metrics)
- [ ] Create `pstrain/lib/tracking.py` for parameter/metric logging
- [ ] Add `pstrain compare-builds` CLI command
- [ ] Document tracking usage

**Deliverables:**
- Enhanced build tracker
- Parameter/metric logging API
- Comparison tool
- Documentation

### Phase 2: Evaluate MLflow (If Needed)

**If web UI or model registry becomes critical:**
- [ ] Evaluate MLflow setup
- [ ] Prototype MLflow integration
- [ ] Compare with build tracker
- [ ] Make decision

## Conclusion

For traditional acoustic model training (HMM/GMM), **MLflow is probably overkill**. Build tracker can handle our needs, and we can add features as needed. Revisit MLflow if:
- Web UI becomes critical
- Model registry is needed
- We switch to neural networks

**Recommendation: Skip MLflow for now, enhance build tracker instead.**
