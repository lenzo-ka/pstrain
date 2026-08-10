# Code Review: pstrain/lib/steps and pstrain/lib/testing

**Date:** 2026-01-15
**Focus:** Correctness, DRY violations
**Scope:** `pstrain/lib/steps/`, `pstrain/lib/testing/`, related modules

---

## Priority 1: Correctness Bugs

### ✅ 1.1 `train.py:224-225` — `TrainingResult.final_frames` not populated — FIXED

**File:** `pstrain/lib/steps/train.py`
**Lines:** 220-226

```python
return TrainingResult(
    iterations=n_iter,
    converged=False,
    final_likelihood=prev_likelihood,
    final_frames=0,  # Will be filled from last iteration  <-- BUG
    final_utts=len(fileids),
)
```

**Issue:** Comment says "Will be filled from last iteration" but value is hardcoded to 0. The last iteration's `stats.total_frames` is never captured for the non-converged case.

**Fix:** Store stats from last iteration before the loop ends:
```python
last_stats = stats  # after stats = trainer.get_stats() in loop
# ...
return TrainingResult(..., final_frames=last_stats.total_frames, ...)
```

---

### ✅ 1.2 `decoder.py:110` — Fragile CI model detection — FIXED

**File:** `pstrain/lib/testing/decoder.py`
**Lines:** 109-110

```python
model_path_str = str(self.model_dir).lower()
is_ci_model = "/ci" in model_path_str or model_path_str.endswith("/ci")
```

**Issue:** String-based detection is fragile. Paths like `/home/citadel/models/cd-8g` would incorrectly match as CI due to "ci" in "citadel".

**Fix:** Check for CI by examining the mdef file (CI models have no triphones), or use a more specific pattern like `/ci-` or `/ci/`:
```python
is_ci_model = "/ci-" in model_path_str or "/ci/" in model_path_str or model_path_str.endswith("/ci")
```

Or better, detect from model structure:
```python
is_ci_model = self._detect_model_type() == "ci"  # Check mdef for triphones
```

---

### ✅ 1.3 `cd_hmm_untied.py:91-98` — Inconsistent return codes — FIXED

**File:** `pstrain/lib/steps/cd_hmm_untied.py`
**Lines:** 91-98

```python
if ctx.dry_run:
    ctx.log_comment("CD HMM untied training not yet implemented")
    return 0  # Success in dry-run

# TODO: Implement actual training
ctx.log("")
ctx.log("CD HMM untied training not yet implemented.")
return 1  # Failure in real execution
```

**Issue:** Dry-run returns success (0), but actual execution returns failure (1) for the same unimplemented feature. This is misleading.

**Fix:** Either raise `NotImplementedError` consistently (like `ci_hmm.py`), or return the same code in both paths.

---

### ✅ 1.4 `cd_pipeline.py:93-137` — `_parse_mdef_stats` reads file twice — FIXED

**File:** `pstrain/lib/steps/cd_pipeline.py`
**Lines:** 93-137

```python
def _parse_mdef_stats(mdef_path: Path) -> dict[str, int]:
    # ...
    with open(mdef_path) as f:
        for line in f:
            # First pass logic

    # Count tied states from the mdef entries
    with open(mdef_path) as f:  # <-- Second open
        content = f.read()
```

**Issue:** File is opened and read twice. The second pass re-reads the entire file.

**Fix:** Read file once and process both passes on the same content.

---

### ✅ 1.5 `ci_hmm.py:132-146` — Unreachable code — FIXED

**File:** `pstrain/lib/steps/ci_hmm.py`
**Lines:** 132-146

```python
raise NotImplementedError(
    "Baum-Welch training not yet implemented via CFFI. "
    "Need to create pstrain.lib.bw module with CFFI bindings to bw functions."
)

# TODO: Future implementation structure:  <-- Unreachable
# for iteration in range(1, merged_params["max_iterations"] + 1):
```

**Issue:** Comments after `raise` are unreachable. Not a runtime bug, but confusing dead code.

**Fix:** Move the TODO comment before the raise, or remove if the implementation now exists in `train.py`.

---

### ✅ 1.6 `package.py:93` vs `filler.dict` — Inconsistent filler entries — FIXED

**File:** `pstrain/lib/steps/package.py:89-93`

```python
# Create minimal noisedict
with open(output_path, "w") as f:
    f.write("<s> SIL\n")
    f.write("</s> SIL\n")
    f.write("<sil> SIL\n")
```

**File:** `pstrain/data/filler.dict`
```
<sil> SIL
<s> SIL
</s> SIL
```

**Issue:** The hardcoded noisedict has the same entries but in a different order than `filler.dict`. While functionally equivalent, inconsistency can cause confusion. More importantly, if `filler.dict` is updated, the hardcoded fallback won't match.

**Fix:** Use a shared constant or always require `filler_dict_path`.

---

## Priority 2: DRY Violations

### ✅ 2.1 `WERResult` and `TestResult` — Duplicate properties — FIXED

**Files:** `pstrain/lib/testing/wer.py`, `pstrain/lib/testing/test.py`

Both classes define nearly identical properties:
- `accuracy` (same implementation)
- `errors` (same implementation)
- `total_words` → `ref_words` alias
- `correct` → `hits` alias
- `to_dict()` with overlapping fields

**Fix:** Either:
1. Have `TestResult` inherit from or compose `WERResult`, or
2. Extract a mixin/base class with shared properties, or
3. Have `TestResult` contain a `WERResult` field and delegate

Recommendation: Option 3 — `TestResult` should have a `wer_result: WERResult` field instead of duplicating all WER fields.

---

### ✅ 2.2 Feature params defaults repeated — FIXED

**Locations:**
- `pstrain/lib/steps/features.py:30-39` (default_params dict)
- `pstrain/lib/steps/package.py:21-28` (function defaults)
- `pstrain/lib/steps/package.py:177-184` (dict.get calls)

```python
# features.py
default_params: dict[str, Any] = {
    "samprate": 16000,
    "nfilt": 40,
    "nfft": 512,
    ...
}

# package.py
def create_feat_params(
    output_path: Path,
    samprate: int = 16000,
    nfilt: int = 40,
    nfft: int = 512,
    ...
)
```

**Fix:** Define canonical defaults in one place (e.g., `pstrain.lib.config.defaults`) and import everywhere:
```python
from pstrain.lib.config import DEFAULT_FEAT_PARAMS
```

---

### ✅ 2.3 Model file lists repeated — FIXED

Added `MODEL_FILES_REQUIRED`, `MODEL_FILES_OPTIONAL`, `MODEL_FILES_ALL` constants to `pstrain/lib/model.py`.

**Locations:**
- `pstrain/lib/steps/ci_hmm.py:42-47` (get_inputs flat model files)
- `pstrain/lib/steps/ci_hmm.py:58-65` (get_outputs hmm model files)
- `pstrain/lib/steps/cd_hmm_untied.py:43-53` (inputs)
- `pstrain/lib/steps/cd_hmm_untied.py:58-64` (outputs)
- `pstrain/lib/steps/split.py:85-91` (required_files)
- `pstrain/lib/steps/package.py:156-162` (model_files)
- `pstrain/lib/testing/test.py:144-145` (required_files)

All refer to the same model files: `mdef`, `means`, `variances`, `mixture_weights`, `transition_matrices`.

**Fix:** Define a constant:
```python
# pstrain/lib/model.py or pstrain/lib/constants.py
MODEL_FILES = ["mdef", "means", "variances", "mixture_weights", "transition_matrices"]
```

---

### 2.4 Path conversion boilerplate — SKIPPED (low value)

**Pattern appearing in many functions:**
```python
def run_something(
    model_dir: Path,
    output_dir: Path,
    ...
) -> ...:
    model_dir = Path(model_dir)
    output_dir = Path(output_dir)
    ...
```

This appears in:
- `train.py:67-69`
- `cd_pipeline.py:59-62, 158-160, 329-331, 371-374, etc.`
- `split.py:77-78`
- `package.py:141-142`

**Fix:** Use a decorator or accept only `Path` (not `Path | str`) and let callers convert. Or use `Path()` at call sites.

---

### ✅ 2.5 Duplicate file existence validation — FIXED

**Pattern:**
```python
required_files = [path1, path2, ...]
for f in required_files:
    if not f.exists():
        raise FileNotFoundError(f"Required file not found: {f}")
```

Appears in: `train.py:72-84`, `cd_pipeline.py:65-67`, `split.py:85-94`, `test.py:144-147`

**Fix:** Extract to utility:
```python
def validate_files_exist(files: list[Path], context: str = "") -> None:
    for f in files:
        if not f.exists():
            raise FileNotFoundError(f"Required file not found: {f}" + (f" ({context})" if context else ""))
```

---

## Priority 3: Minor Issues / Cleanup

### ✅ 3.1 Magic number in `split.py:25` — FIXED

```python
count_per_gaussian: float = 1000.0,
```

**Fix:** Define as named constant with explanation:
```python
DEFAULT_UNIFORM_COUNT = 1000.0  # Minimum occupancy for Gaussian splitting
```

---

### ✅ 3.2 `StepContext` mutable dataclass field — FIXED

**File:** `pstrain/lib/steps/base.py:39`

```python
@dataclass
class StepContext:
    ...
    _header_emitted: bool = False
```

While not a bug (dataclasses handle this correctly), having a mutable-looking field as a default is a code smell. It works because `bool` is immutable.

**Fix:** Use `field(default=False)` for clarity:
```python
_header_emitted: bool = field(default=False, repr=False)
```

---

### ✅ 3.3 Unused backward-compat functions in `ci_hmm.py` — FIXED

`step_20_ci_hmm` and `run_step_20` had no callers anywhere in the codebase
(only `ci_hmm_step.to_dict()` / `ci_hmm_step.run()` are used). Removed.

---

## Prioritized Action Plan

| Priority | Item | Status |
|----------|------|--------|
| P1 | Fix `TrainingResult.final_frames` bug | ✅ Done |
| P1 | Fix CI model detection in decoder | ✅ Done |
| P1 | Fix inconsistent return codes in cd_hmm_untied | ✅ Done |
| P1 | Fix double file read in `_parse_mdef_stats` | ✅ Done |
| P1 | Clean up unreachable code in ci_hmm | ✅ Done |
| P1 | Align noisedict with filler.dict | ✅ Done |
| P2 | Extract `WERResult` composition in `TestResult` | ✅ Done |
| P2 | Create `MODEL_FILES_*` constants | ✅ Done |
| P2 | Create `DEFAULT_FEAT_PARAMS` constant | ✅ Done |
| P2 | Extract `validate_files_exist()` utility | ✅ Done |
| P3 | Add constant for uniform count default | ✅ Done |
| P3 | Path conversion boilerplate | Skipped (low value) |
| P3 | `StepContext` field clarity | ✅ Done |
| P3 | Remove unused `ci_hmm.py` backward-compat shims | ✅ Done |

---

## Summary

**Completed:** 13 items (all P1, all P2, all actionable P3)
**Skipped:** 1 item (path conversion - low value, would require touching many files)
