# Design documentation

Design notes for pstrain. Most of these are background documents written
while making decisions; the current source of truth is the code in
`pstrain/lib/`.

## Current architecture

* [`parity-and-deviations.md`](parity-and-deviations.md) — stock-equivalence
  conditions and the tracked divergence register.
* [`config-unification.md`](config-unification.md) — proposed canonical,
  source-aware configuration schema and migration from the current split
  configuration systems.
* [`one-command-training.md`](one-command-training.md) — proposed
  `pstrain train` workflow from prepared prompts, WAVs, and a dictionary.
* [`native-boundary.md`](native-boundary.md) — the boundary between
  Python and the vendored C library: which operations run inside a
  contained helper process, which are still unguarded, and how native
  failures are classified.
* [`pipeline-runner.md`](pipeline-runner.md) — the task runner that
  orchestrates training (`pstrain.lib.pipeline`). Replaces an earlier
  Snakemake-based design.
* [`training-pipeline.md`](training-pipeline.md) — the full training
  workflow: features → flat → ci-Ng → cd-untied → trees → cd-Ng.
* [`multi-pron-training.md`](multi-pron-training.md) — how Baum-Welch
  training handles multiple pronunciations per word (per-utterance
  graph with parallel variant paths; default on).
* [`task-orchestration.md`](task-orchestration.md) — long-form notes on
  observability, build tracking, and chunk composition. Still useful
  as background; the framework-comparison sections are now stale (we
  rolled our own runner).

## Domain references

* [`project-setup.md`](project-setup.md) — directory structure and
  initial setup.
* [`ci-model-requirements.md`](ci-model-requirements.md) — what CI
  models need (features, dictionary, transcripts, flat init).
* [`ci-training-cli-plan.md`](ci-training-cli-plan.md) — CLI surface
  for stepping through CI training manually.
* [`testing-with-cmu-arctic.md`](testing-with-cmu-arctic.md) —
  end-to-end test corpus.
* [`terminology.md`](terminology.md) — glossary of acoustic-modeling
  terms.

## Past decisions

* [`mlflow-evaluation.md`](mlflow-evaluation.md) — why pstrain doesn't use
  MLflow; we keep a small build tracker instead.

## Decisions reversed (kept here for context)

* **Snakemake vs custom runner.** We initially chose Snakemake; the
  workflow turned out to be small enough that the dependency footprint
  and DSL overhead weren't worth it. See `pipeline-runner.md` for the
  current design. The original `snakemake-vs-dagster.md`,
  `snakemake-implementation.md`, and `framework-evaluation.md` docs
  have been removed; the current rationale lives in
  `pipeline-runner.md` under "Why we built our own."
