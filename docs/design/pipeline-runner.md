# Pipeline runner

pstrain uses a small Python-native task runner in `pstrain.lib.pipeline` to
orchestrate the training workflow. This document describes what it is,
how it works, and why we built it instead of using Snakemake.

## What it is

```
pstrain/lib/pipeline/
  runner.py     # Task, Pipeline, staleness, topo sort, execution
  context.py    # PipelineContext, config loading
  tasks.py      # Concrete tasks for the pstrain workflow
```

* **`Task`** — an immutable dataclass: `name`, `fn` (callable),
  `inputs: tuple[Path, ...]`, `outputs: tuple[Path, ...]`,
  `parallel_group: str`, `description: str`.
* **`Pipeline`** — registers tasks and resolves the DAG by matching
  one task's outputs against another's inputs. Plans, checks staleness,
  topologically sorts, and executes.
* **`PipelineContext`** — per-run configuration (project dir,
  experiment, named config, derived feature/training params). Loaded
  from `project/etc/configs.yaml`.

The CLI entry points are:

* `pstrain build <target>` — build a named target (e.g. `cd-8g`).
* `pstrain features` — shortcut for `pstrain build features`.
* `pstrain step <name>` — single-step debugging entry that delegates to
  the same pipeline.

## How it works

### Dependency resolution

Tasks declare file paths. The pipeline indexes outputs and uses
`inputs → outputs` matching to walk the graph (the same model
Snakemake uses).

### Staleness

A task is **stale** when any of:

1. Any declared output is missing.
2. The newest input mtime is strictly greater than the oldest output
   mtime.
3. **Any upstream task is itself stale** (transitively). The planner
   propagates staleness downstream because an upstream's pending
   re-run will produce outputs newer than this task's existing
   outputs.

`--force` marks every reachable task stale unconditionally.

### Execution

Tasks run sequentially by default. Adjacent tasks sharing a
`parallel_group` are batched together and dispatched to a
`ProcessPoolExecutor`. This is how feature extraction fans out across
the ~1000 audio files in `train.fileids` + `test.fileids`. Set
`-j N` on the CLI to choose worker count.

The linear training chain (flat → ci-1g → ci-2g → ... → cd-32g) runs
in-process because each step depends on the previous one's output.

### Dry-run

`--dry-run` prints the topologically-sorted plan with stale/up-to-date
markers and never executes:

```
# Plan for target: cd-1g
# 1108 task(s); 10 stale

. [   1] extract:arctic_a0001  (up to date)
. [   2] extract:arctic_a0002  (up to date)
...
* [1099] flat  (missing output: shared/models/flat/default/feat.params)
* [1100] ci-1g  (missing output: shared/models/ci-1g/default/feat.params)
...
* [1108] cd-1g  (missing output: shared/models/cd-1g/default/feat.params)

# Legend: * = will run, . = up to date
```

## Why we built our own

A previous iteration used Snakemake. The workflow we actually have is
small enough that Snakemake's pull-ins didn't pay off:

* The DAG has ~15 logical nodes plus a fan-out over fileids. Not the
  large, branching, multi-sample DAG Snakemake is designed for.
* Every Snakefile rule's `run:` block just called into
  `pstrain.lib.steps.run_*` Python functions. Snakemake was a thin shim,
  not actually orchestrating shell commands or managing envs.
* Snakemake pulls ~25 transitive dependencies (gitpython, jinja2,
  pulp, nbformat, ...) for what amounts to "if output is older than
  input, re-run."
* The DSL is not real Python. Hard to test, hard to type-check, hard
  to debug. Inputs/outputs were duplicated between Snakefile rules
  and `Step` classes.

The runner replaces ~1100 lines of `Snakefile` + `features.smk` +
`targets.yaml` with ~400 lines of Python in `pstrain/lib/pipeline/`.
Adds zero runtime dependencies. Everything is one process, importable
and debuggable.

## What we explicitly don't support

* **Cluster execution** (Slurm, Kubernetes, etc.). If you need to run
  training on a cluster, Snakemake or Dagster would be a better fit.
* **Per-task conda envs.** pstrain has one Python environment.
* **Content-hash staleness.** Mtime parity with Snakemake is enough;
  layer a content-hash check on top if a real need shows up.

## Multi-pronunciation training

Baum-Welch training defaults to multi-pronunciation mode: each word
with `k` variants in the dictionary contributes `k` parallel phone
paths to the per-utterance training graph, and forward-backward
sums posteriors across them. Variant arc weights are initialized
uniformly (`1/k`) so dictionary row order doesn't pick the acoustic
targets.

Opt out per-config by setting `training.multipron_training: false`
in `etc/configs.yaml`; that config's runs fall through to the
legacy linear path (bit-identical to pstrain's pre-multipron behavior).

See [`multi-pron-training.md`](multi-pron-training.md) for the full
design and the as-built layout.

## Adding a new pipeline node

1. In `pstrain/lib/pipeline/tasks.py`, write a builder that closes over
   `ctx` and returns a `Task`:

   ```python
   def _make_my_task(ctx: PipelineContext) -> Task:
       src = ctx.model_dir("ci-8g")
       out = ctx.model_dir("my-thing")

       def run() -> None:
           from pstrain.lib.something import do_thing
           do_thing(src=src, out=out)

       return Task(
           name="my-thing",
           fn=run,
           inputs=tuple(ctx.model_files("ci-8g")),
           outputs=tuple(ctx.model_files("my-thing")),
           description="Do my thing",
       )
   ```

2. Register it in `build_pipeline()` and add it to `TARGETS` if it
   should be a named build target.

3. If it's a fan-out (one task per fileid, etc.), make `fn` a
   `functools.partial` over a top-level worker function so it
   pickles for `ProcessPoolExecutor`, and set
   `parallel_group="some-name"`.

## Testing

* `tests/test_pipeline_runner.py` — runner behavior in isolation:
  topo sort, staleness, propagation, dry-run, parallel fan-out,
  cycles, failures.
* `tests/test_pipeline_tasks.py` — task graph validation: every
  registered target has a producer, every declared target is
  registered, the cd-8g plan includes the full chain in dependency
  order.
* `tests/test_pipeline_integration.py` — end-to-end training runs
  against a real audio corpus (CMU Arctic via `PSTRAIN_TEST_PROJECT`).
