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

`--dry-run` prints the topologically-sorted plan and never executes. The
plan uses the same tab-separated shape as the run it predicts: two comment
lines, a header row, then one row per stage.

```
# Plan for target: cd-1g
# 1263 task(s); 1263 stale
index	stage	tasks	status	description
1	provenance:split	1	not built yet	Record effective split configuration
2	split	1	upstream 'provenance:split' will run	Partition all.transcription into train/test fileids + transcripts
3	provenance:features	1	not built yet	Record effective features configuration
4-1135	features	1132	stale
1136	provenance:training	1	not built yet	Record effective training configuration
1137	flat	1	upstream 'split' will run	Initialize flat (uniform) acoustic model
1138	ci-1g	1	upstream 'flat' will run	Train CI-1g (1 Gaussian per state)
...
1263	cd-1g	1	upstream 'cd-1g-init' will run	Train tied CD-1g model
```

`index` is a position in the plan. A collapsed fan-out reports as one row
carrying the range of positions it spans and the number of tasks in it, so
a plan holding 1,132 per-utterance feature tasks still shows the shape of
the build. `--verbose` lists every member instead.

`status` speaks three vocabularies:

* `not built yet` — a declared output does not exist. This is the normal
  state of every stage in a fresh project, so it is not phrased as a fault.
* `stale` — outputs exist but are older than inputs. A partly-cached
  fan-out reports the fraction, `3 of 40 stale`, and a group with nothing
  to do reports `up to date`.
* `upstream '<task>' will run` — this stage is due only because something
  it depends on is.

There are no bullet markers and no continuation line for the description,
so every row carries the same columns and a plan pastes as a TSV beside the
progress rows it foretells.

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

## Stage-specific Baum-Welch control

Training schedules are configured independently because the upstream recipes
do not apply one variance history and endpoint to every stage:

```yaml
training:
  ci: {max_iterations: 10, min_iterations: 1, convergence_ratio: 0.001}
  untied: {max_iterations: 6, min_iterations: 1, convergence_ratio: 0.001}
  tied: {max_iterations: 10, min_iterations: 1, convergence_ratio: 0.001}
```

All three use the SphinxTrain signed likelihood-delta decision and may stop
before their cap after `min_iterations`. The six-pass untied cap records the
effective endpoint of the preserved CMU Arctic SLT run; upstream stage 30 is a
converge-with-cap loop, not a fixed-count loop. CI and tied stages retain the
A7c-matched 0.001 decision threshold and upstream ten-pass cap.

Variance accumulation is deliberately code-defined by stage. CI and each newly
split tied stage use one-pass variance on their first iteration and centered
two-pass variance thereafter. CD-untied uses centered two-pass variance from
its first iteration, matching the unconditional `-2passvar yes` in
`scripts/30.cd_hmm_untied/baum_welch.pl`.

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
