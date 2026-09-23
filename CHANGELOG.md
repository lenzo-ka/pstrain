# Changelog

Release-relevant changes are recorded here. The project is currently an alpha;
the version in `pyproject.toml` is authoritative.

## Unreleased

### Upgrading

- Rebuild source installations and custom native integrations together: the
  native ABI is now 6. Published wheels bundle their matching native library.

### Changes

- Training and `pstrain align` now say when an utterance cannot be aligned at
  any beam width because its transcript needs more frames than the audio has.
  The report names the required minimum and the frames available instead of the
  generic "final state not reached", and the wider-beam retry is skipped, since
  no beam can recover such an utterance. The minimum is measured on the graph
  the engine actually built, so it is exact for that model and dictionary.
- Every Baum-Welch pass now reports how many utterances the wider-beam retry was
  spent on and how many it recovered, and the training telemetry records the same
  pair per pass. The stage summary totals the second forward passes across every
  pass, so an utterance retried on three passes counts three times there.
  Nothing about when the retry runs has changed; it simply says what it bought.
- With `failed_alignment` set to `omit`, an utterance dropped for the same reason
  on later passes is now reported once for the stage rather than once per pass.
  The stage omission summary still lists every pass it was dropped on.
- `pstrain align` now checks the pronunciation dictionary against the phone
  inventory the model was trained on, and prints one collected report before
  the run naming every word whose pronunciation uses a phone the model does
  not define. Those pronunciations were already being dropped silently, so
  the words surfaced much later as alignment failures that read like an
  out-of-vocabulary problem. Such a failure now says which word lost its
  pronunciation and which phone was missing.
- The report distinguishes the case that stops alignment outright. When a
  word's unsuffixed pronunciation is dropped and an alternative survives, the
  aligner refuses to start and no utterance aligns, including utterances that
  never use that word. The report says so before the run and names the word,
  and the resulting failures carry the reason instead of an opaque
  initialization error. Training does not share that rule, and the report
  says which outcome applies.
- A dictionary or model the check cannot read no longer turns the check off
  silently; the reason is reported.
- `pstrain validate` and the input validation run by `pstrain train` name the
  affected words and their pronunciations when the dictionary uses phones
  outside the phoneset, rather than counting the phones alone.
- The native lexicon loader states how many pronunciations it dropped for
  undefined phones, so its end-of-load summary no longer reports only the
  entries that loaded.

## 0.4.1 - 2026-09-16

### Fixes

- Training with `training.split_variance_floor_fraction` above zero no longer
  aborts when normalization leaves small negative variances in sparsely observed
  split densities. The bound now replaces them, storing zero where the reference
  variance is zero. Variances that are NaN or infinite still stop training.

### Changes

- Documentation is published at https://lenzo-ka.github.io/pstrain/ from each
  release, so it matches the installed version. Working plans and progress notes
  remain in the repository but are not part of the published site.
- Removed an unused copy of the sphinxbase bit array from the native library.
  Language model reading is unchanged; it uses PocketSphinx's own implementation.

## 0.4.0 - 2026-09-13

### Upgrading from 0.3.0

- Rebuild source installations and custom native integrations together: the
  native ABI is now 5. Published wheels bundle their matching native library.
- Multipronunciation training now honors multiple requested workers. Different
  shard counts may change floating-point reduction and later training results;
  use one worker when retaining the serial path matters.
- The split-variance bound is opt-in and defaults to zero. Checkpoint restoration
  is also explicit; neither feature silently repairs or rewinds training.
- Training supports 13-coefficient `1s_c_d_dd` features and now rejects unsupported
  front-end configurations early. Standalone feature extraction supports other
  widths; pass their actual `veclen` when reading nondefault Sphinx MFC files.
- Native Windows training remains unsupported despite Windows library and wheel
  support. See [platform support](docs/support.md).

### Changes

- Multipronunciation Baum-Welch training now uses the requested worker count
  through separate processes. Shards merge raw statistics and fallback-state
  activation before applying the survival prior once; one-worker training
  retains its serial path. Interrupted workers and workers whose coordinator
  exits are contained and cleaned up.
- Training no longer reports a negative or nonfinite likelihood change as
  convergence. Optional `training.split_variance_floor_fraction` bounds split
  variances against fixed one-Gaussian references; its default remains zero.
- Added `pstrain checkpoints` for inspection and explicit restoration of a
  retained checkpoint, with dry-run and JSON support. Update N is evaluated in
  pass N+1; evidence binds that model and its associated counts. The final
  update remains unevaluated until scored. Restoration preserves a
  backup and invalidates completion metadata; training never rolls back
  automatically.
- The quick start now explains Python prerequisites and virtual environments,
  and provides an exercised bundled-corpus training, testing, and packaging
  walkthrough. The tutorial uses isolated corpus downloads and is executed
  offline in CI.
- Configuration, corpus validation, feature dimensions, model buffer layouts,
  and pipeline dependency tracking now fail clearly on inconsistent inputs.
  Model updates preserve raw count semantics and recover from caught file-write
  failures instead of leaving partially replaced parameter sets.
- Native float/model storage and byte swaps avoid strict-aliasing violations;
  corpus fetching is isolated from notebook processes. Windows native-worker
  requests use the supported pipe transport. Native Windows training remains
  unsupported; see the platform support guide.
- Runtime tests enforce CLI access through the public API. Commands
  honor JSON output or reject unsupported JSON use explicitly. Benchmark
  adoption rejects incomplete decoding rather than treating it as a valid score.

- The CMU Pronouncing Dictionary can now be fetched, converted, and cached from
  its original upstream repository through `pstrain.api.dictionary`, defaulting
  to the latest revision and accepting a pinned one; the resolved commit and
  per-file digests are recorded alongside the cache (2026-09-08).
- The bundled Arctic dictionary was regenerated so the committed file and the
  stress-stripping conversion agree; the entries are unchanged and only their
  order differs (2026-09-08).
- The pinned Arctic benchmark band was re-measured under the ten-pass untied
  schedule and the regenerated dictionary. The live cells moved in opposite
  directions: SLT-55 improved from 142 to 139 errors, and the larger set moved
  from 22,564 to 22,703. Neither difference against the preserved upstream
  model is statistically significant; the paired intervals span zero
  (2026-09-08).
- CI, untied CD, and tied CD training schedules now share the
  `max_iterations: 10` default instead of applying different pass limits
  (2026-09-07).
- `pstrain tutorial` now copies the bundled HMM/GMM tutorial notebook from an
  installed package, explains how to launch it, and protects an existing copy
  unless `--force` is given (2026-09-06).
- `strip_dictionary_stress()` now merges pronunciations that become duplicates
  and renumbers the surviving variants; the bundled Arctic dictionary and its
  benchmark evidence have been regenerated from that output (2026-09-06).
- The Arctic benchmark documentation now identifies the preserved upstream
  oracle and its provenance limits, compares resource-matched runs, and states
  when a measured difference is not statistically significant (2026-09-06).
- Documentation now accurately describes Windows support, the distinction
  between `pstrain.api` and `pstrain.lib`, the evaluation-metrics extra, project
  validation reports, model parameters, and the supported `LogMath` wrapper
  (2026-09-06).
- Guarded Arctic comparison, adoption, and documentation commands now refuse
  baseline evidence that differs from the committed files unless a local
  experiment explicitly overrides the check (2026-09-06).
- `pstrain package` now provides dry-run planning and packages trained models
  from the command line; it refuses source overlap, unrelated destinations, and
  unsupported markers, requires `--overwrite` for unmarked legacy packages,
  and replaces recognized packages transactionally (2026-09-05).
- Training now supports SphinxTrain-compatible skip-state phone topology through
  the canonical `skip_state` configuration setting (2026-09-05).
- Training output now reports concise tab-separated progress, avoids repeated
  warnings, and summarizes omitted utterances and affected pass ranges
  (2026-09-05).
- Parallel decoding now shuts down its native helpers cleanly instead of hanging
  after decoding has finished (2026-09-05).
- `scripts/procctl.py launch` now tolerates slow identity readers and exec
  wrappers and records untruncated command observations; a refused launch can
  take about sixty-eight seconds while settling and cleanup finish (2026-09-05).
- `pstrain train` and `pstrain.api.one_command.validate_inputs()` now accept
  Festival and FestVox prompt files; `pstrain.api.diagnostics` exposes
  Baum-Welch telemetry readers, while the top-level public API exports CMUdict
  and stress-stripping helpers and remains importable on Windows (2026-09-05).
- Arctic setup now records every missing vocabulary word and the affected
  utterances that remain in the corpus. Segment aggregation and flat
  initialization report omitted utterances and their reasons; standalone
  aggregation fails on omissions unless they are explicitly allowed, and flat
  initialization enforces the configured skip tolerance (2026-09-05).
- Malformed model indexes and inconsistent transition matrices now stop native
  initialization with model-specific diagnostics instead of risking invalid
  memory access or incomplete output (2026-09-05).
- `mllr_transform` now returns failure and explains the problem when required
  transform or model paths are missing (2026-09-05).
- The decoder now logs expected density-probe failures before falling back to
  PocketSphinx's default top-N setting (2026-09-05).
- Big-endian segment reads now use defined unsigned byte swaps, avoiding
  platform-dependent results and native undefined behavior (2026-09-05).
- The `sphinxtrain` profile now uses SphinxTrain's convergence threshold while
  the accuracy-oriented default profile retains its stricter threshold
  (2026-09-04).
- Baum-Welch per-utterance diagnostics now go to tab-separated project log files
  instead of flooding terminal and notebook output (2026-09-04).
- Training now retries an unalignable utterance with a wider beam and, if it
  still fails, reports and omits it while enforcing the configured skip limit;
  strict abort behavior remains available (2026-09-04).
- Native worker failures now preserve their diagnostics across process
  boundaries and distinguish a startup timeout from a worker crash
  (2026-09-03 through 2026-09-04).
- Language-model building now reads both supported transcript forms, and
  malformed transcripts fail with a file and line number. `pstrain test`
  automatically builds a language model from training text when no model option
  is supplied (2026-09-04).
- `pstrain test` now fails with the recorded decoder reasons when requested
  utterances produce no result instead of reporting a fabricated score
  (2026-09-03).
- The tutorial notebook now follows the checkout version, renders its likelihood
  formula correctly, installs its evaluation dependency, tolerates the optional
  PocketSphinx package, and safely replaces stale project configuration
  (2026-09-03).

## 0.3.0 - 2026-08-24

- Forced alignment now retries once with a wider beam when an utterance cannot
  reach its final state, recovering long or acoustically hard utterances that a
  fixed beam would otherwise drop; this mirrors the Baum-Welch training retry.
  Adds `alignment.beam`, `alignment.retry_beam_factor`, and
  `alignment.failed_alignment` configuration and a native live-beam setter, and
  wires the previously accepted-but-ignored `pstrain align --beam` (2026-08-24).
- The decoder now caps its top-N Gaussian count at the model's density, so a
  model with fewer densities than the default no longer logs a spurious `-topn`
  warning on every decoded utterance; decoding results are unchanged (2026-08-24).
- Added an end-to-end HMM/GMM training tutorial notebook covering corpus
  preparation, feature extraction, training, forced alignment, decoding, and
  packaging (2026-08-24).

## 0.2.0 - 2026-08-18

- Added a structured `pstrain.api` as the supported programmatic entry point;
  the command-line interface now reaches the training library only through it,
  enforced by a boundary check in the verified build (2026-08-18).
- Added a `bench` extra providing PocketSphinx for the `pip-en-us` benchmark
  band, and corrected the install hint the benchmark error printed (2026-08-17).
- Split the README quickstart into "from PyPI" and "from a checkout" so a
  pip-installed user can run it without the repository fixtures (2026-08-17).
- `make_quests` now fails loudly on degraded input statistics instead of writing
  a plausible-looking question file (2026-08-17).
- Relocated the Arctic benchmark pin evidence under `evidence/` so a record
  change runs its authenticating gate, and added a generated coverage statement
  and an oracle-provenance note (2026-08-17, 2026-08-18).
- The package version now falls back to `0.0.0+unknown` instead of raising at
  import when the package is relocated without its metadata (2026-08-17).

## 0.1.1 - 2026-08-17

- Added self-contained Windows wheels with bundled native dependencies
  (2026-08-17).
- Extended build and PyPI release workflows to cover Windows wheels
  (2026-08-17).
- Added PE artifact checks to the native floating-point contract gate
  (2026-08-17).

## 0.1.0 - 2026-08-17

- Imported the original pstrain/SphinxTrain-derived codebase (2026-07-03).
- Adopted the BSD 2-Clause license for new pstrain code while preserving the
  notices and license terms of the CMU-derived C sources (2026-07-08).
- Added and pinned the reproducible CMU Arctic benchmark baseline and its
  immutable corpus metadata (2026-08-11).
- Contained fatal native CFFI operations behind a reusable worker boundary and
  completed the audited CFFI operation surface (2026-08-10 through 2026-08-11).
