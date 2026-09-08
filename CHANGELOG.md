# Changelog

Release-relevant changes are recorded here. The project is currently an alpha;
the version in `pyproject.toml` is authoritative.

## Unreleased

- CI, untied CD, and tied CD training schedules now share the
  `max_iterations: 10` default instead of applying different pass limits
  (2026-09-07).
- `pstrain tutorial` now copies the bundled HMM/GMM tutorial notebook from an
  installed package, explains how to launch it, and protects an existing copy
  unless `--force` is given (2026-09-06).
- Stress stripping now merges pronunciations that become duplicates and
  renumbers the surviving variants; the bundled Arctic dictionary and its
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
- `pstrain package` now provides dry-run planning and safe model packaging from
  the command line and public API; replacements are transactional and refuse
  unsafe destinations, unrelated data, and unsupported package markers
  (2026-09-05).
- Training now supports SphinxTrain-compatible skip-state phone topology through
  the canonical `skip_state` configuration setting (2026-09-05).
- Training output now reports concise tab-separated progress, avoids repeated
  warnings, and summarizes omitted utterances and affected pass ranges
  (2026-09-05).
- Parallel decoding now shuts down its native helpers cleanly instead of hanging
  after decoding has finished (2026-09-05).
- Project setup now accepts Festival and FestVox prompt files, and the public API
  exposes Baum-Welch telemetry readers, CMUdict, and stress-stripping helpers
  while remaining importable on Windows (2026-09-05).
- Arctic setup, segment aggregation, and flat initialization now report every
  omitted utterance and its reason; standalone aggregation fails on omissions
  unless they are explicitly allowed, and training enforces its configured skip
  tolerance during flat initialization (2026-09-05).
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
- `pstrain test` now reads both supported transcript forms and automatically
  builds a language model from training text when no model option is supplied;
  malformed transcripts fail with a file and line number (2026-09-04).
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
