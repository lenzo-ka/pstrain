# Canonical configuration

Status: proposed decision for C1. This document defines the target contract;
it does not change current behavior.

## Decision summary

Use one strict Pydantic schema as the canonical configuration model, but replace
the current Pydantic field set with the fields and names actually consumed by
`PipelineContext`. Keep named profiles as the user-facing unit. Resolve every
run into one validated model, then project that model into the immutable
runtime dataclasses used by the pipeline.

The current `PstrainConfig` is not that model. It describes a second,
substantially inactive configuration system. C2 should provide a migration
reader for it and then remove it, rather than preserve two schemas behind a
common facade.

## Current state

### The active schema

`pstrain build` constructs `PipelineContext` from
`project/etc/configs.yaml`. Each top-level key is a named profile. A profile
contains `features`, `training`, `split`, and `runner` blocks plus a
description. `FeatParams`, `TrainParams`, `TrainingSchedule`, `SplitParams`,
and `RunnerParams` in `pstrain/lib/pipeline/context.py` define the accepted
fields and defaults. The loader rejects unknown fields and performs several
additional value and cross-field checks.

This is the active schema because `pstrain/lib/pipeline/tasks.py` reads those
objects when it constructs feature, split, initialization, Baum-Welch,
decision-tree, packaging, and runner tasks. Effective feature, split, and
training values also participate in provenance fingerprints.

Built-in profiles live twice: as `DEFAULT_CONFIGS` in the Python module and in
the repository's `etc/configs.yaml`. Project setup writes the Python copy into
new projects. At load time, project profiles replace built-in profiles by
name; the replacement is shallow at the profile boundary, while missing
fields inside the selected profile fall back to dataclass defaults. Thus a
project `default` profile does not inherit individual values from the built-in
`default` mapping.

### The inactive Pydantic schema

`PstrainConfig` and its nested models in `pstrain/lib/config/models.py`
describe different names and shapes, including `audio.sample_rate`,
`features.num_ceps`, `training.n_states`, and `parallel.n_jobs`. `ConfigManager`
merges user, project, and experiment YAML into this model. `pstrain setup`
writes `etc/config.yaml`, and the `pstrain config show|get|set|list|schema`
commands operate on it.

The training build does not load that merged object. For example, changing
`features.num_ceps` with `pstrain config set` does not change the active
`features.ncep`; changing `parallel.n_jobs` does not set the pipeline runner's
`runner.jobs`. The config-reference generator imports this inactive Pydantic
schema, so `docs/api/config-reference.rst` describes values that the principal
training path does not consume. This is the D-bundle deferral that C1 must
close.

Some validation in the inactive schema now rejects stale feature and training
keys and directs users toward profiles. That reduces one failure mode but does
not make the schema active.

### Entry surfaces and precedence today

There is no single precedence chain today; there are separate consumers:

| Surface | Current consumer | Current precedence and effect |
| --- | --- | --- |
| Dataclass defaults | `PipelineContext` | Lowest active build defaults. |
| Built-in named profiles | `load_configs` | Replace dataclass defaults for fields they contain. |
| `project/etc/configs.yaml` | `load_configs` | A project profile replaces the same-named built-in profile as a whole; other built-ins remain discoverable only indirectly through an unknown-name error. |
| `pstrain build -c/--config NAME` | build CLI | Selects one profile; default is `default`. It does not name a file. |
| `pstrain build -j/--jobs N` | pipeline runner | Overrides `runner.jobs` for that invocation. Auto resolution is CPU count minus two. |
| `pstrain build --experiment NAME` | paths and provenance | Selects experiment outputs; it does not load experiment configuration. |
| `~/.pstrain/config.yaml` | inactive `ConfigManager` and config CLI | Lowest tier in the inactive merge. User defaults are translated into the inactive field names. |
| `project/etc/config.yaml` | setup and inactive config CLI | Overrides user defaults in the inactive merge; ignored by `pstrain build`. |
| `project/experiments/NAME/config.yaml` | inactive config CLI | Overrides project and user values when an experiment is requested; ignored by `pstrain build`. |
| Setup flags | `pstrain setup` | Direct arguments for source files, linking, validation, and overwrite behavior. `--config FILE` is copied and validated as inactive `PstrainConfig`; it does not select a named build profile. |
| Individual command and `pstrain step` flags | their command implementations | Direct per-command values. They bypass profile resolution and may have defaults different from the active profile (for example split flags). |
| Global `--dry-run` and JSON flags | CLI framework | Invocation behavior and output only; they are not training configuration. |
| `PSTRAIN_BIN_DIR`, `PSTRAIN_LIB_PATH`, `PSTRAIN_INCLUDE_DIR`, and platform library-path variables | native path discovery | Process environment overrides installation discovery, not model configuration. |
| `PSTRAIN_BW_CHECKPOINTS` | Baum-Welch training | Debug artifact switch read directly by the engine wrapper. |
| `PSTRAIN_REQUIRE_CLIB`, `PSTRAIN_TIMINGS_FAULT`, `PSTRAIN_GOLDEN_X86_64_STRICT`, and `PSTRAIN_BENCH_CACHE` | tests, fault injection, or benchmark harness | Operational/test controls outside ordinary project configuration. |

The old `pstrain/lib/dictionary.py` module no longer exists. Dictionary code is
already a package at `pstrain/lib/dictionary/`; C2 must reconcile stale
references and compatibility expectations, not delete a second live module.

### The lying-surface archetype

The A4 ledger found that `training.convergence_ratio` could be declared in the
active profile yet never reached `run_bw_training`; commit `71da8c6` repaired
that parameter plumbing along with other training and tree controls. The later
A7b history treated this as a lying configuration surface: accepted syntax
and plausible documentation asserted control that the engine did not honor.
That failure is more serious than an unknown key because it produces a valid
looking, reproducible-looking run with different semantics.

## Truthfulness invariant

**A declared configuration value must reach the engine component it governs,
or resolution must fail loudly before work starts.**

This invariant applies to files, CLI overrides, environment adapters, generated
documentation, provenance, and programmatic entry points. A field is not
complete merely because it validates. Each canonical field must have:

1. a schema definition and documented default;
2. a traceable resolution source;
3. an explicit runtime consumer or an explicit designation as CLI-only
   metadata;
4. a test demonstrating the consumer receives a non-default value; and
5. inclusion in the relevant provenance fingerprint.

CI should fail if a canonical field has no registered consumer, if a runtime
consumer accepts an unregistered configuration value, or if generated
reference output differs from the checked-in document. These checks make the
A7b lying-config-surface class mechanically difficult to reintroduce.

## Proposed model

### Canonical schema and runtime projection

Define a strict Pydantic `Profile` model whose names and nesting initially
match the active profile contract: `features`, `training` with per-stage
schedules, `split`, `runner`, and metadata. Pydantic is the canonical schema
because it supplies validation, descriptions, JSON Schema, migration hooks,
and structured introspection. The existing frozen dataclasses remain useful
runtime value objects, but become generated projections with no independent
defaults or validation policy.

Paths and corpus resources belong in a project section of the same canonical
document, outside named model profiles. Experiment-specific overrides belong
in an explicit experiment overlay. This preserves reusable profiles while
bringing project and experiment settings into the same resolver.

Remove the present `PstrainConfig` after migration. Do not alias its mismatched
fields indefinitely: aliases hide ambiguity about which default and consumer
is authoritative.

### Proposed precedence

From lowest to highest:

1. canonical schema defaults;
2. installed built-in profile;
3. user defaults in `~/.pstrain/config.yaml`;
4. project configuration and the selected project profile;
5. experiment overlay;
6. supported environment overrides, if any are deliberately registered;
7. explicit CLI overrides.

Resolution is a deep, field-level merge after every layer has been migrated to
the current schema version. Unknown keys and type errors fail with the layer's
file and field path. Environment variables must not acquire generic automatic
mapping; each supported variable needs a schema field, parser, documentation,
and provenance policy. Test-only fault switches remain outside the canonical
user contract and are labeled as such.

CLI options that merely select a project, experiment, profile, output format,
or dry-run mode are selectors or presentation controls, not extra schema
layers. Direct semantic flags such as `--jobs` are recorded as overrides.

### Versioning and migration

Every canonical project document carries `config_version`, beginning with
version 1. Readers support the current version and a bounded set of older
versions. Migration is deterministic, side-effect-free during inspection, and
reported as old path, new path, and any changed interpretation.

`pstrain config migrate --check` prints the proposed conversion without
writing. `pstrain config migrate` writes a canonical file atomically and keeps
a timestamped backup. Ambiguous collisions—for example both
`features.num_ceps` and active `features.ncep` with different values—stop and
request a choice. Removed fields stop with a reason rather than being dropped.

Existing directories may contain `etc/configs.yaml`, `etc/config.yaml`, and
experiment configs. The migration reader loads all three, reports which were
actually effective before migration, converts active profiles first, and only
maps inactive values that have an unambiguous canonical consumer. It never
claims that a formerly ignored inactive value affected historical builds.

### Explainability and discovery

`pstrain config explain [KEY]` resolves exactly as the requested build would.
It accepts the same project, experiment, profile, and semantic override
selectors as `pstrain build`. For each field it prints:

- resolved value and canonical type;
- winning source, including file and YAML path or exact CLI flag;
- overridden candidates in precedence order;
- default and validation constraints;
- runtime consumer and relevant provenance scope; and
- a short reason, such as “CLI override wins over project profile.”

Machine-readable JSON contains stable source-kind and field-path identifiers.
An unknown key, invalid layer, or declared field without a runtime consumer is
an error.

`pstrain config profiles` lists built-in and project profiles, descriptions,
origin, schema version, and whether a project profile shadows a built-in.
`pstrain config show --resolved` displays the selected effective profile;
`--sources` adds source annotations. Unknown-profile errors continue to list
available names.

### Generated reference and CI

Generate the configuration reference and JSON Schema from the canonical
Pydantic model. The generator must import the same model used by the resolver,
not a documentation-only facade. Check in the rendered RST for stable docs
builds, and add a CI check that regenerates it into a temporary location and
fails on a diff. The same job runs schema-consumer coverage tests required by
the truthfulness invariant. This completes the config-reference work deferred
by the D-bundle.

## Compatibility and C2 landing order

C1 is documentation only. C2 should land in this order:

1. Add the canonical versioned models, source-aware resolver, and runtime
   projection without changing the build CLI's effective defaults.
2. Add contract tests that inject a non-default value for every semantic field
   and observe it at its registered consumer and in provenance.
3. Add profile discovery, `config explain`, migration check/write commands,
   and canonical reference generation; wire regeneration and consumer coverage
   into CI.
4. Make setup and build use the resolver. During one compatibility window,
   read legacy files with warnings and reject ambiguous combinations.
5. Migrate maintained fixtures and examples, then remove the inactive
   `PstrainConfig`, `ConfigManager`, and old generator.
6. Remove `pstrain/lib/commands.py` and the legacy `Action` layer after all CLI
   verbs use direct library calls and the supported dry-run plan abstraction.
7. Reconcile dictionary module/package compatibility. The standalone module is
   already deleted; remove stale shims, imports, and design notes only after
   verifying the package API covers supported callers.
8. After the announced compatibility window, remove legacy readers and aliases.

Migration warnings must identify the command that performs the conversion and
must not be emitted for a freshly generated project. Historical provenance is
left untouched; new provenance records the canonical version and sources.

## Open decisions for Kevin

1. **Canonical file layout.** Recommendation: keep `etc/configs.yaml` for
   named profiles and add project/experiment overlays in versioned canonical
   files, all parsed by one schema family. A single large file is simpler to
   locate but creates contention and makes reusable profile sets harder to
   share.
2. **User-wide semantic defaults.** Recommendation: allow them, below project
   layers, but require `config explain` and provenance to expose them. Removing
   them improves portability; retaining them supports consistent local policy
   across projects.
3. **Built-in shadowing.** Recommendation: deep-merge project profiles over a
   named built-in only when the project explicitly declares `extends`.
   Otherwise require a complete profile. Implicit deep merge is convenient but
   lets installed-version default changes alter an old project silently.
4. **Compatibility duration.** Recommendation: one minor release with legacy
   reads and loud warnings, followed by removal. A longer window reduces
   immediate migration cost but prolongs the two-schema truthfulness risk.
5. **Environment overrides.** Recommendation: keep native-library discovery
   and diagnostic switches outside semantic model configuration, and add no
   generic `PSTRAIN_*` mapping. Generic mapping is convenient in automation but
   makes provenance and typo detection substantially weaker.
6. **Consumer registration enforcement.** Recommendation: make missing
   consumer coverage a required CI failure for every semantic field. The test
   matrix has maintenance cost, but it directly prevents a repeat of A4/A7b.
