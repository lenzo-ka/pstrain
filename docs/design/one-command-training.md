# One-command training

Status: proposed decision for U1. This document defines a CLI workflow; it does
not implement or change behavior.

## Goal

Given a WAV directory, pronunciation dictionary, and prompt list, create a
valid pstrain project and run a useful training target with one command:

```console
pstrain train ./model \
  --audio ./wav \
  --prompts ./prompts.txt \
  --dictionary ./dictionary.dict
```

The command is orchestration over the existing setup, validation, and build
library boundaries. It must not introduce a second setup implementation or a
second training pipeline.

## Input contract

Required inputs are:

- `PROJECT_DIR`, the new or resumable project directory;
- `--audio DIR`, containing `.wav` files recursively;
- `--prompts FILE`, mapping file IDs to prompt text; and
- `--dictionary FILE`, the pronunciation lexicon.

The preferred prompt format is leading-ID text, one utterance per line:
`fileid WORDS`. File IDs may include relative directory components and must
match WAV paths below `--audio` after removing `.wav`. The command may also
accept Sphinx transcript form and explicit `--prompt-format`, using the
existing parser/converter rather than shell transformations. Format detection
must fail when ambiguous.

Sensible defaults are:

- experiment `default`;
- profile `default`;
- target `ci-1g`, which gives a quick, complete first model;
- deterministic split policy from the selected canonical profile (currently
  95 percent training and seed 42 in the active pipeline defaults);
- copy inputs into the project for portability;
- extract the phoneset from the main and default filler dictionaries;
- validate before scheduling work; and
- runner auto-parallelism, with `-j/--jobs` as an invocation override.

The full proposed surface is:

```text
pstrain train PROJECT_DIR --audio DIR --prompts FILE --dictionary FILE
    [--target TARGET] [--profile NAME] [--experiment NAME]
    [--phoneset FILE] [--filler-dict FILE]
    [--prompt-format auto|leading-id|sphinx|tsv|csv]
    [--link-audio] [--resume] [--force] [-j N]
    [--normalize-with POLICY] [--normalization-report FILE]
    [--dry-run] [--json]
```

`--resume` means preserve the project inputs and continue the dependency-aware
build. `--force` retains the build command's meaning of rebuilding reachable
tasks; it must not imply overwriting source corpus files. Replacing existing
inputs needs a separately explicit setup choice if supported. `--dry-run`
shows setup writes, validation, resolved configuration, and the build plan.

## Pre-normalized prompts are the default

Prompt lists are assumed **pre-normalized**. Built-in normalization is opt-in
only and must match the lexicon's conventions. A normalizer that diverges from
the lexicon silently manufactures out-of-vocabulary words.

Therefore the default path parses and validates text but does not change word
spelling, case, punctuation, Unicode form, number expansion, or token
boundaries. Validation compares prompt tokens exactly with dictionary lookup
semantics and emits an OOV report. It surfaces likely normalization/lexicon
mismatches—such as systematic case differences or punctuation-attached
tokens—as diagnostics, not automatic fixes.

`--normalize-with POLICY` is an explicit transformation. A policy is named,
versioned, described before execution, and checked against the selected
lexicon. The command writes both the original and transformed prompt hashes,
the policy/version, and an OOV before/after report. It refuses a policy whose
declared casing or token conventions conflict with the lexicon unless the user
chooses a future, separately explicit override. A generic silent
`--normalize` switch is not sufficient.

The committed ARCTIC benchmark already proves this principle. Its training and
decoder transcripts are normalized, committed inputs with authenticated
hashes, paired with the exact dictionary used for measurement. The harness
does not reinterpret raw `txt.done.data` at run time. `pstrain train` should
apply the same committed-transcript pattern to ordinary projects: preserve the
prepared prompt list as an input artifact and record its identity.

## Handoff to setup and build

The orchestration sequence is:

1. Parse arguments and resolve all source paths without writing.
2. Parse prompts, inventory WAV file IDs, load the dictionary, and produce a
   validation report: duplicate IDs, missing or extra audio, malformed entries,
   empty prompts, dictionary parse errors, phone errors, and OOV tokens with
   counts and example utterances.
3. If opt-in normalization was requested, transform into a separate staged
   prompt artifact, rerun the same validation, and retain both reports.
4. Call the setup library to create the project, install inputs, install or
   extract phoneset/filler resources, write canonical configuration, and retain
   the pre-normalized prompt identity.
5. Run project validation. No training task starts if validation has errors.
6. Construct `PipelineContext` through the canonical resolver and hand the
   selected target to the existing dependency-aware build pipeline. Setup,
   split, features, and all model stages remain owned by their current library
   functions.
7. On success, print the model path, resolved profile, target, experiment,
   elapsed summary, and the commands for resuming, testing, and inspecting
   configuration.

This sequence also removes a named defect in the current README quickstart:
the user must run an `awk` command and `mv` to rewrite the held-out
leading-ID transcript into Sphinx decoder form before `pstrain test`.
One-command training should produce correctly typed training and decoder
transcript artifacts at the split boundary. Users should never mutate a split
artifact in place to satisfy a later consumer.

## Failure UX

Failures are grouped before work begins where possible. Human output starts
with the blocking cause, then the affected count and a bounded sample. Full
details go to stable report files under the project's reports directory or to
an explicitly requested path. JSON output uses stable error codes.

Important cases include:

- prompt IDs with no WAV and WAVs with no prompt;
- duplicate or unsafe IDs;
- unsupported sample rates or inconsistent WAV properties;
- malformed dictionary entries or phones outside the phoneset;
- exact OOV tokens, counts, and example utterance IDs;
- likely case, punctuation, or Unicode convention mismatches, labeled as
  suggestions rather than corrections;
- an existing project whose installed inputs differ from the requested inputs;
- unknown profile or target, with discoverable alternatives; and
- interrupted builds, with an exact resume command.

No partial training begins after an input validation failure. Setup writes
should be staged and committed atomically enough that a failure leaves either
the old usable project or a resumable new project. Native or training failures
retain the pipeline's failed task name and logs rather than being collapsed to
“training failed.”

## CLI verb choice

Three shapes are plausible:

1. `pstrain setup ...` followed by `pstrain build ...` preserves existing
   primitives but does not meet the one-command goal.
2. `pstrain create ...` followed by `pstrain run ...` is regular and leaves
   room for non-training workflows, but exposes two new verbs and still makes
   the common first-run path multi-command.
3. `pstrain train ...` composes setup, validation, and build in one intent-level
   verb while leaving the lower-level commands available.

Recommendation: add `pstrain train`. “Train” is the outcome users seek, maps
directly to the required three inputs, and can resume through the existing
pipeline. Document `setup`, `validate`, and `build` as advanced/decomposed
equivalents. Do not make `train` a shell subprocess chain; call the shared
libraries so errors, dry-run output, and provenance remain structured.

## Compatibility and rollout

The first release should create the same directory layout and build outputs as
manual setup plus build. Existing projects remain operable with `pstrain
build`; `pstrain train PROJECT --resume ...` may adopt them only after config
migration and input-identity checks succeed. The command should initially
target the canonical configuration work from C1/C2 rather than encode the
current two-schema split.

Documentation should replace the quickstart's manual transcript conversion
with `pstrain train`, while retaining a decomposed example for debugging and
automation. Tests should compare the one-command project and output plan with
the equivalent setup/build calls and verify that the default path never calls
a normalizer.

## Open decisions for Kevin

1. **Default target.** Recommendation: `ci-1g`. It is fast enough for a first
   successful run and proves the complete path. A CD target is more useful for
   production but makes the advertised first command substantially slower and
   more failure-prone on small corpora.
2. **Prompt syntax.** Recommendation: make leading-ID text canonical, accept
   Sphinx/TSV/CSV through explicit or unambiguous detection, and store typed
   training and decoder derivatives. Supporting many formats is convenient but
   increases ambiguity and quoting edge cases.
3. **Input ownership.** Recommendation: copy prompts and dictionary, copy audio
   by default, and offer `--link-audio`. Copying is portable and reproducible;
   linking avoids large duplication but lets external changes invalidate a
   project.
4. **OOV policy.** Recommendation: any OOV blocks training by default, with a
   complete report and a future explicit threshold override if needed. A
   permissive default gets farther but can drop or fail utterances deep in the
   engine.
5. **Normalization policy registry.** Recommendation: ship no implicit default;
   add only named policies with documented lexicon compatibility. A bundled
   general-English normalizer is convenient but cannot truthfully match
   arbitrary user lexicons.
6. **Existing destination behavior.** Recommendation: require `--resume` for a
   compatible project and a separate explicit replacement option for changed
   inputs. Treating any existing directory as resumable risks mixing corpora;
   always refusing it makes recovery unnecessarily awkward.
7. **Default split.** Recommendation: inherit the selected canonical profile
   rather than hard-code a train-specific default. A fixed CLI default is
   easier to explain but creates another configuration surface that can drift.
