# TIMIT canonical-split rung 2 status — 2026-08-16

## State

The branch contains a canonical-split TIMIT benchmark harness and a paired,
fixed-seed dither training stage.  The default bounded stage is `ci-1g`; the
same harness accepts `--target cd-8g` for the full ladder.  Each dither cell has
its own project and feature tree.  The complete resolved profile is identical
between cells except for `features.dither`; both use `features.seed: 243`.

The checkout began at `f81acc7`, the current `origin/main`, rather than the
requested historical `6076381`.  The stage-1 preparation work is present in
the current history as `5c0f072`.  The named stage-2 branch and commit were not
available from the current remote; its exact-phone alignment capabilities are
present in the current alignment implementation, including `94454f4`.

## Canonical split

The shrub preparation run against
`/home/lenzo/shrub-data/ldc-corpora/corpora/timit` produced:

- 3,696 training utterances from 462 speakers
- 1,344 test utterances from 168 speakers
- 192 utterances in the conventional 24-speaker core test slice
- SA1 and SA2 excluded from both partitions
- no random train/test split

The four canonical manifest files are copied into each project as persistent,
authoritative external split files.  `all.transcription` is their ordered
concatenation, and audio file IDs remain rooted at `train/` or `test/`.

## Dither adjudication run

The clean paired run is active on shrub:

- checkout: `/home/lenzo/shrub-data/pstrain-timitcs`
- per-checkout environment: `/home/lenzo/shrub-data/pstrain-timitcs/.venv`
- result directory: `/home/lenzo/shrub-data/timit-canonical-dither-v2`
- PID file: `/home/lenzo/shrub-data/timit-canonical-dither-v2/run.pid`
- top-level log: `/home/lenzo/shrub-data/timit-canonical-dither-v2/run.log`
- workers: 34 on a 36-logical-core host
- target: `ci-1g`
- current cell: `dither-off`
- pending cell: `dither-on`

The system interpreter still raises `ModuleNotFoundError` for ambient
`import pstrain` from a neutral directory.  The run explicitly supplies the
native library built into the checkout venv so spawned workers use the same
installation.

No dither result is claimed yet.  On successful completion,
`/home/lenzo/shrub-data/timit-canonical-dither-v2/results.json` will contain
the paired MFC-tree and acoustic-parameter SHA-256 identities and boolean
`features_changed` and `model_changed` adjudications.

## Verification

- Focused local tests: 11 passed.
- Focused shrub-venv tests: 11 passed.
- Local `make verified`: green; 716 passed, 1 skipped, 1 deselected, with all
  configuration, lint, type, format, native, and ambient-import gates green.

## Remaining steps

1. Monitor PID and logs until both `ci-1g` cells finish; preserve
   `results.json` and both training logs.
2. If the process exits without `results.json`, diagnose the terminal lines in
   the active cell's `training.log` and resume in a new result directory if
   project integrity is uncertain.
3. Copy the completed result record into the local worktree and report whether
   fixed-seed dither changes the feature tree, the trained model tree, or both.
4. Run the paired harness again to confirm same-cell reproducibility before
   treating hash differences as a stable adjudication.
5. Extend the completed pair to `cd-8g`, run exact-phone alignment on the
   canonical full test set and 192-utterance core slice, and score both cells
   with `pstrain.benchmarks.boundaries`.
6. Open the PR only after the measured record is committed and the final
   `make verified` run remains green.
