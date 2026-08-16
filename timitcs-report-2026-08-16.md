# TIMIT canonical-split rung 2 report — 2026-08-16

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

## Dither adjudication

The clean paired run completed on shrub:

- checkout: `/home/lenzo/shrub-data/pstrain-timitcs`
- per-checkout environment: `/home/lenzo/shrub-data/pstrain-timitcs/.venv`
- result directory: `/home/lenzo/shrub-data/timit-canonical-dither-v2`
- PID file: `/home/lenzo/shrub-data/timit-canonical-dither-v2/run.pid`
- top-level log: `/home/lenzo/shrub-data/timit-canonical-dither-v2/run.log`
- workers: 34 on a 36-logical-core host
- target: `ci-1g`
- cells: `dither-off` and `dither-on`, both complete

The system interpreter still raises `ModuleNotFoundError` for ambient
`import pstrain` from a neutral directory.  The run explicitly supplies the
native library built into the checkout venv so spawned workers use the same
installation.

### Conclusion

On the canonical TIMIT split, with `features.seed` fixed at 243, enabling
frontend dither changes both the feature stream and the resulting `ci-1g`
model.  Dither is not a no-op even when the RNG is seeded.  The adjudication is
`features_changed=true` and `model_changed=true`.

The exact SHA-256 values below are conditioned evidence from this shrub host
and build, not reproducible pins.  The boolean adjudication above is the
durable claim.

| Cell | Feature tree SHA-256 | Model tree SHA-256 | Resolved config SHA-256 |
| --- | --- | --- | --- |
| dither off | `30dee11fae4d1def3ff784f5230b41de6d4246ed12ab6e2052d8819235d8c7b2` | `a7fd560ddd34566e9c7189f64c0fedd0d948e690c14d5af7cf0bb91121954df5` | `3c6af89a0e866f9a615181b37dae06eb1d6e6406b3aa6101f36cd3336b6f3f7e` |
| dither on | `ac138281879232d961339e863a541d5212d0e1eecfd5b562c21307d717fa1a6c` | `9e4e2841b2cccf73b92d72ea24be9774094d2a9d792a3b521dbba366ec42f75a` | `b08c91c277cc182e838cfbec2fd9de587a78115be25e7ec81d3cc8d3221ce67e` |

The completed records are committed at
`benchmarks/timit/dither-ci-1g/results.json` and
`benchmarks/timit/dither-ci-1g/run.log`.

## Verification

- Focused local tests: 11 passed.
- Focused shrub-venv tests: 11 passed.
- Local `make verified`: green; 716 passed, 1 skipped, 1 deselected, with all
  configuration, lint, type, format, native, and ambient-import gates green.
