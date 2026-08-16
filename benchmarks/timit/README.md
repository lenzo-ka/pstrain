# TIMIT boundary benchmark

Stage 1 prepares, but does not train, a same-data forced-alignment benchmark.
TIMIT is not redistributed.  Prepare persistent local manifests from an LDC
installation:

```console
python scripts/setup_timit.py /path/to/timit benchmarks/timit/data
```

The split is TIMIT's canonical complete TRAIN/TEST division with SA1 and SA2
excluded: 3,696 training utterances from 462 speakers and 1,344 test
utterances from 168 different speakers.  `core-test.fileids` identifies the
conventional 192-utterance, 24-speaker core reporting slice; it is not the
benchmark's full held-out partition.

The four split files are persistent.  If all four exist, setup validates and
honours them byte-for-byte.  A partial split is rejected.  Speaker overlap,
unknown/missing utterances, transcript drift, and fileid/transcript ordering
differences are errors.

`timit.reduced.dict` contains only corpus words and all their TIMIT dictionary
pronunciations.  Its input hash covers both the source dictionary and corpus
vocabulary, so setup regenerates it after either changes and verifies 100%
word coverage.  `reference-phones.tsv` preserves the hand-labelled sample
boundaries for scoring without rounding.

The metric in `pstrain.benchmarks.boundaries` sequence-aligns phone identities
and scores internal boundaries.  A reference boundary is comparable only if
both adjacent phones match consecutive hypothesis phones.  Mismatch-adjacent
boundaries remain in tolerance-recall denominators, and insertions remain in
precision denominators.  Report matched-boundary mean/median absolute error
together with coverage, edit counts, and recall/precision/F1 at 10, 20, 25,
and 50 ms.

## Canonical-split dither adjudication

`scripts/bench_timit.py` prepares the manifests above, installs them as the
authoritative external split in two isolated pstrain projects, and trains a
fixed-seed pair that differs only in `features.dither`.  Separate projects are
required so neither cell can reuse the other's feature cache.  By default the
stage stops at `ci-1g`, which is a useful, bounded training adjudication; pass
`--target cd-8g` for the full acoustic-model ladder.

On a 36-logical-core compute host, leaving two cores free:

```console
python scripts/bench_timit.py /path/to/timit --work-dir /path/to/results -j 34
```

`results.json` records the canonical split counts, a content hash of each
cell's MFC tree, and a content hash of the acoustic parameter files (excluding
configuration/provenance files).  Equal seeds make reruns repeatable; different
feature or model hashes adjudicate whether enabling dither changes that
stage's numeric result.

### Completed `ci-1g` result

On the canonical TIMIT split, with `features.seed` fixed at 243, enabling
frontend dither changes both the feature stream and the resulting `ci-1g`
model.  Dither is not a no-op even when the RNG is seeded.  The paired run used
3,696 training utterances from 462 speakers and 1,344 test utterances from 168
speakers, with SA1 and SA2 excluded; the core test slice contains 192
utterances.  Dither was the only varying profile field.  The committed
[`results.json`](dither-ci-1g/results.json) and
[`run.log`](dither-ci-1g/run.log) preserve the adjudication record.

The following SHA-256 values are conditioned evidence from this host and build,
not reproducible pins.  The durable result is the boolean adjudication
`features_changed=true` and `model_changed=true`.

| Cell | Feature tree SHA-256 | Model tree SHA-256 | Resolved config SHA-256 |
| --- | --- | --- | --- |
| dither off | `30dee11fae4d1def3ff784f5230b41de6d4246ed12ab6e2052d8819235d8c7b2` | `a7fd560ddd34566e9c7189f64c0fedd0d948e690c14d5af7cf0bb91121954df5` | `3c6af89a0e866f9a615181b37dae06eb1d6e6406b3aa6101f36cd3336b6f3f7e` |
| dither on | `ac138281879232d961339e863a541d5212d0e1eecfd5b562c21307d717fa1a6c` | `9e4e2841b2cccf73b92d72ea24be9774094d2a9d792a3b521dbba366ec42f75a` | `b08c91c277cc182e838cfbec2fd9de587a78115be25e7ec81d3cc8d3221ce67e` |
