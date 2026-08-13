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
