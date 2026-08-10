# M4b SLT run evidence

Corpus: the fixed 1,043-utterance training split in
`pstrain-cmu-arctic-slt`. The source log was
`/Volumes/experiments/pstrain-parity/pstrain-cmu-arctic-slt/training.log`.

## Skip counts

| Run | Result |
|---|---:|
| terminal-filler fix only (baseline) | 224 / 1,043 skipped at CD-untied iteration 3 |
| two-sided graph only | 198 / 1,043 skipped |
| graph plus dictionary-domain initialization | 10 failed attempts / 1,043 (five utterances, original plus widened retry) |

The five utterance identities and the `1e-100`/`1e-200` direct-probe
results are in `m4b-slt-skips.tsv`. The normal `1e-90` run automatically
retries once at `1e-100`, hence ten failed attempts but five skipped
utterances. All five also fail with the effectively-off `1e-200` beam.

The baseline raw log was not retained as a standalone lane artifact, so the
224 identities cannot be reconstructed unambiguously from the shared log,
which contains interleaved output from several forced runs. This file records
that evidence limitation rather than manufacturing an identity list.

## Reproduction commands

```bash
PSTRAIN_LIB_PATH=/path/to/libpstrainc.dylib \
  pstrain build cd-untied --project /path/to/pstrain-cmu-arctic-slt \
  --config parity --force

# Direct stability probes use the saved cd-untied model and each utterance
# listed in m4b-slt-skips.tsv with BWConfig(a_beam=BEAM, topn=1,
# multipron=True), for BEAM in 1e-90, 1e-100, and 1e-200.
```

## Growth

| Metric | Before | After | Change |
|---|---:|---:|---:|
| Untied triphone rows | 9,786 | 10,052 | +266 (+2.72%) |
| mdef | 491,359 B | 504,926 B | +13,567 B |
| means | 4,598,636 B | 4,723,124 B | +124,488 B |
| variances | 4,598,636 B | 4,723,124 B | +124,488 B |
| mixture weights | 117,976 B | 121,168 B | +3,192 B |
| transition matrices | 1,984 B | 1,984 B | 0 |
| parameter files total | 9,309,224 B | 9,561,400 B | +252,176 B (+2.71%) |
| CD-untied wall time | 2.4 s | 2.2 s | -0.2 s (run noise) |

Peak RSS was not captured. At the engineered two-variant boundary, the
one-sided graph had 12 phone slots/14 directed edges and the two-sided graph
has 14/16: two local additions across two ambiguous positions.
