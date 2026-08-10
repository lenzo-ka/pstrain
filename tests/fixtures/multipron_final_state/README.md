# Multipron final-state regression fixtures

These three MFCC files are copied byte-for-byte from the read-only CMU ARCTIC
SLT parity run at:

`/Volumes/experiments/pstrain-parity/pstrain-v5-on/shared/features/v4/`

They were selected from the 12 utterances classified in
`experiments/v6/m4-taxonomy.json` on 2026-08-10. `arctic_a0257` and
`arctic_b0424` were also skipped by upstream multipron training;
`arctic_a0336` was pstrain-only. The companion dictionary contains exactly
the corpus pronunciations required by these transcripts and retains all
variants.

`model/` is the flat-model snapshot from the same parity run. It is included
because the smaller numeric-harness phoneset does not contain every phone in
these utterances (notably CH).
