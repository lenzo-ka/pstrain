# Lane M4b fold-in report

1. The in-process tree builder now applies `cntthresh` with upstream
   `bldtree` semantics: exclude a model when any emitting-state/stream count
   is below the threshold. The value is `1e-5`, from upstream
   `csrc/programs/bldtree/parse_cmd_ln.c`'s `-cntthresh` default and confirmed
   by the A7 audit. The exclusion gap predates `fix/untied-skips` and may
   contribute to the remaining pstrain/upstream tree-quality and CLB gap.
2. Dictionary-wide untied enumeration is explicitly scoped to multipron
   mode. Linear mode uses upstream `mk_mdef_gen` occurrence pruning; universal
   enumeration would be an unnecessary upstream deviation there.
3. SLT inventory and parameter growth is +2.7%; measured details and the
   transcript-reachable follow-up option are in the design and evidence docs.
4. The engineered graph grows 12→14 slots and 14→16 directed edges. The
   measured CD-untied wall time is 2.4→2.2 seconds, supporting position-local,
   additive growth rather than an utterance-wide explosion.
5. The boundary regression resolves all four graph copies to the four exact
   expected triphone identities.
6. Compact skip and beam-stability evidence plus reproduction commands are
   versioned under `docs/evidence`. The missing standalone 224-identity raw
   list is called out explicitly in the evidence record.
7. `phone_graph_triphone.c` now describes the two-sided cross-product.
