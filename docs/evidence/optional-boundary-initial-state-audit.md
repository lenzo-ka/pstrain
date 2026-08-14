# Optional boundary silence: initial-state assumption audit

This audit was completed before the round-3 implementation changes. Line numbers
refer to `origin/main`, whose `backward.c` contains 18 literal uses of
`state_seq[0]` and whose `forward.c` contains seven.

## `backward.c`

All 18 literal uses occur in the frame-zero finalization block. In each row,
"multiple breaks it" means that using only state zero omits every bypass entry.

| Upstream line | Use and assumption | Multiple breaks it? | Required form |
|---|---|---:|---|
| 1144 | Select density buffer by state zero's local codebook; assumes it is the only frame-zero emitter. | Yes | Compute each distinct initial emitter's local codebook. |
| 1145 | Select density-index buffer by the same codebook assumption. | Yes | Pair each computed initial codebook with its density-index buffer. |
| 1148 | Compute state zero's codebook; assumes no other initial codebook is needed. | Yes | Compute every distinct initial emitter codebook. |
| 1151 | Make only state zero's local codebook active for scaling. | Yes | Build the deduplicated set of all initial local codebooks. |
| 1157 | Read densities for state zero when calculating its output probability. | Yes | Calculate output probability per initial state. |
| 1158 | Read density indices for state zero. | Yes | Use the current initial state's local codebook. |
| 1159 | Use state zero's mixture weights. | Yes | Use the current initial state's mixture weights. |
| 1187 | Select state zero's local codebook for reestimation. | Yes | Iterate active initial states and select each state's codebook. |
| 1188 | Select state zero's CI local codebook. | Yes | Select each initial state's CI codebook. |
| 1194 | Use state zero's mixture weights for partial output probabilities. | Yes | Use each initial state's mixture weights. |
| 1203 | Use state zero's mixture weights for CD density terms. | Yes | Use each initial state's weights and posterior contribution. |
| 1207 (twice) | Compare state zero's CD and CI codebooks. | Yes | Compare them for each initial state. |
| 1211 | Use state zero's CI mixture weights for partial CI output probabilities. | Yes | Use each initial state's CI weights. |
| 1220 | Use state zero's CI mixture weights for CI density terms. | Yes | Use each initial state's CI weights and posterior. |
| 1227 | Accumulate CD counts into state zero's local mixture-weight accumulator. | Yes | Accumulate into each initial state's local accumulator. |
| 1234 (twice) | Compare state zero's CI and CD mixture-weight identities. | Yes | Compare them per initial state. |
| 1238 | Accumulate tied/discrete CI counts into state zero's local CI accumulator. | Yes | Accumulate into each initial state's local CI accumulator. |
| 1242 | Accumulate continuous CI counts into state zero's local CI accumulator. | Yes | Accumulate into each initial state's local CI accumulator. |

The repeated expressions at lines 1207 and 1234 account for the source's 18
literal occurrences while sharing one semantic operation each.

Related non-literal assumptions in the same block:

- **[read]** Line 1169, `beta[0] = prior_beta[0] * op`, is the complete
  alpha/beta check only with one unit-weight initial state. It must become a sum
  over initial states of `prior_beta[i] * op_i * entry_weight_i`.
- **[read]** Line 1186, `asf[0]`, gates reestimation only for state zero. It must
  become a per-initial-state activity test.
- **[read]** Line 1216 explicitly says `ASSUMPTION: 1 initial state`. Its `1.0`
  is correct under that precondition because the sole initial state's posterior
  is one. With multiple entries, the CI density term must receive
  `prior_beta[i] * op_i * entry_weight_i / final_alpha`, the same posterior
  represented by the CD path.
- **[read]** Line 577 says `Process non-emitting initial states first`, but here
  “initial” describes the first backward-pass worklist, seeded with the single
  final non-emitting state at lines 558–563. It does not select HMM entry state
  zero and is safe for multiple forward initial states; clarify the wording to
  “initial backward worklist” if this area is edited.

## `forward.c`

| Upstream line | Use and assumption | Multiple breaks it? | Required form |
|---|---|---:|---|
| 281 | Select density buffer by state zero's codebook; assumes it is the only frame-zero emitter. | Yes | Compute each distinct initial emitter's local codebook. |
| 282 | Select density-index buffer by the same assumption. | Yes | Pair each initial codebook with its density-index buffer. |
| 285 | Compute state zero's codebook only. | Yes | Compute every distinct initial emitter codebook. |
| 287 | Make only state zero's local codebook active for scaling. | Yes | Build the deduplicated set of all initial local codebooks. |
| 293 | Read densities only for state zero's output probability. | Yes | Calculate output probability per initial state. |
| 294 | Read density indices only for state zero. | Yes | Use each initial state's local codebook. |
| 295 | Use only state zero's mixture weights. | Yes | Use each initial state's mixture weights. |

Related non-literal assumptions in the same block:

- **[read]** Lines 270–335 consistently say and implement “the initial state”:
  scalar allocation, `outprob[0]`, one scale basis, one alpha entry, one active
  state, and state index zero. The explicit line-330 comment, `Only one initial
  state (for now)`, confirms the precondition. Required form: allocate for the
  possible initial set, initialize every flagged initial emitting state with its
  entry weight, and record every such state in the active arrays. A shared scale
  may still use the best weighted frame-zero output, matching later frames.
- **[read]** Line 576 says `Assumptions about topology that might not be valid`
  above debug assertions that a non-emitting successor is ordered after its
  predecessor and at most two state slots away. Multiple initial states alone do
  not break this. The optional *exit* bypass does: it is an actual arc that can
  jump over the final SIL states to the final non-emitting exit, so the distance
  assertion is invalid. Required form: remove the index-distance assertion; the
  predecessor/successor relationship is already established by traversing the
  graph arc. This is debug-only but must not encode the invalid topology.

## Scope and alternative

- **[read]** The existing PR implementation has already replaced the two
  frame-zero single-state regions with loops over flagged initial emitters. The
  audit therefore calls for validating and correcting that bounded
  generalization, not another broad forward/backward rewrite.
- **[inference]** Preserving exactly one *emitting* initial state cannot express
  both “consume frame zero as SIL” and “consume frame zero as first-word speech”:
  those paths require different acoustic states. A single virtual non-emitting
  start would preserve a single graph entry but would require new frame-zero
  epsilon-closure semantics in the battle-tested forward/backward core, plus
  accumulator and Viterbi handling. That is materially larger and riskier than
  the current bounded multi-entry initialization.
- **[inference]** A feature-specific wrapper that expands the virtual start into
  the existing multiple emitting entries before calling forward/backward merely
  relocates the same multiple-initial-state semantics and does not restore the
  vendored precondition.
