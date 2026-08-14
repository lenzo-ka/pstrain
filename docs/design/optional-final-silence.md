# Optional final transcript silence

## Deliberate vendored divergence

Decision date: 2026-08-14.

Upstream SphinxTrain requires the filler-dictionary HMMs represented by
`<s>` and `</s>` on every utterance path. Pstrain deliberately diverges:
`training.optional_final_silence` defaults to `true`, permitting the final
boundary silence HMM to consume zero frames. Initial silence remains mandatory.
The option is declared in the versioned configuration schema; setting it to
`false` reproduces upstream's mandatory initial and final boundary behavior
exactly.

The reason for the divergence is training contamination. A recording without
boundary silence otherwise contributes speech frames to `SIL` merely to meet
the HMM's minimum duration. Those posteriors are accumulated into the silence
model on every Baum-Welch pass.

## Graph ownership

The forced aligner's `align_build_sent_hmm()` separately adds mandatory start
and finish words. A separately gated bypass prototype was observed to make
every one of the 1,132 measured SLT alignments fail. That observation was the
removal criterion; the failure mechanism was not diagnosed, so the graph is
deliberately unchanged.
This divergence belongs only to the training state sequence, where posterior
accumulation creates the contamination being corrected.

With the option enabled, both single-pronunciation and multipron training use
the owned graph builder. The graph retains the initial and final `SIL` HMMs and
adds direct arcs from the last spoken HMM exits to the existing final
non-emitting sentence exit. The ordinary arcs through final `SIL` remain
available. The added final alternative consumes zero frames; initial and
interior silence behavior is unchanged. The graph has exactly one frame-zero
emitting entry, so `forward.c` and `backward.c` retain their upstream
single-initial-state preconditions.

The bypass is a probabilistic alternative, not an unnormalized free path. Each
last-spoken exit splits half of its outgoing mass across its retained successors
and assigns half to the direct final-exit bypass. The retained and bypass
alternatives therefore total one, matching the graph builder's normalized
fan-out convention.

## Scope

This divergence is decode-affecting within BW's training alignment: its
utterance graph now permits the last spoken HMM as its exit, which the upstream
graph does not. It changes the
frames eligible for `SIL` accumulation. It does not manufacture, generate,
score, or exclude frames and does not change dither, scoring, evaluation,
beams, or any other default.
