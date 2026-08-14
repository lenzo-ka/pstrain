# Optional transcript boundary silence

## Deliberate vendored divergence

Decision date: 2026-08-14.

Upstream SphinxTrain requires the filler-dictionary HMMs represented by
`<s>` and `</s>` on every utterance path. Pstrain deliberately diverges:
`training.optional_boundary_silence` defaults to `true`, permitting both
boundary silence HMMs to consume zero frames. The option is declared in the
versioned configuration schema; setting it to `false` restores the upstream
mandatory-boundary behavior.

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
the owned graph builder. The graph retains the initial and final `SIL` HMMs,
marks the first spoken HMMs as additional forward/backward entries, and adds
direct arcs from the last spoken HMM exits to the existing final non-emitting
sentence exit. The ordinary arcs through both `SIL` HMMs remain available.
The added alternatives consume zero frames; interior optional silence behavior
is unchanged.

The retained initial `SIL` path has virtual-start weight 1. The zero-frame
bypass also has total virtual-start weight 1, distributed uniformly across the
initial phone graph's successors. This preserves the graph builder's existing
`1 / n_next` pronunciation fan-out convention.

## Scope

This divergence is decode-affecting within BW's training alignment: its
utterance graph now permits the first spoken HMM as its entry and the last
spoken HMM as its exit, which the upstream graph does not. It changes the
frames eligible for `SIL` accumulation. It does not manufacture, generate,
score, or exclude frames and does not change dither, scoring, evaluation,
beams, or any other default.
