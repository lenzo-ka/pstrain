# Optional transcript boundary silence

## Deliberate vendored divergence

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
and finish words. A separately gated bypass prototype made every one of the
1,132 measured SLT alignments fail, so that graph is deliberately unchanged.
This divergence belongs only to the training state sequence, where posterior
accumulation creates the contamination being corrected.

The Baum-Welch state-sequence engine has one hard-wired emitting entry state
and one hard-wired non-emitting terminal state. It cannot express an epsilon
alternative before that entry without changing the numerical forward/backward
contract. With the option enabled, both its historical linear builder and its
multipron graph builder therefore omit only explicit leading `<s>` and trailing
`</s>` transcript words. This is the exact zero-frame boundary path; interior
optional silence behavior is unchanged.

## Scope

This divergence is decode-affecting within BW's training alignment: its
utterance graph now permits the first spoken HMM as its entry and the last
spoken HMM as its exit, which the upstream graph does not. It changes the
frames eligible for `SIL` accumulation. It does not manufacture, generate,
score, or exclude frames and does not change dither, scoring, evaluation,
beams, or any other default.
