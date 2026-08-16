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

## What mandatory final silence can measure

The forced-aligner topology also limits what alignment-derived trailing-silence
durations can establish. A successful path through a topology ending in a
mandatory three-state `SIL` must traverse that model, so every aligned utterance
is assigned a positive final-`SIL` duration. Exact zero is outside the
instrument's support: widening the beam cannot produce a value that the topology
forbids. Consequently, these alignments cannot measure the prevalence of
utterances with no trailing silence or provide that population as a stratifier.

The observed shape is still useful, within that limitation. Pstrain's native
forced aligner, `pstrain_align_mfcc`, aligned 1,092 of the 1,132 CMU Arctic SLT
utterances at the shipped `1e-64` beam and another 6 only at `1e-120`, a beam 56
orders of magnitude wider. The remaining 34 did not align at either beam; they
are an observed class, not zero-duration observations, so the duration
distribution has a conditional denominator of 1,098.

Of those 1,098 alignments, 344 (31.33%) assigned final `SIL` exactly 3 frames,
the minimum permitted by its three emitting states. The median was 5 frames,
the nearest-rank 90th percentile was 11, and the range was 3 to 22. At 10 ms per
frame, this is a 30 ms floor, a 50 ms median, and a 220 ms maximum. The pileup at
the model-imposed floor is consistent with the aligner absorbing the boundary
into mandatory silence for a population with little or no real trailing
silence, but it does not count that population.

These measurements characterize pstrain's forced aligner, not PocketSphinx's
two-pass Viterbi alignment in `state_align_search.c`. No arm of the PocketSphinx
PR #468 experiment was built or run, so they provide no evidence for or against
that change. They are also distinct from the separately gated forced-aligner
bypass prototype described above, which produced no successful duration
observations.

A reproduction that selects utterances with no trailing silence therefore needs
an independent criterion, such as a signal-level detector, hand annotation, or
a corpus with boundary labels. An energy-based detector over this corpus
produced a wildly different population from the alignment; substituting either
for the other would change what the experiment tests.

## Scope

This divergence is decode-affecting within BW's training alignment: its
utterance graph now permits the last spoken HMM as its exit, which the upstream
graph does not. It changes the
frames eligible for `SIL` accumulation. It does not manufacture, generate,
score, or exclude frames and does not change dither, scoring, evaluation,
beams, or any other default.
