# Arctic benchmark pin

This pins the from-checkout, turnkey Arctic benchmark run by
`scripts/bench_arctic.py` at engine commit `740f112`. It covers multipron
training off and on in the SLT-55 same-speaker and big cross-speaker cells. The
machine-readable [record](arctic-pin/record.json) and upstream-oracle
[sidecar](arctic-pin/oracle-sidecar.json) preserve the per-utterance rows and
the complete engine, model, corpus, transcript, language-model, dictionary,
and decoder identities.

The pin was accepted on a re-derived basis: its measured deltas are the
documented baseline. They are not a claim that every cell is at zero delta
from the preserved upstream model.

## Measurement identity

The decode path is a defining condition of this measurement. Audio is decoded
from WAV through pinned PocketSphinx 5.1.1 using Python 3.12.12 and native
library SHA-256
`6a5da2377c3b2b033b35d93a12a57bb869413bbc98f045d6c3f3652585792be3`.
The engine is pstrain 0.1.0 at `740f112`, with no tracked modifications. A
result obtained through another decode path is not the same measurement even
when the acoustic-model bytes are identical.

## Pin conditions

| Condition | Pinned value |
|---|---|
| Band | BM1 |
| Language model | SHA-256 `2cf11ab0474a0bdd165cbee59db674b05764fdb00bf6f9824c0dccce571637b5` |
| Decode dictionary | SHA-256 `24ff2852a707b63f499fd968294d5e4c02d44e0eb1ec511e40be1f380d785846` |
| Filler dictionary | SHA-256 `fb50883998c41a5030c2a602965935c647563321e84a86f2adabb377ec24b49c` |
| Shared training | 3 states, 200 senones, `a_beam=1e-90`, `b_beam=1e-10`, maximum skip fraction 0.05, retry beam factor `1e10`, tree state weights `[1.0, 0.05, 0.0]`, `ssplitmax=7`, `ssplitthr=0`, `csplitmax=2000`, `csplitthr=0`, `mwfloor=1e-8`, 12 question permutations, 20 questions/state, 1 question iteration |
| Multipron off training | `multipron_training=false`, linear untied inventory; CI/tied convergence 0.1, 1–10 iterations; untied convergence 0.1, 1–6 iterations |
| Multipron on training | `multipron_training=true`, transcript-reachable untied inventory; CI/tied/untied convergence 0.001, 1–10 iterations |
| Acoustic features | 16 kHz, 13 cepstra, 25 filters, 512-point FFT, 130–6800 Hz, alpha 0.97, `1s_c_d_dd`, lifter 22, DCT, no AGC, batch CMN, no variance normalization |
| Split | Seed 42, test count 0, both modes |
| Decoder | `beam=pbeam=lpbeam=lponlybeam=fwdflatbeam=1e-80`; `wbeam=fwdflatwbeam=1e-40`; `pl_window=5`, `lw=10`, `wip=0.2` |
| Bootstrap | Matched-pair percentile, 100,000 resamples, seed 7; big cells speaker-stratified |

The corpus archive identities are BDL
`26b91aaf48b2799b2956792b4632c2f926cd0542f402b5452d5adecb60942904`,
CLB `3f16dc3f3b97955ea22623efb33b444341013fc660677b2e170efdcc959fa7c6`,
RMS `c6dc11235629c58441c071a7ba8a2d067903dfefbaabc4056d87da35b72ecda4`,
and SLT `7c173297916acf3cc7fcab2713be4c60b27312316765a90934651d367226b4ea`
(SHA-256; 1,132 WAVs each). The transcript identities are big
`5737c7296d491df39aaa2db24ab03e6acf6e058c10bd7d6b331ffdbf242c5b58`,
SLT-55 `1de4e31c934ea6c5cd414307b8cf4f71c0d846adfd68edf04c4ececaafa4c532`,
full SLT `cce3d341c02445d3aa91453f1d7f0fa3097f6ab6df76264336277ca2c54f3085`,
and pin training
`28788cd1ce2269d344b50420d74007fa8c443778680724f6334e2712ea110959`.
The 1,043-utterance training fileids hash is
`8ce9a55c5929f6f86579ee1b244c38fd4d0a9d41e436e2057337f74c1bb4d631`.
The JSON record is authoritative for the complete frozen configuration,
model-file identities, and resource metadata.

## Baseline

Delta is pstrain minus the preserved upstream oracle in WER percentage points.
The bands are paired 95% bootstrap summaries, not iid confidence intervals:
utterances are clustered by speaker rather than exchangeable independent
observations. The big cells therefore resample within speaker strata.

| Mode | Cell | pstrain WER | Oracle WER | Delta pp | Paired 95% CI | Interpretation |
|---|---|---:|---:|---:|---:|---|
| off | SLT-55 | 28.8499 | 28.8499 | +0.0000 | [-4.7059, +4.7619] | statistical parity |
| on | SLT-55 | 30.0195 | 26.9006 | +3.1189 | [-0.3906, +6.6667] | statistical parity |
| off | big | 76.6393 | 74.7915 | +1.8478 | [+1.3257, +2.3646] | documented baseline gap |
| on | big | 76.2157 | 75.0017 | +1.2141 | [+0.6922, +1.7350] | documented baseline gap |

### Gap composition

Cross-decoding the preserved earlier-era pstrain models through this pin's
decode path shows that both model instance and decode path contribute. In the
big off cell, the preserved model is +1.344 pp behind the oracle (95% CI
[+0.832, +1.855]) and is -0.504 pp better than the pin-retrained model (95% CI
[-0.958, -0.050]): 72.7% of the pin gap is present in the preserved model on
the modern path and 27.3% is the retraining increment. In the big on cell, the
corresponding values are +0.787 pp versus oracle (95% CI [+0.253, +1.320]) and
-0.427 pp versus the pin model (95% CI [-0.881, +0.027]), or 64.8% and 35.2%
of the pin gap respectively.

Across eras, moving the preserved pstrain model to the pin path changes WER by
only +0.070 pp off and +0.157 pp on, while the byte-identical upstream oracle
improves by 0.597 pp off and 0.504 pp on. The decode path is therefore
model-sensitive; the differential decode-path response supplies most of the
era-to-era delta movement, and pin retraining supplies a further roughly
0.43–0.50 pp.

## Forward gate

Future runs compare matched pairs against the pinned per-utterance rows. The
acceptance bar is no statistically significant regression. The big-cell gap
against the preserved upstream models under the modern decoder is an open,
tracked improvement target; it is part of this documented baseline and is not
itself a regression.

## Decode-path transport

Earlier program measurements used precomputed-feature batch decoding rather
than this WAV decode path. Those measurements remain bound to that path: a
byte-identical acoustic model shifts by 0.50–0.60 pp WER between the two paths.
This pin consequently records its full decode identity. Future comparisons
must either hold that identity fixed or cross-decode the compared models.

## Known training skips

These are documented training-set composition facts, not decode failures:

| Mode | Utterance | Stage and pass | Mechanism |
|---|---|---|---|
| off | `arctic_a0587` | `cd-2g`, pass 1 | Beam failure on a hard utterance after the permitted retry; mirrored upstream, whose preserved build ignores it at CI passes 5–6. |
| on | `arctic_a0302` | `cd-untied`, passes 3–10 | Beam failure on a known-hard utterance after the permitted retry in the multipron posture; part of the recorded on-mode remainder class as the reachable inventory shifted. |
| on | `arctic_a0587` | `cd-1g`, pass 6 | Beam failure on a known-hard utterance after the permitted retry in the multipron posture; part of the same recorded on-mode remainder mechanism. |

## Replicability

The benchmark uses canonical ordering and is deterministic: the same checkout
and the same resource band produce the same record, modulo the explicitly
recorded environment identity.
