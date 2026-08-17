# Arctic benchmark pin

This pins the from-checkout, turnkey Arctic benchmark run by
`scripts/bench_arctic.py` on current main. Its basis is `MULTIPRON-ONLY`: the
live cells are multipron-on SLT-55 and big; the two off-mode cells are retained
as `retired/historical` and are neither trained nor decoded. The
machine-readable [record](arctic-pin/record.json), upstream-oracle
[sidecar](arctic-pin/oracle-sidecar.json), and generated
[paired analysis](arctic-pin/paired-analysis.json) preserve the per-utterance
rows, live comparisons, and the complete engine, model, corpus, transcript,
language-model, dictionary, and decoder identities.

The pin was accepted on a re-derived basis: its measured deltas are the
documented baseline. They are not a claim that every cell is at zero delta
from the preserved upstream model.

## Measurement identity

The decode path is a defining condition of this measurement. Audio is decoded
from WAV through pinned PocketSphinx 5.1.1 using Python 3.12.3 and native
library SHA-256
`5ed31754a35151f9c3ff0feed011ee35ee0fe1f4e83d5d3868c50d9e25b89132`.
The engine is pstrain 0.1.0. Its exact commit and artifact hashes are recorded
in the machine-readable record. A
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
| Basis | `MULTIPRON-ONLY`; off cells retained as retired history |
| Multipron on training | Product defaults: `multipron_training=true`, transcript-reachable untied inventory, optional final silence, one retry at a beam widened by `1e10`; CI/tied 1–10 iterations and untied 1–6, convergence 0.001 |
| Acoustic features | 16 kHz, 13 cepstra, 25 filters, 512-point FFT, 130–6800 Hz, alpha 0.97, `1s_c_d_dd`, lifter 22, DCT, no AGC, batch CMN, no variance normalization |
| Split | Seed 42, test count 0 |
| Decoder | `beam=pbeam=lpbeam=lponlybeam=fwdflatbeam=1e-80`; `wbeam=fwdflatwbeam=1e-40`; `pl_window=5`, `lw=10`, `wip=0.2` |
| Bootstrap | Matched-pair percentile, 100,000 resamples, seed 7; big cells speaker-stratified |

### Configuration provenance by result cell

The live cells `on/slt55` and `on/big` come from the named benchmark profile
`on`. The remeasurement set `runner.jobs=32` at the command line. These are
the complete differences from the shipped schema defaults; an unlisted setting
equals its shipped default. The record's
conditions and each cell's provenance come from the same resolved build-child
snapshot, and validation rejects disagreement between them. Its only semantic
difference from shipped product defaults is `split.test_count=0`, which keeps
the established external evaluation cells intact.

| Cells | Setting | Shipped default | Cell value | Winning source kind |
|---|---|---:|---:|---|
| on/slt55, on/big | `runner.jobs` | `null` | `32` | `cli` |
| on/slt55, on/big | `split.test_count` | `null` | `0` | `project-profile` |

## Provenance correction

The previous record carried its on-mode numbers from a `7f13286`
all-triphone run but mislabeled them as a `578f6a9` transcript-reachable run.
On 2026-08-17, the pin was remeasured on shrub at `bbb2fef` under the declared
transcript-reachable profile. The retained run includes its resolved
configuration, training log, and an 11,883-row CD-untied mdef. It reproduced
the published rows byte-for-byte: `on/big` SHA-256
`11d512dbd1c412bd56a94717b3d91b9fbfdb2ee97c2c169db9c0a9e749e4a977`
and `on/slt55` SHA-256
`f3fa77a1a2cbf51138f0f7c375b8392a9ce707d41a3bba41613cfbb44cdd0d54`.
The WER results are unchanged; the record now carries the measured engine,
configuration, and native-library provenance.

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
| off (retired) | SLT-55 | 28.8499 | 28.8499 | +0.0000 | [-4.7059, +4.7619] | historical only |
| on | SLT-55 | 26.9006 | 26.9006 | +0.0000 | [-2.5831, +2.5000] | no statistically significant regression |
| off (retired) | big | 76.6393 | 74.7915 | +1.8478 | [+1.3257, +2.3646] | historical only |
| on | big | 75.2685 | 75.0017 | +0.2668 | [-0.1991, +0.7309] | no statistically significant regression |

The live intervals are generated from the current record and preserved oracle
rows by `scripts/regenerate_arctic_paired_analysis.py`; they are not
hand-entered into the machine-readable analysis.

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

## Training skips and decode coverage

The live pin has no accepted training skips. In particular, `arctic_a0587`
trains under product defaults and neither former exception fires. The
`training.accept_arctic_a0587_known_skip` knob is retained solely to describe
the retired `off` profile's provenance; that profile carries the true value,
while the live `on` profile disables it (and the a0302 exception band). Thus
all live cells run exception-free. Decode shortfalls are not gates: coverage
is a recorded comparison field, and drift from its pinned
`decoded/denominator` is surfaced in `field_differences` for deliberate record
adoption without raising or failing the WER gate.

## Condition contract maintenance

The record owns the condition fields it contains. A changed or removed pinned value is fatal;
a live field added after the record was written is reported as uncovered but does not masquerade
as benchmark drift. `make config-check` runs this authentication in the ordinary suite. Adopt
newly added fields deliberately, without changing existing pins or any measured result, with
`python scripts/check_arctic_pin.py --adopt-uncovered`.

Fresh-record adoption also requires exact full-cell equality for both retired
`off` cells. Every serialized field in each retired cell—including bootstrap
summaries and any future field—is stable; any addition, removal, or value
change is refused.

## Replicability

The benchmark uses canonical ordering and is deterministic. Native Mach-O
fingerprints omit randomized UUID and derived code-signature bytes, and model
identity covers every acoustic-model artifact while excluding timing,
provenance, and completion metadata. The same checkout and resource band
therefore produce the same pinned content identities; environment identity is
recorded separately.
