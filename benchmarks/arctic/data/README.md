# Ratified Arctic band data

The PIN band uses `cmu_arctic_slt.dict` and `training-unigram.lm` byte-for-byte.
The dictionary is CMUdict-derived Arctic data regenerated from the upstream SLT
setup's dictionary: the same entry list, restricted to the installed CMUdict
source and deduplicated so that pronunciations differing only by stress do not
survive stress stripping as duplicate variants. It is distributed under the CMU
license in `csrc/LICENSE.sphinx`.

- Dictionary SHA-256: `b9d8271957f978287620d9b20a79e12b0b84470f520942c580145570021d0588`
- LM SHA-256: `2cf11ab0474a0bdd165cbee59db674b05764fdb00bf6f9824c0dccce571637b5`
- Generated filler SHA-256: `fb50883998c41a5030c2a602965935c647563321e84a86f2adabb377ec24b49c`

**Every benchmark number was measured with an Arctic dictionary, never with the
pip PocketSphinx dictionary.** The banked intent was “matched at decode time,”
and the actually matched resource was the Arctic dictionary; the harness honors
that intent over the letter. The pip dictionary is preserved only as the labeled
`--band=pip-en-us` alternative. Kevin reviews this interpretation at the pin PR.

Which Arctic dictionary differs by result:

- The live `on/slt55` and `on/big` cells, and the re-decoded upstream oracle
  rows they are compared against, were measured with the dictionary above.
- The retired `off/slt55` and `off/big` cells, the `off/big` +1.0-point floor,
  and the preserved historical oracle rows reported beside them were measured
  with the earlier Arctic dictionary, SHA-256
  `24ff2852a707b63f499fd968294d5e4c02d44e0eb1ec511e40be1f380d785846`. That
  identity travels with those results in the pin record's
  `historical_provenance` and in the oracle sidecar; the live pin does not
  restate it.

The canonical LM was generated with deterministic add-one smoothing from the
earlier 1,043-prompt training partition. The committed transcript contains all
1,132 prompts, so the retained `arpabo` builder produces the known
`43ea28991421...` model instead of the canonical LM. A regression test records
this understood mismatch; decode always consumes the committed canonical LM.
