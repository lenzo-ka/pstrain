# Source-determinable re-roll dispositions

Decision date: 2026-08-14.

All six surfaces are **keep and declare**. An uncalled implementation is not residue when it correctly
preserves a capability represented by a vendored core program. The in-process wrappers retain the PR
#92 containment boundary, typed Python exceptions, and direct in-process array handling; they do not
replace the upstream command position.

| ID | Capability and core program | User-facing wiring | Why keep the in-process wrapper | Upstream-compatible position |
| --- | --- | --- | --- | --- |
| R04 | Deleted interpolation (`delint`) | Unwired. A semi-continuous-model final smoothing pipeline step plus CLI/config would wire it. | `pstrain_delint` remains behind `@contained`, preserving typed failures and in-process accumulator/path arrays without a subprocess boundary. | Exists by construction: core `delint` and `CommandBuilder.delint()` remain callable. |
| R07 | Flat initialization (`mk_flat`, `init_gau`, `norm`) | Wired through the `pstrain flat` CLI and flat-model pipeline task. | The live `pstrain_flat_tmat`, `pstrain_flat_mixw`, `pstrain_init_gau`, and `pstrain_norm_gau` calls remain behind `@contained`, preserving typed failures and direct model-array assembly. | Exists by construction: all three core programs remain callable. |
| R08 | Gaussian splitting (`inc_comp`) | Wired through the Gaussian-splitting CLI and density-growth training step. | The live `pstrain_inc_comp` call remains behind `@contained`, preserving typed failures and direct density-count/model-array handling. | Exists by construction: core `inc_comp` remains callable. |
| R09 | KD-tree construction (`kdtree`) | Unwired. A semi-continuous-model packaging/decoder-acceleration step plus CLI/config would wire it. | `pstrain_kdtree_build` remains behind `@contained`, preserving typed failures and in-process model data handling without a subprocess boundary. | Exists by construction: core `kdtree` and `CommandBuilder.kdtree()` remain callable. |
| R11 | MAP adaptation (`map_adapt`) | Unwired. An adaptation pipeline/CLI stage consuming adaptation-data BW accumulators would wire it. | `pstrain_map_adapt` remains behind `@contained`, preserving typed failures and in-process accumulator/model arrays without a subprocess boundary. | Exists by construction: core `map_adapt` and `CommandBuilder.map_adapt()` remain callable. |
| R13 | Corpus parameter counting (`param_cnt`) | Unwired. A corpus-diagnostics/state-tying preparation step plus CLI/config would wire it. | `pstrain_param_cnt` remains behind `@contained`, preserving typed failures and the in-process native data boundary. Its current Python contract writes the core text count format and returns `None`; it does not return typed counts. | Exists by construction: core `param_cnt` and `CommandBuilder.param_cnt()` remain callable. |

The containment-routing gate is unchanged. Each Python-to-CFFI entry point above remains `@contained`;
no declared exception is added or weakened.
