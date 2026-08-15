# Source-determinable re-roll dispositions

Decision date: 2026-08-14.

The native-worker boundary is part of the supported behavior of retained CFFI operations. A retained
wrapper is therefore a deliberate divergence from the upstream command boundary even when its numeric
implementation is equivalent to the core program.

| ID | Operation | Disposition | Declaration | Upstream-compatible position |
| --- | --- | --- | --- | --- |
| R04 | Deleted interpolation | Delete and use core | The uncalled `pstrain_delint` and Python wrapper were removed. `CommandBuilder.delint()` invokes the core program, so failures use its process exit and stderr contract. | Core is the only supported position. |
| R07 | Flat initialization | Keep and declare | The live flat pipeline keeps `pstrain_flat_tmat`, `pstrain_flat_mixw`, `pstrain_init_gau`, and `pstrain_norm_gau` behind `@contained`. This preserves one in-process-shaped workflow, typed containment failures, and direct model-array assembly. | The shipped `mk_flat`, `init_gau`, and `norm` programs remain callable; no additional compatibility position is owed because the retained path adds a boundary without changing the established model semantics. |
| R08 | Gaussian splitting | Keep and declare | The live density-growth step keeps `pstrain_inc_comp` behind `@contained`. This preserves typed containment failures and direct use of the retained density-count path. | The shipped `inc_comp` program remains callable; no additional compatibility position is owed because both positions implement the same split semantics. |
| R09 | KD-tree builder | Delete and use core | The uncalled `pstrain_kdtree_build` and Python wrapper were removed. `CommandBuilder.kdtree()` invokes the core program. | Core is the only supported position. |
| R11 | MAP adaptation | Delete and use core | The uncalled `pstrain_map_adapt` and Python wrapper were removed. `CommandBuilder.map_adapt()` invokes the core program. | Core is the only supported position. |
| R13 | Parameter counting | Delete and use core | The uncalled `pstrain_param_cnt` and Python wrapper were removed. `CommandBuilder.param_cnt()` invokes the core program. The removed wrapper wrote the core text format and returned `None`; it did not return typed counts to Python. | Core is the only supported position. |

The containment-routing gate is unchanged. No deleted operation required an exception: its Python-to-CFFI
callsite and native symbol were removed together. The retained R07 and R08 callsites remain `@contained`.
