# W-E report — August 16, 2026

## Scope and parity classification

W-E is on the build-only side of the line. It consolidates platform spellings and headers without changing training arithmetic, model data, algorithms, or parity-sensitive output. There is no parity impact. The historical Windows no-op delay in `sphinx3_align` remains a no-op through `sys_compat_optional_sleep`; other sleep behavior is unchanged.

`csrc/include/sys_compat/misc.h` is now the single routing point for `strdup`, `popen`, `pclose`, and sleep. `csrc/include/sys_compat/file.h` is the single routing point for `unlink` and `access`. The Windows routes are `_strdup`, `_popen`, `_pclose`, `Sleep`, `_unlink`, and `_access`; POSIX routes retain the original APIs. Generic translation units no longer guard `<unistd.h>` themselves: `pio.c` and `cmd_ln.c` use `sys_compat/file.h`, `profile.c` uses it for unlink, and `sphinx3_align/corpus.c` uses `sys_compat/misc.h`. The remaining direct `<unistd.h>` includes are internal to `sys_compat` or in `mmio.c`'s platform-specific memory-mapping implementation, outside these wrappers.

## Call sites moved

- `strdup` (37): `programs/mk_mdef_gen/hash.c` lines 94–97, 165, and 231; `programs/mk_mdef_gen/mk_mdef_gen.c` lines 175, 294, 489, 671–674, 711, 733–736, 911, and 916–919; `libs/libcommon/lexicon.c` line 256; `libs/libcommon/acmod_set.c` lines 212, 273, and 885; `libs/libio/pset_io.c` lines 97 and 111; `libs/libio/corpus.c` lines 239, 250, 272, 1114, 1280, and 1485; and `libs/libio/s3io.c` lines 120–121.
- `popen`/`pclose` (5): `programs/sphinx_fe/sphinx_fe.c` lines 239 and 268; `libs/libsphinxbase/util/pio.c` lines 138, 161, and 187. CMake declares `HAVE_POPEN` on Windows because `sys_compat` supplies the MSVC CRT route.
- Sleep (12): `programs/norm/main.c` lines 311, 356, and 379; `programs/agg_seg/agg_all_seg.c` line 163; `programs/bw/main.c` lines 931, 983, and 1816; `libs/libio/corpus.c` line 1332; `libs/libsphinxbase/util/pio.c` lines 436 and 506; and the `SLEEP_SEC` route in `programs/sphinx3_align/corpus.c` line 102. The latter preserves its prior Windows no-op behavior.
- `unlink` (8): `programs/bw/accum.c` lines 776, 781, 787, and 807; `libs/libsphinxbase/util/profile.c` lines 318, 325, 331, and 335.
- `access` (1): `programs/inc_comp/accum_wt_param.c` line 90.

## MSVC CRT warning policy

The selected policy is a scoped `_CRT_SECURE_NO_WARNINGS` definition in the CMake `MSVC` branch. This is justified for a build-only consolidation: the existing code's allocation, ownership, pipe, and file semantics remain intact, while changing these legacy interfaces to the secure CRT variants would introduce signature, buffer-management, and error-handling changes beyond W-E. All scattered warning 4996 pragmas were removed; unrelated conversion-warning pragmas remain scoped at their existing sites.

## Validation

- `make verified`: passed on macOS, including the full 719-test verified suite, 10 CTest tests, floating-point contraction gate, configuration checks, generated-document check, Arctic pin checks, Ruff, mypy, and format check.
- shrub MSVC-under-Wine accelerator: its documented smoke compile first exposed an incomplete Wine prefix left by setup; the broken prefix was retained as `.wine.broken-win-e-20260816`, a fresh prefix was generated, and the documented smoke compile passed.
- MSVC 19.51 wrapper probe: compiled and linked `strdup`, `popen`/`pclose`, sleep, `unlink`, and `access` routes into `win-e-probe.exe` with `/W3` and `_CRT_SECURE_NO_WARNINGS`.
- MSVC 19.51 touched translation units: all 17 wrapper-bearing units compiled to objects. The compiler emitted only pre-existing conversion, unused, and non-scoped POSIX-name warnings; the consolidated wrapper calls emitted no errors. The linked probe confirmed the wrapper definitions and CRT symbols resolve together.
- The never-gating Windows scoping workflow remains `continue-on-error`; W-E does not change its gating status.
