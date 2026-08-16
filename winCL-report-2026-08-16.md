# Windows clang-cl build-closure report — August 16, 2026

## Scope and parity classification

This is a syntax/build-only change. It changes executable source ownership and namespaces one program-local helper; it does not change training arithmetic, algorithms, model data, CLI behavior, or output formats. There is no numeric or parity impact.

## Standalone program de-duplication

- `mk_s2sendump`: retained its genuinely program-specific `senone.c`, but renamed its externally visible `senone_init` definition to `mk_s2sendump_senone_init` so it cannot collide with the differently implemented core/dependency senone routine. No translation unit was removed from this target.
- `param_cnt`: removed `cb_cnt.c`, `enum_corpus.c`, `param_cnt.c`, `phone_cnt.c`, and `ts_cnt.c` from the executable source list. The target now uses the copies already compiled into `pstrainc`; only `main.c` and the standalone command-line parser remain executable-owned.
- `agg_seg`: removed `agg_all_seg.c`, `agg_phn_seg.c`, `agg_st_seg.c`, `cnt_phn_seg.c`, `cnt_st_seg.c`, and `mk_seg.c` from the executable source list. The target now uses the copies already compiled into `pstrainc`; only `main.c` and the standalone command-line parser remain executable-owned.
- `mk_mdef_gen`: removed `hash.c`, `heap.c`, and `mk_mdef_gen.c` from the executable source list. The target now uses the copies already compiled into `pstrainc`; only `main.c` and the standalone command-line parser remain executable-owned.
- `bw`: removed `accum.c`, `backward.c`, `baum_welch.c`, `forward.c`, `next_utt_states.c`, and `viterbi.c` from the executable source list. The target now uses the copies already compiled into `pstrainc`; only `main.c` and `train_cmd_ln.c` remain executable-owned.

No `/FORCE`, `/FORCE:MULTIPLE`, or other duplicate-tolerant linker option was added.

## Validation

- macOS AppleClang: `make verified` passed outside the process sandbox, including all 719 selected Python tests, all 10 CTest tests, the floating-point contraction gate, configuration checks, Ruff, mypy, and format checks. The complete `BUILD_CLI` program set built, including all five affected programs.
- Targeted macOS rebuild: `mk_s2sendump`, `param_cnt`, `agg_seg`, `mk_mdef_gen`, and `bw` all compiled and linked from the reduced source graphs.
- Shrub MSVC 19.51: the affected tree and pinned PocketSphinx checkout were rsynced to `~/shrub-data/win-cl-floor`. The documented `mscl.sh` compiler successfully entered MSVC compilation, but shrub does not provide an integrated CMake/Wine linker driver; Linux-host CMake stopped in its manifest/type feature-probe layer before generating project targets. A full five-target MSVC link was therefore not completed locally.
- Shrub clang-cl: no `clang-cl` or `lld-link` driver is installed. Clang-cl confirmation requires one Windows CI probe run. The fix is compiler-agnostic because it removes duplicate translation units at the CMake graph level and namespaces the one genuine program-local variant.
