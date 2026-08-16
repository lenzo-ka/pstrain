# Windows Python CI report — August 16, 2026

## Scope and parity classification

This change adds Windows native-library loading and Python CI coverage. It does not change training arithmetic, algorithms, model data, output formats, or numeric policy. There is no numeric or parity impact.

## Native library and program discovery

- The CFFI loader now selects the exact platform library name: `pstrainc.dll` on Windows, `libpstrainc.dylib` on macOS, and `libpstrainc.so` on other POSIX systems.
- Windows build discovery checks configuration-specific runtime directories, including `build/bin/Release`, before the generic `build/bin` directory. Package-local `bin` discovery remains supported.
- CLI resolution checks the `.exe` form on Windows, including programs found through the configured binary directory. POSIX program names remain unchanged.
- Loader tests assert that Windows does not select a POSIX library and that POSIX systems do not select the Windows DLL.

## Windows Python gate

The `windows-latest` job uses Python 3.13 and MSVC. It performs one Release build of the shared native library and CLI programs, installs the package with its development dependencies, sets `PSTRAIN_REQUIRE_CLIB=1`, and runs the Windows-compatible pytest selection.

The passing selection contains 39 tests:

- Seven library-path tests covering `build/bin/Release`, `build/bin`, package-local `bin`, and platform-specific filenames.
- One Windows `.exe` resolution test and one CFFI ABI-skew rejection test.
- Nine package and binding-structure tests covering a real DLL load, required symbols, singleton behavior, CFFI type creation, and the native floating-point contraction declaration.
- Fourteen direct native-binding tests covering library loading, logmath operations and edge cases, exported constants, error macros, and frontend creation with default, custom, and 8 kHz configurations.
- Seven platform-agnostic feature tests covering parameter objects, Sphinx MFC file round trips, and WAV input.

The full-suite probe was not used as the gate. It reached 283 passes but also produced 112 failures and 10 errors. Higher-level model, frontend, splitting, training, pipeline, subprocess, and parity tests currently cross POSIX-only process boundaries, including the native worker's use of POSIX nonblocking file descriptors, signal and process-group APIs, shell assumptions, and Unix path behavior. Their assertions were not weakened.

## Explicit Windows skips

Eight POSIX-only modules are still collected by the Windows selection so pytest reports their reasons:

- `test_bw_sharding.py`, `test_dtree.py`, `test_e2e_training.py`, `test_feat_params.py`, `test_train_convergence.py`, and `test_train_retry.py`: skipped because training resource accounting requires the POSIX `resource` module.
- `test_numeric_harness.py` and `test_pipeline_tasks.py`: skipped because pipeline locking requires the POSIX `fcntl` module.

## Validation

- Local macOS `make verified`: passed, including 720 Python tests passed, one skipped, one deselected, 10 CTest tests, configuration checks, the floating-point contraction gate, Ruff, mypy, and formatting checks.
- Local selected pytest run: 39 passed.
- Pull request #126 Windows Python 3.13 job: 39 passed and eight skipped.
- Pull request #126 Windows native build, MSVC: passed.
- Pull request #126 Windows native build, clang-cl: passed.
