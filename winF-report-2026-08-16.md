# W-F report — August 16, 2026

## Scope and parity classification

W-F is a build/CI-configuration-only change. It does not change training arithmetic, algorithms, model data, CLI behavior, or output formats. There is no numeric or parity impact.

## Windows native build gate

- Promoted the Windows native workflow from a dispatch-only diagnostic probe to a normal blocking workflow on pull requests and pushes to the default branches.
- Kept MSVC as the primary matrix row and clang-cl as the second row. Each row configures the complete native graph and performs a Release build of the static core, shared core, and CLI programs.
- Removed job-level error tolerance and MSBuild's `ContinueOnError` setting, so any configuration, compilation, or link error fails its matrix row and the workflow.
- Removed probe-only graph inspection, floating-point canaries, object staging, and artifact upload. The gate runs only the native build; it does not run Python or pytest on Windows.

## Validation

- `make verified`: passed on macOS, including all 719 selected Python tests, all 10 CTest tests, the floating-point contraction gate, configuration checks, generated-document checks, Arctic pin checks, Ruff, mypy, and format checks.
- Pull request Windows native build, MSVC: pending.
- Pull request Windows native build, clang-cl: pending.

Making the Windows native build check required for merge is a branch-protection setting and is Kevin's follow-up.
