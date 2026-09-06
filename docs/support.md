# Support and dependency policy

## Platforms

pstrain supports macOS, Linux, and Windows. CI builds wheels on all three
platforms, and release builds publish those wheels to PyPI. On Windows, CI
builds the native library and command-line programs with both MSVC and clang-cl,
then runs a Windows-compatible subset of the Python tests against an MSVC build
on Python 3.13. The full Python test suite is run on macOS and Linux; Windows CI
does not currently claim that broader coverage.

## Dependencies

Runtime dependencies declare tested minimum versions and may float within their
compatible major releases; lock files are intentionally not used for library
consumers. Development, documentation, and CI tools are constrained in
`pyproject.toml`, while GitHub Actions and pre-commit hook revisions are pinned
in their workflow files and advanced through reviewed dependency updates.
Security and compatibility fixes may raise a minimum version; unnecessary
runtime dependencies should not be added.
