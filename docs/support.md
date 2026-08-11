# Support and dependency policy

## Platforms

pstrain supports macOS and Linux. CI builds wheels and runs the C and Python
test suites on both platforms. Native Windows/MSVC support is future work: the
vendored CMU Sphinx C currently relies on POSIX APIs and does not build cleanly
with MSVC. WSL is the current Windows-hosted route to a supported Linux build.

## Dependencies

Runtime dependencies declare tested minimum versions and may float within their
compatible major releases; lock files are intentionally not used for library
consumers. Development, documentation, and CI tools are constrained in
`pyproject.toml`, while GitHub Actions and pre-commit hook revisions are pinned
in their workflow files and advanced through reviewed dependency updates.
Security and compatibility fixes may raise a minimum version; unnecessary
runtime dependencies should not be added.
