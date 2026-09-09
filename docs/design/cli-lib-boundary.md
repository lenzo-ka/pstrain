# CLI-to-library boundary

The command-line package, `pstrain.cli`, reaches training code through the
public `pstrain.api` package. It must not import `pstrain.lib` directly or ask
another module to import the library on its behalf.

## Enforcement

The authoritative gate is installed by `tests/conftest.py` for the pytest
session, which is the same continuous-integration path as the process-pool
constructor invariant and the native-stdout scanner. It wraps Python's
`__import__` and `importlib.import_module` entry points and checks the runtime
module name against the active call stack. If the nearest pstrain boundary
frame is a CLI frame, the import fails with the target and source location. An
API frame permits the import, as does a library frame performing a subsequent
internal import.

The pull-request workflow runs that pytest session on every pull request. Code
changes use the normal three-leg PR matrix; a documentation-only change still
runs the Ubuntu/Python 3.11 leg so the runtime gate cannot disappear behind the
path classifier.

Wrapping the name-based entry points is deliberate. Meta-path finders and
Python's import audit event run only when import machinery must load a module;
they do not run when `sys.modules` already contains it. The wrappers run before
that cache lookup, so package-attribute imports, aliases, and computed names do
not create a cache-dependent hole.

`scripts/check_cli_lib_boundary.py` remains as a cheap static early warning. It
finds ordinary direct imports in CLI files even when tests do not execute the
affected path. It is not authoritative: computed names and transitive imports
are intentionally outside its scope.

## Exact limits

The runtime gate observes only imports executed in the pytest process after the
root conftest installs it. It does not observe:

- an unexecuted CLI path;
- a spawned child or subprocess;
- a module object taken directly from `sys.modules` or retained before guard
  installation, because no import occurs;
- a custom loader, private importlib entry point, or import callable captured
  before guard installation; or
- native-extension loading that does not use either wrapped Python entry point.

The stack rule is intentionally narrow. It decides import provenance from the
nearest `pstrain.api`, `pstrain.lib`, or `pstrain.cli` frame. It does not try to
prove that arbitrary objects previously handed to CLI code originated in the
public API.
