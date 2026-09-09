# CLI-to-library boundary

The command-line package, `pstrain.cli`, is intended to reach training code
through `pstrain.api`, not directly through `pstrain.lib`. The repository uses
two deliberately limited regression checks for that architecture. Neither is
an authoritative whole-program boundary proof.

## Runtime observation

`tests/conftest.py` installs the runtime observation for every pytest session,
alongside the process-pool constructor invariant and native-stdout scanner. It
wraps Python's `__import__` and `importlib.import_module` entry points. These
callables run before the module-cache lookup, unlike a meta-path finder or
import audit event, so a supported import request remains visible when its
target is already in `sys.modules`.

When such a request targets `pstrain.lib` and has CLI provenance, every
application frame between the CLI and import must be in `pstrain.api` or
`pstrain.lib`, and the segment must contain an API frame. Import bootstrap
frames are transparent. So are `multiprocessing` frames, which re-import the
defining module of a process target chosen by an already validated library
frame during serialization. Checking the rest of the whole segment rejects an
API function that calls a neutral callback or resumes a neutral generator. It
also rejects a lazy library import made by a library function that CLI code
obtained without an API frame on the call path.

Violations use a dedicated `BaseException` subclass so an application's broad
`except Exception` handler cannot turn a failed boundary check into an ordinary
fallback result.

CLI provenance is copied into direct `threading.Thread` targets and
`ThreadPoolExecutor.submit` work. The worker validates its callable-side stack
against that carried origin; executor infrastructure is outside the validated
segment. This closes the ordinary thread-dispatch loss of Python call frames.

The pull-request workflow runs pytest on every pull request. Code changes use
the normal three-leg PR matrix; a documentation-only change still runs the
Ubuntu/Python 3.11 leg.

## Static addition

`scripts/check_cli_lib_boundary.py` scans every CLI source file whether or not
tests execute it. It rejects ordinary imports targeting `pstrain.lib`, common
literal dynamic-import spellings, and attribute access such as `package.lib`
when `package` is bound by a direct `import pstrain as package`. That last rule
covers the important import-order case in which another module has already
loaded `pstrain.lib`: `import pstrain` is the only runtime import request, but
the forbidden attribute access remains visible in CLI source.

The static check is an early-warning ratchet, not a Python data-flow analysis.
Its allowlist is empty.

## Exact combined guarantee and limits

The runtime check rejects supported name-based `pstrain.lib` import requests
executed with a visible or carried CLI origin unless a continuous API/library
stack segment establishes their route. The static check additionally rejects
the source constructions described above, including unexecuted paths and a
directly imported alias touching an already-loaded `pstrain.lib` attribute.

Neither check observes or resolves all Python behavior. In particular:

- a module object can be taken from `sys.modules`, retained before guard
  installation, returned by arbitrary code, or reached through an alias that
  the static scan does not trace;
- `getattr`, assignment data flow, function returns, generated code, custom
  loaders, private importlib entry points, and import callables captured before
  installation remain outside one or both checks;
- runtime observations cover only paths executed in the pytest process after
  installation; spawned processes and subprocesses install no guard;
- provenance propagation covers direct `threading.Thread` targets and the
  installed `ThreadPoolExecutor.submit` method, not every scheduler, a submit
  override, native thread, or process pool;
- trusted import-bootstrap and `multiprocessing` infrastructure frames are
  omitted from the stack segment; they are not treated as application code;
- an explicit `except BaseException` can intercept the runtime violation; and
- re-exported classes and previously handed-off objects do not themselves
  record that they were obtained from the public API. A later import is allowed
  only when its executed stack independently satisfies the runtime rule.

Under pytest-xdist, each worker installs and enforces its own wrappers. Route
observations are not aggregated into the controller, so the controller summary
says that explicitly instead of displaying an empty observation list.

The test lifecycle asserts ownership of every installed wrapper before and
after each test call and again before rollback. Replacing a wrapper is therefore
a test failure rather than a silent gap in later enforcement. Deliberately
replacing and restoring a wrapper entirely inside one test operation remains
outside what a periodic lifecycle assertion can observe.
