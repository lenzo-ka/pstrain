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
`pstrain.lib`, and the segment must contain an API frame.

Every role the rule reasons about — command line, public API, library — is
established the same way, and from something the code being judged cannot
rewrite. The directory of each of the three packages is derived at installation
from the checkout root the test session already trusts, and is then fixed for
the session. The guard never reads a package's `__path__` afterwards, because
`__path__` is an ordinary mutable list: code that could rewrite it before the
first check could move the anchor under files of its own and authenticate from
there. Installation resolves each package through the import system once, before
any wrapper is in place and before any test module is imported, and fails loudly
if it does not land on the derived directory, so an interpreter that would
import a different `pstrain` cannot quietly authenticate the wrong tree.

A frame belongs to a package only when the live `sys.modules` entry for its
`__name__` owns that frame's globals and its code was compiled from a file under
that frozen directory. `__name__` is an ordinary writable string, so a module
that merely calls itself `pstrain.api.something` is not boundary code and cannot
establish a route. The same holds for the command line, and there it matters in
the permissive direction: a command-line frame is what ends the stack walk, so a
frame that merely claimed a `pstrain.cli` name would end the walk early and hide
both itself and the real command-line frame beyond it, turning a refused import
into an allowed one. An unauthenticated frame claiming such a name is therefore
an ordinary neutral frame: it stays in the segment, and the walk continues past
it to whatever really dispatched the call.

Requiring a live `sys.modules` entry also matters in the other direction. That
entry is ordinary mutable process state, so genuine command-line code can be
running with no entry under its name, or with the name bound to some other
object — deleted, rebound, or a real command-line file executed through a
loader and never registered. No frame then authenticates as the command line,
and a stack with no recognized command-line frame yielded no origin at all.
With no origin there is no rule to apply, so the import was allowed: the
absence of provenance switched enforcement off rather than on. A computed
import target evades the static scan as well, so nothing else caught it.

The stack walk therefore also remembers a frame whose code was compiled from a
file under the frozen command-line directory, together with the segment
collected up to that frame, and falls back to that source location when the
walk finds no authenticated origin. A source location is not
authentication — `co_filename` is chosen by whoever compiled the code — and it
is not used as one: an authenticated origin always wins, and the fallback is
consulted only after the walk has run to the end without finding one. That is
exactly the set of stacks that previously yielded no origin and were allowed
unconditionally.

Where several frames carry such a source location, the walk keeps the outermost
one. Keeping the innermost was itself a way to make enforcement *more*
permissive by forging a `co_filename`. An inner candidate's segment stops short
of every frame beyond it, so a generated frame placed just below the public API
could name itself into the command-line directory, become the nearest
candidate, and discard both the neutral frames that interrupt the route and the
genuine command-line frame that follows them — and the import refused without
that forgery was allowed with it. Keeping the outermost candidate leaves the
intervening frames in the segment, where an inner candidate is judged as the
ordinary neutral frame it is. That is exactly how such a frame is already
judged when an authenticated command-line frame lies further out, so the
verdict no longer depends on which command-line frames happened to keep their
registration. A forged source location can then only lengthen the segment, so
it can make the check stricter or leave it unchanged, never weaker.

Only two kinds of infrastructure are transparent, and both are held by identity
rather than by module name. The import machinery is recognized by
module-dictionary identity. A short list of exact `multiprocessing` code
objects covers serialization, where the two directions are not equivalent.
Pickling re-imports the defining module of an object its caller already chose,
so it is transparent when an authenticated boundary frame invoked it.
Unpickling takes its target from the incoming bytes and so can never establish
a crossing; it is transparent only inside `pstrain.lib`, where the import it
triggers is library-to-library and crosses nothing. Every other frame,
including the rest of `multiprocessing`, stays in the segment.

Checking the whole remaining segment rejects an API function that calls a
neutral callback or resumes a neutral generator. It also rejects a lazy library
import made by a library function that CLI code obtained without an API frame
on the call path.

The rule needs an API frame, not merely an API name, so `pstrain.api` forwards
through concrete functions wherever the command line reaches work that imports
lazily. A bare re-export leaves nothing on the stack and its callers would be
refused. The forwarders are additions to the public surface, never
replacements for what it already promised: each one's parameters are pinned to
the library callable it fronts and each publishes that callable's
documentation. A public class is not subclassed to obtain a frame, because that
would cost the identity, equality, `isinstance` and pickle relationships
callers already rely on. Where the command line needs an API frame to build a
re-exported class, the API adds a factory instead —
`create_pipeline_context` for `PipelineContext` — and the public class stays
the same object the library exports.

Violations use a dedicated `BaseException` subclass so an application's broad
`except Exception` handler cannot turn a failed boundary check into an ordinary
fallback result.

CLI provenance is copied into direct `threading.Thread` targets and
`ThreadPoolExecutor.submit` work. The worker validates its callable-side stack
against that carried origin; executor infrastructure is outside the validated
segment. This closes the ordinary thread-dispatch loss of Python call frames.

Detecting a violation on a worker thread does not by itself fail the run. A
`ThreadPoolExecutor` submission re-raises through its future, but a bare
`threading.Thread` prints the exception and discards it, which pytest reports
only as a warning. Violations raised off the main thread are therefore recorded
as well, and the test lifecycle fails on any record still outstanding after a
test or at session end.

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
executed with a CLI origin — one carried from a dispatch, one authenticated on
the stack, or, failing both, one taken from the source location of the
outermost frame on the stack compiled under the frozen command-line
directory — unless a continuous API/library stack segment
establishes their route. Every frame of that segment, and an authenticated
origin, are established the same way: by module identity plus a source location
under a package directory frozen from the checkout root at installation. The
static check additionally rejects the source constructions
described above, including unexecuted paths and a directly imported alias
touching an already-loaded `pstrain.lib` attribute.

Neither check observes or resolves all Python behavior. In particular:

- a module object can be taken from `sys.modules`, retained before guard
  installation, returned by arbitrary code, or reached through an alias that
  the static scan does not trace;
- a frame is authenticated from the live `sys.modules` entry for its name plus
  a source location under the frozen directory. Code that both writes
  `sys.modules` under a real package name and compiles itself with a
  `co_filename` inside that package's directory can therefore still present
  itself as code of that package. Both halves are required, neither is
  something ordinary command-line code does, and the anchor itself is no longer
  reachable — but this is a residual forgery the runtime rule does not detect,
  and it is the mechanism the guard's own tests use to build genuine-shaped
  frames;
- a command-line origin always requires a source location under the frozen
  command-line directory. A live `sys.modules` entry owning the frame's globals
  never supplies origin coverage by itself: it authenticates a source location
  the frame already has, and it stands in for nothing when that source location
  is absent. So a module registered under a real `pstrain.cli` name, whose live
  entry does own its frame's globals, is still not a command-line origin when
  its code was compiled outside that directory — and neither is code with
  neither half. In both cases the runtime rule does not apply to that stack's
  imports at all: the check is not merely weaker there, it is absent, and a
  `pstrain.lib` import made on such a stack is allowed without being examined.
  Ordinary command-line code is compiled under the directory, and the fallback
  origin covers the remaining case, where only the live module identity is
  missing or has been replaced;
- `getattr`, assignment data flow, function returns, generated code, custom
  loaders, private importlib entry points, and import callables captured before
  installation remain outside one or both checks;
- runtime observations cover only paths executed in the pytest process after
  installation; spawned processes and subprocesses install no guard;
- provenance propagation covers direct `threading.Thread` targets and the
  installed `ThreadPoolExecutor.submit` method, not every scheduler, a submit
  override, native thread, or process pool;
- the import machinery and an enumerated set of `multiprocessing`
  serialization code objects are omitted from the stack segment; they are not
  treated as application code. A rename in a future CPython makes installation
  fail rather than quietly changing what is trusted;
- exactly one unpickling code object is transparent — the enumerated
  `multiprocessing.connection._ConnectionBase.recv` — and only when it is
  reached directly from an authenticated `pstrain.lib` frame. A library frame
  that receives on such a connection can therefore import library modules its
  own source never names, because the incoming bytes choose them. Nothing wider
  is conceded: `pickle.loads`, every other deserialization path, and that same
  code object reached from anywhere else all remain ordinary segment frames. The
  rule constrains routes from the command line into the library, not
  library-to-library imports;
- an explicit `except BaseException` can intercept the runtime violation, and
  so can C code that discards the failed import and raises its own error: a
  refused library import during pickling surfaces as `PicklingError`. The
  crossing is still refused, but the violation type does not survive that
  conversion; and
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

The whole rule now rests on the one checkout root frozen at installation, so
installing a second time under a different root fails loudly. It was previously
ignored in favor of the root already frozen, which would have enforced the
boundary of one checkout while the session ran another.
