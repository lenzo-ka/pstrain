# CLI-to-library boundary

The command-line package, `pstrain.cli`, is intended to reach training code
through `pstrain.api`, not directly through `pstrain.lib`. Two checks in the
repository catch a crossing that has been written by accident.

## What these checks are, and are not

They are development-time checks. `tests/conftest.py` installs the runtime one
for the pytest session; nothing in the shipped package imports it, and `tests`
is not in the wheel. Neither check ever runs in a user's process, and neither
guards anything at run time. What they guard is the architecture, while it is
being written.

So the only person in a position to defeat them is a contributor working in
their own pull request — and that contributor can delete the test, empty the
static allowlist, or edit the rule. Nothing here is designed to withstand
someone who wants to get around it, and nothing needs to be. The limits below
are about what ordinary code does that the checks cannot see, not about what a
determined author could arrange.

Neither check is a whole-program boundary proof.

## Runtime observation

`tests/conftest.py` installs the runtime observation for every pytest session,
alongside the process-pool constructor invariant and native-stdout scanner. It
wraps Python's `__import__` and `importlib.import_module` entry points. These
callables run before the module-cache lookup, unlike a meta-path finder or
import audit event, so a supported import request remains visible when its
target is already in `sys.modules`.

When such a request targets `pstrain.lib` and has CLI provenance, every
application frame between the CLI and the import must be in `pstrain.api` or
`pstrain.lib`, and the segment must contain an API frame.

### Establishing the three roles

Every role the rule reasons about — command line, public API, library — is
established the same way. The directory of each of the three packages is derived
at installation from the checkout root the test session already trusts, and is
then fixed for the session. The guard does not read a package's `__path__`
afterwards: that is an ordinary mutable list, and a test may legitimately
rearrange it, so reading it later would make the answer depend on when the
question was asked. Installation resolves each package through the import system
once, before any wrapper is in place and before any test module is imported, and
fails loudly if it does not land on the derived directory. An interpreter that
would import a different `pstrain` therefore fails at installation rather than
reporting on the wrong tree.

A frame belongs to a package when the live `sys.modules` entry for its
`__name__` owns that frame's globals and its code was compiled from a file under
that frozen directory. `__name__` alone is an ordinary writable string, so a
module that merely calls itself `pstrain.api.something` is not API code and
cannot establish a route; a frame that claims a `pstrain.cli` name with neither
half is an ordinary neutral frame that stays in the segment while the walk
continues past it.

### Command-line origins

Ordinary command-line code can hold just one half of that ownership, and each
half can go missing for reasons that have nothing to do with anyone trying to
hide.

The live `sys.modules` entry can be deleted or rebound while the very same
function keeps running, and a real command-line file executed through a loader
is never registered at all. The source location can be absent too: a zipapp, a
frozen bundle, or any loader that compiles from something other than a file
under the checkout gives a genuine `pstrain.cli` module a `co_filename` that is
not a path under the frozen directory.

Either half on its own makes a frame a *candidate* origin. What that buys is not
a stronger check but a check at all. With no origin anywhere on the stack there
is no rule to apply, so the library import was simply allowed and never
examined — the absence of provenance switched enforcement off rather than on,
and a computed import target evades the static scan as well, so nothing else
caught it. A frame holding both halves always wins; candidates are consulted
only after the walk has run to the end without one, which is exactly the set of
stacks that used to yield no origin.

Where several frames are candidates, the walk keeps the outermost. Keeping the
innermost made the verdict depend on which command-line frames happened to keep
their registration or their source location: an inner candidate's segment stops
short of every frame beyond it, so it discarded both the neutral frames that
interrupt the route and the outer command-line frame that followed them, and the
same import was refused or allowed according to that accident.

Keeping the outermost cannot go the other way. Every candidate is a non-boundary
frame — `pstrain.cli` is neither `pstrain.api` nor `pstrain.lib`, and the three
frozen directories are disjoint — and a candidate is added to the segment before
the walk moves past it. A segment reaching a further-out candidate therefore
contains the nearer one and is interrupted there. Moving the origin outwards can
turn an acceptance into a refusal, never a refusal into an acceptance.

### Route state across a dispatch

CLI provenance is copied into direct `threading.Thread` targets and
`ThreadPoolExecutor.submit` work, so a worker does not lose the origin merely by
losing the submitting stack. The worker validates its callable-side stack
against the carried origin; executor infrastructure is outside the validated
segment.

What is carried is three-valued, not two: the route so far is clean but has not
reached an API frame, or established through one, or interrupted. An
interruption is permanent within the route that carries it — no API frame on a
worker's stack can re-establish a route that a non-boundary frame has already
broken.

Permanence is not absolute, and should not be read that way. A worker frame
holding both halves of command-line ownership ends the stack walk where it
stands, so it supersedes the carried origin instead of continuing it, and the
segment inside it is judged on its own from a clean start. That is not an escape
from the carried state: it takes a genuine command-line frame running on the
worker's own stack, which is a new route by any reading, and it is exactly what
the submitting stack does with the same frame. Nothing weaker — a candidate
origin, an API frame, a helper — supersedes anything.

The distinction is the whole point. While the carried state was a single
"reached an API frame continuously" boolean, "not through an API frame yet" and
"already interrupted" shared its `False`, and a worker-side API frame turned the
second into an established route. Command-line code that called a neutral helper
and let the helper submit an API call to a thread or an executor was therefore
allowed, exactly like the same code submitting that API call itself — and only
the second is the route the boundary permits. Two flags, one of them sticky,
would record the distinction as well, but they admit a combination that means
nothing, and every reader would have to remember to consult the second before
trusting the first. Reading one and not the other was the defect.

### Transparent infrastructure

Only two kinds of infrastructure are transparent, and both are held by identity
rather than by module name. The import machinery is recognized by
module-dictionary identity. A short list of exact `multiprocessing` code objects
covers serialization, where the two directions are not equivalent. Pickling
re-imports the defining module of an object its caller already chose, so it is
transparent when a boundary frame invoked it. Unpickling takes its target from
the incoming bytes and so can never establish a crossing; it is transparent only
inside `pstrain.lib`, where the import it triggers is library-to-library and
crosses nothing. Every other frame, including the rest of `multiprocessing`,
stays in the segment.

Checking the whole remaining segment rejects an API function that calls a
neutral callback or resumes a neutral generator. It also rejects a lazy library
import made by a library function that CLI code obtained without an API frame on
the call path.

### Consequences for the public API

The rule needs an API frame, not merely an API name, so `pstrain.api` forwards
through concrete functions wherever the command line reaches work that imports
lazily. A bare re-export leaves nothing on the stack and its callers would be
refused. The forwarders are additions to the public surface, never replacements
for what it already promised: each one's parameters are pinned to the library
callable it fronts and each publishes that callable's documentation. A public
class is not subclassed to obtain a frame, because that would cost the identity,
equality, `isinstance` and pickle relationships callers already rely on. Where
the command line needs an API frame to build a re-exported class, the API adds a
factory instead — `create_pipeline_context` for `PipelineContext` — and the
public class stays the same object the library exports.

### Reporting

Violations use a dedicated `BaseException` subclass so an application's broad
`except Exception` handler cannot turn a failed boundary check into an ordinary
fallback result.

Detecting a violation on a worker thread does not by itself fail the run. A
`ThreadPoolExecutor` submission re-raises through its future, but a bare
`threading.Thread` prints the exception and discards it, which pytest reports
only as a warning. Violations raised off the main thread are therefore recorded
as well, and the test lifecycle fails on any record still outstanding after a
test or at session end.

The test lifecycle also asserts ownership of every installed wrapper before and
after each test call and again before rollback, so a test that replaces one is a
failure rather than a silent gap in later enforcement. Under pytest-xdist each
worker installs and enforces its own wrappers; route observations are not
aggregated into the controller, so the controller summary says so instead of
displaying an empty list.

The whole rule rests on the one checkout root frozen at installation, so
installing a second time under a different root fails loudly rather than being
ignored in favor of the root already frozen.

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

## Combined coverage and limits

The runtime check rejects supported name-based `pstrain.lib` import requests
executed with a CLI origin — one carried from a dispatch, one taken from a frame
holding both halves of command-line ownership, or, failing both, one taken from
the outermost frame holding either half — unless a continuous API/library stack
segment establishes their route. The static check additionally rejects the
source constructions described above, including unexecuted paths and a directly
imported alias touching an already-loaded `pstrain.lib` attribute.

What ordinary code can do that neither check sees:

- a module object can be taken from `sys.modules`, retained before guard
  installation, returned by a function, or reached through an alias the static
  scan does not trace. No import request is made, so there is nothing for the
  runtime wrappers to observe;
- `getattr`, assignment data flow, function returns, generated code, custom
  loaders, private importlib entry points, and import callables captured before
  installation remain outside one or both checks;
- only two callables are wrapped, `builtins.__import__` and
  `importlib.import_module`. Native or extension code that reaches the import
  machinery by another route — `PyImport_ImportModule` and its relatives called
  from C, or an extension module importing during its own initialization —
  never passes through either one, so the import happens and nothing observes
  it. "Supported name-based imports" is about which callable the import goes
  through, not about how the module name is spelled;
- runtime observations cover only paths executed in the pytest process after
  installation. Spawned processes and subprocesses install no guard, so a
  crossing that happens only in a child is not observed;
- provenance propagation covers direct `threading.Thread` targets and the
  installed `ThreadPoolExecutor.submit` method — not every scheduler, a submit
  override, a native thread, or a process pool. Work reaching a worker by any
  other route arrives with no origin and is not examined;
- the import machinery and an enumerated set of `multiprocessing` serialization
  code objects are omitted from the stack segment; they are not application
  code. A rename in a future CPython makes installation fail rather than quietly
  changing what is omitted;
- exactly one unpickling code object is transparent — the enumerated
  `multiprocessing.connection._ConnectionBase.recv` — and only when it is
  reached directly from a `pstrain.lib` frame. A library frame that receives on
  such a connection can therefore import library modules its own source never
  names, because the incoming bytes choose them. Nothing wider is conceded:
  `pickle.loads`, every other deserialization path, and that same code object
  reached from anywhere else all remain ordinary segment frames. The rule
  constrains routes from the command line into the library, not
  library-to-library imports;
- an explicit `except BaseException` intercepts the violation, and so does C
  code that discards the failed import and raises its own error: a refused
  library import during pickling surfaces as `PicklingError`. The crossing is
  still refused, but the violation type does not survive that conversion; and
- re-exported classes and previously handed-off objects do not record that they
  were obtained from the public API. A later import is allowed only when its
  executed stack independently satisfies the rule.

The check errs toward refusing in one known shape. A command-line module
recognized only by its module identity — one loaded from a zip or a custom
loader — that calls a second command-line module recognized only by its source
location, which then reaches the library through the public API, is refused: the
outermost rule takes the first module as the origin and leaves the second in the
segment, where it is a non-boundary frame and interrupts the route. The route is
a genuine API route, so the refusal is wrong on its merits. Producing it takes
two differently loaded, partially recognized command-line modules in one stack,
which this repository does not do, and a development-time check that fails
closed here costs a contributor a message to read rather than a missed crossing.
The rule is not relaxed to allow it.

Two mechanical properties are worth stating plainly, because they are easy to
assume away. A frame is recognized as command-line code by module identity or by
source location, so code that arranges either — writing `sys.modules` under a
`pstrain.cli` name, or compiling itself with a matching `co_filename` — is
treated as command-line code; that is exactly the mechanism the guard's own
tests use to build genuine-shaped frames. And a lifecycle assertion between test
operations cannot observe a wrapper replaced and restored entirely inside one
operation. Neither is a weakness in the check, which has no adversary to
withstand; both are simply the shape of what it can see.
