# Native boundary: what is contained, and what is not

`libpstrainc` is vendored C. It reports errors by ending the process: roughly
495 `E_FATAL` / `E_FATAL_SYSTEM` / `exit` / `abort` sites across 69 of its 157
compiled files, plus the segfaults that malformed input provokes in parsers
that do not check their own reads. None of that is catchable in the calling
interpreter, so any operation that dlopens the library in-process can end a
Python program that merely handed it a bad file.

The containment mechanism is a **persistent helper process**. It is not the
pipeline's `ProcessPoolExecutor`: it is a separate, long-lived child that holds
the library, services one request at a time over a pipe, and is respawned
whenever it dies. Its address space is the blast radius. See
`pstrain.lib.native_worker`.

## Phase contract: `contained-all-operations`

Supported Python operations implemented by pstrain route their known direct CFFI call
expressions through the helper, except for the decoder described below. The static gate
checks specified syntactic patterns under a reserved `lib`/`_lib` name convention: literal
name and attribute-chain calls to the CDEF-derived native surface and loader helpers;
statically resolvable `getattr` and literal-key dictionary selections; and direct dynamic
`getattr`, `builtins.getattr`, `__getattribute__`, subscript, and explicit `__call__` forms
on those reserved handle names. A visibly local assignment of an ordinary Python value to
`lib` or `_lib` suppresses the convention for that scope.

This is a source-pattern gate, not proof of containment for every possible Python-to-CFFI
call. It is silent on multi-step dataflow, attribute binding provenance (including a Python
registry stored as `self._lib`), function pointers from `ffi.addressof`, cross-module aliases,
dynamic imports, and semantically equivalent spellings outside the enumerated patterns.
The shipped PocketSphinx decoder in `pstrain.lib.testing.decoder` remains in-process and
is used by benchmark, CLI testing, and decode-shard paths; decoder containment is not
certified and decoder behavior is unchanged. This decoder-exemption wording is
documentation-only; the exempt path and enforcement are unchanged.
The original three operations retain their individual protocol names:

| Python entry point | native entry point |
| --- | --- |
| `pstrain.lib.dtree.prune_tree` | `pstrain_prune_tree` |
| `pstrain.lib.dtree.make_quests` | `pstrain_make_quests` |
| `pstrain.lib.mdef.generate_ci_mdef` | `pstrain_mdef_gen_ci` |

For these three, malformed user input cannot terminate the calling process.
Each failure arrives as a typed exception carrying the operation, the input
paths, and the diagnostic text the native code wrote to its stderr.

The remaining one-shot wrappers use the generic `python_call` route. Stateful
BW, alignment and logmath objects live in the helper and are addressed through
opaque object handles. Raw model I/O is exposed only through coarse complete
read/write operations, and feature extraction through its complete operation;
raw C pointers never cross the process boundary. Direct access through the
private `_pstrainc` implementation module is not a supported public operation.

## Complete-model value validation

`require_complete_model()` does not claim that every Python rejection is also a native
parser rejection. Its range and enumeration checks are a conservative subset of observed
native rejection, but its numeric spelling check is deliberately stricter. Native command-line
conversion accepts a numeric prefix and truncates fractional spellings supplied to integer
options (for example, `-ncep 13.5` becomes `13`). The complete-model boundary rejects such
tokens because accepting them would let the recorded front end differ from the value actually
used. Native parsing and feature-layout acceptance remain authoritative outside this stated
exception.

Across every phase there is one standing rule: **one native call at a time per
process.** Multi-threaded use of the library is unsupported and undefined —
`global_cmdln`, the error-callback slot, the allocator's jump target and some
forty per-program statics are all process-global. Inside a contained operation
this holds structurally, because the helper is single-threaded and services one
request at a time.

## Exception hierarchy

```
PstrainError
└── PstrainNativeError                operation, input_path(s), diagnostic, returncode
    ├── PstrainNativeFatalError       helper exited nonzero, or the call reported a diagnosed failure
    ├── PstrainNativeCrashError       helper died on a signal (adds .signal)
    ├── PstrainWorkerProtocolError    helper exited 0 with a request outstanding
    └── PstrainInvalidInputError      refused by Python-side validation, before any helper ran
PstrainWorkerError                    helper unstartable, timed out, or died between requests
```

`PstrainNativeError` subclasses `RuntimeError`, so callers that already catch
`RuntimeError` keep working.

`input_path` names the file that failed, not the output path.

A clean `exit(0)` while a request is outstanding is a **protocol violation, not
a success**. `agg_seg`'s count path used to do exactly this — write the count
file, then `exit(0)` with status 0 and no diagnostic on an ordinary first run.
Those two sites are now compiled out of the library build
(`PSTRAIN_LIBRARY_BUILD`), so control returns to the caller;
`PstrainWorkerProtocolError` remains as the backstop for any site not yet found.

## Lifecycle

- One helper per owning process, created lazily on the first guarded call and
  reused for every call after it. Under the pipeline's fan-out, each pool worker
  gets its own helper, so parallelism is not funnelled through a single child.
- The `spawn` start method is used explicitly, everywhere. Construction state is
  picklable.
- At each complete-operation boundary the helper calls `pstrain_session_reset()`, which frees and
  NULLs `global_cmdln` (whose keys would otherwise accumulate across eight
  different argument tables) and clears the `pstrain_dtree` and `pstrain_kmeans`
  module statics. Methods on a live stateful object form one coarse operation,
  so reset occurs when that object closes rather than between its methods.
  Worker death remains the hard floor: the OS reclaims
  everything.
- Any death — signal, nonzero exit, clean exit mid-request, transport EOF — is
  classified, raised, and followed by a fresh helper on the next call.
- Control requests are limited to a 64 KiB serialized payload. Coarse operations
  that explicitly accept arrays (model I/O and in-memory BW/alignment calls) have
  a 256 MiB ceiling. One deadline covers both a non-blocking pipe send and the
  response wait; expiry kills the helper so the next call starts fresh.

## No silent fallback

When process isolation cannot be established, guarded work **fails**; it never
quietly runs in-process instead. The pipeline runner used to fall back to inline
execution when its process pool could not start. Under containment that would be
a silent revert to exactly the unsafe path containment exists to remove, so the
batch is aborted with a `PstrainWorkerError` instead. The startup probe proves
that the pool can start before task submission, not that every executor worker
has already started. A lazily started worker can still fail after other tasks
complete; that aborts the batch, with completed tasks and their provenance and
completion manifests remaining valid while unfinished tasks remain stale.
