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

## Phase contract: `contained-3`

Exactly three operations are routed through the helper today:

| Python entry point | native entry point |
| --- | --- |
| `pstrain.lib.dtree.prune_tree` | `pstrain_prune_tree` |
| `pstrain.lib.dtree.make_quests` | `pstrain_make_quests` |
| `pstrain.lib.mdef.generate_ci_mdef` | `pstrain_mdef_gen_ci` |

For these three, malformed user input cannot terminate the calling process.
Each failure arrives as a typed exception carrying the operation, the input
paths, and the diagnostic text the native code wrote to its stderr.

**Everything else is unguarded and keeps today's behaviour.** That includes the
other 65 `pstrain_*` operations — among them `pstrain_agg_seg`, `pstrain_bw_*`,
`pstrain_align_*`, `pstrain_tie_states`, `pstrain_init_mixw`, `pstrain_norm`,
`pstrain_param_cnt` — and the 91 raw vendored functions declared in
`pstrain/lib/_cffi/cdef.py` (`model_def_read`, the `s3gau_*` readers/writers,
`fe_*`, `feat_*`, `logmath_*`, `acmod_set_*`). Calling any of them with input
you have not validated can still end the interpreter. Extending containment to
the full surface is the next phase.

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
- Between requests the helper calls `pstrain_session_reset()`, which frees and
  NULLs `global_cmdln` (whose keys would otherwise accumulate across eight
  different argument tables) and clears the `pstrain_dtree` and `pstrain_kmeans`
  module statics. Worker death remains the hard floor: the OS reclaims
  everything.
- Any death — signal, nonzero exit, clean exit mid-request, transport EOF — is
  classified, raised, and followed by a fresh helper on the next call.

## No silent fallback

When process isolation cannot be established, guarded work **fails**; it never
quietly runs in-process instead. The pipeline runner used to fall back to inline
execution when its process pool could not start. Under containment that would be
a silent revert to exactly the unsafe path containment exists to remove, so the
batch is aborted with a `PstrainWorkerError` instead.
