# Package replacement safety

`package_model()` protects a caller from accidentally naming the wrong output
directory. It replaces only a recognizable pstrain package, requires explicit
consent for a recognizable pre-marker package, never accepts an invalid or
unsupported marker, restricts a package name to one path component, and refuses
known source/destination overlap. Those checks apply through the public Python API
as well as the command line.

That mistake-protection guarantee applies on every supported platform. It assumes
that another process is not actively changing the same directory entries while a
package operation is running.

## Concurrent pathname changes

Protection against a process deliberately or accidentally racing the transaction
is a stronger, platform-dependent guarantee.

On macOS and Linux, pstrain anchors the destination parent, source model, staging
directory, retained package, and recovery directory with open descriptors. It
derives identity from `fstat()` before moving an object, uses descriptor-relative
exclusive renames, validates the retained descriptor rather than reopening its
backup name, and performs recursive traversal relative to descriptors. After every
rename return, including an exceptional return, it compares the open identity with
both directory entries before deciding what moved. Recovery repeats that
reconciliation rather than trusting in-memory transition bookkeeping. An outer
pathname substitution therefore cannot redirect validation or recursive cleanup
to the substituted tree.

Windows does not expose the descriptor-relative exclusive directory rename used by
that transaction through Python or the system interface used here. Packaging keeps
the same ownership checks and atomic no-replace publication there, but its retain,
validation, recovery, and cleanup steps remain path-based. A process that changes
those names between system calls can redirect an operation on Windows. Stop writers
and package from a directory that other processes cannot modify when replacement
must be protected from concurrent interference.

Windows also cannot reconcile an asynchronous interruption that arrives after an
unnamed-package rename has completed but before Python records its result. The
public output can then contain new entries already published, such as `acoustic`,
beside old entries not yet replaced, such as `README.txt`, `dict`, and
`pstrain-package.json`. Old entries already retained may remain in a sibling
`.<output-name>-old-*` directory, and a later recovery error can obscure the
original interruption. Stop writers, move the mixed public output aside without
deleting it, and inspect that recovery directory. Restore its retained entries to
reconstruct the old package, or publish again into an empty controlled output and
move unrelated entries back afterward. Do not infer from the old public marker that
the mixed tree is internally consistent.

Even on macOS and Linux, the system interfaces do not provide a portable operation
meaning “unlink the directory entry only if it still names this open descriptor.”
Cleanup uses `unlinkat()` or directory-relative `rmdir()` against an open parent and
checks the entry identity immediately beforehand. Another process with write access
to that same open directory can still substitute the final component between the
check and the removal call. A non-empty substituted directory normally makes
`rmdir()` fail, but a substituted file, symlink, or empty directory can be removed.
The same mkdir-to-open interval exists when creating a private staging or recovery
directory: an active process able to discover and replace its unpredictable,
mode-0700 name in that interval can cause pstrain to adopt the replacement.

Package structure and marker reads are descriptor-relative, but open descriptors
do not lock a directory's contents. A concurrent writer can still change children
inside the retained package while validation or cleanup walks them. pstrain fails
closed when it observes an identity mismatch or cannot reconcile an operation, but
it cannot guarantee detection of every change made and reversed entirely between
two system calls.

For an operation exposed to untrusted or concurrent writers, place the output in a
directory writable only by the packaging process, stop those writers for the
duration, or publish into a separately controlled directory and hand off the final
package after `package_model()` returns. Recovery directories named
`.<package>-old-*` or `.pstrain-package-old-*` are intentionally retained whenever
the transaction cannot prove that cleanup or restoration is safe; inspect them
before removing them.

Cleanup occurs after publication. If cleanup cannot prove that a recovery object is
still the one it owns, `package_model()` raises even though the new package may
already be fully published at the requested destination. The recovery directory is
preserved when it may still contain data. A caller must inspect both the destination
and the reported recovery directory after such an exception; treating every
exception as “nothing was published” is incorrect.
