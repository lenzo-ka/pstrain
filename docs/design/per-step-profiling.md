# Per-step profiling

Pipeline builds persist one JSON document per run under
`.pstrain/timings/<run-id>.json`. Each task record contains its name, inferred
stage, parallel group, UTC start/end timestamps, wall time, and the four CPU
counters returned by `os.times()`. The stage rollup sums task wall time and CPU
time; wall therefore represents aggregate task occupancy (including parallel
work), while CPU/wall makes serialized and I/O-bound stages visible.

Measurements happen around the task function in the process that executes it.
Pool workers return their record with the task result and the parent merges and
writes the run document. There are no worker-side shared files.

CPU total is worker self user/system plus reaped-child user/system. A spawned
child that exits during a task is attributed to that task. The native helper is
persistent and handles requests synchronously, but operating systems generally
do not add an unreaped, still-running child's CPU to `os.times()` child counters.
Its per-request attribution is consequently best-effort and may remain absent
until the helper is reaped; the artifact reports the measured counters without
claiming stronger attribution.

`pstrain timings [RUN_ID] --project-dir PROJECT` prints the latest run by
default or a named run. A successful build prints the same stage table when it
takes more than three seconds, and always when `pstrain build -v` is used.
Timing persistence is atomic and observational: failures only log a warning and
never change a task or build result. Persistence is a single end-of-run write,
so a hard process crash leaves no artifact. A subsequent write removes stale
temporary timing files left by an interrupted write.
