# Support and dependency policy

## Platforms

pstrain supports macOS, Linux, and Windows. CI builds wheels on all three
platforms, and release builds publish those wheels to PyPI.

Windows support does not extend to training. Running any stage of the training
pipeline on Windows raises `RuntimeError: pipeline provenance locking requires
the POSIX fcntl module; pipeline execution is unavailable on this platform`,
because provenance replacement locks its critical section with `fcntl.flock`
and Windows has no `fcntl`. What does work there is the package itself, the
native library and command-line programs, and the native worker that carries
requests to them. Training a model requires macOS or Linux.

On Windows, CI builds the native library and command-line programs with both
MSVC and clang-cl, then runs a Windows-compatible subset of the Python tests
against an MSVC build on Python 3.13. The full Python test suite is run on
macOS and Linux; Windows CI does not currently claim that broader coverage.

### Windows Python coverage

This has been measured. The Windows Python 3.13 job requested 146 test units
and reported 101 passed, 45 skipped. Those 45 are the 39 cases and the six
module-level collection skips below, and 101 plus 45 is the job's 146, so the
breakdown here and the CI summary line describe the same run.

Six POSIX resource-accounting modules skip during collection, leaving 140
collected items. Of those 140, 39 skip and 101 pass:

- Six whole modules skip because POSIX training resource accounting uses
  `resource`: `test_bw_sharding.py`, `test_dtree.py`, `test_e2e_training.py`,
  `test_feat_params.py`, `test_train_convergence.py`, and
  `test_train_retry.py`.
- 39 cases skip because pipeline provenance replacement locks its critical
  section with `fcntl.flock`, which Windows does not have. In
  `tests/test_numeric_harness.py` that is every case built on the
  `flat_project` or `full_project` fixtures, since creating either one runs a
  pipeline (30 cases). In `tests/test_pipeline_tasks.py` it is
  `test_meaningful_feature_config_change_rebuilds_features`,
  `test_reverting_feature_config_rebuilds_features`,
  `test_irrelevant_config_edit_does_not_rebuild_features`,
  `test_model_and_package_copy_build_provenance`,
  `test_split_runs_end_to_end_and_partitions`,
  `test_lm_target_succeeds_on_setup_project_layout`,
  `test_editing_persistent_split_revalidates_and_changes_membership`,
  `test_configured_bw_parameters_reach_training_call`, and
  `test_configured_untied_schedule_and_variance_reach_training_call`.
- The remaining 101 cases pass: everything selected from `tests/test_paths.py`,
  `tests/test_commands.py`, `tests/test_cffi_abi.py`,
  `tests/test_lib_structure.py`, `tests/test_pstrainc.py` and
  `tests/test_features.py`; all of `tests/test_pipeline_tasks.py` outside the
  nine provenance-lock cases above; and the four cases in
  `tests/test_numeric_harness.py` that do not build a pipeline fixture.

No case is skipped any longer for native-worker request transport, but that
buys less than the count suggests, so it is worth stating what it does buy.
Sixty-nine cases carried the transport skip. Thirty-nine of them now skip for
the provenance lock instead, and thirty run. Twenty-seven of those thirty never
reach the native worker at all: they build and inspect pipeline plans, or check
provenance and fingerprint payloads (the one that executes a task,
`test_linear_default_untied_stage_builds_occurrence_inventory`, reaches the C
library in this process rather than through the worker). The three that
genuinely drive the worker are the
`test_m4_real_utterances_reach_shared_final_state` cases, whose per-utterance
feature payload is larger than the pipe buffer and so takes the pending-write
path rather than completing at once. One further case,
`tests/test_native_worker.py::test_windows_write_request_gives_up_at_the_deadline_on_a_full_pipe`,
has since been added to the Windows selection to cover the request deadline
directly; the job now requests 147 units, and that case is not part of the
measurement above.

The transport skip existed because the transport did not work on Windows at
all: the spawned child reported ready, and the parent then failed before
sending its first request with `OSError: [Errno 9] Bad file descriptor`, raised
by `os.set_blocking()` on the connection handle. On Windows a
`multiprocessing` duplex pipe is an overlapped, message-mode named pipe and
`Connection.fileno()` returns a Win32 `HANDLE`, which `os.set_blocking`,
`select.select` and `os.write` all reject: they take a C-runtime file
descriptor or a socket. The request write is now expressed in the terms each
platform actually offers -- a non-blocking descriptor and `select` on POSIX, a
bounded wait on an overlapped completion event on Windows -- and the
deadline that a full pipe must not outlive is preserved on both.

What that fix moved is where Windows stops, not whether it stops. The wall is
now `fcntl`, one layer past the transport, and no amount of further work on
the request pipe reaches it.

## Dependencies

Runtime dependencies declare tested minimum versions and may float within their
compatible major releases; lock files are intentionally not used for library
consumers. Development, documentation, and CI tools are constrained in
`pyproject.toml`, while GitHub Actions and pre-commit hook revisions are pinned
in their workflow files and advanced through reviewed dependency updates.
Security and compatibility fixes may raise a minimum version; unnecessary
runtime dependencies should not be added.
