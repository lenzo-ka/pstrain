# Support and dependency policy

## Platforms

pstrain supports macOS, Linux, and Windows. CI builds wheels on all three
platforms, and release builds publish those wheels to PyPI. On Windows, CI
builds the native library and command-line programs with both MSVC and clang-cl,
then runs a Windows-compatible subset of the Python tests against an MSVC build
on Python 3.13. The full Python test suite is run on macOS and Linux; Windows CI
does not currently claim that broader coverage.

### Windows Python coverage

The Windows Python 3.13 job explicitly requests 146 test units. Six POSIX
resource-accounting modules skip during collection, leaving 140 collected
items. Of those 140, 39 skip and 101 run:

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
- The remaining 101 cases run: everything selected from `tests/test_paths.py`,
  `tests/test_commands.py`, `tests/test_cffi_abi.py`,
  `tests/test_lib_structure.py`, `tests/test_pstrainc.py` and
  `tests/test_features.py`; all of `tests/test_pipeline_tasks.py` outside the
  nine provenance-lock cases above; and the four cases in
  `tests/test_numeric_harness.py` that do not build a pipeline fixture.

No case is skipped any longer for native-worker request transport. That skip
existed because the transport did not work on Windows at all: the spawned
child reported ready, and the parent then failed before sending its first
request with `OSError: [Errno 9] Bad file descriptor`, raised by
`os.set_blocking()` on the connection handle. On Windows a
`multiprocessing` duplex pipe is an overlapped, message-mode named pipe and
`Connection.fileno()` returns a Win32 `HANDLE`, which `os.set_blocking`,
`select.select` and `os.write` all reject: they take a C-runtime file
descriptor or a socket. The request write is now expressed in the terms each
platform actually offers -- a non-blocking descriptor and `select` on POSIX, a
bounded wait on an overlapped completion event on Windows -- and the
deadline that a full pipe must not outlive is preserved on both.

The skip accounting above is exact: it follows from the skip conditions in the
test files, which do not depend on the platform that computes them. The pass
and fail split behind the 101 running cases has not been measured on Windows
and is not claimed here; the Windows CI leg is what establishes it. This
section should be rewritten from that job's output once it has run.

## Dependencies

Runtime dependencies declare tested minimum versions and may float within their
compatible major releases; lock files are intentionally not used for library
consumers. Development, documentation, and CI tools are constrained in
`pyproject.toml`, while GitHub Actions and pre-commit hook revisions are pinned
in their workflow files and advanced through reviewed dependency updates.
Security and compatibility fixes may raise a minimum version; unnecessary
runtime dependencies should not be added.
