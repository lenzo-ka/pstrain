# Support and dependency policy

## Platforms

pstrain supports macOS, Linux, and Windows. CI builds wheels on all three
platforms, and release builds publish those wheels to PyPI. On Windows, CI
builds the native library and command-line programs with both MSVC and clang-cl,
then runs a Windows-compatible subset of the Python tests against an MSVC build
on Python 3.13. The full Python test suite is run on macOS and Linux; Windows CI
does not currently claim that broader coverage.

### Measured Windows Python coverage

The Windows Python 3.13 job explicitly requests 146 test cases. Six POSIX
resource-accounting modules skip during collection, leaving 140 collected
items. The measured result is 71 passing cases and 75 honest skips:

- 41 passing cases cover `tests/test_paths.py`, the Windows executable lookup
  case in `tests/test_commands.py`, `tests/test_cffi_abi.py`,
  `tests/test_lib_structure.py`, the selected load/logmath/feature cases in
  `tests/test_pstrainc.py`, and the selected parameter, MFC, and WAV cases in
  `tests/test_features.py`.
- `tests/test_numeric_harness.py` runs and passes
  `test_bw_discrete_contract_negative_control_rejects_dropped_identity`.
- `tests/test_pipeline_tasks.py` runs and passes the target-registry default;
  native-library identity and fingerprint cases; recursive audio-fileid case;
  named, unknown, and shipped-profile configuration cases; multipron,
  final-silence, failed-alignment, untied-inventory, and transcript-inventory
  configuration cases; BW default/floor/schedule/variance-policy cases; and
  `test_bw_config_requires_explicit_normalization_policies` (29 cases total).
- Six whole modules remain skipped because POSIX training resource accounting
  uses `resource`: `test_bw_sharding.py`, `test_dtree.py`,
  `test_e2e_training.py`, `test_feat_params.py`,
  `test_train_convergence.py`, and `test_train_retry.py`.
- No Windows case is currently skipped for POSIX provenance locking. The
  provenance-lock markers remain accurate on a non-Windows platform without
  `fcntl`, but on Windows native-worker transport is encountered first.
- 69 cases skip because native-worker request transport is unavailable on
  Windows. In `tests/test_numeric_harness.py` that is every case except the
  discrete-contract negative control (33 cases). In
  `tests/test_pipeline_tasks.py` it is the following 36 cases:
  `test_pipeline_builds_without_error`,
  `test_all_registered_targets_have_producers`,
  `test_every_target_in_TARGETS_is_registered`,
  `test_can_plan_each_ci_and_cd_target`,
  `test_cd_8g_plan_includes_full_chain`,
  `test_fanout_tasks_share_parallel_group`,
  `test_extract_task_forwards_lifter`,
  `test_extract_task_forwards_every_recorded_waveform_field`,
  `test_extract_task_forwards_preemphasis_alpha`,
  `test_meaningful_feature_config_change_rebuilds_features`,
  `test_reverting_feature_config_rebuilds_features`,
  `test_irrelevant_config_edit_does_not_rebuild_features`,
  `test_model_and_package_copy_build_provenance`,
  `test_stage_fingerprints_cover_only_effective_relevant_values`,
  `test_skip_state_change_invalidates_training_fingerprint`,
  `test_project_sharding_policy_changes_training_provenance`,
  `test_training_fingerprint_excludes_config_source_metadata`,
  `test_training_fingerprint_payload_composition_is_pinned`, both parameter
  cases of
  `test_training_provenance_declares_requested_and_effective_bw_shard_count`,
  `test_exclusion_schedule_config_and_provenance_are_verbatim`,
  `test_exclusion_schedule_does_not_change_decode_eval_inputs`,
  `test_provenance_rejects_non_finite_config`,
  `test_nested_and_flat_audio_fanout_uses_relative_fileids`,
  `test_empty_audio_directory_fails_during_pipeline_construction`,
  `test_missing_audio_directory_fails_during_pipeline_construction`,
  `test_split_task_produces_fileid_files`,
  `test_split_runs_end_to_end_and_partitions`,
  `test_lm_target_succeeds_on_setup_project_layout`,
  `test_editing_persistent_split_revalidates_and_changes_membership`,
  `test_tree_building_is_fanned_out`,
  `test_model_tasks_depend_on_split_outputs`,
  `test_resolved_untied_inventory_appears_in_provenance`,
  `test_linear_default_untied_stage_builds_occurrence_inventory`,
  `test_configured_bw_parameters_reach_training_call`, and
  `test_configured_untied_schedule_and_variance_reach_training_call`.

The observed native-worker failure is in request transport, not process spawn
or child start-up. The spawned child reports ready, then the parent gets
`OSError: [Errno 9] Bad file descriptor` while applying `os.set_blocking()` to
the Windows pipe handle before sending the first request. The retry does the
same. No child diagnostic is produced because the failure occurs in the parent.
The retry path then discards the worker and its diagnostic file before calling
`_raise_death()`, whose bare `assert self._process is not None` masks the
original transport error. That assertion is also a known diagnostic defect:
under `python -O` it disappears and execution continues past an already
discarded process. Fixing Windows native-worker transport and that production
assertion are separate work from the support measurement recorded here.

## Dependencies

Runtime dependencies declare tested minimum versions and may float within their
compatible major releases; lock files are intentionally not used for library
consumers. Development, documentation, and CI tools are constrained in
`pyproject.toml`, while GitHub Actions and pre-commit hook revisions are pinned
in their workflow files and advanced through reviewed dependency updates.
Security and compatibility fixes may raise a minimum version; unnecessary
runtime dependencies should not be added.
