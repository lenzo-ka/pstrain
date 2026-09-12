Checkpoint recovery API
=======================

Inspect saved updates and explicitly restore a selected checkpoint. Update N is
evaluated by pass N+1; the final update normally remains unevaluated. Evaluation
health describes configured skip limits, nonzero processed counts and finite
likelihood statistics, not recognition quality. Legacy telemetry without matching hashes is unverified.

Restoration never selects a checkpoint automatically. Stop concurrent training
and readers first. The previous model is backed up under ``recovery-history``;
completion markers, provenance and cached ``sendump`` are invalidated before the
selected parameters and counts are published. Training diagnostics and original
checkpoints remain available. ``dry_run=True`` performs inspection only.

.. automodule:: pstrain.api.checkpoints
   :members:
   :undoc-members:
   :show-inheritance:
