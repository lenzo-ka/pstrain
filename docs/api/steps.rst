Training Steps API
==================

.. automodule:: pstrain.api.steps
   :members:
   :exclude-members: run_build_lm, run_step_cd_hmm_untied, run_step_ci_hmm, run_step_features, step_cd_hmm_untied, step_ci_hmm, step_features
   :undoc-members:
   :show-inheritance:

The package root also exports these public convenience functions:

* ``pstrain.api.steps.step_features``: :func:`pstrain.api.step_features`
* ``pstrain.api.steps.run_step_features``: :func:`pstrain.api.run_step_features`
* ``pstrain.api.steps.step_ci_hmm``: :func:`pstrain.api.step_ci_hmm`
* ``pstrain.api.steps.run_step_ci_hmm``: :func:`pstrain.api.run_step_ci_hmm`
* ``pstrain.api.steps.step_cd_hmm_untied``: :func:`pstrain.api.step_cd_hmm_untied`
* ``pstrain.api.steps.run_step_cd_hmm_untied``: :func:`pstrain.api.run_step_cd_hmm_untied`
* ``pstrain.api.steps.run_build_lm``: :func:`pstrain.api.run_build_lm`

.. currentmodule:: pstrain.api.steps

.. py:data:: features_step

   Public feature-extraction step instance.

.. py:data:: ci_hmm_step

   Public context-independent HMM training step instance.

.. py:data:: cd_hmm_untied_step

   Public untied context-dependent HMM training step instance.
