Configuration API
=================

.. automodule:: pstrain.api.config
   :members:
   :undoc-members:
   :show-inheritance:

.. currentmodule:: pstrain.api.config

.. py:data:: CURRENT_CONFIG_VERSION

   Current version number for the canonical configuration schema.

User settings
-------------

``~/.pstrain/config.yaml`` holds both semantic overrides and presentation
preferences. ``PSTRAIN_USER_CONFIG`` selects an alternate file for both readers.
For example:

.. code-block:: yaml

   config_version: 1
   features:
     alpha: 0.9
   training:
     tied:
       max_iterations: 7
   json_output:
     indent: 4

``get_user_config().save()`` preserves the semantic blocks while saving changed
preferences. It validates the document and replaces the file atomically;
omitted semantic blocks remain omitted, so saving preferences does not freeze
unrelated defaults. Project and experiment overlays accept semantic settings
only. Unknown fields in a versioned user document are errors.

The legacy ``defaults`` block remains readable for compatibility but is not a
training override. Use ``features``, ``training``, ``split``, ``runner``,
``sharding``, and ``alignment`` for effective settings. Saving a legacy document
writes version 1 and preserves the legacy fields that the resolver actually
consumed, such as ``features.num_ceps`` as ``features.ncep``.

Exclusion schedules accept positive pass numbers or ``"*"``. Numeric strings
are normalized to integer pass numbers; aliases for the same pass in one
schedule are rejected rather than silently overwriting an exclusion list.
