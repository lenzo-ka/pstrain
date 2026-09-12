Configuration API
=================

.. automodule:: pstrain.api.config
   :members:
   :undoc-members:
   :show-inheritance:

.. currentmodule:: pstrain.api.config

.. py:data:: CURRENT_CONFIG_VERSION

   Current version number for the canonical configuration schema.

Exclusion schedules accept positive pass numbers or ``"*"``. Numeric strings
are normalized to integer pass numbers; aliases for the same pass in one
schedule are rejected rather than silently overwriting an exclusion list.
