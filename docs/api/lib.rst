Library API
===========

Core library functionality.

The ``pstrain.api`` module is the recommended public API. It re-exports all
functions from ``pstrain.lib`` and adds step functions for training workflows.

Public API
----------

.. automodule:: pstrain.api
   :members:
   :undoc-members:
   :show-inheritance:

Project Setup
-------------

.. automodule:: pstrain.lib.setup
   :members:
   :undoc-members:

Project Validation
------------------

.. automodule:: pstrain.lib.validate
   :members:
   :undoc-members:

Configuration
-------------

.. automodule:: pstrain.lib.config
   :members:
   :undoc-members:
   :no-index:

Data Structures
---------------

Dictionary
~~~~~~~~~~~

.. automodule:: pstrain.lib.dictionary
   :members:
   :undoc-members:

Phoneset
~~~~~~~~

.. automodule:: pstrain.lib.phoneset
   :members:
   :undoc-members:

Transcription
~~~~~~~~~~~~~

.. automodule:: pstrain.lib.transcription
   :members:
   :undoc-members:

Models
------

.. automodule:: pstrain.lib.model
   :members:
   :undoc-members:
   :show-inheritance:

Native Worker
-------------

Crash containment for the guarded native operations. See
:doc:`../design/native-boundary` for which operations are guarded, which are
not, and what each exception means.

.. automodule:: pstrain.lib.native_worker
   :members:
   :undoc-members:
   :show-inheritance:

Guarded Logmath Wrapper
-----------------------

``LogMath`` is the supported wrapper for native log-domain arithmetic. Other
contents of the private ``_pstrainc`` module, including raw library handles and
symbols, are implementation details and are not supported application APIs.

.. automodule:: pstrain.lib._pstrainc
   :members: LogMath
   :undoc-members:
   :show-inheritance:
