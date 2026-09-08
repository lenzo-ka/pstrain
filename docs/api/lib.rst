Library API
===========

Core library functionality.

The ``pstrain.api`` module is the recommended public API. It re-exports most of
``pstrain.lib`` and adds higher-level entry points; the modules are not
interchangeable.

Public API
----------

.. automodule:: pstrain.api
   :members:
   :undoc-members:
   :show-inheritance:

.. currentmodule:: pstrain.api

.. py:data:: TUTORIAL_FILENAME

   The filename used for the bundled tutorial notebook.

Project Setup
-------------

.. automodule:: pstrain.lib.setup
   :members:
   :undoc-members:
   :no-index:

Project Validation
------------------

.. automodule:: pstrain.lib.validate
   :members:
   :undoc-members:
   :no-index:

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
   :no-index:

Phoneset
~~~~~~~~

.. automodule:: pstrain.lib.phoneset
   :members:
   :undoc-members:
   :no-index:

Transcription
~~~~~~~~~~~~~

.. automodule:: pstrain.lib.transcription
   :members:
   :undoc-members:
   :no-index:

Models
------

.. automodule:: pstrain.lib.model
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Native Worker
-------------

Crash containment for the guarded native operations. See
:doc:`../design/native-boundary` for which operations are guarded, which are
not, and what each exception means.

.. automodule:: pstrain.lib.native_worker
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Guarded Logmath Wrapper
-----------------------

``LogMath`` is the supported wrapper for native log-domain arithmetic. Other
contents of the private ``_pstrainc`` module, including raw library handles and
symbols, are implementation details and are not supported application APIs.

.. automodule:: pstrain.lib._pstrainc
   :members: LogMath
   :undoc-members:
   :show-inheritance:
   :no-index:
