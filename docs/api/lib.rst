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

Low-level C Bindings
--------------------

For advanced users who need direct access to C functions:

.. automodule:: pstrain.lib._pstrainc
   :members:
   :undoc-members:
   :show-inheritance:
