CLI API
=======

Command-line interface.

JSON output
-----------

``--json`` produces a single machine-readable result on standard output for
``train``, ``tutorial``, ``info``, ``config explain``, ``config profiles``,
``config show``, ``config get``, ``config schema``, and ``config list``. The
formatting controls ``--json-indent`` and ``--json-ascii`` apply to these
commands. Human-readable progress that accompanies a JSON result is written to
standard error.

The other commands do not yet have a stable machine-readable result contract:
``setup``, ``build``, ``split``, ``features``, ``flat``, ``clean``,
``validate-project``, ``test``, ``package``, ``align``, ``compare``,
``timings``, ``config migrate``, ``config path``, ``step ci_hmm``, and
``step cd_hmm_untied``. They reject ``--json`` with a nonzero exit status
instead of silently producing human output. Place the global flag before the
command name when checking support; commands that support JSON also advertise
it in their command help.

.. automodule:: pstrain.cli
   :members:
   :undoc-members:
   :show-inheritance:
