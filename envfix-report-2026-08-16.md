# Ambient-import gate fixture correction — 2026-08-16

## Outcome

- The checkout-local acceptance case now runs with the test's isolated virtual-environment
  interpreter instead of the ambient interpreter that launched pytest.
- The isolated interpreter does not inherit system site-packages, and its explicit `PYTHONPATH`
  contains the checkout root. Consequently, `pstrain.__file__` and every `pstrain.__path__` entry
  resolve within the checkout in the acceptance case.
- The production ambient-import gate and all rejection cases are unchanged. A `pstrain` package or
  package-path entry that resolves through site-packages outside the checkout remains an error.
- The Python 3.11 installed-package job now skips the in-tree ambient prerequisite when it runs
  `config-check`. The config checks still run, while the expected editable installation is not
  incorrectly treated as an in-tree verification environment.

## Validation

- `make verified`: passed. Native C tests passed 6/6; the strict Python suite passed 684
  tests with one expected skip and one deselection; the focused configuration suite passed 41/41;
  and the ambient-import, floating-point contract, generated documentation, Arctic pin, paired
  analysis, Ruff, formatting, and mypy checks passed.
- PR #115 Python 3.13 passed on commit `06181da`. Python 3.11 passed its test suite, then its
  installed-package `config-check` invocation correctly rejected the job's editable installation;
  verification of the CI-only prerequisite correction is pending.
