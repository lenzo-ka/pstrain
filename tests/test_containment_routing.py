"""Construction-based gate for the Python-to-CFFI routing boundary."""

from __future__ import annotations

import ast

from scripts.check_containment_routing import Scanner, scan


def test_every_literal_cffi_callsite_is_contained_or_declared() -> None:
    violations = [item for item in scan() if item.disposition == "violation"]
    assert not violations, "\n" + "\n".join(
        f"{item.path}:{item.line}:{item.column + 1}: {item.symbol}" for item in violations
    )


def test_proxy_mention_does_not_exempt_an_uncontained_callsite() -> None:
    source = """
class Escape:
    def marker(self):
        return NativeObjectProxy("unused")

    def direct_native(self):
        return self._lib.pstrain_bw_free(self._ctx)
"""
    scanner = Scanner("pstrain/_construction.py")
    scanner.visit(ast.parse(source))

    assert [(item.symbol, item.disposition) for item in scanner.callsites] == [
        ("self._lib.pstrain_bw_free", "violation")
    ]


def test_dynamic_callees_are_an_explicitly_silent_axis() -> None:
    source = """
getattr(_pstrainc, "_" + "init")()
getattr(lib, "pstrain_" + "bw_free")(None)
"""
    scanner = Scanner("pstrain/_construction.py")
    scanner.visit(ast.parse(source))

    assert scanner.callsites == []
