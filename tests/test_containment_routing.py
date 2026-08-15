"""Construction-based gate for the Python-to-CFFI routing boundary."""

from __future__ import annotations

import ast

from scripts.check_containment_routing import CFFI_FUNCTION_NAMES, Scanner, scan


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


def test_statically_resolvable_indirect_callees_are_detected() -> None:
    source = """
CALLEE = "pstrain_bw_free"
getattr(lib, "pstrain_" + "bw_free")(ctx)
getattr(lib, CALLEE)(ctx)
{"free": lib.pstrain_bw_free}["free"](ctx)
"""
    scanner = Scanner("pstrain/_construction.py")
    scanner.visit(ast.parse(source))

    assert [(item.symbol, item.disposition) for item in scanner.callsites] == [
        ("lib.pstrain_bw_free", "violation"),
        ("lib.pstrain_bw_free", "violation"),
        ("lib.pstrain_bw_free", "violation"),
    ]


def test_native_leaf_population_is_derived_from_cffi_not_a_name_prefix() -> None:
    assert {"s3mixw_read", "semi_ts2cb", "cont_ts2cb"} <= CFFI_FUNCTION_NAMES

    source = """
lib.s3mixw_read(path, out, n_mixw, n_feat, n_density)
lib.semi_ts2cb(n_tied_state)
lib.cont_ts2cb(n_tied_state)
"""
    scanner = Scanner("pstrain/_construction.py")
    scanner.visit(ast.parse(source))

    assert [(item.symbol, item.disposition) for item in scanner.callsites] == [
        ("lib.s3mixw_read", "violation"),
        ("lib.semi_ts2cb", "violation"),
        ("lib.cont_ts2cb", "violation"),
    ]


def test_unresolvable_indirect_callee_is_an_explicitly_silent_axis() -> None:
    scanner = Scanner("pstrain/_construction.py")
    scanner.visit(ast.parse("getattr(lib, runtime_name)(ctx)"))
    assert scanner.callsites == []


def test_proxy_branch_must_forward_before_native_fallback_is_worker_only() -> None:
    source = """
class Escape:
    def close(self):
        if hasattr(self, "_proxy"):
            return None
        return self._lib.pstrain_bw_free(self._ctx)
"""
    scanner = Scanner("pstrain/_construction.py")
    scanner.visit(ast.parse(source))
    assert [(item.symbol, item.disposition) for item in scanner.callsites] == [
        ("self._lib.pstrain_bw_free", "violation")
    ]


def test_nested_function_does_not_inherit_worker_or_proxy_depth() -> None:
    source = """
class Escape:
    def method(self):
        if hasattr(self, "_proxy"):
            return self._proxy.call("method")
        def thunk():
            return self._lib.pstrain_bw_free(self._ctx)
        return thunk
"""
    scanner = Scanner("pstrain/_construction.py")
    scanner.visit(ast.parse(source))
    assert [(item.symbol, item.disposition) for item in scanner.callsites] == [
        ("self._lib.pstrain_bw_free", "violation"),
    ]
