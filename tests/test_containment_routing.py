"""Construction-based gate for the Python-to-CFFI routing boundary."""

from __future__ import annotations

from scripts.check_containment_routing import scan


def test_every_cffi_callsite_is_contained_proxied_or_declared() -> None:
    violations = [item for item in scan() if item.disposition == "violation"]
    assert not violations, "\n" + "\n".join(
        f"{item.path}:{item.line}:{item.column + 1}: {item.symbol}" for item in violations
    )
