"""Checks for contract documentation generated from gate declarations."""

from pathlib import Path

from pstrain.lib.contract_docs import generate_bw_sharding_contract


def test_bw_sharding_contract_matches_gate_declarations() -> None:
    root = Path(__file__).parents[1]
    expected = (root / "docs/design/bw-sharding-contract.md").read_text()
    assert generate_bw_sharding_contract(root) == expected
