"""Checks for contract documentation generated from gate declarations."""

import difflib
from pathlib import Path

import pytest

from pstrain.lib.contract_docs import (
    _declarations,
    contract_check_fields,
    contract_check_files,
    generate_bw_sharding_contract,
)


def test_bw_sharding_contract_matches_gate_declarations() -> None:
    root = Path(__file__).parents[1]
    expected = (root / "docs/design/bw-sharding-contract.md").read_text()
    actual = generate_bw_sharding_contract(root)
    diff = "".join(
        difflib.unified_diff(
            expected.splitlines(True),
            actual.splitlines(True),
            fromfile="checked-in contract",
            tofile="generated contract",
        )
    )
    if actual != expected:
        pytest.fail(f"contract document differs from generated gate scope:\n{diff}", pytrace=False)


def test_checked_file_helper_negative_control(tmp_path: Path) -> None:
    left, right = tmp_path / "left", tmp_path / "right"
    left.mkdir()
    right.mkdir()
    (left / "artifact").write_bytes(b"left")
    (right / "artifact").write_bytes(b"right")
    with pytest.raises(AssertionError, match="artifact"):
        contract_check_files(left=left, right=right, artifacts=("artifact",), scope=1)


def test_checked_field_helper_negative_control() -> None:
    with pytest.raises(AssertionError, match="field"):
        contract_check_fields(left={"field": 1}, right={"field": 2}, artifacts=("field",), scope=1)


def test_checked_cannot_be_declared_without_an_assertion_helper(tmp_path: Path) -> None:
    gate = tmp_path / "gate.py"
    gate.write_text(
        "@contract_scope(order=1, kind='example', checked_files=('ghost',))\n"
        "def test_gate():\n"
        "    pass\n"
    )
    with pytest.raises(ValueError, match="CHECKED cannot be declared"):
        _declarations(tmp_path, (gate,))
