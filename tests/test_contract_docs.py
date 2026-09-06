"""Checks for contract documentation generated from gate declarations."""

import difflib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from pstrain.lib.contract_docs import (
    _check_pinned_resource_rows,
    _declarations,
    contract_check_fields,
    contract_check_files,
    generate_arctic_pin_document,
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


def test_arctic_pin_document_matches_its_evidence() -> None:
    root = Path(__file__).parents[1]
    expected = (root / "docs/benchmarks/arctic-pin.md").read_text()
    assert generate_arctic_pin_document(root) == expected


@pytest.mark.parametrize("artifact", ["record.json", "oracle-sidecar.json", "paired-analysis.json"])
def test_arctic_pin_document_refuses_modified_evidence(tmp_path: Path, artifact: str) -> None:
    source = Path(__file__).parents[1]
    root = tmp_path / "repo"
    evidence = root / "evidence/arctic-pin"
    docs = root / "docs/benchmarks"
    evidence.mkdir(parents=True)
    docs.mkdir(parents=True)
    for name in ("record.json", "oracle-sidecar.json", "paired-analysis.json"):
        shutil.copyfile(source / "evidence/arctic-pin" / name, evidence / name)
    shutil.copyfile(source / "docs/benchmarks/arctic-pin.md", docs / "arctic-pin.md")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "evidence", "docs"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "baseline evidence",
        ],
        check=True,
    )
    path = evidence / artifact
    path.write_bytes(path.read_bytes() + b"\n")

    with pytest.raises(RuntimeError, match=f"baseline .*{artifact} has uncommitted modifications"):
        generate_arctic_pin_document(root)


def test_arctic_pin_document_uncommitted_override_is_visible(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = Path(__file__).parents[1]

    generate_arctic_pin_document(root, allow_uncommitted_baseline=True)

    captured = capsys.readouterr()
    assert captured.err.count("WARNING: --allow-uncommitted-baseline used") == 3


def test_arctic_pin_document_refuses_an_unstated_resource_digest() -> None:
    root = Path(__file__).parents[1]
    record = json.loads((root / "evidence/arctic-pin/record.json").read_text())
    record["resources"]["dictionary_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="decode dictionary digest"):
        _check_pinned_resource_rows((root / "docs/benchmarks/arctic-pin.md").read_text(), record)


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
