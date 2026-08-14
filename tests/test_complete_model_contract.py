"""Executable contract tests for complete acoustic-model consumers."""

import ast
import shutil
from pathlib import Path

import pytest

from pstrain.lib.model import require_complete_model


def test_decoder_rejects_feat_params_missing_front_end_field(tmp_path: Path) -> None:
    """An existing file is incomplete when any training front-end field is absent."""
    fixture = Path(__file__).parent / "fixtures" / "multipron_final_state"
    model = tmp_path / "model"
    shutil.copytree(fixture / "model", model)
    feat_params = model / "feat.params"
    feat_params.write_text(
        "\n".join(
            line for line in feat_params.read_text().splitlines() if not line.startswith("-nfilt ")
        )
        + "\n"
    )

    with pytest.raises(ValueError, match=r"feat\.params.*missing required.*-nfilt"):
        require_complete_model(model)


def test_production_code_does_not_construct_pocketsphinx_directly() -> None:
    """New in-tree consumers must not bypass pstrain's complete-model boundary."""
    root = Path(__file__).parent.parent / "pstrain"
    violations: list[str] = []
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Import)
                and any(
                    alias.name == "pocketsphinx" and (alias.asname or alias.name) == "Decoder"
                    for alias in node.names
                )
                or (
                    isinstance(node, ast.ImportFrom)
                    and node.module == "pocketsphinx"
                    and any(alias.name == "Decoder" for alias in node.names)
                )
            ):
                violations.append(str(path.relative_to(root.parent)))

    assert violations == []
