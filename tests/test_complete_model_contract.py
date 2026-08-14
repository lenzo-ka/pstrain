"""Executable contract tests for complete acoustic-model consumers."""

import ast
import json
import shutil
from pathlib import Path

import pytest

from pstrain.lib.model import COMPLETE_MODEL_FEAT_PARAMS_REQUIRED, require_complete_model
from pstrain.lib.pipeline.context import FeatParams
from pstrain.lib.pipeline.feat_params import feat_params_lines


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
        module_aliases = {
            alias.asname or alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
            if alias.name == "pocketsphinx"
        }
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module == "pocketsphinx"
                and any(alias.name == "Decoder" for alias in node.names)
            ) or (
                isinstance(node, ast.Attribute)
                and node.attr == "Decoder"
                and isinstance(node.value, ast.Name)
                and node.value.id in module_aliases
            ):
                violations.append(str(path.relative_to(root.parent)))

    assert violations == []


def test_required_field_artifact_matches_code_and_canonical_writer() -> None:
    artifact = Path(__file__).parent.parent / "complete-model-required-fields.json"
    inventory = json.loads(artifact.read_text())
    documented = {field["name"] for field in inventory["fields"]}
    emitted = {line.split(maxsplit=1)[0] for line in feat_params_lines(FeatParams())}

    assert inventory["required_count"] == len(inventory["fields"])
    assert inventory["optional_fields"] == []
    assert documented == COMPLETE_MODEL_FEAT_PARAMS_REQUIRED == emitted
