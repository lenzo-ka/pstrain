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


@pytest.mark.parametrize("invalid_field", sorted(COMPLETE_MODEL_FEAT_PARAMS_REQUIRED))
def test_complete_model_rejects_invalid_value_in_each_required_field(
    tmp_path: Path, invalid_field: str
) -> None:
    """Each of the 21 required values has a semantic or representation check."""
    model = tmp_path / "model"
    model.mkdir()
    feat_params = model / "feat.params"
    valid = dict(line.split(maxsplit=1) for line in feat_params_lines(FeatParams()))
    valid[invalid_field] = "invalid-value\n"
    feat_params.write_text("".join(f"{name} {value}" for name, value in valid.items()))

    with pytest.raises(ValueError, match=r"Invalid feat\.params field"):
        require_complete_model(model)


def _literal_decoder_constructions(tree: ast.AST) -> list[ast.AST]:
    """Return literal PocketSphinx Decoder imports/constructions in *tree*."""
    module_aliases = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
        if alias.name == "pocketsphinx"
    }
    importlib_aliases = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
        if alias.name == "importlib"
    }
    import_module_aliases = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "importlib"
        for alias in node.names
        if alias.name == "import_module"
    }
    return [
        node
        for node in ast.walk(tree)
        if (
            isinstance(node, ast.ImportFrom)
            and node.module == "pocketsphinx"
            and any(alias.name == "Decoder" for alias in node.names)
        )
        or (
            isinstance(node, ast.Attribute)
            and node.attr == "Decoder"
            and isinstance(node.value, ast.Name)
            and node.value.id in module_aliases
        )
        or (
            isinstance(node, ast.Attribute)
            and node.attr == "Decoder"
            and isinstance(node.value, ast.Call)
            and len(node.value.args) >= 1
            and isinstance(node.value.args[0], ast.Constant)
            and node.value.args[0].value == "pocketsphinx"
            and (
                (
                    isinstance(node.value.func, ast.Attribute)
                    and node.value.func.attr == "import_module"
                    and isinstance(node.value.func.value, ast.Name)
                    and node.value.func.value.id in importlib_aliases
                )
                or (
                    isinstance(node.value.func, ast.Name)
                    and node.value.func.id in import_module_aliases
                )
            )
        )
    ]


def test_literal_importlib_decoder_construction_is_detected() -> None:
    construction = ast.parse(
        'import importlib\nimportlib.import_module("pocketsphinx").Decoder()\n'
    )

    assert _literal_decoder_constructions(construction)


def test_production_code_has_no_literal_pocketsphinx_decoder_construction() -> None:
    """Literal imports must not bypass pstrain's complete-model boundary.

    This source gate is silent on dynamically computed module and attribute names.
    """
    root = Path(__file__).parent.parent / "pstrain"
    violations: list[str] = []
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        if _literal_decoder_constructions(tree):
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
