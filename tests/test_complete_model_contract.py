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


@pytest.mark.parametrize(
    "invalid_field", sorted(COMPLETE_MODEL_FEAT_PARAMS_REQUIRED - {"-alpha", "-cmninit"})
)
def test_complete_model_rejects_invalid_value_in_each_required_field(
    tmp_path: Path, invalid_field: str
) -> None:
    """Fields native rejects syntactically retain a construction-based check."""
    model = tmp_path / "model"
    model.mkdir()
    feat_params = model / "feat.params"
    valid = dict(line.rstrip("\n").split(maxsplit=1) for line in feat_params_lines(FeatParams()))
    valid[invalid_field] = "invalid-value"
    feat_params.write_text("".join(f"{name} {value}\n" for name, value in valid.items()))

    with pytest.raises(ValueError, match=r"Invalid feat\.params field"):
        require_complete_model(model)


def _write_feat_params(model: Path, **updates: str) -> None:
    valid = dict(line.rstrip("\n").split(maxsplit=1) for line in feat_params_lines(FeatParams()))
    valid.update(updates)
    (model / "feat.params").write_text(
        "".join(f"{name} {value}\n" for name, value in valid.items())
    )


@pytest.mark.parametrize(
    "updates",
    [
        {"-nfft": "513"},
        {"-samprate": "16000", "-frate": "12000"},
        {"-samprate": "16000", "-frate": "16001"},
        {"-samprate": "16000", "-frate": "100", "-wlen": "0.005"},
    ],
)
def test_complete_model_rejects_native_front_end_failures(
    tmp_path: Path, updates: dict[str, str]
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    _write_feat_params(model, **updates)
    with pytest.raises(ValueError, match=r"Invalid feat\.params field"):
        require_complete_model(model)


def test_complete_model_rejects_float_spelled_nfft_as_an_integer_error(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    _write_feat_params(model, **{"-nfft": "512.0"})

    with pytest.raises(
        ValueError,
        match=r"Invalid feat\.params field -nfft='512\.0'.*: must be an integer",
    ):
        require_complete_model(model)


@pytest.mark.parametrize(
    "updates",
    [
        {"-samprate": "16000.5"},
        {"-lowerf": "130.5"},
        {"-dither": "Y"},
        {"-dither": "false"},
        {"-feat": "1s_3c"},
        {"-feat": "1s_4c"},
        {"-upperf": "8001"},
        {"-alpha": "native-atof-compatible"},
        {"-cmninit": "native,does,not,reject,or,check,length"},
    ],
)
def test_complete_model_does_not_reject_native_accepted_values(
    tmp_path: Path, updates: dict[str, str]
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    _write_feat_params(model, **updates)
    assert require_complete_model(model) == model / "feat.params"


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
