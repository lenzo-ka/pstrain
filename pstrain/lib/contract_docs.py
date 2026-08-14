"""Generate contract scope documentation from executable gate behavior."""

from __future__ import annotations

import ast
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from typing import TypeVar

_F = TypeVar("_F", bound=Callable[..., object])
_START = "<!-- BEGIN GENERATED GATE SCOPE -->"
_END = "<!-- END GENERATED GATE SCOPE -->"


def contract_scope(**_scope: object) -> Callable[[_F], _F]:
    """Attach non-certifying descriptive context to a test gate."""

    def decorate(function: _F) -> _F:
        return function

    return decorate


def contract_check_files(*, left: Path, right: Path, artifacts: Sequence[str], scope: int) -> None:
    """Byte-compare artifacts and expose the checked set to the generator."""
    del scope
    for artifact in artifacts:
        assert (left / artifact).read_bytes() == (right / artifact).read_bytes(), artifact


def contract_check_fields(
    *, left: Mapping[str, object], right: Mapping[str, object], artifacts: Sequence[str], scope: int
) -> None:
    """Value-compare fields and expose the checked set to the generator."""
    del scope
    for artifact in artifacts:
        assert left[artifact] == right[artifact], artifact


def _literal(node: ast.expr) -> object:
    return ast.literal_eval(node)


def _literal_bindings(statements: Iterable[ast.stmt]) -> dict[str, object]:
    bindings: dict[str, object] = {}
    for statement in statements:
        if isinstance(statement, ast.Assign):
            targets, value = statement.targets, statement.value
        elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
            targets, value = [statement.target], statement.value
        else:
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                with suppress(ValueError, TypeError):
                    bindings[target.id] = _literal(value)
    return bindings


def _declarations(root: Path, paths: Iterable[Path]) -> list[dict[str, object]]:
    declarations: list[dict[str, object]] = []
    for path in paths:
        tree = ast.parse(path.read_text(), filename=str(path))
        module_bindings = _literal_bindings(tree.body)
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            function_scopes: list[dict[str, object]] = []
            for decorator in node.decorator_list:
                if not (
                    isinstance(decorator, ast.Call)
                    and isinstance(decorator.func, ast.Name)
                    and decorator.func.id == "contract_scope"
                ):
                    continue
                if decorator.args:
                    raise ValueError(f"{path}:{node.lineno}: contract_scope accepts keywords only")
                if any(keyword.arg is None for keyword in decorator.keywords):
                    raise ValueError(
                        f"{path}:{node.lineno}: expanded scope arguments are forbidden"
                    )
                scope: dict[str, object] = {
                    keyword.arg: _literal(keyword.value)
                    for keyword in decorator.keywords
                    if keyword.arg is not None
                }
                forbidden = {name for name in scope if name.startswith("checked")}
                if forbidden:
                    names = ", ".join(sorted(forbidden))
                    raise ValueError(
                        f"{path}:{node.lineno}: CHECKED cannot be declared ({names}); "
                        "use contract_check_files or contract_check_fields"
                    )
                scope["gate"] = f"{path.relative_to(root).as_posix()}::{node.name}"
                scope["checked_files"] = []
                scope["checked_fields"] = []
                function_scopes.append(scope)

            bindings = module_bindings | _literal_bindings(node.body)
            for call in (item for item in ast.walk(node) if isinstance(item, ast.Call)):
                if not (
                    isinstance(call.func, ast.Name)
                    and call.func.id in {"contract_check_files", "contract_check_fields"}
                ):
                    continue
                keywords = {item.arg: item.value for item in call.keywords if item.arg}
                if "scope" not in keywords or "artifacts" not in keywords:
                    raise ValueError(
                        f"{path}:{call.lineno}: contract check requires scope and artifacts"
                    )
                call_scope = _literal(keywords["scope"])
                artifact_node = keywords["artifacts"]
                artifacts = (
                    bindings.get(artifact_node.id)
                    if isinstance(artifact_node, ast.Name)
                    else _literal(artifact_node)
                )
                if not isinstance(artifacts, (tuple, list)) or not all(
                    isinstance(item, str) for item in artifacts
                ):
                    raise ValueError(
                        f"{path}:{call.lineno}: artifacts must resolve to literal strings"
                    )
                matches = [scope for scope in function_scopes if scope.get("order") == call_scope]
                if len(matches) != 1:
                    raise ValueError(
                        f"{path}:{call.lineno}: contract check scope {call_scope!r} is not declared once"
                    )
                key = (
                    "checked_files" if call.func.id == "contract_check_files" else "checked_fields"
                )
                checked_artifacts = matches[0][key]
                if not isinstance(checked_artifacts, list):
                    raise TypeError(f"internal contract scope error: {key} is not a list")
                checked_artifacts.extend(
                    item for item in artifacts if item not in checked_artifacts
                )
            declarations.extend(function_scopes)

    def order(item: dict[str, object]) -> int:
        value = item.get("order")
        if not isinstance(value, int):
            raise ValueError("contract scope order must be an integer")
        return value

    return sorted(declarations, key=order)


def _names(value: object) -> str:
    items = (
        tuple(str(item) for item in value) if isinstance(value, (tuple, list)) else (str(value),)
    )
    quoted = [f"`{item}`" for item in items]
    if len(quoted) == 1:
        return quoted[0]
    if len(quoted) == 2:
        return " and ".join(quoted)
    return f"{', '.join(quoted[:-1])}, and {quoted[-1]}"


def _render(scope: dict[str, object]) -> str:
    gate = f"`{scope['gate']}`"
    checked_files_value = scope["checked_files"]
    checked_fields_value = scope["checked_fields"]
    copied_outputs = scope["copied_outputs"]
    if not isinstance(checked_files_value, list) or not all(
        isinstance(item, str) for item in checked_files_value
    ):
        raise TypeError("internal contract scope error: checked_files")
    if not isinstance(checked_fields_value, list) or not all(
        isinstance(item, str) for item in checked_fields_value
    ):
        raise TypeError("internal contract scope error: checked_fields")
    if not isinstance(copied_outputs, tuple):
        raise TypeError("internal contract scope error: copied_outputs")
    checked_files: tuple[str, ...] = tuple(checked_files_value)
    checked_fields: tuple[str, ...] = tuple(checked_fields_value)
    copied = tuple(name for name in checked_files if name in copied_outputs)
    produced = tuple(name for name in checked_files if name not in copied)
    checked_parts = []
    if produced:
        checked_parts.append(f"produced files {_names(produced)}")
    if copied:
        checked_parts.append(f"copied inputs {_names(copied)}")
    if checked_fields:
        checked_parts.append(f"fields {_names(checked_fields)}")
    checked = "; ".join(checked_parts) or "nothing"
    declared_counts = scope.get("shard_counts", scope.get("requested_shards", "not declared"))
    describes = f"DESCRIBES (certifies nothing): kind `{scope['kind']}`, declared shard counts {_names(declared_counts)}"
    if "passes" in scope:
        describes += f", and {scope['passes']} passes"
    result = f"{gate}. CHECKED (mechanically asserted): {checked}. {describes}."
    if scope["kind"] == "fixed-count-reproducibility":
        result += " It makes no cross-architecture or cross-operating-system comparison."
    elif scope["kind"] == "cross-count-discrete-state":
        result += " It makes no cross-count floating-parameter comparison."
    elif scope["kind"] == "one-shard-reference" and "mdef" in copied:
        result += " The compared `mdef` is copied from the same input model in both arms."
    elif scope["kind"] == "artifact-validation":
        result += " The declared mutations are orientation, not a certified artifact inventory."
    return result


def _copied_outputs(root: Path) -> tuple[str, ...]:
    tree = ast.parse((root / "pstrain/lib/steps/train.py").read_text())
    value = _literal_bindings(tree.body).get("_COPIED_TRAINING_OUTPUTS")
    if not isinstance(value, tuple) or not all(isinstance(item, str) for item in value):
        raise ValueError("training copy behavior must define literal _COPIED_TRAINING_OUTPUTS")
    return value


def generate_bw_sharding_contract(root: Path | None = None) -> str:
    root = root or Path.cwd()
    document = root / "docs/design/bw-sharding-contract.md"
    paths = (root / "tests/test_numeric_harness.py", root / "tests/test_bw_sharding.py")
    scopes = _declarations(root, paths)
    if [scope.get("order") for scope in scopes] != list(range(1, len(scopes) + 1)):
        raise ValueError("contract scope order values must be contiguous from 1")
    copied_outputs = _copied_outputs(root)
    for scope in scopes:
        scope["copied_outputs"] = copied_outputs
    generated = [
        _START,
        "## Generated gate scope",
        "",
        "`CHECKED` entries are derived from assertion-helper calls in the named gates. `DESCRIBES` "
        "entries come from decorators and certify nothing. No comprehensive `CONSUMES` inventory "
        "is claimed because the harness does not mechanically trace inputs.",
        "",
    ]
    generated.extend(f"{index}. {_render(scope)}\n" for index, scope in enumerate(scopes, 1))
    generated.append(_END)
    text = document.read_text()
    before, separator, remainder = text.partition(_START)
    if not separator:
        raise ValueError(f"missing generated scope start marker in {document}")
    _, separator, after = remainder.partition(_END)
    if not separator:
        raise ValueError(f"missing generated scope end marker in {document}")
    return before + "\n".join(generated) + after


def write_bw_sharding_contract(root: Path | None = None) -> None:
    root = root or Path.cwd()
    path = root / "docs/design/bw-sharding-contract.md"
    path.write_text(generate_bw_sharding_contract(root))
