"""Generate contract scope documentation from declarations on executable gates."""

from __future__ import annotations

import ast
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TypeVar

_F = TypeVar("_F", bound=Callable[..., object])
_START = "<!-- BEGIN GENERATED GATE SCOPE -->"
_END = "<!-- END GENERATED GATE SCOPE -->"


def contract_scope(**_scope: object) -> Callable[[_F], _F]:
    """Attach documentation-only scope metadata to a test gate."""

    def decorate(function: _F) -> _F:
        return function

    return decorate


def _literal(node: ast.expr) -> object:
    """Read a scope value while rejecting executable decorator expressions."""
    return ast.literal_eval(node)


def _declarations(root: Path, paths: Iterable[Path]) -> list[dict[str, object]]:
    declarations: list[dict[str, object]] = []
    for path in paths:
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for decorator in node.decorator_list:
                if not isinstance(decorator, ast.Call):
                    continue
                if (
                    not isinstance(decorator.func, ast.Name)
                    or decorator.func.id != "contract_scope"
                ):
                    continue
                if decorator.args:
                    raise ValueError(f"{path}:{node.lineno}: contract_scope accepts keywords only")
                scope: dict[str, object] = {}
                for keyword in decorator.keywords:
                    if keyword.arg is None:
                        raise ValueError(
                            f"{path}:{node.lineno}: expanded scope arguments are forbidden"
                        )
                    scope[keyword.arg] = _literal(keyword.value)
                scope["gate"] = f"{path.relative_to(root).as_posix()}::{node.name}"
                declarations.append(scope)

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
    kind = scope["kind"]
    gate = f"`{scope['gate']}`"
    if kind == "fixed-count-reproducibility":
        return (
            f"At exactly {_names(scope['shard_counts'])} shards and {scope['passes']} passes, {gate} "
            f"repeats the same seeded manifest twice. It compares produced model files "
            f"{_names(scope['produced_files'])}, the copied input {_names(scope['copied_files'])}, "
            f"and per-shard files {_names(scope['accumulator_files'])} byte-for-byte. The scope is "
            "the architecture and operating system executing the test; it does not compare across "
            "architectures or operating systems."
        )
    if kind == "cross-count-discrete-state":
        return (
            f"Across shard counts {_names(scope['shard_counts'])}, {gate} compares "
            f"{_names(scope['fields'])} for exactly {scope['passes']} passes on the same seeded "
            "manifest. It makes no cross-count floating-parameter comparison."
        )
    if kind == "one-shard-reference":
        return (
            f"At exactly {_names(scope['shard_counts'])} shard and {scope['passes']} passes, {gate} "
            f"compares the reducer path with the in-process reference. It compares files "
            f"{_names(scope['files'])} byte-for-byte and telemetry fields {_names(scope['fields'])} "
            "value-for-value."
        )
    if kind == "provenance-comparison":
        return (
            f"{gate} compares model directories whose {_names(scope['file'])} records effective "
            f"shard counts {_names(scope['shard_counts'])} and requires the comparison to report "
            "that file as different."
        )
    if kind == "multipron-fallback":
        return (
            f"For a request of {_names(scope['requested_shards'])} shards, {gate} requires multipron "
            f"training to select {_names(scope['effective_shards'])} effective shard and emit reason "
            f"{_names(scope['reason'])}; it also requires non-multipron training to retain the request."
        )
    if kind == "artifact-validation":
        return (
            f"{gate} exercises rejection of {_names(scope['mutations'])} in the shard-artifact "
            "validator. These are unit-level mutations, not externally supplied production artifacts."
        )
    raise ValueError(f"Unknown contract scope kind: {kind!r}")


def generate_bw_sharding_contract(root: Path | None = None) -> str:
    """Return the BW sharding contract with its generated scope replaced."""
    root = root or Path.cwd()
    document = root / "docs/design/bw-sharding-contract.md"
    paths = (root / "tests/test_numeric_harness.py", root / "tests/test_bw_sharding.py")
    scopes = _declarations(root, paths)
    if [scope.get("order") for scope in scopes] != list(range(1, len(scopes) + 1)):
        raise ValueError("contract scope order values must be contiguous from 1")
    generated = [
        _START,
        "## Generated gate scope",
        "",
        "This section is generated from `@contract_scope` declarations on the named test gates.",
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
    """Regenerate the BW sharding contract in place."""
    root = root or Path.cwd()
    path = root / "docs/design/bw-sharding-contract.md"
    path.write_text(generate_bw_sharding_contract(root))
