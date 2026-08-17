#!/usr/bin/env python3
"""Statically enforce containment of Python-to-CFFI call expressions.

This scanner certifies calls whose callee is a literal name or attribute chain, a
literal/module-constant ``getattr`` name, a literal-key dictionary selection, a direct
``FFI().dlopen``, or a single-assignment local alias of a known native leaf/loader. It
also flags runtime ``getattr`` and subscript dispatch directly on conventionally named
library handles. It is silent on multi-step dataflow, function pointers produced by
``ffi.addressof``, and aliases that cross module boundaries or arise from dynamic
imports.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from cffi import FFI

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pstrain.lib._cffi.cdef import CDEF  # noqa: E402


def _cffi_function_names() -> frozenset[str]:
    """Derive every callable native leaf from the ABI-mode CFFI surface."""
    ffi = FFI()
    ffi.cdef(CDEF)
    return frozenset(
        key.removeprefix("function ")
        for key in ffi._parser._declarations  # type: ignore[attr-defined]
        if key.startswith("function ")
    )


CFFI_FUNCTION_NAMES = _cffi_function_names()
# These Python helpers reach ``dlopen`` directly or transitively.  They are not
# C functions, so they do not occur in CDEF, but they are native-entry leaves.
NATIVE_ENTRY_NAMES = CFFI_FUNCTION_NAMES | {
    "_init",
    "dlopen",
    "get_ffi",
    "get_lib",
    "path_or_null",
}
INFRASTRUCTURE = {
    "pstrain/lib/_cffi/core.py",
    "pstrain/lib/_pstrainc.py",
    "pstrain/lib/native_worker.py",
}
DECLARED_EXCEPTIONS = {
    "pstrain/lib/testing/decoder.py": (
        "The PocketSphinx decoder used by shipped benchmark, CLI testing, and decode-shard "
        "paths deliberately remains in-process; changing its lifecycle is outside this gate."
    )
}


@dataclass(frozen=True)
class Callsite:
    path: str
    line: int
    column: int
    function: str
    symbol: str
    disposition: str
    reason: str


def _name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _terminates(statements: list[ast.stmt]) -> bool:
    return bool(statements) and isinstance(statements[-1], (ast.Return, ast.Raise))


def _worker_test(node: ast.AST) -> bool | None:
    if isinstance(node, ast.Call) and _name(node.func).endswith("in_worker"):
        return True
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        value = _worker_test(node.operand)
        return None if value is None else not value
    return None


def _proxy_test(node: ast.AST) -> bool:
    """Recognize the proxy branch used at the start of stateful-object methods."""
    return (
        isinstance(node, ast.Call)
        and _name(node.func) == "hasattr"
        and len(node.args) == 2
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "self"
        and isinstance(node.args[1], ast.Constant)
        and node.args[1].value == "_proxy"
    )


def _proxy_forward(statements: list[ast.stmt]) -> bool:
    """Require the guarded branch to actually invoke the object's proxy."""
    return any(
        isinstance(item, ast.Call) and _name(item.func).startswith("self._proxy.")
        for statement in statements
        for item in ast.walk(statement)
    )


class Scanner(ast.NodeVisitor):
    def __init__(self, path: str) -> None:
        self.path = path
        self.callsites: list[Callsite] = []
        self.functions: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
        self.worker_depth = 0
        self.constants: dict[str, str] = {}
        self.aliases: list[dict[str, str]] = [{}]

    def _decorated(self) -> bool:
        return any(
            any(_name(decorator).endswith("contained") for decorator in function.decorator_list)
            for function in self.functions
        )

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        prior = self.worker_depth
        self.worker_depth = 0
        self.functions.append(node)
        self.aliases.append({})
        self._visit_statements(node.body)
        self.aliases.pop()
        self.functions.pop()
        self.worker_depth = prior

    visit_AsyncFunctionDef = visit_FunctionDef

    def _visit_statements(self, statements: list[ast.stmt]) -> None:
        worker_only = False
        for statement in statements:
            prior = self.worker_depth
            if worker_only:
                self.worker_depth += 1
            if isinstance(statement, ast.If):
                state = _worker_test(statement.test)
                if state is not None:
                    self.visit(statement.test)
                    self.worker_depth += 1 if state else 0
                    self._visit_statements(statement.body)
                    self.worker_depth = prior + (1 if worker_only else 0) + (1 if not state else 0)
                    self._visit_statements(statement.orelse)
                    self.worker_depth = prior + (1 if worker_only else 0)
                    if not state and _terminates(statement.body):
                        worker_only = True
                    continue
            self.visit(statement)
            self.worker_depth = prior
            if (
                isinstance(statement, ast.If)
                and _proxy_test(statement.test)
                and _terminates(statement.body)
                and _proxy_forward(statement.body)
            ):
                worker_only = True

    def visit_Assign(self, node: ast.Assign) -> None:
        if (
            not self.functions
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.constants[target.id] = node.value.value
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target = node.targets[0].id
            value = self._callee(node.value)
            leaf = value.rsplit(".", 1)[-1]
            if value and leaf in NATIVE_ENTRY_NAMES:
                self.aliases[-1][target] = value
            else:
                self.aliases[-1].pop(target, None)
        self.generic_visit(node)

    def _string(self, node: ast.AST) -> str | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name):
            return self.constants.get(node.id)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left, right = self._string(node.left), self._string(node.right)
            return left + right if left is not None and right is not None else None
        return None

    def _callee(self, node: ast.AST) -> str:
        if (
            isinstance(node, ast.Attribute)
            and node.attr == "dlopen"
            and isinstance(node.value, ast.Call)
            and _name(node.value.func).endswith("FFI")
        ):
            return "FFI().dlopen"
        literal = _name(node)
        if literal:
            if isinstance(node, ast.Name):
                return self.aliases[-1].get(literal, literal)
            return literal
        if isinstance(node, ast.Call) and _name(node.func) == "getattr" and len(node.args) >= 2:
            attribute = self._string(node.args[1])
            base = _name(node.args[0])
            if base and attribute is not None:
                return f"{base}.{attribute}"
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Dict):
            key = self._string(node.slice)
            for candidate_key, candidate_value in zip(
                node.value.keys, node.value.values, strict=True
            ):
                if candidate_key is not None and self._string(candidate_key) == key:
                    return _name(candidate_value)
        return ""

    def _dynamic_native_callee(self, node: ast.AST) -> str:
        """Describe runtime dispatch directly on a conventionally named CFFI handle."""
        if isinstance(node, ast.Call) and _name(node.func) == "getattr" and len(node.args) >= 2:
            base = _name(node.args[0])
            if self._native_handle(base) and self._string(node.args[1]) is None:
                return f"getattr({base}, <dynamic>)"
        if isinstance(node, ast.Subscript):
            base = _name(node.value)
            if self._native_handle(base):
                return f"{base}[<dynamic>]"
        return ""

    @staticmethod
    def _native_handle(name: str) -> bool:
        return name.rsplit(".", 1)[-1] in {"lib", "_lib"}

    def visit_Call(self, node: ast.Call) -> None:
        symbol = self._callee(node.func)
        leaf = symbol.rsplit(".", 1)[-1]
        dynamic_symbol = self._dynamic_native_callee(node.func)
        if leaf in NATIVE_ENTRY_NAMES or dynamic_symbol:
            symbol = dynamic_symbol or symbol
            if self.path in INFRASTRUCTURE:
                disposition, reason = "infrastructure", "implements the low-level worker boundary"
            elif self.path in DECLARED_EXCEPTIONS:
                disposition, reason = "declared_exception", DECLARED_EXCEPTIONS[self.path]
            elif self._decorated():
                disposition, reason = "contained", "enclosing callable has @contained"
            elif self.worker_depth:
                disposition, reason = "worker_only", "control flow requires in_worker()"
            else:
                disposition, reason = "violation", "CFFI is reachable in the caller process"
            function = self.functions[-1].name if self.functions else "<module>"
            self.callsites.append(
                Callsite(
                    self.path, node.lineno, node.col_offset, function, symbol, disposition, reason
                )
            )
        self.generic_visit(node)


def scan() -> list[Callsite]:
    callsites: list[Callsite] = []
    for source in sorted((ROOT / "pstrain").rglob("*.py")):
        relative = source.relative_to(ROOT).as_posix()
        scanner = Scanner(relative)
        scanner.visit(ast.parse(source.read_text(), filename=relative))
        callsites.extend(scanner.callsites)
    return callsites


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    callsites = scan()
    if args.json:
        args.json.write_text(json.dumps([asdict(item) for item in callsites], indent=2) + "\n")
    violations = [item for item in callsites if item.disposition == "violation"]
    for item in violations:
        print(f"{item.path}:{item.line}:{item.column + 1}: uncontained CFFI call {item.symbol}")
    return bool(violations)


if __name__ == "__main__":
    raise SystemExit(main())
