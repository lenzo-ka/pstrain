#!/usr/bin/env python3
"""Statically enforce containment of Python-to-CFFI routing callsites."""

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INFRASTRUCTURE = {
    "pstrain/lib/_cffi/core.py",
    "pstrain/lib/_pstrainc.py",
    "pstrain/lib/native_worker.py",
}
DECLARED_EXCEPTIONS = {
    "pstrain/lib/testing/decoder.py": (
        "The test-only PocketSphinx decoder deliberately remains in-process; changing its "
        "lifecycle is outside the containment-routing lane."
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


class Scanner(ast.NodeVisitor):
    def __init__(self, path: str) -> None:
        self.path = path
        self.callsites: list[Callsite] = []
        self.functions: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
        self.classes: list[ast.ClassDef] = []
        self.worker_depth = 0

    def _decorated(self) -> bool:
        return any(
            any(_name(decorator).endswith("contained") for decorator in function.decorator_list)
            for function in self.functions
        )

    def _proxied_class(self) -> bool:
        if not self.classes:
            return False
        return any(
            isinstance(node, ast.Call) and _name(node.func).endswith("NativeObjectProxy")
            for node in ast.walk(self.classes[-1])
        )

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.classes.append(node)
        self.generic_visit(node)
        self.classes.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.functions.append(node)
        self._visit_statements(node.body)
        self.functions.pop()

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

    def visit_Call(self, node: ast.Call) -> None:
        symbol = _name(node.func)
        leaf = symbol.rsplit(".", 1)[-1]
        if leaf in {"get_lib", "_init"} or leaf.startswith("pstrain_"):
            if self.path in INFRASTRUCTURE:
                disposition, reason = "infrastructure", "implements the low-level worker boundary"
            elif self.path in DECLARED_EXCEPTIONS:
                disposition, reason = "declared_exception", DECLARED_EXCEPTIONS[self.path]
            elif self._decorated():
                disposition, reason = "contained", "enclosing callable has @contained"
            elif self.worker_depth:
                disposition, reason = "worker_only", "control flow requires in_worker()"
            elif self._proxied_class() and symbol.startswith("self._lib."):
                disposition, reason = "proxied", "enclosing class is constructed via NativeObjectProxy"
            else:
                disposition, reason = "violation", "CFFI is reachable in the caller process"
            function = self.functions[-1].name if self.functions else "<module>"
            self.callsites.append(
                Callsite(self.path, node.lineno, node.col_offset, function, symbol, disposition, reason)
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
