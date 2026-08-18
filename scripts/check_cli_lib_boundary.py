#!/usr/bin/env python3
"""Enforce the closed boundary between pstrain.cli and pstrain.lib.

No direct imports are permitted; the empty allowlist makes this boundary
zero-tolerance.

The scanner fully covers ordinary ``import`` and ``from ... import`` statements
that target ``pstrain.lib``, including relative imports. Dynamic-import
detection is defense-in-depth for common ``importlib`` spellings: it is
best-effort over statically resolvable importlib bindings with string-literal
targets. Reflective access (for example, ``getattr(importlib,
"import_module")``), data flow through assignments or function returns (for
example, ``f = importlib; f.import_module(...)``), and non-literal or computed
import targets are intentionally outside this static analysis's scope. Those
forms rely on code review rather than a gate failure.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLI_DIR = ROOT / "pstrain" / "cli"
ALLOWLIST = ROOT / "scripts" / "cli_lib_boundary_allowlist.txt"


def _lib_module(module: str | None) -> bool:
    return module == "pstrain.lib" or bool(module and module.startswith("pstrain.lib."))


def _from_import_edges(module: str, names: list[ast.alias]) -> set[str]:
    if module == "pstrain.lib":
        return {f"{module}.{alias.name}" for alias in names}
    return {module} if _lib_module(module) else set()


def _absolute_from_module(node: ast.ImportFrom, package: str) -> str:
    if node.level == 0:
        return node.module or ""
    package_parts = package.split(".") if package else []
    keep = len(package_parts) - (node.level - 1)
    base_parts = package_parts[: max(keep, 0)]
    if node.module:
        base_parts.extend(node.module.split("."))
    return ".".join(base_parts)


def _dynamic_import_bindings(tree: ast.AST) -> tuple[set[str], set[str], bool]:
    """Return local importlib module/function names and whether ``__import__`` is rebound."""
    module_names: set[str] = set()
    function_names: set[str] = set()
    import_builtin_rebound = False
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                local_name = alias.asname or alias.name
                if node.level == 0 and node.module == "importlib" and alias.name == "import_module":
                    function_names.add(local_name)
                if local_name == "__import__":
                    import_builtin_rebound = True
        elif isinstance(node, ast.Import):
            for alias in node.names:
                local_name = alias.asname or alias.name.split(".", 1)[0]
                # Best-effort: recognize direct importlib bindings only; aliases
                # to its submodules do not bind the top-level importlib module.
                if alias.asname is None and (
                    alias.name == "importlib" or alias.name.startswith("importlib.")
                ):
                    module_names.add("importlib")
                elif alias.name == "importlib" and alias.asname is not None:
                    module_names.add(alias.asname)
                if local_name == "__import__":
                    import_builtin_rebound = True
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            import_builtin_rebound |= node.name == "__import__"
        elif isinstance(node, ast.arg):
            import_builtin_rebound |= node.arg == "__import__"
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            import_builtin_rebound |= node.id == "__import__"
        elif isinstance(node, ast.ExceptHandler) and node.name:
            import_builtin_rebound |= node.name == "__import__"
    return module_names, function_names, import_builtin_rebound


def discover_edges(source: str, package: str, filename: str = "<unknown>") -> set[str]:
    """Extract unique ``pstrain.lib`` module edges from one Python source string."""
    tree = ast.parse(source, filename=filename)
    module_names, function_names, import_builtin_rebound = _dynamic_import_bindings(tree)
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = _absolute_from_module(node, package)
            if node.module is None:
                for alias in node.names:
                    candidate = f"{module}.{alias.name}" if module else alias.name
                    if _lib_module(candidate):
                        imports.add(candidate)
            elif _lib_module(module):
                imports.update(_from_import_edges(module, node.names))
        elif isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names if _lib_module(alias.name))
        elif isinstance(node, ast.Call) and node.args:
            callee = node.func
            is_import = isinstance(callee, ast.Name) and (
                (callee.id == "__import__" and not import_builtin_rebound)
                or callee.id in function_names
            )
            is_importlib_attribute = (
                isinstance(callee, ast.Attribute)
                and callee.attr == "import_module"
                and isinstance(callee.value, ast.Name)
                and callee.value.id in module_names
            )
            target = node.args[0]
            if (
                (is_import or is_importlib_attribute)
                and isinstance(target, ast.Constant)
                and isinstance(target.value, str)
                and _lib_module(target.value)
            ):
                imports.add(target.value)
    return imports


def find_imports(cli_dir: Path | None = None, root: Path | None = None) -> set[str]:
    """Return all unique ``cli path::lib module`` import edges."""
    cli_dir = CLI_DIR if cli_dir is None else cli_dir
    root = ROOT if root is None else root
    imports: set[str] = set()
    for path in sorted(cli_dir.rglob("*.py")):
        relative_path = path.relative_to(root).as_posix()
        package = ".".join(path.parent.relative_to(root).parts)
        modules = discover_edges(path.read_text(encoding="utf-8"), package, str(path))
        imports.update(f"{relative_path}::{module}" for module in modules)
    return imports


def read_allowlist(allowlist: Path | None = None) -> set[str]:
    """Read non-comment allowlist entries, discarding inline justifications."""
    allowlist = ALLOWLIST if allowlist is None else allowlist
    entries: set[str] = set()
    for raw_line in allowlist.read_text(encoding="utf-8").splitlines():
        entry = raw_line.split("#", 1)[0].strip()
        if entry:
            entries.add(entry)
    return entries


def main() -> int:
    discovered = find_imports()
    allowlisted = read_allowlist()
    new = sorted(discovered - allowlisted)
    stale = sorted(allowlisted - discovered)

    for entry in new:
        path, module = entry.split("::", 1)
        print(
            f"NEW: {path} imports {module} directly; route it through pstrain.api "
            "or, if unavoidable for now, add it to the allowlist with justification",
            file=sys.stderr,
        )
    for entry in stale:
        print(
            f"STALE: allowlist entry {entry} no longer matches any import; "
            "remove it — the allowlist may only shrink",
            file=sys.stderr,
        )

    status = "boundary clean" if not new and not stale else f"{len(new)} new / {len(stale)} stale"
    print(f"{len(discovered)} cli→lib imports found, {len(allowlisted)} allowlisted, {status}")
    return 1 if new or stale else 0


if __name__ == "__main__":
    raise SystemExit(main())
