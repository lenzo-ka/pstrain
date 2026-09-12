"""No pstrain module may open a connection in a process that trains.

On macOS the first connection a process opens installs atfork handlers, and a
process that has opened one can no longer spawn the native worker: the worker
dies of SIGSEGV during startup without writing a diagnostic. The crash lands
during training, far from the download that caused it, which is what makes this
class of defect expensive rather than merely wrong. Downloads therefore run in
short-lived children, and this test keeps the list of modules that may perform
one from growing by accident.
"""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "pstrain"

#: Calls that open a connection. Names, not modules: an alias or a
#: ``from urllib.request import urlopen`` is the same hazard.
CONNECTING_CALLS = frozenset(
    {"urlopen", "urlretrieve", "create_connection", "HTTPConnection", "HTTPSConnection"}
)

#: The only modules that may open a connection. Each one does so exclusively
#: inside a helper entry point that runs in a spawned child, never in the
#: process its callers train in.
NETWORK_HELPER_MODULES = frozenset(
    {
        Path("benchmarks/corpora.py"),
        Path("lib/dictionary/cmudict_source.py"),
    }
)


def _called_name(node: ast.Call) -> str:
    target = node.func
    if isinstance(target, ast.Attribute):
        return target.attr
    if isinstance(target, ast.Name):
        return target.id
    return ""


def _modules_that_connect() -> set[Path]:
    found: set[Path] = set()
    for source in sorted(PACKAGE.rglob("*.py")):
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _called_name(node) in CONNECTING_CALLS:
                found.add(source.relative_to(PACKAGE))
    return found


def test_only_the_network_helpers_open_connections() -> None:
    assert _modules_that_connect() == set(NETWORK_HELPER_MODULES), (
        "A module outside the network helpers opens a connection. Route the fetch "
        "through a child process -- see pstrain.benchmarks.corpora.fetch_pinned_archive "
        "-- rather than adding the module here."
    )


def test_every_network_helper_spawns_the_child_that_connects() -> None:
    for module in NETWORK_HELPER_MODULES:
        source = (PACKAGE / module).read_text(encoding="utf-8")
        assert "subprocess.run(" in source, (
            f"{module} opens connections but spawns nothing; its callers would be "
            "the ones left unable to start a native worker"
        )
