"""Documentation coverage gates for the supported Python API."""

from __future__ import annotations

import importlib
import io
import pkgutil
from pathlib import Path
from types import ModuleType

import pytest
from sphinx.application import Sphinx
from sphinx.util.inventory import InventoryFile

ROOT = Path(__file__).parents[1]


def _public_api_modules() -> list[ModuleType]:
    package = importlib.import_module("pstrain.api")
    modules = [package]
    for info in pkgutil.walk_packages(package.__path__, prefix=f"{package.__name__}."):
        relative_name = info.name.removeprefix(f"{package.__name__}.")
        if any(part.startswith("_") for part in relative_name.split(".")):
            continue
        modules.append(importlib.import_module(info.name))
    return modules


def _reachable_documents(toctree_includes: dict[str, list[str]], root: str) -> set[str]:
    reachable: set[str] = set()
    pending = [root]
    while pending:
        document = pending.pop()
        if document in reachable:
            continue
        reachable.add(document)
        pending.extend(toctree_includes.get(document, ()))
    return reachable


@pytest.mark.filterwarnings("ignore::sphinx.deprecation.RemovedInSphinx11Warning")
def test_public_api_is_present_in_built_reference(tmp_path: Path) -> None:
    """Every declared public API module and export must be reachable in built docs.

    The expected exports and autodoc both ultimately consume ``__all__``, so this
    does not independently prove that those declarations are complete. It does
    detect missing modules and pages, qualification or exclusion mistakes,
    inventory omissions, and inventory entries whose documents are unreachable.
    """
    output = tmp_path / "html"
    warnings = io.StringIO()
    app = Sphinx(
        srcdir=ROOT / "docs",
        confdir=ROOT / "docs",
        outdir=output,
        doctreedir=tmp_path / "doctrees",
        buildername="html",
        confoverrides={"intersphinx_mapping": {}},
        status=io.StringIO(),
        warning=warnings,
        freshenv=True,
        warningiserror=True,
    )
    app.build(force_all=True)
    assert app.statuscode == 0, warnings.getvalue()

    inventory = InventoryFile.loads((output / "objects.inv").read_bytes(), uri="").data
    python_object_locations: dict[str, set[str]] = {}
    for role, objects in inventory.items():
        if not role.startswith("py:"):
            continue
        for name, item in objects.items():
            document = item.uri.partition("#")[0].removesuffix(".html")
            python_object_locations.setdefault(name, set()).add(document)
    documented_modules = inventory.get("py:module", {})
    reachable = _reachable_documents(app.env.toctree_includes, app.config.root_doc)

    for module in _public_api_modules():
        exports = getattr(module, "__all__", None)
        assert exports is not None, f"{module.__name__} must declare its public names in __all__"

        reference = documented_modules.get(module.__name__)
        assert reference is not None, f"{module.__name__} has no API reference page"
        reference_document = reference.uri.partition("#")[0].removesuffix(".html")
        assert reference_document in reachable, (
            f"{module.__name__} reference page {reference_document!r} is not reachable "
            "from the documentation toctree"
        )

        expected = {f"{module.__name__}.{name}" for name in exports}
        missing = sorted(expected - python_object_locations.keys())
        assert not missing, (
            f"{module.__name__} public names are missing from objects.inv: {', '.join(missing)}"
        )

        unreachable = {
            name: sorted(python_object_locations[name] - reachable)
            for name in expected
            if name in python_object_locations and python_object_locations[name] - reachable
        }
        assert not unreachable, (
            f"{module.__name__} public names have unreachable inventory locations: "
            + ", ".join(
                f"{name} ({', '.join(documents)})"
                for name, documents in sorted(unreachable.items())
            )
        )
