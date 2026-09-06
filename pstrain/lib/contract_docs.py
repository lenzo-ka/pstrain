"""Generate contract scope documentation from executable gate behavior."""

from __future__ import annotations

import ast
import json
import textwrap
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from typing import TypeVar

_F = TypeVar("_F", bound=Callable[..., object])
_START = "<!-- BEGIN GENERATED GATE SCOPE -->"
_END = "<!-- END GENERATED GATE SCOPE -->"
_COVERAGE_START = "<!-- BEGIN GENERATED COVERAGE -->"
_COVERAGE_END = "<!-- END GENERATED COVERAGE -->"
_IDENTITY_START = "<!-- BEGIN GENERATED MEASUREMENT IDENTITY -->"
_IDENTITY_END = "<!-- END GENERATED MEASUREMENT IDENTITY -->"
_BASELINE_START = "<!-- BEGIN GENERATED BASELINE -->"
_BASELINE_END = "<!-- END GENERATED BASELINE -->"
_CELL_NAMES = {"slt55": "SLT-55", "big": "big"}


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


def _replace_block(text: str, start: str, end: str, generated: Sequence[str], label: str) -> str:
    before, separator, remainder = text.partition(start)
    if not separator:
        raise ValueError(f"missing generated {label} start marker")
    _, separator, after = remainder.partition(end)
    if not separator:
        raise ValueError(f"missing generated {label} end marker")
    return before + "\n".join([start, *generated, end]) + after


def _coverage_block(record: Mapping[str, object]) -> list[str]:
    results = record["results"]
    if not isinstance(results, Mapping):
        raise ValueError("Arctic record results must be an object")
    cells_by_status: dict[str, list[str]] = {}
    for mode, datasets in results.items():
        for dataset, cell in datasets.items():
            cells_by_status.setdefault(cell["status"], []).append(f"{mode}/{dataset}")

    live = sorted(cells_by_status.pop("live", ()))
    retired = sorted(cells_by_status.pop("retired/historical", ()))
    if cells_by_status or not live or not retired:
        raise ValueError(f"unexpected Arctic coverage statuses: {cells_by_status}")
    basis = record["basis"]
    if not isinstance(basis, Mapping):
        raise ValueError("Arctic record basis must be an object")
    declared_retired = sorted(basis["retired_cells"])
    if retired != declared_retired:
        raise ValueError(
            f"retired result cells {retired!r} disagree with basis {declared_retired!r}"
        )
    if any(not cell.startswith("off/") for cell in retired):
        raise ValueError(f"retired Arctic cells are not all off-mode: {retired!r}")
    return [
        f"The live cells are {_names(live)}. The {_names(retired)} cells are retained as",
        "`retired/historical`; the off-mode cells are neither trained nor decoded by the",
        "current benchmark run.",
    ]


def _wrap(paragraph: str) -> list[str]:
    """Wrap generated prose without breaking a backticked digest across lines."""
    return textwrap.wrap(paragraph, width=88, break_long_words=False, break_on_hyphens=False)


def _identity_block(record: Mapping[str, object]) -> list[str]:
    """State the live decode identity from the record rather than from memory."""
    engine = record["engine"]
    if not isinstance(engine, Mapping):
        raise ValueError("Arctic record engine must be an object")
    retained = record["historical_provenance"]
    if not isinstance(retained, Mapping):
        raise ValueError("Arctic record historical provenance must be an object")
    historical_engine = retained["engine"]
    python_version = str(engine["python_version"]).split()[0]
    pocketsphinx = str(engine["pocketsphinx_version"]).split("+")[0]
    historical_python = str(historical_engine["python_version"]).split()[0]
    return [
        *_wrap(
            "The decode path is a defining condition of this measurement. The live cells are "
            f"decoded from WAV through pinned PocketSphinx {pocketsphinx} using Python "
            f"{python_version}, native library SHA-256 "
            f"`{engine['native_library_sha256']}`, and decode dictionary SHA-256 "
            f"`{engine['decode_dictionary_sha256']}`. The engine is pstrain "
            f"{engine['version']} at `{engine['git_describe']}`. A result obtained through "
            "another decode path is not the same measurement even when the acoustic-model "
            "bytes are identical."
        ),
        "",
        *_wrap(
            "The retired off-mode cells were not measured on that path. They were decoded by "
            f"pstrain {historical_engine['version']} at `{historical_engine['git_describe']}` "
            f"using Python {historical_python}, native library SHA-256 "
            f"`{historical_engine['native_library_sha256']}`, and decode dictionary SHA-256 "
            f"`{historical_engine['decode_dictionary_sha256']}`. That identity travels with "
            "them in the record's `historical_provenance` and is never inherited from a "
            "later run."
        ),
    ]


def _row(
    mode: str,
    dataset: str,
    pstrain: float,
    oracle: float,
    interval: Sequence[float],
    comparability: Mapping[str, object],
) -> str:
    delta = pstrain - oracle
    if mode == "off":
        interpretation = "historical only"
    elif interval[0] > 0:
        interpretation = "statistically significant regression"
    elif interval[1] < 0:
        interpretation = "statistically significant improvement"
    else:
        # A favorable point estimate whose interval spans zero is not an
        # improvement. The column must not let a negative delta read as one.
        interpretation = "no statistically significant difference"
    label = "off (retired)" if mode == "off" else mode
    paired_axis = comparability.get("paired_decode")
    attribution_axis = comparability.get("implementation_attribution")
    if not isinstance(paired_axis, Mapping) or not isinstance(attribution_axis, Mapping):
        raise ValueError("Arctic comparison has invalid comparability axes")
    paired_decode = paired_axis.get("status")
    attribution = attribution_axis.get("status")
    if paired_decode not in ("COMPARABLE", "NOT COMPARABLE") or attribution not in (
        "COMPARABLE",
        "NOT COMPARABLE",
    ):
        raise ValueError("Arctic comparison has invalid comparability status")
    return (
        f"| {label} | {_CELL_NAMES[dataset]} | {pstrain:.4f} | {oracle:.4f} | {delta:+.4f} "
        f"| [{interval[0]:+.4f}, {interval[1]:+.4f}] | {paired_decode} "
        f"| {attribution} | {interpretation} |"
    )


def _baseline_block(
    record: Mapping[str, object],
    oracle: Mapping[str, object],
    analysis: Mapping[str, object],
) -> list[str]:
    """Render the baseline table from the measured artifacts, never by hand."""
    results = record["results"]
    oracle_results = oracle["results"]
    oracle_verdicts = oracle["pstrain_vs_oracle"]
    if not isinstance(results, Mapping) or not isinstance(oracle_results, Mapping):
        raise ValueError("Arctic results must be objects")
    if not isinstance(oracle_verdicts, Sequence) or not isinstance(
        analysis["pstrain_vs_oracle"], Sequence
    ):
        raise ValueError("Arctic comparisons must be sequences")
    live = {
        (row["mode"], row["dataset"]): row
        for row in analysis["pstrain_vs_oracle"]
        if isinstance(row, Mapping)
    }
    retired = {
        (row["mode"], row["dataset"]): row
        for row in oracle_verdicts
        if isinstance(row, Mapping) and row["mode"] == "off"
    }
    rows = []
    for dataset in ("slt55", "big"):
        verdict = retired[("off", dataset)]
        rows.append(
            _row(
                "off",
                dataset,
                float(results["off"][dataset]["wer"]),
                float(oracle_results["off"][dataset]["wer"]),
                verdict["paired_ci_95_pp"],
                verdict["comparability"],
            )
        )
        comparison = live[("on", dataset)]
        rows.append(
            _row(
                "on",
                dataset,
                float(comparison["pstrain_wer"]),
                float(comparison["oracle_wer"]),
                comparison["paired_ci_95_pp"],
                comparison["comparability"],
            )
        )
    return [
        "| Mode | Cell | pstrain WER | Oracle WER | Delta pp | Paired 95% CI "
        "| Paired decode | Implementation attribution | Interpretation |",
        "|---|---|---:|---:|---:|---:|---|---|---|",
        *rows,
        "",
        "The live rows come from the record and the resource-matched oracle sidecar through",
        "`scripts/regenerate_arctic_paired_analysis.py`; the retired rows come from the",
        "sidecar's preserved historical comparison. Both are regenerated from the checked-in",
        "artifacts, so an amended measurement cannot leave this table stale.",
    ]


def _check_pinned_resource_rows(text: str, record: Mapping[str, object]) -> None:
    """Refuse a pin document whose hand-written resource digests left the record."""
    resources = record["resources"]
    if not isinstance(resources, Mapping):
        raise ValueError("Arctic record resources must be an object")
    expected = {
        "Language model": resources["lm_sha256"],
        "Decode dictionary": resources["dictionary_sha256"],
        "Filler dictionary": resources["filler_dictionary_sha256"],
    }
    for label, digest in expected.items():
        row = f"| {label} | SHA-256 `{digest}` |"
        if row not in text:
            raise ValueError(
                f"pin document does not state the recorded {label.lower()} digest: expected row "
                f"{row!r}"
            )


def generate_arctic_pin_document(root: Path | None = None) -> str:
    """Generate the Arctic pin's coverage, decode identity, and baseline table."""
    root = root or Path.cwd()
    document = root / "docs/benchmarks/arctic-pin.md"
    evidence = root / "evidence/arctic-pin"
    record = json.loads((evidence / "record.json").read_text())
    oracle = json.loads((evidence / "oracle-sidecar.json").read_text())
    analysis = json.loads((evidence / "paired-analysis.json").read_text())
    text = document.read_text()
    _check_pinned_resource_rows(text, record)
    text = _replace_block(text, _COVERAGE_START, _COVERAGE_END, _coverage_block(record), "coverage")
    text = _replace_block(
        text, _IDENTITY_START, _IDENTITY_END, _identity_block(record), "measurement identity"
    )
    return _replace_block(
        text,
        _BASELINE_START,
        _BASELINE_END,
        _baseline_block(record, oracle, analysis),
        "baseline",
    )


def write_arctic_pin_document(root: Path | None = None) -> None:
    root = root or Path.cwd()
    path = root / "docs/benchmarks/arctic-pin.md"
    path.write_text(generate_arctic_pin_document(root))
