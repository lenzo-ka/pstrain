#!/usr/bin/env python3
"""Compare Sphinx trees literally or tied-mdef allocations by partition.

Tree node numbers, file order, likelihood gains, and occupancies are
serialization details. Child ordering and the normalized-whitespace question
expression at every internal node are retained by the literal-tree metric.

Mdef partition comparison ignores senone labels and asks only which contexts
share a senone within each base-phone/emitting-state subject. Keyed-row
agreement is available as a separate, explicitly label-sensitive diagnostic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from math import comb
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Node:
    left: int | None
    right: int | None
    question: str | None


Subject = tuple[str, int]
Context = tuple[str, ...]
Assignments = dict[Subject, dict[Context, int]]
MdefRows = dict[Context, tuple[int, ...]]
MDEF_COUNT_HEADERS = {
    "n_base",
    "n_tri",
    "n_state_map",
    "n_tied_state",
    "n_tied_ci_state",
    "n_tied_tmat",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_tree(path: Path) -> tuple[int, dict[int, Node]]:
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not lines or lines[0].split()[:1] != ["n_node"]:
        raise ValueError(f"{path}: missing n_node header")
    declared = int(lines[0].split()[1])
    nodes: dict[int, Node] = {}
    for lineno, line in enumerate(lines[1:], 2):
        fields = line.split(maxsplit=5)
        if len(fields) < 5:
            raise ValueError(f"{path}:{lineno}: malformed node")
        node_id = int(fields[0])
        if node_id in nodes:
            raise ValueError(f"{path}:{lineno}: duplicate node {node_id}")
        if fields[1] == fields[2] == "-":
            if len(fields) != 5:
                raise ValueError(f"{path}:{lineno}: leaf has a question")
            nodes[node_id] = Node(None, None, None)
        else:
            if fields[1] == "-" or fields[2] == "-" or len(fields) != 6:
                raise ValueError(f"{path}:{lineno}: incomplete internal node")
            question = " ".join(fields[5].split())
            nodes[node_id] = Node(int(fields[1]), int(fields[2]), question)
    if len(nodes) != declared:
        raise ValueError(f"{path}: declares {declared} nodes, contains {len(nodes)}")
    return declared, nodes


def canonical_tree(path: Path) -> tuple[Any, dict[str, Any]]:
    declared, nodes = parse_tree(path)
    if 0 not in nodes:
        raise ValueError(f"{path}: root node 0 missing")
    visiting: set[int] = set()
    visited: set[int] = set()

    def walk(node_id: int) -> Any:
        if node_id not in nodes:
            raise ValueError(f"{path}: missing child node {node_id}")
        if node_id in visiting:
            raise ValueError(f"{path}: cycle at node {node_id}")
        if node_id in visited:
            raise ValueError(f"{path}: node {node_id} is reused")
        visiting.add(node_id)
        node = nodes[node_id]
        if node.left is None:
            result: Any = ["leaf"]
        else:
            result = ["split", node.question, walk(node.left), walk(node.right)]
        visiting.remove(node_id)
        visited.add(node_id)
        return result

    canonical = walk(0)
    if len(visited) != declared:
        extra = sorted(set(nodes) - visited)
        raise ValueError(f"{path}: unreachable nodes {extra}")
    internal = sum(node.question is not None for node in nodes.values())
    return canonical, {"nodes": declared, "internal_nodes": internal, "leaves": declared - internal}


def first_difference(left: Any, right: Any, route: str = "root") -> dict[str, Any] | None:
    if left[0] != right[0]:
        return {"route": route, "kind": "topology", "left": left[0], "right": right[0]}
    if left[0] == "leaf":
        return None
    if left[1] != right[1]:
        return {"route": route, "kind": "question_label", "left": left[1], "right": right[1]}
    return first_difference(left[2], right[2], route + ".left") or first_difference(
        left[3], right[3], route + ".right"
    )


def compare_files(left_path: Path, right_path: Path) -> dict[str, Any]:
    left, left_stats = canonical_tree(left_path)
    right, right_stats = canonical_tree(right_path)
    difference = first_difference(left, right)
    return {
        "metric": "literal_tree",
        "equal": difference is None,
        "first_difference": difference,
        "left": {"path": str(left_path), "sha256": sha256(left_path), **left_stats},
        "right": {"path": str(right_path), "sha256": sha256(right_path), **right_stats},
    }


def compare_directories(left_dir: Path, right_dir: Path) -> dict[str, Any]:
    left_names = {p.name for p in left_dir.glob("*.dtree")}
    right_names = {p.name for p in right_dir.glob("*.dtree")}
    common = sorted(left_names & right_names)
    trees = {name: compare_files(left_dir / name, right_dir / name) for name in common}
    left_only = sorted(left_names - right_names)
    right_only = sorted(right_names - left_names)
    return {
        "metric": "literal_tree_directory",
        "definition": "rooted ordered topology + exact normalized-whitespace question expression",
        "equal": not left_only and not right_only and all(item["equal"] for item in trees.values()),
        "left_only": left_only,
        "right_only": right_only,
        "common_tree_count": len(common),
        "equal_tree_count": sum(item["equal"] for item in trees.values()),
        "different_tree_count": sum(not item["equal"] for item in trees.values()),
        "trees": trees,
    }


def parse_mdef_rows(path: Path) -> MdefRows:
    rows: MdefRows = {}
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        fields = line.split()
        if not fields or fields[0].startswith("#"):
            continue
        if (len(fields) == 1 and fields[0] == "0.3") or (
            len(fields) == 2 and fields[0].isdigit() and fields[1] in MDEF_COUNT_HEADERS
        ):
            continue
        if len(fields) < 10 or fields[-1] != "N":
            raise ValueError(f"{path}:{lineno}: malformed mdef row: {line.strip()}")
        context = tuple(fields[:6])
        try:
            senones = [int(value) for value in fields[6:-1]]
        except ValueError as error:
            raise ValueError(f"{path}:{lineno}: invalid senone ID") from error
        if context in rows:
            raise ValueError(f"{path}:{lineno}: duplicate context row")
        rows[context] = tuple(senones)
    if not rows:
        raise ValueError(f"{path}: no context-dependent emitting-state assignments")
    return rows


def parse_mdef_assignments(path: Path) -> Assignments:
    assignments: dict[Subject, dict[Context, int]] = defaultdict(dict)
    for context, senones in parse_mdef_rows(path).items():
        if context[1] == "-":
            continue
        for state, senone in enumerate(senones):
            assignments[(context[0], state)][context] = senone
    return dict(assignments)


def _subject_name(subject: Subject) -> str:
    return f"{subject[0]}-{subject[1]}"


def _disagreement_samples(
    common: list[Context],
    left_rows: dict[Context, int],
    right_rows: dict[Context, int],
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Find representative disagreeing pairs without enumerating every pair."""
    by_left: dict[int, dict[int, Context]] = defaultdict(dict)
    by_right: dict[int, dict[int, Context]] = defaultdict(dict)
    for context in common:
        left_label = left_rows[context]
        right_label = right_rows[context]
        by_left[left_label].setdefault(right_label, context)
        by_right[right_label].setdefault(left_label, context)

    samples: list[dict[str, Any]] = []
    seen: set[tuple[Context, Context]] = set()
    for groups, shared_side in ((by_left, "left"), (by_right, "right")):
        for opposing in groups.values():
            representatives = list(opposing.values())
            for first, second in zip(representatives, representatives[1:], strict=False):
                pair = tuple(sorted((first, second)))
                if pair in seen:
                    continue
                seen.add(pair)
                samples.append(
                    {
                        "contexts": [list(pair[0]), list(pair[1])],
                        "share_senone_on": shared_side,
                    }
                )
                if len(samples) == limit:
                    return samples
    return samples


def compare_partitions(left_path: Path, right_path: Path) -> dict[str, Any]:
    """Compare context equivalence classes, independently of senone IDs."""
    left = parse_mdef_assignments(left_path)
    right = parse_mdef_assignments(right_path)
    subjects: dict[str, Any] = {}
    total_pairs = 0
    total_disagreements = 0
    equal_subjects = 0
    for subject in sorted(left.keys() | right.keys()):
        left_rows = left.get(subject, {})
        right_rows = right.get(subject, {})
        common = sorted(left_rows.keys() & right_rows.keys())
        left_only = sorted(left_rows.keys() - right_rows.keys())
        right_only = sorted(right_rows.keys() - left_rows.keys())
        left_labels = [left_rows[key] for key in common]
        right_labels = [right_rows[key] for key in common]
        left_counts = Counter(left_labels)
        right_counts = Counter(right_labels)
        joint_counts = Counter(zip(left_labels, right_labels, strict=True))
        disagreements = (
            sum(comb(count, 2) for count in left_counts.values())
            + sum(comb(count, 2) for count in right_counts.values())
            - 2 * sum(comb(count, 2) for count in joint_counts.values())
        )
        pairs = comb(len(common), 2)
        equal = not left_only and not right_only and disagreements == 0
        disagreement_samples = _disagreement_samples(common, left_rows, right_rows)
        differing_contexts = sorted(
            {tuple(context) for sample in disagreement_samples for context in sample["contexts"]}
        )
        subjects[_subject_name(subject)] = {
            "equal": equal,
            "common_contexts": len(common),
            "left_only_contexts": [list(context) for context in left_only],
            "right_only_contexts": [list(context) for context in right_only],
            "left_parts": len(left_counts),
            "right_parts": len(right_counts),
            "context_pairs": pairs,
            "pair_disagreements": disagreements,
            "differing_contexts": [list(context) for context in differing_contexts],
            "pair_disagreement_samples": disagreement_samples,
        }
        total_pairs += pairs
        total_disagreements += disagreements
        equal_subjects += int(equal)
    return {
        "metric": "partition",
        "definition": "within-subject context pairs that share a senone; senone IDs ignored",
        "equal": equal_subjects == len(subjects),
        "left": {"path": str(left_path), "sha256": sha256(left_path)},
        "right": {"path": str(right_path), "sha256": sha256(right_path)},
        "summary": {
            "subjects": len(subjects),
            "equal_subjects": equal_subjects,
            "context_pairs": total_pairs,
            "pair_disagreements": total_disagreements,
        },
        "subjects": subjects,
    }


def compare_keyed_rows(left_path: Path, right_path: Path) -> dict[str, Any]:
    """Count exact context-to-ID matches; this is not an allocation metric."""
    left = parse_mdef_rows(left_path)
    right = parse_mdef_rows(right_path)
    common = sorted(left.keys() & right.keys())
    matches = sum(left[context] == right[context] for context in common)
    return {
        "metric": "keyed_row_agreement",
        "label_sensitive": True,
        "allocation_metric": False,
        "common_rows": len(common),
        "matching_rows": matches,
        "left": {"path": str(left_path), "sha256": sha256(left_path)},
        "right": {"path": str(right_path), "sha256": sha256(right_path)},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument(
        "--mode",
        choices=("literal-tree", "partition", "keyed-row"),
        default="literal-tree",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.mode == "partition":
        result = compare_partitions(args.left, args.right)
    elif args.mode == "keyed-row":
        result = compare_keyed_rows(args.left, args.right)
    else:
        result = (
            compare_directories(args.left, args.right)
            if args.left.is_dir() and args.right.is_dir()
            else compare_files(args.left, args.right)
        )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
