import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from tree_compare import compare_files, compare_keyed_rows, compare_partitions  # noqa: E402


def write(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


def test_known_equal_ignores_ids_order_and_scores(tmp_path: Path) -> None:
    a = write(tmp_path / "a", "n_node 3\n0 1 2 9 8 ((Q 1))\n1 - - 7 6\n2 - - 5 4\n")
    b = write(tmp_path / "b", "n_node 3\n0 8 4 1 2 ((Q 1))\n4 - - 3 4\n8 - - 5 6\n")
    assert compare_files(a, b)["equal"]


def test_known_different_question_label(tmp_path: Path) -> None:
    a = write(tmp_path / "a", "n_node 3\n0 1 2 9 8 ((Q 1))\n1 - - 7 6\n2 - - 5 4\n")
    b = write(tmp_path / "b", "n_node 3\n0 1 2 9 8 ((!Q 1))\n1 - - 7 6\n2 - - 5 4\n")
    result = compare_files(a, b)
    assert not result["equal"]
    assert result["first_difference"]["kind"] == "question_label"


def test_known_different_topology(tmp_path: Path) -> None:
    a = write(tmp_path / "a", "n_node 1\n0 - - 9 8\n")
    b = write(tmp_path / "b", "n_node 3\n0 1 2 9 8 ((Q 1))\n1 - - 7 6\n2 - - 5 4\n")
    result = compare_files(a, b)
    assert not result["equal"]
    assert result["first_difference"]["kind"] == "topology"


def test_senone_id_permutation_changes_keyed_rows_not_partition(tmp_path: Path) -> None:
    header = "0.3\n2 n_base\n"
    rows = [
        "AA B C i n/a 0 {first} {second} {third} N",
        "AA D E i n/a 0 {first} {second} {third} N",
        "AA F G i n/a 0 {other} {second} {third} N",
    ]
    left = write(
        tmp_path / "left.mdef",
        header + "\n".join(row.format(first=10, other=11, second=20, third=30) for row in rows),
    )
    right = write(
        tmp_path / "right.mdef",
        header + "\n".join(row.format(first=91, other=90, second=20, third=30) for row in rows),
    )

    partition = compare_partitions(left, right)
    keyed_rows = compare_keyed_rows(left, right)

    assert partition["equal"]
    assert partition["summary"]["pair_disagreements"] == 0
    assert keyed_rows["matching_rows"] < keyed_rows["common_rows"]
    assert keyed_rows["label_sensitive"]
    assert not keyed_rows["allocation_metric"]


def test_truncated_mdef_row_is_rejected_in_both_modes(tmp_path: Path) -> None:
    valid = "AA B C i n/a 0 10 20 30 N\n"
    malformed = "AA D E i n/a 0 10 20\n"
    left = write(tmp_path / "left.mdef", "0.3\n2 n_base\n" + valid + malformed)
    right = write(tmp_path / "right.mdef", "0.3\n2 n_base\n" + valid)

    for compare in (compare_partitions, compare_keyed_rows):
        with pytest.raises(ValueError, match=r"left\.mdef:4: malformed mdef row: AA D E"):
            compare(left, right)


def test_shared_leaf_is_rejected_as_node_reuse(tmp_path: Path) -> None:
    shared = write(tmp_path / "shared", "n_node 2\n0 1 1 9 8 ((Q 1))\n1 - - 7 6\n")
    tree = write(tmp_path / "tree", "n_node 3\n0 1 2 9 8 ((Q 1))\n1 - - 7 6\n2 - - 5 4\n")

    with pytest.raises(ValueError, match=r"shared: node 1 is reused"):
        compare_files(shared, tree)


def test_partition_difference_names_moved_context(tmp_path: Path) -> None:
    header = "0.3\n2 n_base\n"
    contexts = ("AA B C i n/a 0", "AA D E i n/a 0", "AA F G i n/a 0")
    left = write(
        tmp_path / "left.mdef",
        header
        + "".join(
            f"{context} {senone} 20 30 N\n"
            for context, senone in zip(contexts, (10, 10, 11), strict=True)
        ),
    )
    right = write(
        tmp_path / "right.mdef",
        header
        + "".join(
            f"{context} {senone} 20 30 N\n"
            for context, senone in zip(contexts, (90, 91, 91), strict=True)
        ),
    )

    result = compare_partitions(left, right)

    assert not result["equal"]
    assert result["summary"]["pair_disagreements"] == 2
    assert list(contexts[1].split()) in result["subjects"]["AA-0"]["differing_contexts"]
    assert result["subjects"]["AA-0"]["pair_disagreement_samples"]
