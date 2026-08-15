import sys
from pathlib import Path

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
