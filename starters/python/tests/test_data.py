import json

import pytest

from evals.data import class_counts, dumps, load_jsonl, split_rows


def _write(tmp_path, rows):
    p = tmp_path / "rows.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return p


def _rows(n_pass, n_fail, prefix=""):
    return [{"id": f"{prefix}p{i}", "query": "q", "response": "r", "label": "PASS"} for i in range(n_pass)] + [
        {"id": f"{prefix}f{i}", "query": "q", "response": "r", "label": "FAIL"} for i in range(n_fail)
    ]


def test_load_rejects_bad_rows(tmp_path):
    with pytest.raises(ValueError, match="label"):
        load_jsonl(_write(tmp_path, [{"id": "a", "query": "q", "response": "r", "label": "MAYBE"}]))
    with pytest.raises(ValueError, match="duplicate"):
        load_jsonl(_write(tmp_path, [{"id": "a", "query": "q", "response": "r"}] * 2))
    with pytest.raises(ValueError, match="required"):
        load_jsonl(_write(tmp_path, [{"id": "a", "query": "q", "response": "r"}]), require_label=True)
    (tmp_path / "list.jsonl").write_text("[1, 2]\n")
    with pytest.raises(ValueError, match="object"):
        load_jsonl(tmp_path / "list.jsonl")


def test_load_normalizes_label_case(tmp_path):
    rows = load_jsonl(_write(tmp_path, [{"id": "a", "query": "q", "response": "r", "label": "pass"}]))
    assert rows[0]["label"] == "PASS"


def test_split_is_disjoint_complete_and_near_the_fractions():
    rows = _rows(600, 400)
    train, dev, test = split_rows(rows)
    ids = [r["id"] for r in train + dev + test]
    assert sorted(ids) == sorted(r["id"] for r in rows)
    assert len(set(ids)) == len(ids)
    for part, share in ((train, 0.15), (dev, 0.40), (test, 0.45)):
        assert abs(len(part) / 1000 - share) < 0.04
        assert abs(class_counts(part)["FAIL"] / len(part) - 0.4) < 0.06


def test_adding_rows_never_moves_an_existing_row():
    # The review's case: labeling one more row must not move a test row into train.
    before = split_rows(_rows(30, 20))
    after = split_rows(_rows(30, 20) + _rows(5, 5, prefix="new-"))
    for old, new in zip(before, after):
        assert {r["id"] for r in old} <= {r["id"] for r in new}


def test_split_ignores_input_order_and_changes_with_the_seed():
    rows = _rows(30, 20)
    a = [[r["id"] for r in part] for part in split_rows(rows, seed=7)]
    b = [[r["id"] for r in part] for part in split_rows(list(reversed(rows)), seed=7)]
    c = [[r["id"] for r in part] for part in split_rows(rows, seed=8)]
    assert a == b and a != c


def test_split_matches_the_typescript_starter():
    # Same hash in both languages, so the same ids land in the same split.
    train, dev, test = split_rows(_rows(10, 10), seed=42)
    assert [r["id"] for r in train] == ["f5", "p1", "p5", "p9"]
    assert len(dev) == 12 and len(test) == 4


def test_example_dataset_splits_into_both_classes():
    from pathlib import Path

    rows = load_jsonl(Path(__file__).parents[1] / "data" / "labeled.example.jsonl", require_label=True)
    train, dev, test = split_rows(rows)
    assert len(rows) == 40
    for part in (train, dev, test):
        assert class_counts(part)["PASS"] > 0 and class_counts(part)["FAIL"] > 0


def test_dumps_writes_nan_as_null():
    assert dumps({"x": float("nan"), "y": [1.0, float("inf")]}) == '{"x": null, "y": [1.0, null]}'
