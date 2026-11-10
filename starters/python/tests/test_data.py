import json

import pytest

from evals.data import load_jsonl, stratified_split


def _write(tmp_path, rows):
    p = tmp_path / "rows.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return p


def _rows(n_pass, n_fail):
    return [{"id": f"p{i}", "query": "q", "response": "r", "label": "PASS"} for i in range(n_pass)] + [
        {"id": f"f{i}", "query": "q", "response": "r", "label": "FAIL"} for i in range(n_fail)
    ]


def test_load_rejects_bad_rows(tmp_path):
    with pytest.raises(ValueError, match="label"):
        load_jsonl(_write(tmp_path, [{"id": "a", "query": "q", "response": "r", "label": "MAYBE"}]))
    with pytest.raises(ValueError, match="duplicate"):
        load_jsonl(_write(tmp_path, [{"id": "a", "query": "q", "response": "r"}] * 2))
    with pytest.raises(ValueError, match="required"):
        load_jsonl(_write(tmp_path, [{"id": "a", "query": "q", "response": "r"}]), require_label=True)


def test_load_normalizes_label_case(tmp_path):
    rows = load_jsonl(_write(tmp_path, [{"id": "a", "query": "q", "response": "r", "label": "pass"}]))
    assert rows[0]["label"] == "PASS"


def test_split_is_stratified_disjoint_and_complete():
    rows = _rows(60, 40)
    train, dev, test = stratified_split(rows)
    ids = [r["id"] for r in train + dev + test]
    assert sorted(ids) == sorted(r["id"] for r in rows)
    assert len(set(ids)) == len(ids)
    assert (len(train), len(dev), len(test)) == (15, 40, 45)
    for part in (train, dev, test):
        fails = sum(r["label"] == "FAIL" for r in part)
        assert abs(fails / len(part) - 0.4) < 0.05


def test_split_is_stable_for_a_seed_and_ignores_input_order():
    rows = _rows(30, 20)
    a = stratified_split(rows, seed=7)
    b = stratified_split(list(reversed(rows)), seed=7)
    assert [[r["id"] for r in part] for part in a] == [[r["id"] for r in part] for part in b]


def test_example_dataset_loads_and_splits():
    from pathlib import Path

    rows = load_jsonl(Path(__file__).parents[1] / "data" / "labeled.example.jsonl", require_label=True)
    train, dev, test = stratified_split(rows)
    assert len(rows) == 40
    assert {r["label"] for r in dev} == {"PASS", "FAIL"}
    assert {r["label"] for r in test} == {"PASS", "FAIL"}
