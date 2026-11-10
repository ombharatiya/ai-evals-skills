import json

from evals.calibrate import calibrate
from evals.data import load_jsonl, stratified_split
from evals.judge import JudgeConfig, pick_few_shots
from evals.run_eval import corrected_rate, score_rows
from evals.runner import Budget, judge_rows
from tests.fakes import FakeJudge

from pathlib import Path

DATA = Path(__file__).parents[1] / "data"


def test_calibrate_writes_a_run_folder(tmp_path):
    rows = load_jsonl(DATA / "labeled.example.jsonl", require_label=True)
    train, dev, _ = stratified_split(rows)
    # A judge that agrees with the humans everywhere except one real failure.
    fails = [r for r in dev if r["label"] == "FAIL"]
    labels = {r["id"]: r["label"] for r in dev}
    labels[fails[0]["id"]] = "PASS"
    config = JudgeConfig(rubric="R", few_shots=pick_few_shots(train))
    summary = calibrate(rows, FakeJudge(labels), config, "dev", dev, tmp_path, repeats=2, workers=2)

    report = summary["report"]
    assert report["pass_recall"] == 1.0
    assert report["fail_recall"] == (len(fails) - 1) / len(fails)
    assert summary["label_flips_across_repeats"] == 0
    assert summary["judge"]["few_shot_ids"] == [r["id"] for r in config.few_shots]
    disagreements = (tmp_path / "disagreements.jsonl").read_text().splitlines()
    assert [json.loads(x)["id"] for x in disagreements] == [fails[0]["id"]]
    # Few-shots come from train and never from the split being scored.
    assert not set(summary["judge"]["few_shot_ids"]) & {r["id"] for r in dev}


def test_score_rows_runs_code_checks_before_the_judge():
    rows = load_jsonl(DATA / "outputs.example.jsonl")
    judge = FakeJudge({"o-008": "FAIL"})
    results = {r["id"]: r for r in score_rows(rows, judge, workers=2)}
    assert "o-009" not in judge.seen and results["o-009"]["passed"] is False  # empty response
    assert "o-010" not in judge.seen and results["o-010"]["passed"] is False  # over max_words
    assert results["o-008"]["judge"] == "FAIL" and results["o-008"]["passed"] is False
    assert results["o-001"]["passed"] is True and results["o-001"]["must_pass"] is True


def test_budget_stops_new_calls():
    rows = [{"id": str(i), "query": "q", "response": "r"} for i in range(10)]
    budget = Budget(limit_usd=0.03)
    outcomes = judge_rows(FakeJudge(cost=0.01), rows, workers=1, budget=budget)
    ran = [o for o in outcomes if o.verdict]
    assert len(ran) == 3
    assert all(o.error and "budget" in o.error for o in outcomes if not o.verdict)


def test_rogan_gladen_correction():
    # A judge with 0.9 PASS recall and 0.8 FAIL recall reading 0.75 observed -> 0.7857 true.
    assert abs(corrected_rate(0.75, 0.9, 0.8) - (0.75 + 0.8 - 1) / 0.7) < 1e-12
    assert corrected_rate(0.5, 0.5, 0.4) is None
