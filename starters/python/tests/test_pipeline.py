import json
from pathlib import Path

import pytest

from evals.calibrate import bar_failures, calibrate
from evals.data import load_jsonl, split_rows
from evals.judge import JudgeConfig, JudgeError, Verdict, pick_few_shots
from evals.run_eval import annotation_text, corrected_pass_rate, main as run_eval_main, rogan_gladen, score_rows
from evals.runner import Budget, judge_rows
from tests.fakes import FakeJudge

DATA = Path(__file__).parents[1] / "data"


def test_calibrate_writes_a_run_folder(tmp_path):
    rows = load_jsonl(DATA / "labeled.example.jsonl", require_label=True)
    train, dev, _ = split_rows(rows)
    fails = [r for r in dev if r["label"] == "FAIL"]
    labels = {r["id"]: r["label"] for r in dev}
    labels[fails[0]["id"]] = "PASS"  # the judge misses one real failure
    config = JudgeConfig(rubric="R", few_shots=pick_few_shots(train))
    summary = calibrate(rows, FakeJudge(labels), config, "dev", dev, tmp_path, repeats=2, workers=2,
                        dataset={"sha256": "x"})

    report = summary["report"]
    assert report["pass_recall"] == 1.0
    assert report["fail_recall"] == (len(fails) - 1) / len(fails)
    assert summary["label_flips_across_repeats"] == 0
    assert summary["split_class_counts"] == {"PASS": len(dev) - len(fails), "FAIL": len(fails)}
    disagreements = (tmp_path / "disagreements.jsonl").read_text().splitlines()
    assert [json.loads(x)["id"] for x in disagreements] == [fails[0]["id"]]
    assert not set(summary["judge"]["few_shot_ids"]) & {r["id"] for r in dev}
    # The example set is too small to clear the bar, and the run says why.
    reasons = bar_failures(summary, 0.9, 0.6, 30)
    assert any("FAIL rows in this split" in r for r in reasons)


def test_test_split_skips_disagreements_and_fails_on_missing_verdicts(tmp_path):
    rows = load_jsonl(DATA / "labeled.example.jsonl", require_label=True)
    _, _, test = split_rows(rows)

    class Declines(FakeJudge):
        def __call__(self, row):
            if row["label"] == "FAIL" and row["id"] != next(r["id"] for r in test if r["label"] == "FAIL"):
                raise JudgeError("declined", cost_usd=0.01)
            return super().__call__(row)

    labels = {r["id"]: r["label"] for r in test}
    summary = calibrate(rows, Declines(labels), JudgeConfig(rubric="R"), "test", test, tmp_path)
    assert not (tmp_path / "disagreements.jsonl").exists()
    assert summary["report"]["fail_recall"] == 1.0  # one FAIL row, judged right
    reasons = bar_failures(summary, 0.9, 0.6, 1)
    assert any("no verdict" in r for r in reasons)  # the rest were declined: no pass


def test_score_rows_runs_code_checks_before_the_judge():
    rows = load_jsonl(DATA / "outputs.example.jsonl")
    judge = FakeJudge({"o-008": "FAIL"})
    results = {r["id"]: r for r in score_rows(rows, judge, workers=2)}
    assert "o-009" not in judge.seen and results["o-009"]["passed"] is False  # empty response
    assert "o-010" not in judge.seen and results["o-010"]["passed"] is False  # over max_words
    assert results["o-008"]["judge"] == "FAIL" and results["o-008"]["passed"] is False
    assert results["o-001"]["passed"] is True and results["o-001"]["must_pass"] is True


def test_budget_charges_failed_calls_and_stops_new_ones():
    rows = [{"id": str(i), "query": "q", "response": "r"} for i in range(10)]

    def always_declines(row):
        raise JudgeError("declined", cost_usd=0.01)

    budget = Budget(limit_usd=0.03)
    outcomes = judge_rows(always_declines, rows, workers=1, budget=budget)
    assert round(budget.spent_usd, 2) == 0.03
    assert sum(1 for o in outcomes if o.error == "declined") == 3
    assert all("budget" in o.error for o in outcomes[3:])


def test_budget_rejects_a_value_that_is_not_a_number(monkeypatch):
    monkeypatch.setenv("EVAL_BUDGET_USD", "5 USD")
    with pytest.raises(ValueError, match="number"):
        Budget.from_env()


def test_a_setup_error_stops_new_calls():
    rows = [{"id": str(i), "query": "q", "response": "r"} for i in range(50)]
    calls = []

    def bad_key(row):
        calls.append(row["id"])
        raise PermissionError("bad key")

    with pytest.raises(PermissionError):
        judge_rows(bad_key, rows, workers=1)
    assert len(calls) == 1


def test_correction_applies_to_judged_rows_only():
    # 2 code-check failures, then 6 of 8 judged rows PASS.
    results = [{"passed": False, "judged": False, "check_failures": ["x"], "judge": None} for _ in range(2)]
    results += [{"passed": i < 6, "judged": True, "check_failures": [], "judge": "PASS" if i < 6 else "FAIL"} for i in range(8)]
    expected = rogan_gladen(6 / 8, 0.9, 0.8) * 8 / 10
    assert corrected_pass_rate(results, 0.9, 0.8) == pytest.approx(expected)
    assert rogan_gladen(0.5, float("nan"), 0.8) is None
    assert rogan_gladen(0.5, 0.5, 0.4) is None


def _results_file(path, ids, passed=True):
    path.write_text("".join(json.dumps({"id": i, "passed": passed}) + "\n" for i in ids))
    return path


def test_run_eval_is_invalid_on_a_tiny_or_unpaired_comparison(tmp_path, monkeypatch):
    # The review's case: a candidate sharing one id with the baseline used to ship.
    # A broken comparison fails even when CI only fails on investigate.
    import evals.run_eval as run_eval

    rows = [{"id": f"r{i}", "query": "q", "response": "fine"} for i in range(10)]
    data = tmp_path / "data.jsonl"
    data.write_text("".join(json.dumps(r) + "\n" for r in rows))
    rubric = tmp_path / "rubric.md"
    rubric.write_text("R")
    baseline = _results_file(tmp_path / "base.jsonl", ["r0"])
    monkeypatch.setattr(run_eval, "ClaudeJudge", lambda config: FakeJudge())
    code = run_eval_main(["--data", str(data), "--rubric", str(rubric), "--baseline", str(baseline),
                          "--fail-on", "investigate", "--out", str(tmp_path / "runs")])
    summary = json.loads(next((tmp_path / "runs").glob("*-eval/summary.json")).read_text())
    assert code == 1 and summary["verdict"] == "invalid"
    assert summary["vs_baseline"]["unpaired_ids"] == [f"r{i}" for i in range(1, 10)]
    assert any("paired rows" in h for h in summary["problems"])
    assert (next((tmp_path / "runs").glob("*-eval")) / "results.jsonl").exists()


def test_run_eval_checks_inputs_before_spending(tmp_path, monkeypatch, capsys):
    import evals.run_eval as run_eval

    judge = FakeJudge()
    monkeypatch.setattr(run_eval, "ClaudeJudge", lambda config: judge)
    data = tmp_path / "data.jsonl"
    data.write_text(json.dumps({"id": "a", "query": "q", "response": "r"}) + "\n")
    rubric = tmp_path / "rubric.md"
    rubric.write_text("R")
    bad = tmp_path / "summary.json"
    bad.write_text('{"not": "a results file"}')
    code = run_eval_main(["--data", str(data), "--rubric", str(rubric), "--baseline", str(bad)])
    assert code == 2 and judge.seen == []


def _write_rows(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def _run(tmp_path, monkeypatch, args, judge=None):
    import evals.run_eval as run_eval

    monkeypatch.setattr(run_eval, "ClaudeJudge", lambda config: judge or FakeJudge())
    out = tmp_path / "runs"
    code = run_eval_main(args + ["--out", str(out)])
    summary = json.loads(sorted(out.glob("*-eval/summary.json"))[-1].read_text())
    return code, summary


def test_run_eval_is_invalid_when_the_candidate_drops_baseline_rows(tmp_path, monkeypatch):
    # The re-review's case: 500 baseline rows, the candidate produced only 400.
    rows = [{"id": f"r{i}", "query": "q", "response": "fine"} for i in range(400)]
    data = _write_rows(tmp_path / "data.jsonl", rows)
    rubric = tmp_path / "rubric.md"
    rubric.write_text("R")
    base = _write_rows(tmp_path / "base.jsonl", [{"id": f"r{i}", "passed": True} for i in range(500)])
    code, summary = _run(tmp_path, monkeypatch, ["--data", str(data), "--rubric", str(rubric),
                                                 "--baseline", str(base), "--indifference", "0.2",
                                                 "--fail-on", "investigate"])
    assert code == 1 and summary["verdict"] == "invalid"
    assert summary["vs_baseline"]["band"] == "ship"  # the interval alone would have shipped
    assert len(summary["vs_baseline"]["missing_ids"]) == 100

    # A missing regression row blocks.
    base = _write_rows(tmp_path / "base2.jsonl",
                       [{"id": f"r{i}", "passed": True, "must_pass": i == 450} for i in range(500)])
    code, summary = _run(tmp_path, monkeypatch, ["--data", str(data), "--rubric", str(rubric),
                                                 "--baseline", str(base), "--indifference", "0.2"])
    assert code == 1 and summary["verdict"] == "block" and "r450" in summary["must_pass_failures"]


def test_run_eval_refuses_a_calibration_that_did_not_pass(tmp_path, monkeypatch):
    from evals.judge import JudgeConfig

    data = _write_rows(tmp_path / "data.jsonl", [{"id": "a", "query": "q", "response": "r"}])
    rubric = tmp_path / "rubric.md"
    rubric.write_text("R")
    failed = {"judge": JudgeConfig(rubric="R").describe(), "split": "test",
              "report": {"pass_recall": 0.9, "fail_recall": 0.143}, "bar": {"passed": False}}
    cal = tmp_path / "summary.json"
    cal.write_text(json.dumps(failed))
    code, summary = _run(tmp_path, monkeypatch, ["--data", str(data), "--rubric", str(rubric), "--calibration", str(cal),
                                                 "--fail-on", "investigate"])
    assert code == 1 and summary["verdict"] == "invalid"
    assert "pass_rate_corrected" not in summary
    assert any("passing test read" in h for h in summary["problems"])


def test_calibrate_records_whether_it_passed(tmp_path, monkeypatch):
    import evals.calibrate as calibrate_mod

    monkeypatch.setattr(calibrate_mod, "ClaudeJudge", lambda config: FakeJudge())
    code = calibrate_mod.main(["--data", str(DATA / "labeled.example.jsonl"), "--rubric", str(DATA.parent / "rubrics" / "refund_policy.md"),
                               "--out", str(tmp_path)])
    summary = json.loads(next(tmp_path.glob("*/summary.json")).read_text())
    assert code == 1 and summary["bar"]["passed"] is False and summary["bar"]["reasons"]


def test_calibrate_rejects_zero_repeats(capsys):
    import evals.calibrate as calibrate_mod

    with pytest.raises(SystemExit):
        calibrate_mod.main(["--data", "x", "--rubric", "y", "--repeats", "0"])


def test_fail_on_lets_ci_report_a_hold_without_failing(tmp_path, monkeypatch, capsys):
    # 40 unchanged rows: a clean comparison whose interval is too wide to ship on.
    rows = [{"id": f"r{i}", "query": "q", "response": "fine"} for i in range(40)]
    data = _write_rows(tmp_path / "data.jsonl", rows)
    rubric = tmp_path / "rubric.md"
    rubric.write_text("R")
    base = _write_rows(tmp_path / "base.jsonl", [{"id": f"r{i}", "passed": True} for i in range(40)])
    code, summary = _run(tmp_path, monkeypatch, ["--data", str(data), "--rubric", str(rubric), "--baseline", str(base)])
    assert summary["verdict"] == "hold" and summary["problems"] == [] and code == 1
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    capsys.readouterr()
    code, summary = _run(tmp_path, monkeypatch, ["--data", str(data), "--rubric", str(rubric), "--baseline", str(base),
                                                 "--fail-on", "investigate"])
    assert summary["verdict"] == "hold" and code == 0
    annotation = [line for line in capsys.readouterr().out.splitlines() if line.startswith("::")]
    # GitHub decodes %25 back to %.
    assert annotation == ["::warning title=Release gate HOLD::delta +0.000, 95%25 CI [-0.088, +0.088], n=40; "
                          "read the rows in results.jsonl (the eval-run artifact in CI)"]


def test_annotations_escape_text_from_the_data():
    assert annotation_text("id 100%\nnext") == "id 100%25%0Anext"
