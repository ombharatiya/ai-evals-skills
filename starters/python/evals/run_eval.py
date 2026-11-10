"""Score new outputs and gate a release on them.

    python -m evals.run_eval --data data/outputs.example.jsonl --rubric rubrics/refund_policy.md \\
        --calibration runs/<calibrate-test-run>/summary.json --baseline runs/<previous-eval>/results.jsonl

Each row runs the code checks first. A failed check fails the row and skips
the judge. Rows that pass every check go to the calibrated judge. The run
reports the pass-all rate with a Wilson interval and a per-check drill-down,
and writes runs/<timestamp>-eval/ with results.jsonl and summary.json.

With --baseline (an earlier results.jsonl on the same ids), it computes the
paired difference and its interval and prints a ship / hold / investigate /
block verdict. Exit code 0 for ship (or a first run with no baseline), 1
otherwise, so CI can use it directly. Rows with no verdict turn a ship into
a hold: nothing passes silently.
Rows with "must_pass": true are the regression set: one failure blocks.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import anthropic

from .checks import run_checks
from .data import load_jsonl, stratified_split, write_jsonl
from .gate import band, paired_delta
from .judge import ClaudeJudge, Judge, JudgeConfig, pick_few_shots
from .metrics import wilson
from .runner import Budget, judge_rows

# A wrong key or model raises an API error. A missing key raises a TypeError in the
# SDK, matched by its message so real TypeErrors still surface.
SETUP_ERRORS = (anthropic.AuthenticationError, anthropic.PermissionDeniedError,
                anthropic.NotFoundError, anthropic.BadRequestError, TypeError)


def is_setup_error(exc: Exception) -> bool:
    return not isinstance(exc, TypeError) or "authentication" in str(exc).lower()


def score_rows(rows: list[dict], judge: Judge, workers: int = 4, budget: Budget | None = None) -> list[dict]:
    results: dict[str, dict] = {}
    to_judge = []
    for row in rows:
        checks = run_checks(row)
        failed = [c for c in checks if not c.passed]
        results[row["id"]] = {
            "id": row["id"],
            "must_pass": bool(row.get("must_pass")),
            "checks": {c.name: c.passed for c in checks},
            "check_failures": [f"{c.name}: {c.detail}" for c in failed],
            "judge": None,
            "critique": None,
            "served_by": None,
            "error": None,
        }
        if not failed:
            to_judge.append(row)
    for outcome in judge_rows(judge, to_judge, workers=workers, budget=budget):
        r = results[outcome.row["id"]]
        if outcome.verdict:
            r.update(judge=outcome.verdict.label, critique=outcome.verdict.critique,
                     served_by=outcome.verdict.model, fallback_used=outcome.verdict.fallback_used)
        else:
            r["error"] = outcome.error
    for r in results.values():
        if r["error"]:
            r["passed"] = None  # unknown: excluded from rates, reported separately
        else:
            r["passed"] = not r["check_failures"] and r["judge"] == "PASS"
    return [results[row["id"]] for row in rows]


def corrected_rate(observed: float, pass_recall: float, fail_recall: float) -> float | None:
    """Rogan-Gladen correction for judge error (point estimate only)."""
    denom = pass_recall + fail_recall - 1
    if denom <= 0:
        return None
    return min(1.0, max(0.0, (observed + fail_recall - 1) / denom))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="JSONL of outputs to score (id, query, response)")
    p.add_argument("--rubric", required=True)
    p.add_argument("--labeled", default=None, help="labeled JSONL whose train split supplies few-shots")
    p.add_argument("--calibration", default=None, help="calibrate summary.json, used to correct the pass rate")
    p.add_argument("--baseline", default=None, help="earlier results.jsonl to compare against")
    p.add_argument("--window", type=float, default=0.02, help="indifference / minor-regression window")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--model", default=None)
    p.add_argument("--effort", default=None)
    p.add_argument("--out", default="runs")
    args = p.parse_args(argv)

    rows = load_jsonl(args.data)
    config = JudgeConfig(rubric=Path(args.rubric).read_text(encoding="utf-8"))
    if args.model:
        config.model = args.model
    if args.effort:
        config.effort = args.effort
    if args.labeled:
        train, _, _ = stratified_split(load_jsonl(args.labeled, require_label=True))
        config.few_shots = pick_few_shots(train)

    calibration = json.loads(Path(args.calibration).read_text()) if args.calibration else None
    if calibration and calibration["judge"] != config.describe():
        print("WARNING: this judge (model, effort, rubric or few-shots) differs from the calibrated one.")
        print("         The calibration numbers do not apply to it. Calibrate again.")

    budget = Budget.from_env()
    try:
        results = score_rows(rows, ClaudeJudge(config), workers=args.workers, budget=budget)
    except SETUP_ERRORS as exc:
        if not is_setup_error(exc):
            raise
        print(f"Can't run the judge: {exc}", file=sys.stderr)
        print("Set ANTHROPIC_API_KEY (or run `ant auth login`) and check --model.", file=sys.stderr)
        return 2

    known = [r for r in results if r["passed"] is not None]
    passed = sum(1 for r in known if r["passed"])
    rate = passed / len(known) if known else float("nan")
    lo, hi = wilson(passed, len(known))
    check_fails = Counter(name for r in results for name, ok in r["checks"].items() if not ok)
    judge_fails = sum(1 for r in results if r["judge"] == "FAIL")
    must_fail = [r["id"] for r in results if r["must_pass"] and r["passed"] is False]
    errors = [r for r in results if r["error"]]

    summary = {
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "judge": config.describe(),
        "n": len(results),
        "scored": len(known),
        "pass_rate": rate,
        "pass_rate_ci": [lo, hi],
        "check_failures": dict(check_fails),
        "judge_failures": judge_fails,
        "must_pass_failures": must_fail,
        "errors": len(errors),
        "estimated_cost_usd": round(budget.spent_usd, 4),
    }
    if calibration:
        cr = calibration["report"]
        summary["pass_rate_corrected"] = corrected_rate(rate, cr["pass_recall"], cr["fail_recall"])

    # Without a baseline there is nothing to compare: this run becomes the baseline.
    verdict = "baseline"
    if args.baseline:
        base = {}
        for line in Path(args.baseline).read_text().splitlines():
            if line.strip():
                b = json.loads(line)
                if b.get("passed") is not None:
                    base[b["id"]] = b["passed"]
        cand = {r["id"]: r["passed"] for r in known}
        d = paired_delta(base, cand)
        verdict = band(d, window=args.window)
        summary["vs_baseline"] = {"n": d.n, "baseline": d.baseline_rate, "candidate": d.candidate_rate,
                                  "delta": d.delta, "ci": list(d.ci), "band": verdict}
    if must_fail:
        verdict = "block"
    if errors and verdict in ("ship", "baseline"):
        verdict = "hold"
    summary["verdict"] = verdict

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out) / f"{stamp}-eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "results.jsonl", results)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(f"pass-all rate {rate:.3f}  95% CI [{lo:.2f}, {hi:.2f}]  n={len(known)}  judge={config.model}@{config.effort}")
    if "pass_rate_corrected" in summary and summary["pass_rate_corrected"] is not None:
        print(f"  corrected for judge error: {summary['pass_rate_corrected']:.3f} (point estimate)")
    for name, count in sorted(check_fails.items()):
        print(f"  code check {name}: {count} failed")
    print(f"  judge: {judge_fails} failed")
    if "vs_baseline" in summary:
        v = summary["vs_baseline"]
        print(f"  vs baseline: {v['baseline']:.3f} -> {v['candidate']:.3f}  delta {v['delta']:+.3f}  "
              f"95% CI [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]  n={v['n']}")
    if must_fail:
        print(f"  regression set failures: {', '.join(must_fail)}")
    if errors:
        print(f"  {len(errors)} rows have no verdict (refusal, budget, or bad output); see results.jsonl")
    print(f"verdict: {verdict.upper()}   wrote {out_dir}")
    return 0 if verdict in ("ship", "baseline") else 1


if __name__ == "__main__":
    sys.exit(main())
