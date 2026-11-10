"""Score new outputs and gate a release on them.

    python -m evals.run_eval --data data/outputs.example.jsonl --rubric rubrics/refund_policy.md \\
        --labeled data/labeled.example.jsonl --calibration runs/<calibrate-test-run>/summary.json \\
        --baseline runs/<previous-eval>/results.jsonl

Each row runs the code checks first. A failed check fails the row and skips
the judge. Rows that pass every check go to the calibrated judge. The run
reports the pass-all rate with a Wilson interval and a per-check drill-down,
and writes runs/<timestamp>-eval/ with results.jsonl, summary.json and a copy
of the rubric. results.jsonl is written as soon as scoring finishes.

With --baseline (an earlier results.jsonl), it pairs the two runs on their
shared ids, computes the difference in pass rate with Newcombe's paired
interval, and prints ship, hold, investigate or block. Exit code 0 for ship
(or a first run with no baseline), 1 otherwise, so CI can use it directly.

Nothing ships silently:
- any failing row marked "must_pass": true (the regression set) blocks
- rows with no verdict, ids missing from the baseline, fewer than
  --min-paired paired rows, or a judge that differs from the calibrated one
  turn a ship into a hold
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
from .data import dumps, file_sha256, load_jsonl, split_rows, write_jsonl
from .gate import band, no_change_halfwidth, paired_delta, rows_for_indifference
from .judge import ClaudeJudge, Judge, JudgeConfig, pick_few_shots, price_known
from .metrics import wilson
from .runner import Budget, judge_rows

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
            "judged": False,
            "judge": None,
            "critique": None,
            "served_by": None,
            "fallback_used": None,
            "error": None,
        }
        if not failed:
            to_judge.append(row)
    for outcome in judge_rows(judge, to_judge, workers=workers, budget=budget):
        r = results[outcome.row["id"]]
        r["judged"] = True
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


def rogan_gladen(observed: float, pass_recall: float | None, fail_recall: float | None) -> float | None:
    """Rogan-Gladen correction for judge error (point estimate only)."""
    values = (observed, pass_recall, fail_recall)
    if any(v is None or v != v for v in values):
        return None
    denom = pass_recall + fail_recall - 1
    if denom <= 0:
        return None
    return min(1.0, max(0.0, (observed + fail_recall - 1) / denom))


def corrected_pass_rate(results: list[dict], pass_recall: float | None, fail_recall: float | None) -> float | None:
    """Correct only the judged rows, then put the code-check failures back in.

    Rows that failed a code check never reached the judge, so judge error
    doesn't apply to them: they stay failed.
    """
    known = [r for r in results if r["passed"] is not None]
    judged = [r for r in known if r["judged"] and not r["check_failures"]]
    if not known or not judged:
        return None
    observed = sum(1 for r in judged if r["judge"] == "PASS") / len(judged)
    corrected = rogan_gladen(observed, pass_recall, fail_recall)
    if corrected is None:
        return None
    return corrected * len(judged) / len(known)


def read_baseline(path: str) -> tuple[dict[str, bool], set[str]]:
    """The baseline's pass/fail per id, and the ids it marks must_pass."""
    base: dict[str, bool] = {}
    must_pass: set[str] = set()
    for lineno, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{lineno}: not valid JSON; --baseline takes a results.jsonl from an earlier run") from exc
        if not isinstance(row, dict) or "id" not in row or "passed" not in row:
            raise ValueError(f"{path}:{lineno}: missing 'id' or 'passed'; --baseline takes a results.jsonl from an earlier run")
        if row["passed"] is not None:
            base[row["id"]] = bool(row["passed"])
        if row.get("must_pass"):
            must_pass.add(row["id"])
    if not base:
        raise ValueError(f"{path}: no scored rows")
    return base, must_pass


def read_calibration(path: str) -> dict:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: not valid JSON; --calibration takes a calibrate summary.json") from exc
    if not isinstance(data, dict) or "judge" not in data or "report" not in data:
        raise ValueError(f"{path}: --calibration takes the summary.json from a calibrate run")
    return data


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="JSONL of outputs to score (id, query, response)")
    p.add_argument("--rubric", required=True)
    p.add_argument("--labeled", default=None, help="labeled JSONL whose train split supplies few-shots")
    p.add_argument("--seed", type=int, default=42, help="split seed; match the calibration run")
    p.add_argument("--few-shots", type=int, default=4, help="few-shot count; match the calibration run")
    p.add_argument("--calibration", default=None, help="calibrate summary.json, used to correct the pass rate")
    p.add_argument("--baseline", default=None, help="earlier results.jsonl to compare against")
    p.add_argument("--indifference", type=float, default=0.01, help="ship when the whole interval is within this")
    p.add_argument("--minor-regression", type=float, default=0.02, help="a sure drop up to this is investigate")
    p.add_argument("--min-paired", type=int, default=30, help="fewest paired rows a ship can rest on")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--model", default=None)
    p.add_argument("--effort", default=None)
    p.add_argument("--out", default="runs")
    args = p.parse_args(argv)

    # Check every input before spending anything on judge calls.
    try:
        budget = Budget.from_env()
        rows = load_jsonl(args.data)
        base, base_must_pass = read_baseline(args.baseline) if args.baseline else (None, set())
        calibration = read_calibration(args.calibration) if args.calibration else None
        rubric = Path(args.rubric).read_text(encoding="utf-8")
        labeled = load_jsonl(args.labeled, require_label=True) if args.labeled else None
    except (ValueError, OSError) as exc:
        print(exc, file=sys.stderr)
        return 2
    config = JudgeConfig(rubric=rubric)
    if args.model:
        config.model = args.model
    if args.effort:
        config.effort = args.effort
    if labeled is not None:
        train, _, _ = split_rows(labeled, seed=args.seed)
        config.few_shots = pick_few_shots(train, n=args.few_shots)
    for label in ("PASS", "FAIL"):
        if labeled is not None and config.few_shots and label not in {r.get("label") for r in config.few_shots}:
            print(f"warning: the few-shots have no {label} example: train holds none; label more rows", file=sys.stderr)
    if budget.limit_usd is not None and not price_known(config.model):
        print(f"EVAL_BUDGET_USD is set but there's no price for {config.model} in evals/judge.py PRICES.", file=sys.stderr)
        return 2

    holds: list[str] = []
    calibration_ok = False
    flip_rate = 0.0
    if calibration:
        bar = calibration.get("bar") or {}
        if calibration["judge"] != config.describe():
            holds.append("this judge (model, effort, prompt or few-shots) differs from the calibrated one; calibrate again")
        elif calibration.get("split") != "test" or not bar.get("passed"):
            holds.append("the calibration record isn't a passing test read; calibrate on dev until it clears the bar, then read test")
        else:
            calibration_ok = True
        flips, scored = calibration.get("label_flips_across_repeats"), calibration.get("report", {}).get("n")
        if flips is not None and scored:
            flip_rate = flips / scored

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out) / f"{stamp}-eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "rubric.md").write_text(rubric)

    try:
        results = score_rows(rows, ClaudeJudge(config), workers=args.workers, budget=budget)
    except SETUP_ERRORS as exc:
        if not is_setup_error(exc):
            raise
        print(f"Can't run the judge: {exc}", file=sys.stderr)
        print("Set ANTHROPIC_API_KEY (or run `ant auth login`) and check --model.", file=sys.stderr)
        return 2
    write_jsonl(out_dir / "results.jsonl", results)  # paid for: save before anything else can fail

    known = [r for r in results if r["passed"] is not None]
    passed = sum(1 for r in known if r["passed"])
    rate = passed / len(known) if known else float("nan")
    lo, hi = wilson(passed, len(known))
    check_fails = Counter(name for r in results for name, ok in r["checks"].items() if not ok)
    judge_fails = sum(1 for r in results if r["judge"] == "FAIL")
    must_fail = [r["id"] for r in results if r["must_pass"] and r["passed"] is False]
    errors = [r for r in results if r["error"]]
    fallbacks = sum(1 for r in results if r.get("fallback_used"))
    if errors:
        holds.append(f"{len(errors)} rows have no verdict (refusal, budget or bad output); see results.jsonl")
    if fallbacks:
        holds.append(f"{fallbacks} verdicts came from a fallback model, not the calibrated judge")

    summary: dict = {
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "judge": config.describe(),
        "dataset": {"path": args.data, "sha256": file_sha256(args.data)},
        "n": len(results),
        "scored": len(known),
        "pass_rate": rate,
        "pass_rate_ci": [lo, hi],
        "check_failures": dict(check_fails),
        "judge_failures": judge_fails,
        "fallback_verdicts": fallbacks,
        "must_pass_failures": must_fail,
        "errors": len(errors),
        "estimated_cost_usd": round(budget.spent_usd, 4),
    }
    if calibration_ok:
        cr = calibration["report"]
        summary["pass_rate_corrected"] = corrected_pass_rate(results, cr.get("pass_recall"), cr.get("fail_recall"))

    # Without a baseline there is nothing to compare: this run becomes the baseline.
    verdict = "baseline"
    if base is not None:
        cand = {r["id"]: bool(r["passed"]) for r in known}
        unpaired = sorted(set(cand) - set(base))
        if unpaired:
            holds.append(f"{len(unpaired)} ids aren't in the baseline and weren't compared: {', '.join(unpaired[:10])}")
        absent = sorted(set(base) - {r["id"] for r in results})
        if absent:
            holds.append(f"{len(absent)} baseline ids are missing from this run: {', '.join(absent[:10])}")
        absent_must = sorted(set(absent) & base_must_pass)
        if absent_must:
            # A regression row the candidate didn't produce can't count as passing.
            must_fail = must_fail + absent_must
            summary["must_pass_failures"] = must_fail
        if not set(cand) & set(base):
            print("The baseline shares no scored ids with this run.", file=sys.stderr)
            (out_dir / "summary.json").write_text(dumps(summary, indent=2) + "\n")
            return 1
        d = paired_delta(base, cand)
        verdict = band(d, indifference=args.indifference, minor_regression=args.minor_regression)
        if d.n < args.min_paired:
            holds.append(f"only {d.n} paired rows; a ship needs at least {args.min_paired}")
        summary["vs_baseline"] = {
            "n": d.n, "baseline_rate": d.baseline_rate, "candidate_rate": d.candidate_rate,
            "delta": d.delta, "ci": list(d.ci), "band": verdict, "unpaired_ids": unpaired, "missing_ids": absent,
            "paired_counts": {"both_pass": d.both_pass, "baseline_only": d.baseline_only,
                              "candidate_only": d.candidate_only, "both_fail": d.both_fail},
        }
    # The gate set's resolution: how wide the interval is for an unchanged
    # candidate, and how many rows the indifference window needs.
    res_n, res_rate = (summary["vs_baseline"]["n"], summary["vs_baseline"]["baseline_rate"]) if "vs_baseline" in summary else (len(known), rate)
    notes: list[str] = []
    if res_n and res_rate == res_rate:
        half = no_change_halfwidth(res_n, res_rate, flip_rate)
        needed = rows_for_indifference(res_rate, args.indifference, start=res_n, flip_rate=flip_rate)
        summary["resolution"] = {"n": res_n, "flip_rate": flip_rate, "no_change_halfwidth": half,
                                 "rows_for_indifference": needed}
        flips_text = (f" with the judge flipping {flip_rate:.1%} of labels between runs" if flip_rate
                      else " (a floor: it assumes the judge never flips a label; calibrate with --repeats 2 to measure flips)")
        if half > args.indifference:
            notes.append(
                f"with {res_n} rows at this pass rate{flips_text}, an unchanged candidate's interval is about +-{half:.3f}, "
                f"wider than the {args.indifference:.3f} indifference window, so no candidate can ship by staying the same. "
                + (f"Grow the gate set to about {needed} rows, or pass --indifference {half:.3f} or more" if needed
                   else f"Pass --indifference {half:.3f} or more")
            )
    summary["notes"] = notes
    if must_fail:
        verdict = "block"
    elif holds and verdict in ("ship", "baseline"):
        verdict = "hold"
    summary["verdict"] = verdict
    summary["holds"] = holds
    (out_dir / "summary.json").write_text(dumps(summary, indent=2) + "\n")

    print(f"pass-all rate {rate:.3f}  95% CI [{lo:.2f}, {hi:.2f}]  n={len(known)}  judge={config.model}@{config.effort}")
    if summary.get("pass_rate_corrected") is not None:
        print(f"  corrected for judge error: {summary['pass_rate_corrected']:.3f} (point estimate)")
    for name, count in sorted(check_fails.items()):
        print(f"  code check {name}: {count} failed")
    print(f"  judge: {judge_fails} failed")
    if "vs_baseline" in summary:
        v = summary["vs_baseline"]
        print(f"  vs baseline: {v['baseline_rate']:.3f} -> {v['candidate_rate']:.3f}  delta {v['delta']:+.3f}  "
              f"95% CI [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]  n={v['n']}  band={v['band']}")
    if must_fail:
        print(f"  regression set failures: {', '.join(must_fail)}")
    for reason in holds:
        print(f"  hold: {reason}")
    for note in notes:
        print(f"  note: {note}")
    print(f"  estimated cost ${summary['estimated_cost_usd']:.2f}")
    print(f"verdict: {verdict.upper()}   wrote {out_dir}")
    return 0 if verdict in ("ship", "baseline") else 1


if __name__ == "__main__":
    sys.exit(main())
