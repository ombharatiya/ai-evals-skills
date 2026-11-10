"""Calibrate a judge against human labels.

    python -m evals.calibrate --data data/labeled.example.jsonl --rubric rubrics/refund_policy.md

Splits the labeled set into train / dev / test (stratified, seeded), builds
few-shots from train, runs the judge on dev, and reports PASS recall, FAIL
recall, Cohen's kappa and agreement, each with N. Read the disagreements,
edit the rubric, run again. When dev clears the bar, run once with
`--split test` and keep that number. Reading test again after more edits
turns it into a second dev set.

Every run writes runs/<timestamp>-calibrate-<split>/ with summary.json,
predictions.jsonl, disagreements.jsonl and a copy of the rubric.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import anthropic

from .data import load_jsonl, stratified_split, write_jsonl
from .judge import ClaudeJudge, Judge, JudgeConfig, pick_few_shots
from .metrics import judge_report
from .runner import Budget, judge_rows

# A wrong key or model raises an API error. A missing key raises a TypeError in the
# SDK, matched by its message so real TypeErrors still surface.
SETUP_ERRORS = (anthropic.AuthenticationError, anthropic.PermissionDeniedError,
                anthropic.NotFoundError, anthropic.BadRequestError, TypeError)


def is_setup_error(exc: Exception) -> bool:
    return not isinstance(exc, TypeError) or "authentication" in str(exc).lower()


def calibrate(
    rows: list[dict],
    judge: Judge,
    config: JudgeConfig,
    split_name: str,
    split_rows: list[dict],
    out_dir: Path,
    repeats: int = 1,
    workers: int = 4,
    budget: Budget | None = None,
) -> dict:
    budget = budget or Budget()
    runs = [judge_rows(judge, split_rows, workers=workers, budget=budget) for _ in range(repeats)]
    first = runs[0]
    errors = [o for o in first if o.error]
    scored = [o for o in first if o.verdict]
    report = judge_report([o.row["label"] for o in scored], [o.verdict.label for o in scored])

    flips = 0
    if repeats > 1:
        by_id = [{o.row["id"]: o.verdict.label for o in run if o.verdict} for run in runs]
        common = set.intersection(*(set(m) for m in by_id))
        flips = sum(1 for i in common if len({m[i] for m in by_id}) > 1)

    predictions = [
        {
            "id": o.row["id"],
            "human": o.row["label"],
            "judge": o.verdict.label if o.verdict else None,
            "critique": o.verdict.critique if o.verdict else None,
            "served_by": o.verdict.model if o.verdict else None,
            "fallback_used": o.verdict.fallback_used if o.verdict else None,
            "error": o.error,
            "query": o.row["query"],
            "response": o.row["response"],
            "human_critique": o.row.get("critique"),
        }
        for o in first
    ]
    disagreements = [p for p in predictions if p["judge"] and p["judge"] != p["human"]]

    summary = {
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "split": split_name,
        "judge": config.describe(),
        "dataset_rows": len(rows),
        "report": asdict(report),
        "errors": len(errors),
        "fallback_verdicts": sum(1 for o in scored if o.verdict.fallback_used),
        "repeats": repeats,
        "label_flips_across_repeats": flips if repeats > 1 else None,
        "estimated_cost_usd": round(budget.spent_usd, 4),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out_dir / "rubric.md").write_text(config.rubric)
    write_jsonl(out_dir / "predictions.jsonl", predictions)
    write_jsonl(out_dir / "disagreements.jsonl", disagreements)
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="labeled JSONL (id, query, response, label)")
    p.add_argument("--rubric", required=True, help="rubric file: role, definitions, PASS/FAIL criteria")
    p.add_argument("--split", choices=["dev", "test"], default="dev")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--few-shots", type=int, default=4, help="few-shot examples from train (0 to disable)")
    p.add_argument("--repeats", type=int, default=1, help="run the split N times and count label flips")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--min-recall", type=float, default=0.90, help="bar for PASS and FAIL recall")
    p.add_argument("--min-kappa", type=float, default=0.60)
    p.add_argument("--model", default=None, help="judge model (default: $JUDGE_MODEL or claude-opus-5-5)")
    p.add_argument("--effort", default=None, help="judge effort (default: $JUDGE_EFFORT or medium)")
    p.add_argument("--out", default="runs")
    args = p.parse_args(argv)

    rows = load_jsonl(args.data, require_label=True)
    train, dev, test = stratified_split(rows, seed=args.seed)
    config = JudgeConfig(rubric=Path(args.rubric).read_text(encoding="utf-8"))
    if args.model:
        config.model = args.model
    if args.effort:
        config.effort = args.effort
    config.few_shots = pick_few_shots(train, n=args.few_shots)
    split_rows = dev if args.split == "dev" else test

    if args.split == "test":
        print("Reading the test split. Do this once, after dev clears the bar; keep this number.")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out) / f"{stamp}-calibrate-{args.split}"
    try:
        summary = calibrate(
            rows, ClaudeJudge(config), config, args.split, split_rows, out_dir,
            repeats=args.repeats, workers=args.workers, budget=Budget.from_env(),
        )
    except SETUP_ERRORS as exc:
        if not is_setup_error(exc):
            raise
        print(f"Can't run the judge: {exc}", file=sys.stderr)
        print("Set ANTHROPIC_API_KEY (or run `ant auth login`) and check --model.", file=sys.stderr)
        return 2

    r = summary["report"]
    c = r["confusion"]
    print(f"split={args.split}  n={r['n']}  train/dev/test={len(train)}/{len(dev)}/{len(test)}  judge={config.model}@{config.effort}")
    print(f"  PASS recall (TPR) {r['pass_recall']:.3f}  95% CI [{r['pass_recall_ci'][0]:.2f}, {r['pass_recall_ci'][1]:.2f}]  n={c['tp'] + c['fn']}")
    print(f"  FAIL recall (TNR) {r['fail_recall']:.3f}  95% CI [{r['fail_recall_ci'][0]:.2f}, {r['fail_recall_ci'][1]:.2f}]  n={c['tn'] + c['fp']}")
    print(f"  kappa {r['kappa']:.3f}   agreement {r['agreement']:.3f} (not the headline)")
    print(f"  confusion TP={c['tp']} FN={c['fn']} FP={c['fp']} TN={c['tn']}")
    if summary["errors"]:
        print(f"  {summary['errors']} rows had no verdict; see predictions.jsonl")
    if summary["fallback_verdicts"]:
        print(f"  {summary['fallback_verdicts']} verdicts came from a fallback model, not the calibrated judge")
    if summary["label_flips_across_repeats"]:
        print(f"  {summary['label_flips_across_repeats']} labels flipped between repeats: treat small moves as noise")
    print(f"  wrote {out_dir}")

    ok = r["pass_recall"] >= args.min_recall and r["fail_recall"] >= args.min_recall and r["kappa"] >= args.min_kappa
    if ok and args.split == "dev":
        print("Dev clears the bar. Run once with --split test and keep that number.")
    elif not ok:
        print("Not there yet. Read disagreements.jsonl, change one thing in the rubric, run again.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
