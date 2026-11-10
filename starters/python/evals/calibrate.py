"""Calibrate a judge against human labels.

    python -m evals.calibrate --data data/labeled.example.jsonl --rubric rubrics/refund_policy.md

Splits the labeled set into train / dev / test by a stable hash of each id,
builds few-shots from train, runs the judge on dev, and reports PASS recall,
FAIL recall, Cohen's kappa and agreement, each with N. Read the
disagreements, edit the rubric, run again. When dev clears the bar, run once
with `--split test` and keep that number. Reading test again after more
edits turns it into a second dev set.

The bar: every split row gets a verdict, each class has at least
--min-per-class rows, PASS and FAIL recall reach --min-recall, and kappa
reaches --min-kappa. Every run writes runs/<timestamp>-calibrate-<split>/
with summary.json, predictions.jsonl, a copy of the rubric, and (on dev)
disagreements.jsonl.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import anthropic

from .data import class_counts, dumps, file_sha256, load_jsonl, read_text, split_rows, write_jsonl
from .judge import ClaudeJudge, Judge, JudgeConfig, pick_few_shots, price_known
from .metrics import judge_report
from .runner import Budget, judge_rows

TRAIN_FRACTION, DEV_FRACTION = 0.15, 0.40

# A wrong key or model raises an API error. A missing key raises a TypeError in the
# SDK, matched by its message so real TypeErrors still surface.
SETUP_ERRORS = (anthropic.AuthenticationError, anthropic.PermissionDeniedError,
                anthropic.NotFoundError, anthropic.BadRequestError, TypeError)


def is_setup_error(exc: Exception) -> bool:
    return not isinstance(exc, TypeError) or "authentication" in str(exc).lower()


def few_shot_warnings(few_shots: list[dict]) -> list[str]:
    """The split isn't stratified, so a small train split can miss a class."""
    labels = {r.get("label") for r in few_shots}
    missing = [label for label in ("PASS", "FAIL") if label not in labels]
    if not few_shots:
        return []
    return [f"the few-shots have no {label} example: train holds none; label more rows" for label in missing]


def calibrate(
    rows: list[dict],
    judge: Judge,
    config: JudgeConfig,
    split_name: str,
    split_rows_: list[dict],
    out_dir: Path,
    repeats: int = 1,
    workers: int = 4,
    budget: Budget | None = None,
    dataset: dict | None = None,
) -> dict:
    budget = budget or Budget()
    runs = [judge_rows(judge, split_rows_, workers=workers, budget=budget) for _ in range(repeats)]
    first = runs[0]
    errors = [o for o in first if o.error]
    scored = [o for o in first if o.verdict]
    report = judge_report([o.row["label"] for o in scored], [o.verdict.label for o in scored])

    flips = None
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
        "dataset": dataset or {},
        "dataset_rows": len(rows),
        "split_rows": len(split_rows_),
        "split_class_counts": class_counts(split_rows_),
        "report": asdict(report),
        "errors": len(errors),
        "fallback_verdicts": sum(1 for o in scored if o.verdict.fallback_used),
        "repeats": repeats,
        "label_flips_across_repeats": flips,
        "estimated_cost_usd": round(budget.spent_usd, 4),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(dumps(summary, indent=2) + "\n")
    (out_dir / "rubric.md").write_text(config.rubric)
    write_jsonl(out_dir / "predictions.jsonl", predictions)
    if split_name == "dev":
        # Test disagreements are left out on purpose: reading them invites
        # tuning the rubric on test.
        write_jsonl(out_dir / "disagreements.jsonl", disagreements)
    return summary


def bar_failures(summary: dict, min_recall: float, min_kappa: float, min_per_class: int) -> list[str]:
    r = summary["report"]
    reasons = []
    if summary["errors"]:
        reasons.append(f"{summary['errors']} rows got no verdict (see predictions.jsonl)")
    if summary.get("fallback_verdicts"):
        reasons.append(f"{summary['fallback_verdicts']} verdicts came from a fallback model, not this judge")
    for label, n in summary["split_class_counts"].items():
        if n < min_per_class:
            reasons.append(f"only {n} {label} rows in this split; the bar needs {min_per_class} per class")
    for name in ("pass_recall", "fail_recall"):
        value = r[name]
        if value is None or value != value or value < min_recall:  # missing, NaN or below the bar
            reasons.append(f"{name.replace('_', ' ')} below {min_recall:.2f}")
    kappa = r["kappa"]
    if kappa is None or kappa != kappa or kappa < min_kappa:
        reasons.append(f"kappa below {min_kappa:.2f}")
    return reasons


def _fmt(x: float | None) -> str:
    return "n/a" if x is None or x != x else f"{x:.3f}"


def _ci(pair: tuple[float, float] | list[float]) -> str:
    lo, hi = pair
    return "[n/a]" if lo is None or lo != lo else f"[{lo:.2f}, {hi:.2f}]"


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
    p.add_argument("--min-per-class", type=int, default=30, help="PASS and FAIL rows the split needs")
    p.add_argument("--model", default=None, help="judge model (default: $JUDGE_MODEL or claude-opus-5-5)")
    p.add_argument("--effort", default=None, help="judge effort (default: $JUDGE_EFFORT or medium)")
    p.add_argument("--out", default="runs")
    args = p.parse_args(argv)
    for name, value, low in (("repeats", args.repeats, 1), ("workers", args.workers, 1),
                             ("few-shots", args.few_shots, 0), ("min-per-class", args.min_per_class, 0)):
        if value < low:
            p.error(f"--{name} must be at least {low}")

    try:
        budget = Budget.from_env()
        rows = load_jsonl(args.data, require_label=True)
        rubric = read_text(args.rubric)
    except (ValueError, OSError) as exc:
        print(exc, file=sys.stderr)
        return 2
    train, dev, test = split_rows(rows, seed=args.seed, train=TRAIN_FRACTION, dev=DEV_FRACTION)
    config = JudgeConfig(rubric=rubric)
    if args.model:
        config.model = args.model
    if args.effort:
        config.effort = args.effort
    config.few_shots = pick_few_shots(train, n=args.few_shots)
    if budget.limit_usd is not None and not price_known(config.model):
        print(f"EVAL_BUDGET_USD is set but there's no price for {config.model} in evals/judge.py PRICES.", file=sys.stderr)
        return 2
    split_rows_ = dev if args.split == "dev" else test

    if args.split == "test":
        print("Reading the test split. Do this once, after dev clears the bar; keep this number.")

    dataset = {
        "path": args.data,
        "sha256": file_sha256(args.data),
        "seed": args.seed,
        "fractions": {"train": TRAIN_FRACTION, "dev": DEV_FRACTION, "test": round(1 - TRAIN_FRACTION - DEV_FRACTION, 2)},
        "few_shots": args.few_shots,
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out) / f"{stamp}-calibrate-{args.split}"
    try:
        summary = calibrate(
            rows, ClaudeJudge(config), config, args.split, split_rows_, out_dir,
            repeats=args.repeats, workers=args.workers, budget=budget, dataset=dataset,
        )
    except SETUP_ERRORS as exc:
        if not is_setup_error(exc):
            raise
        print(f"Can't run the judge: {exc}", file=sys.stderr)
        print("Set ANTHROPIC_API_KEY (or run `ant auth login`) and check --model.", file=sys.stderr)
        return 2

    reasons = bar_failures(summary, args.min_recall, args.min_kappa, args.min_per_class)
    warnings = few_shot_warnings(config.few_shots)
    # Record the result, so run_eval can refuse a calibration that didn't pass.
    summary["bar"] = {
        "passed": not reasons,
        "reasons": reasons,
        "min_recall": args.min_recall,
        "min_kappa": args.min_kappa,
        "min_per_class": args.min_per_class,
    }
    summary["warnings"] = warnings
    (out_dir / "summary.json").write_text(dumps(summary, indent=2) + "\n")

    r = summary["report"]
    c = r["confusion"]
    counts = summary["split_class_counts"]
    print(f"split={args.split}  n={r['n']}  train/dev/test={len(train)}/{len(dev)}/{len(test)}  judge={config.model}@{config.effort}")
    print(f"  PASS recall (TPR) {_fmt(r['pass_recall'])}  95% CI {_ci(r['pass_recall_ci'])}  n={c['tp'] + c['fn']} of {counts['PASS']} PASS rows")
    print(f"  FAIL recall (TNR) {_fmt(r['fail_recall'])}  95% CI {_ci(r['fail_recall_ci'])}  n={c['tn'] + c['fp']} of {counts['FAIL']} FAIL rows")
    print(f"  kappa {_fmt(r['kappa'])}   agreement {_fmt(r['agreement'])} (not the headline)")
    print(f"  confusion TP={c['tp']} FN={c['fn']} FP={c['fp']} TN={c['tn']}")
    if summary["fallback_verdicts"]:
        print(f"  {summary['fallback_verdicts']} verdicts came from a fallback model, not the calibrated judge")
    if summary["label_flips_across_repeats"]:
        print(f"  {summary['label_flips_across_repeats']} labels flipped between repeats: treat small moves as noise")
    for warning in warnings:
        print(f"  warning: {warning}")
    print(f"  estimated cost ${summary['estimated_cost_usd']:.2f}   wrote {out_dir}")

    if not reasons and args.split == "dev":
        print("Dev clears the bar. Run once with --split test and keep that number.")
    elif not reasons:
        print("Test clears the bar. This is the judge's calibration record; keep summary.json.")
    else:
        print("Not there yet:")
        for reason in reasons:
            print(f"  - {reason}")
        if args.split == "dev":
            print("Read disagreements.jsonl, change one thing in the rubric (or label more rows), run again.")
        else:
            print("Don't tune on test. Label a fresh batch, calibrate on dev again, and treat the next test read as a re-read: it includes rows you have already seen.")
    return 0 if not reasons else 1


if __name__ == "__main__":
    sys.exit(main())
