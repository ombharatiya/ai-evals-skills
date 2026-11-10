"""Code-graded checks: the cheap, deterministic half of an eval.

Run these before the LLM judge. Anything a string match, a parser or a
counter can decide should not cost a model call, and a failed code check
fails the row without asking the judge.

Each check reads optional fields from the trace row, so the expectations live
with the data:

    {"id": "t-007", "query": "...", "response": "...",
     "must_include": ["30 days"], "must_not_include": ["90 days"], "max_words": 120}
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

Row = dict[str, Any]


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


def not_empty(row: Row) -> CheckResult | None:
    ok = bool(row["response"].strip())
    return CheckResult("not_empty", ok, "" if ok else "response is empty")


def must_include(row: Row) -> CheckResult | None:
    needles = row.get("must_include")
    if not needles:
        return None
    text = row["response"].lower()
    missing = [s for s in needles if s.lower() not in text]
    return CheckResult("must_include", not missing, f"missing: {missing}" if missing else "")


def must_not_include(row: Row) -> CheckResult | None:
    needles = row.get("must_not_include")
    if not needles:
        return None
    text = row["response"].lower()
    found = [s for s in needles if s.lower() in text]
    return CheckResult("must_not_include", not found, f"found: {found}" if found else "")


def max_words(row: Row) -> CheckResult | None:
    limit = row.get("max_words")
    if limit is None:
        return None
    count = len(row["response"].split())
    ok = count <= int(limit)
    return CheckResult("max_words", ok, "" if ok else f"{count} words, limit {limit}")


CHECKS: list[Callable[[Row], CheckResult | None]] = [not_empty, must_include, must_not_include, max_words]


def run_checks(row: Row) -> list[CheckResult]:
    results = []
    for check in CHECKS:
        result = check(row)
        if result is not None:
            results.append(result)
    return results
