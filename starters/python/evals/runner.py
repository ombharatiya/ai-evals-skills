"""Run a judge over many rows with a worker pool and a hard budget cap."""
from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from .judge import Judge, JudgeError, Verdict

Row = dict[str, Any]


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Budget:
    """Stops new judge calls once the estimated spend reaches `limit_usd`.

    The cap is enforced in the runner, not printed as a warning, which is
    what the CI chapter on aievals.co asks for. Set it with EVAL_BUDGET_USD.
    Calls already in flight finish, so the real spend can overshoot by up to
    one call per worker.
    """

    limit_usd: float | None = None
    spent_usd: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @classmethod
    def from_env(cls) -> "Budget":
        raw = os.environ.get("EVAL_BUDGET_USD")
        return cls(limit_usd=float(raw) if raw else None)

    def check(self) -> None:
        with self._lock:
            if self.limit_usd is not None and self.spent_usd >= self.limit_usd:
                raise BudgetExceeded(f"spent ${self.spent_usd:.2f} of ${self.limit_usd:.2f}")

    def add(self, usd: float) -> None:
        with self._lock:
            self.spent_usd += usd


@dataclass
class Outcome:
    row: Row
    verdict: Verdict | None = None
    error: str | None = None


def judge_rows(judge: Judge, rows: list[Row], workers: int = 4, budget: Budget | None = None) -> list[Outcome]:
    budget = budget or Budget()

    def one(row: Row) -> Outcome:
        try:
            budget.check()
        except BudgetExceeded as exc:
            return Outcome(row, error=f"not run: budget reached ({exc})")
        try:
            verdict = judge(row)
        except JudgeError as exc:
            return Outcome(row, error=str(exc))
        budget.add(verdict.cost_usd)
        return Outcome(row, verdict=verdict)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        return list(pool.map(one, rows))
