"""Run a judge over many rows with a worker pool and a hard budget cap."""
from __future__ import annotations

import math
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
    Failed calls (refusals, truncations, bad output) are charged too. Calls
    already in flight finish, so the real spend can overshoot by up to one
    call per worker.
    """

    limit_usd: float | None = None
    spent_usd: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @classmethod
    def from_env(cls) -> "Budget":
        raw = os.environ.get("EVAL_BUDGET_USD", "").strip()
        if not raw:
            return cls()
        try:
            limit = float(raw)
        except ValueError:
            raise ValueError(f"EVAL_BUDGET_USD must be a number of US dollars, got {raw!r}") from None
        if not math.isfinite(limit) or limit < 0:
            raise ValueError(f"EVAL_BUDGET_USD must be a positive number, got {raw!r}")
        return cls(limit_usd=limit)

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
    """Judge every row. A row-level failure becomes an Outcome with an error.

    Any other exception (a bad key, an unknown model) stops new calls at once
    and is raised after the calls already in flight finish.
    """
    budget = budget or Budget()
    stop = threading.Event()

    def one(row: Row) -> Outcome:
        if stop.is_set():
            return Outcome(row, error="not run: the run stopped on an earlier error")
        try:
            budget.check()
        except BudgetExceeded as exc:
            return Outcome(row, error=f"not run: budget reached ({exc})")
        try:
            verdict = judge(row)
        except JudgeError as exc:
            budget.add(exc.cost_usd)
            return Outcome(row, error=str(exc))
        except Exception:
            stop.set()
            raise
        budget.add(verdict.cost_usd)
        return Outcome(row, verdict=verdict)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        return list(pool.map(one, rows))
