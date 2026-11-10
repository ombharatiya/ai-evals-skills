"""Judge-versus-human metrics and confidence intervals.

PASS is the positive class here, so PASS recall is the true positive rate
(TPR) and FAIL recall is the true negative rate (TNR). Some writing, Hamel
Husain's FAQ among it, treats FAIL as positive and calls FAIL recall the TPR.
The code names the two rates by class to avoid that trap.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Confusion:
    tp: int  # human PASS, judge PASS
    fn: int  # human PASS, judge FAIL
    fp: int  # human FAIL, judge PASS (the miss that hurts: a real failure let through)
    tn: int  # human FAIL, judge FAIL

    @property
    def n(self) -> int:
        return self.tp + self.fn + self.fp + self.tn


def confusion(human: list[str], judge: list[str]) -> Confusion:
    if len(human) != len(judge):
        raise ValueError("human and judge label lists differ in length")
    tp = fn = fp = tn = 0
    for h, j in zip(human, judge):
        if h == "PASS" and j == "PASS":
            tp += 1
        elif h == "PASS":
            fn += 1
        elif j == "PASS":
            fp += 1
        else:
            tn += 1
    return Confusion(tp=tp, fn=fn, fp=fp, tn=tn)


def _ratio(num: int, den: int) -> float:
    return num / den if den else float("nan")


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion (95% by default).

    Better behaved than the normal approximation at small n and near 0 or 1,
    which is exactly where judge recalls and pass rates live.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def cohen_kappa(c: Confusion) -> float:
    """Agreement corrected for chance. 1 is perfect, 0 is no better than chance."""
    n = c.n
    if n == 0:
        return float("nan")
    observed = (c.tp + c.tn) / n
    human_pass = (c.tp + c.fn) / n
    judge_pass = (c.tp + c.fp) / n
    expected = human_pass * judge_pass + (1 - human_pass) * (1 - judge_pass)
    if expected == 1:
        return float("nan")
    return (observed - expected) / (1 - expected)


@dataclass(frozen=True)
class JudgeReport:
    n: int
    confusion: dict
    pass_recall: float  # TPR with PASS as positive
    fail_recall: float  # TNR with PASS as positive: share of real failures the judge catches
    pass_recall_ci: tuple[float, float]
    fail_recall_ci: tuple[float, float]
    agreement: float
    kappa: float


def judge_report(human: list[str], judge: list[str]) -> JudgeReport:
    c = confusion(human, judge)
    return JudgeReport(
        n=c.n,
        confusion=asdict(c),
        pass_recall=_ratio(c.tp, c.tp + c.fn),
        fail_recall=_ratio(c.tn, c.tn + c.fp),
        pass_recall_ci=wilson(c.tp, c.tp + c.fn),
        fail_recall_ci=wilson(c.tn, c.tn + c.fp),
        agreement=_ratio(c.tp + c.tn, c.n),
        kappa=cohen_kappa(c),
    )
