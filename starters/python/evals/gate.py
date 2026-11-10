"""Compare a candidate run with a baseline: ship, hold, investigate or block.

The bands and the `band()` logic follow the release-gates chapter on
aievals.co (https://www.aievals.co/learn/production/release-gates). They read
the interval on the paired difference in pass rate (candidate minus baseline,
on the same ids), not the absolute pass rate:

- ship:        the interval sits above zero, or both ends are inside the
               indifference window (default 1 point)
- hold:        the interval contains zero and reaches outside the window
- investigate: the interval sits below zero and the drop is within the
               minor-regression window (default 2 points)
- block:       the interval sits below zero and the drop is larger

The interval is Newcombe's score interval for a paired difference of
proportions (Newcombe 1998, method 10). Unlike a bootstrap, it stays wide
when the two runs agree on every row, so a tiny comparison can't ship on a
[0, 0] interval.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .metrics import wilson


@dataclass(frozen=True)
class Delta:
    n: int
    baseline_rate: float
    candidate_rate: float
    delta: float
    ci: tuple[float, float]
    # Paired counts: both pass, baseline-only pass, candidate-only pass, both fail.
    both_pass: int
    baseline_only: int
    candidate_only: int
    both_fail: int


def newcombe_paired(a: int, b: int, c: int, d: int, z: float = 1.96) -> tuple[float, float, float]:
    """Difference p_candidate - p_baseline and its interval from paired counts.

    a = both pass, b = baseline passes and candidate fails,
    c = baseline fails and candidate passes, d = both fail.
    """
    n = a + b + c + d
    if n == 0:
        raise ValueError("no paired rows")
    p_base, p_cand = (a + b) / n, (a + c) / n
    lo_base, hi_base = wilson(a + b, n, z)
    lo_cand, hi_cand = wilson(a + c, n, z)
    denom = math.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    if denom == 0:
        phi = 0.0
    else:
        # Newcombe's continuity-adjusted correlation for method 10.
        num = a * d - b * c
        if num > 0:
            num = max(num - n / 2, 0.0)
        phi = num / denom
    delta = p_cand - p_base
    down = math.sqrt(max(0.0, (p_cand - lo_cand) ** 2 - 2 * phi * (p_cand - lo_cand) * (hi_base - p_base) + (hi_base - p_base) ** 2))
    up = math.sqrt(max(0.0, (hi_cand - p_cand) ** 2 - 2 * phi * (hi_cand - p_cand) * (p_base - lo_base) + (p_base - lo_base) ** 2))
    return delta, max(-1.0, delta - down), min(1.0, delta + up)


def paired_delta(baseline: dict[str, bool], candidate: dict[str, bool]) -> Delta:
    """Pair the runs on their shared ids. The caller reports ids that didn't pair."""
    ids = sorted(set(baseline) & set(candidate))
    if not ids:
        raise ValueError("baseline and candidate share no ids")
    a = sum(1 for i in ids if baseline[i] and candidate[i])
    b = sum(1 for i in ids if baseline[i] and not candidate[i])
    c = sum(1 for i in ids if not baseline[i] and candidate[i])
    d = len(ids) - a - b - c
    delta, lo, hi = newcombe_paired(a, b, c, d)
    n = len(ids)
    return Delta(n=n, baseline_rate=(a + b) / n, candidate_rate=(a + c) / n, delta=delta, ci=(lo, hi),
                 both_pass=a, baseline_only=b, candidate_only=c, both_fail=d)


def band(delta: Delta, indifference: float = 0.01, minor_regression: float = 0.02) -> str:
    """The guide's gate(): decide a band from the delta and its interval."""
    lo, hi = delta.ci
    if hi < 0:
        return "block" if -delta.delta > minor_regression else "investigate"
    if lo > 0:
        return "ship"
    if max(abs(lo), abs(hi)) <= indifference:
        return "ship"
    return "hold"
