"""Compare a candidate run with a baseline and decide ship, hold, investigate or block.

The bands follow the release-gates chapter on aievals.co. They read the
interval on the paired difference in pass rate (candidate minus baseline, on
the same ids), not the absolute pass rate:

- ship:        the interval sits above zero, or entirely inside the indifference window
- hold:        the interval contains zero and reaches outside the window
- investigate: the interval sits below zero but inside the minor-regression window
- block:       the interval sits below zero and reaches past the window

Any failing row marked `must_pass` (the regression set) blocks on its own.
"""
from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Delta:
    n: int
    baseline_rate: float
    candidate_rate: float
    delta: float
    ci: tuple[float, float]


def paired_delta(
    baseline: dict[str, bool], candidate: dict[str, bool], reps: int = 2000, seed: int = 0
) -> Delta:
    """Paired bootstrap interval on candidate pass rate minus baseline pass rate."""
    ids = sorted(set(baseline) & set(candidate))
    if not ids:
        raise ValueError("baseline and candidate share no ids")
    diffs = [int(candidate[i]) - int(baseline[i]) for i in ids]
    n = len(diffs)
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(diffs) for _ in range(n)) / n for _ in range(reps))
    lo = means[int(0.025 * reps)]
    hi = means[min(reps - 1, int(0.975 * reps))]
    return Delta(
        n=n,
        baseline_rate=sum(baseline[i] for i in ids) / n,
        candidate_rate=sum(candidate[i] for i in ids) / n,
        delta=sum(diffs) / n,
        ci=(lo, hi),
    )


def band(delta: Delta, window: float = 0.02) -> str:
    lo, hi = delta.ci
    # A regression the interval is sure of comes first, even a small one.
    if hi < 0:
        return "investigate" if lo >= -window else "block"
    if lo > 0 or (lo >= -window and hi <= window):
        return "ship"
    return "hold"
