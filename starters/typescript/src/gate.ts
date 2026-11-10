/**
 * Compare a candidate run with a baseline: ship, hold, investigate or block.
 *
 * The bands and band() follow the release-gates chapter on aievals.co
 * (https://www.aievals.co/learn/production/release-gates). They read the
 * interval on the paired difference in pass rate, not the absolute rate:
 * ship when the interval sits above zero or inside the indifference window
 * (default 1 point); hold when it contains zero and reaches outside it;
 * investigate when it sits below zero and the drop is within the
 * minor-regression window (default 2 points); block when the drop is larger.
 *
 * The interval is Newcombe's score interval for a paired difference of
 * proportions (Newcombe 1998, method 10). It stays wide when the two runs
 * agree on every row, so a tiny comparison can't ship on a [0, 0] interval.
 */
import { wilson } from "./metrics.js";

export interface Delta {
  n: number;
  baseline_rate: number;
  candidate_rate: number;
  delta: number;
  ci: [number, number];
  both_pass: number;
  baseline_only: number;
  candidate_only: number;
  both_fail: number;
}

/**
 * Difference p_candidate - p_baseline and its interval from paired counts:
 * a = both pass, b = baseline passes and candidate fails,
 * c = baseline fails and candidate passes, d = both fail.
 */
export function newcombePaired(a: number, b: number, c: number, d: number, z = 1.96): [number, number, number] {
  const n = a + b + c + d;
  if (n === 0) throw new Error("no paired rows");
  const pBase = (a + b) / n;
  const pCand = (a + c) / n;
  const [loBase, hiBase] = wilson(a + b, n, z);
  const [loCand, hiCand] = wilson(a + c, n, z);
  const denom = Math.sqrt((a + b) * (c + d) * (a + c) * (b + d));
  let phi = 0;
  if (denom > 0) {
    // Newcombe's continuity-adjusted correlation for method 10.
    let num = a * d - b * c;
    if (num > 0) num = Math.max(num - n / 2, 0);
    phi = num / denom;
  }
  const delta = pCand - pBase;
  const down = Math.sqrt(Math.max(0, (pCand - loCand) ** 2 - 2 * phi * (pCand - loCand) * (hiBase - pBase) + (hiBase - pBase) ** 2));
  const up = Math.sqrt(Math.max(0, (hiCand - pCand) ** 2 - 2 * phi * (hiCand - pCand) * (pBase - loBase) + (pBase - loBase) ** 2));
  return [delta, Math.max(-1, delta - down), Math.min(1, delta + up)];
}

/** Pair the runs on their shared ids. The caller reports ids that didn't pair. */
export function pairedDelta(baseline: Map<string, boolean>, candidate: Map<string, boolean>): Delta {
  const ids = [...baseline.keys()].filter((id) => candidate.has(id)).sort();
  if (ids.length === 0) throw new Error("baseline and candidate share no ids");
  let a = 0;
  let b = 0;
  let c = 0;
  for (const id of ids) {
    const base = baseline.get(id)!;
    const cand = candidate.get(id)!;
    if (base && cand) a++;
    else if (base) b++;
    else if (cand) c++;
  }
  const d = ids.length - a - b - c;
  const [delta, lo, hi] = newcombePaired(a, b, c, d);
  const n = ids.length;
  return {
    n,
    baseline_rate: (a + b) / n,
    candidate_rate: (a + c) / n,
    delta,
    ci: [lo, hi],
    both_pass: a,
    baseline_only: b,
    candidate_only: c,
    both_fail: d,
  };
}

/**
 * How wide the interval is when a candidate matches the baseline on every
 * row: the gate set's resolution. A window narrower than this can never ship
 * an unchanged candidate.
 */
export function noChangeHalfwidth(n: number, rate: number, flipRate = 0): number {
  // A judge that flips a share of its labels between identical runs adds that
  // many disagreeing pairs, split both ways. flipRate = 0 gives a floor.
  const passes = Math.floor(rate * n + 0.5);
  // The expected number of flips, not rounded to whole rows: rounding makes the
  // width jump up and down as n grows, and rowsForIndifference needs it to shrink.
  const halfFlips = (flipRate * n) / 2;
  const b = Math.min(halfFlips, passes, n - passes);
  const [, lo, hi] = newcombePaired(passes - b, b, b, n - passes - b);
  return Math.max(-lo, hi);
}

/** Smallest gate-set size whose no-change interval fits inside the window. */
export function rowsForIndifference(rate: number, indifference: number, start = 1, limit = 1_000_000, flipRate = 0): number | null {
  const fits = (n: number) => noChangeHalfwidth(n, rate, flipRate) <= indifference;
  let n = Math.max(1, start);
  if (fits(n)) return n;
  let lo = n;
  while (!fits(n)) {
    lo = n;
    n *= 2;
    if (n > limit) return null;
  }
  while (lo + 1 < n) {
    const mid = Math.floor((lo + n) / 2);
    if (fits(mid)) n = mid;
    else lo = mid;
  }
  return n;
}

export type Band = "ship" | "hold" | "investigate" | "block";

/** The guide's gate(): decide a band from the delta and its interval. */
export function band(delta: Delta, indifference = 0.01, minorRegression = 0.02): Band {
  const [lo, hi] = delta.ci;
  if (hi < 0) return -delta.delta > minorRegression ? "block" : "investigate";
  if (lo > 0) return "ship";
  if (Math.max(Math.abs(lo), Math.abs(hi)) <= indifference) return "ship";
  return "hold";
}
