/**
 * Compare a candidate run with a baseline: ship, hold, investigate or block.
 * The bands follow the release-gates chapter on aievals.co and read the
 * interval on the paired difference in pass rate, not the absolute rate.
 */
import { seededRandom } from "./data.js";

export interface Delta {
  n: number;
  baseline_rate: number;
  candidate_rate: number;
  delta: number;
  ci: [number, number];
}

/** Paired bootstrap interval on candidate pass rate minus baseline pass rate. */
export function pairedDelta(
  baseline: Map<string, boolean>,
  candidate: Map<string, boolean>,
  reps = 2000,
  seed = 0,
): Delta {
  const ids = [...baseline.keys()].filter((id) => candidate.has(id)).sort();
  if (ids.length === 0) throw new Error("baseline and candidate share no ids");
  const diffs = ids.map((id) => Number(candidate.get(id)) - Number(baseline.get(id)));
  const n = diffs.length;
  const rand = seededRandom(seed);
  const means: number[] = [];
  for (let r = 0; r < reps; r++) {
    let sum = 0;
    for (let i = 0; i < n; i++) sum += diffs[Math.floor(rand() * n)]!;
    means.push(sum / n);
  }
  means.sort((a, b) => a - b);
  const lo = means[Math.floor(0.025 * reps)]!;
  const hi = means[Math.min(reps - 1, Math.floor(0.975 * reps))]!;
  const rate = (m: Map<string, boolean>) => ids.filter((id) => m.get(id)).length / n;
  return {
    n,
    baseline_rate: rate(baseline),
    candidate_rate: rate(candidate),
    delta: diffs.reduce((a, b) => a + b, 0) / n,
    ci: [lo, hi],
  };
}

export type Band = "ship" | "hold" | "investigate" | "block";

export function band(delta: Delta, window = 0.02): Band {
  const [lo, hi] = delta.ci;
  // A regression the interval is sure of comes first, even a small one.
  if (hi < 0) return lo >= -window ? "investigate" : "block";
  if (lo > 0 || (lo >= -window && hi <= window)) return "ship";
  return "hold";
}
