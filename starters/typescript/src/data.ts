/**
 * Load traces and split a labeled set.
 *
 * A trace file is JSON Lines, one object per line:
 *
 *   {"id": "t-001", "query": "...", "response": "...", "label": "PASS"}
 *
 * `label` is the human verdict and is required for calibration. Optional
 * code-check fields: `must_include`, `must_not_include` (string arrays) and
 * `max_words`. A row with `"must_pass": true` belongs to the regression set.
 */
import { readFileSync, writeFileSync } from "node:fs";

export type Label = "PASS" | "FAIL";

export interface Row {
  id: string;
  query: string;
  response: string;
  label?: Label;
  critique?: string;
  must_include?: string[];
  must_not_include?: string[];
  max_words?: number;
  must_pass?: boolean;
  [key: string]: unknown;
}

export function loadJsonl(path: string, requireLabel = false): Row[] {
  const rows: Row[] = [];
  const seen = new Set<string>();
  readFileSync(path, "utf8")
    .split("\n")
    .forEach((line, index) => {
      if (!line.trim()) return;
      const where = `${path}:${index + 1}`;
      let row: Row;
      try {
        row = JSON.parse(line) as Row;
      } catch {
        throw new Error(`${where}: not valid JSON`);
      }
      for (const field of ["id", "query", "response"] as const) {
        if (typeof row[field] !== "string") throw new Error(`${where}: '${field}' must be a string`);
      }
      if (seen.has(row.id)) throw new Error(`${where}: duplicate id ${row.id}`);
      seen.add(row.id);
      if (row.label !== undefined) {
        const label = String(row.label).toUpperCase();
        if (label !== "PASS" && label !== "FAIL") throw new Error(`${where}: label must be PASS or FAIL`);
        row.label = label;
      } else if (requireLabel) {
        throw new Error(`${where}: 'label' is required here`);
      }
      rows.push(row);
    });
  if (rows.length === 0) throw new Error(`${path}: no rows`);
  return rows;
}

export function writeJsonl(path: string, rows: unknown[]): void {
  writeFileSync(path, rows.map((r) => JSON.stringify(r) + "\n").join(""));
}

/** Small seeded PRNG (mulberry32) so a seed always gives the same split. */
export function seededRandom(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function shuffle<T>(items: T[], rand: () => number): T[] {
  const out = [...items];
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [out[i], out[j]] = [out[j]!, out[i]!];
  }
  return out;
}

/**
 * Split labeled rows into train / dev / test, keeping the PASS:FAIL ratio.
 * Train holds few-shots, dev is where you iterate, test is read once. The
 * default 15 / 40 / 45 split follows the calibration recipe on aievals.co.
 * The split depends only on the ids and the seed, never on input order.
 */
export function stratifiedSplit(rows: Row[], seed = 42, train = 0.15, dev = 0.4): [Row[], Row[], Row[]] {
  if (!(train > 0 && dev > 0 && train + dev < 1)) throw new Error("train and dev must be fractions with train + dev < 1");
  const rand = seededRandom(seed);
  const out: [Row[], Row[], Row[]] = [[], [], []];
  for (const label of ["PASS", "FAIL"] as const) {
    const group = shuffle(
      rows.filter((r) => r.label === label).sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0)),
      rand,
    );
    const nTrain = Math.round(group.length * train);
    const nDev = Math.round(group.length * dev);
    out[0].push(...group.slice(0, nTrain));
    out[1].push(...group.slice(nTrain, nTrain + nDev));
    out[2].push(...group.slice(nTrain + nDev));
  }
  return out;
}
