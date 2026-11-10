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
import { createHash } from "node:crypto";
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

const JSON_SPACE = " \t\r\n";

/** Trim JSON whitespace (space, tab, CR, LF) only, the way the Python starter's strip(" \t\r\n") does. */
export function trimSpace(text: string): string {
  let start = 0;
  let end = text.length;
  while (start < end && JSON_SPACE.includes(text[start]!)) start++;
  while (end > start && JSON_SPACE.includes(text[end - 1]!)) end--;
  return text.slice(start, end);
}

/**
 * Read a UTF-8 file the way the Python starter does: a leading byte-order mark
 * is dropped and CRLF or CR line endings become LF, so a rubric saved on
 * Windows hashes the same in both languages.
 */
export function readText(path: string): string {
  const text = readFileSync(path, "utf8");
  return (text.startsWith("\ufeff") ? text.slice(1) : text).replace(/\r\n?/g, "\n");
}

/** [line number, line] for each non-blank line of a JSONL file. */
export function jsonlLines(path: string): [number, string][] {
  return readText(path)
    .split("\n")
    .map((line, index): [number, string] => [index + 1, line])
    .filter(([, line]) => trimSpace(line) !== "");
}

export function loadJsonl(path: string, requireLabel = false): Row[] {
  const rows: Row[] = [];
  const seen = new Set<string>();
  jsonlLines(path).forEach(([lineno, line]) => {
    const where = path + ":" + lineno;
    let row: Row;
    try {
      row = JSON.parse(line) as Row;
    } catch {
      throw new Error(where + ": not valid JSON");
    }
    if (row === null || typeof row !== "object" || Array.isArray(row)) throw new Error(where + ": each line must be a JSON object");
    for (const field of ["id", "query", "response"] as const) {
      if (typeof row[field] !== "string") throw new Error(where + ": '" + field + "' must be a string");
    }
    // JSON-quoted, so an id can't start an output line of its own.
    if (seen.has(row.id)) throw new Error(where + ": duplicate id " + JSON.stringify(row.id));
    seen.add(row.id);
    if (row.label !== undefined && row.label !== null) {
      const label = String(row.label).toUpperCase();
      if (label !== "PASS" && label !== "FAIL") throw new Error(where + ": label must be PASS or FAIL");
      row.label = label;
    } else if (requireLabel) {
      throw new Error(where + ": 'label' is required here");
    }
    rows.push(row);
  });
  if (rows.length === 0) throw new Error(path + ": no rows");
  return rows;
}

/** NaN and infinity aren't valid JSON. Write them as null. */
export function jsonSafe(value: unknown): unknown {
  if (typeof value === "number" && !Number.isFinite(value)) return null;
  if (Array.isArray(value)) return value.map(jsonSafe);
  if (value && typeof value === "object") {
    return Object.fromEntries(Object.entries(value).map(([k, v]) => [k, jsonSafe(v)]));
  }
  return value;
}

export function dumps(value: unknown, indent?: number): string {
  return JSON.stringify(jsonSafe(value), null, indent);
}

export function writeJsonl(path: string, rows: unknown[]): void {
  writeFileSync(path, rows.map((r) => dumps(r) + "\n").join(""));
}

export function fileSha256(path: string): string {
  return createHash("sha256").update(readFileSync(path)).digest("hex");
}

/** A number in [0, 1) that depends only on the id and the seed. Same hash as the Python starter. */
export function splitUnit(id: string, seed: number): number {
  const digest = createHash("sha256").update(seed + ":" + id).digest();
  return Number(digest.readBigUInt64BE(0)) / 2 ** 64;
}

const byId = (a: Row, b: Row) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/**
 * Split labeled rows into train / dev / test by a stable hash of each id.
 * Train holds few-shots, dev is where you iterate, test is read once. The
 * default 15 / 40 / 45 fractions follow the calibration recipe on aievals.co.
 *
 * A row's split depends only on its id and the seed, so labeling more rows
 * never moves an existing row from test into train or dev, and each split's
 * PASS:FAIL ratio matches the whole set in expectation. Split sizes are
 * approximate on small sets; the calibration run prints per-class counts.
 */
export function splitRows(rows: Row[], seed = 42, train = 0.15, dev = 0.4): [Row[], Row[], Row[]] {
  if (!(train > 0 && dev > 0 && train + dev < 1)) throw new Error("train and dev must be fractions with train + dev < 1");
  const out: [Row[], Row[], Row[]] = [[], [], []];
  for (const row of [...rows].sort(byId)) {
    const u = splitUnit(row.id, seed);
    out[u < train ? 0 : u < train + dev ? 1 : 2].push(row);
  }
  return out;
}

export function classCounts(rows: Row[]): Record<Label, number> {
  return {
    PASS: rows.filter((r) => r.label === "PASS").length,
    FAIL: rows.filter((r) => r.label === "FAIL").length,
  };
}
