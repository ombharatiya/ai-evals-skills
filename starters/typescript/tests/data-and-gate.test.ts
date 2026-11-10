import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe as group, expect, it } from "vitest";
import { runChecks } from "../src/checks.js";
import { numberFlag } from "../src/cli.js";
import { classCounts, dumps, loadJsonl, splitRows, writeJsonl, type Row } from "../src/data.js";
import { band, newcombePaired, noChangeHalfwidth, pairedDelta, rowsForIndifference, type Delta } from "../src/gate.js";

const rows = (nPass: number, nFail: number, prefix = ""): Row[] => [
  ...Array.from({ length: nPass }, (_, i) => ({ id: prefix + "p" + i, query: "q", response: "r", label: "PASS" as const })),
  ...Array.from({ length: nFail }, (_, i) => ({ id: prefix + "f" + i, query: "q", response: "r", label: "FAIL" as const })),
];

group("data", () => {
  it("keeps Unicode line separators and reads files saved on Windows", () => {
    const dir = mkdtempSync(join(tmpdir(), "evals-"));
    const row = { id: "a", query: "q", response: "line one\u2028line two\u0085three" };
    writeJsonl(join(dir, "x.jsonl"), [row]);
    expect(loadJsonl(join(dir, "x.jsonl"))).toEqual([row]);
    writeFileSync(join(dir, "w.jsonl"), '\ufeff{"id": "a", "query": "q", "response": "r"}\r\n\r\n');
    expect(loadJsonl(join(dir, "w.jsonl"))).toEqual([{ id: "a", query: "q", response: "r" }]);
  });

  it("quotes a duplicate id in the error, so it can't emit a workflow command", () => {
    const dir = mkdtempSync(join(tmpdir(), "evals-"));
    const file = join(dir, "dup.jsonl");
    const row = JSON.stringify({ id: "g\n::warning::x", query: "q", response: "r" }) + "\n";
    writeFileSync(file, row + row);
    expect(() => loadJsonl(file)).toThrow(/: duplicate id "g\\n::warning::x"$/);
  });

  it("rejects bad rows", () => {
    const dir = mkdtempSync(join(tmpdir(), "evals-"));
    const file = join(dir, "bad.jsonl");
    writeFileSync(file, JSON.stringify({ id: "a", query: "q", response: "r", label: "MAYBE" }) + "\n");
    expect(() => loadJsonl(file)).toThrow(/label/);
    writeFileSync(file, JSON.stringify({ id: "a", query: "q", response: "r" }) + "\n");
    expect(() => loadJsonl(file, true)).toThrow(/required/);
    writeFileSync(file, "[1, 2]\n");
    expect(() => loadJsonl(file)).toThrow(/object/);
  });

  it("splits disjoint, complete and near the fractions", () => {
    const all = rows(600, 400);
    const [train, dev, test] = splitRows(all);
    expect(new Set([...train, ...dev, ...test].map((r) => r.id)).size).toBe(1000);
    for (const [part, share] of [[train, 0.15], [dev, 0.4], [test, 0.45]] as const) {
      expect(Math.abs(part.length / 1000 - share)).toBeLessThan(0.04);
      expect(Math.abs(classCounts(part).FAIL / part.length - 0.4)).toBeLessThan(0.06);
    }
  });

  it("never moves an existing row when rows are added", () => {
    const before = splitRows(rows(30, 20));
    const after = splitRows([...rows(30, 20), ...rows(5, 5, "new-")]);
    before.forEach((part, i) => {
      const ids = new Set(after[i]!.map((r) => r.id));
      expect(part.every((r) => ids.has(r.id))).toBe(true);
    });
  });

  it("matches the Python starter's split", () => {
    const [train, dev, test] = splitRows(rows(10, 10), 42);
    expect(train.map((r) => r.id)).toEqual(["f5", "p1", "p5", "p9"]);
    expect([dev.length, test.length]).toEqual([12, 4]);
  });

  it("writes NaN as null", () => {
    expect(dumps({ x: NaN, y: [1, Infinity] })).toBe('{"x":null,"y":[1,null]}');
  });

  it("loads the example dataset into both classes per split", () => {
    const data = loadJsonl(new URL("../data/labeled.example.jsonl", import.meta.url).pathname, true);
    expect(data).toHaveLength(40);
    for (const part of splitRows(data)) {
      expect(classCounts(part).PASS).toBeGreaterThan(0);
      expect(classCounts(part).FAIL).toBeGreaterThan(0);
    }
  });
});

group("checks and flags", () => {
  it("reads expectations from the row", () => {
    const row: Row = { id: "x", query: "q", response: "Refunds take 5 to 7 business days." };
    expect(runChecks(row).map((c) => c.name)).toEqual(["not_empty"]);
    const failed = runChecks({ ...row, must_include: ["30 days"], must_not_include: ["business"], max_words: 3 });
    expect(failed.filter((c) => !c.passed).map((c) => c.name)).toEqual(["must_include", "must_not_include", "max_words"]);
  });

  it("rejects bad numeric flags", () => {
    expect(numberFlag("workers", "4", { integer: true, min: 1 })).toBe(4);
    expect(() => numberFlag("workers", "0", { integer: true, min: 1 })).toThrow(/workers/);
    expect(() => numberFlag("seed", "1.5", { integer: true })).toThrow(/integer/);
    expect(() => numberFlag("min-recall", "high")).toThrow(/number/);
  });
});

group("gate", () => {
  const d = (delta: number, lo: number, hi: number): Delta => ({
    n: 100, baseline_rate: 0.8, candidate_rate: 0.8 + delta, delta, ci: [lo, hi],
    both_pass: 0, baseline_only: 0, candidate_only: 0, both_fail: 0,
  });

  it("follows the guide's gate()", () => {
    expect(band(d(0.03, 0.01, 0.05))).toBe("ship");
    expect(band(d(0, -0.008, 0.009))).toBe("ship");
    expect(band(d(0, -0.018, 0.018))).toBe("hold");
    expect(band(d(-0.012, -0.025, -0.001))).toBe("investigate");
    expect(band(d(-0.05, -0.08, -0.03))).toBe("block");
  });

  it("keeps the interval wide when the runs agree on every row", () => {
    const [delta, lo, hi] = newcombePaired(1, 0, 0, 0);
    expect(delta).toBe(0);
    expect(lo).toBeLessThan(-0.5);
    expect(hi).toBeGreaterThan(0.5);
  });

  it("matches the Python interval", () => {
    const [, lo, hi] = newcombePaired(160, 40, 0, 0);
    expect(lo).toBeCloseTo(-0.2608563961117042, 12);
    expect(hi).toBeCloseTo(-0.14698816374338808, 12);
  });

  it("reports the gate set's resolution like the Python starter", () => {
    expect(noChangeHalfwidth(50, 0.76)).toBeCloseTo(0.0529, 3);
    expect(rowsForIndifference(0.76, 0.01)).toBe(277);
    expect(rowsForIndifference(0.97, 0.01)).toBe(408);
    // The review's high-pass-rate cases, where flips are capped by the FAIL rows.
    for (const [rate, flip] of [[0.99, 0.03], [0.99, 0.1], [0.97, 0.1]] as const) {
      const n = rowsForIndifference(rate, 0.01, 1, 1_000_000, flip)!;
      expect(noChangeHalfwidth(n - 1, rate, flip)).toBeGreaterThan(0.01);
      for (let m = n; m < n + 400; m++) expect(noChangeHalfwidth(m, rate, flip)).toBeLessThanOrEqual(0.01);
    }
  });

  it("detects a clear paired regression", () => {
    const base = new Map(Array.from({ length: 200 }, (_, i) => [String(i), true] as [string, boolean]));
    const cand = new Map(Array.from({ length: 200 }, (_, i) => [String(i), i >= 40] as [string, boolean]));
    const r = pairedDelta(base, cand);
    expect(r.delta).toBeCloseTo(-0.2, 10);
    expect(r.baseline_only).toBe(40);
    expect(band(r)).toBe("block");
  });
});
