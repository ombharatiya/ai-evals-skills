import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe as group, expect, it } from "vitest";
import { runChecks } from "../src/checks.js";
import { loadJsonl, stratifiedSplit, type Row } from "../src/data.js";
import { band, pairedDelta, type Delta } from "../src/gate.js";

const rows = (nPass: number, nFail: number): Row[] => [
  ...Array.from({ length: nPass }, (_, i) => ({ id: "p" + i, query: "q", response: "r", label: "PASS" as const })),
  ...Array.from({ length: nFail }, (_, i) => ({ id: "f" + i, query: "q", response: "r", label: "FAIL" as const })),
];

group("data", () => {
  it("rejects bad rows", () => {
    const dir = mkdtempSync(join(tmpdir(), "evals-"));
    const file = join(dir, "bad.jsonl");
    writeFileSync(file, JSON.stringify({ id: "a", query: "q", response: "r", label: "MAYBE" }) + "\n");
    expect(() => loadJsonl(file)).toThrow(/label/);
    writeFileSync(file, JSON.stringify({ id: "a", query: "q", response: "r" }) + "\n");
    expect(() => loadJsonl(file, true)).toThrow(/required/);
  });

  it("splits stratified, disjoint and complete", () => {
    const [train, dev, test] = stratifiedSplit(rows(60, 40));
    expect([train.length, dev.length, test.length]).toEqual([15, 40, 45]);
    const ids = [...train, ...dev, ...test].map((r) => r.id);
    expect(new Set(ids).size).toBe(100);
    for (const part of [train, dev, test]) {
      const fails = part.filter((r) => r.label === "FAIL").length;
      expect(Math.abs(fails / part.length - 0.4)).toBeLessThan(0.05);
    }
  });

  it("gives the same split for a seed regardless of input order", () => {
    const a = stratifiedSplit(rows(30, 20), 7).map((p) => p.map((r) => r.id));
    const b = stratifiedSplit(rows(30, 20).reverse(), 7).map((p) => p.map((r) => r.id));
    expect(a).toEqual(b);
  });

  it("loads the example dataset", () => {
    const data = loadJsonl(new URL("../data/labeled.example.jsonl", import.meta.url).pathname, true);
    expect(data).toHaveLength(40);
  });
});

group("checks", () => {
  it("reads expectations from the row", () => {
    const row: Row = { id: "x", query: "q", response: "Refunds take 5 to 7 business days." };
    expect(runChecks(row).map((c) => c.name)).toEqual(["not_empty"]);
    const failed = runChecks({ ...row, must_include: ["30 days"], must_not_include: ["business"], max_words: 3 });
    expect(failed.filter((c) => !c.passed).map((c) => c.name)).toEqual(["must_include", "must_not_include", "max_words"]);
  });
});

group("gate", () => {
  const d = (lo: number, hi: number): Delta => ({ n: 100, baseline_rate: 0.8, candidate_rate: 0.8, delta: (lo + hi) / 2, ci: [lo, hi] });

  it("follows the four release-gate bands", () => {
    expect(band(d(0.01, 0.05))).toBe("ship");
    expect(band(d(-0.01, 0.015))).toBe("ship");
    expect(band(d(-0.04, 0.03))).toBe("hold");
    expect(band(d(-0.015, -0.005))).toBe("investigate");
    expect(band(d(-0.08, -0.03))).toBe("block");
  });

  it("detects a clear paired regression", () => {
    const base = new Map(Array.from({ length: 200 }, (_, i) => [String(i), true] as [string, boolean]));
    const cand = new Map(Array.from({ length: 200 }, (_, i) => [String(i), i >= 40] as [string, boolean]));
    const r = pairedDelta(base, cand);
    expect(r.delta).toBeCloseTo(-0.2, 10);
    expect(r.ci[1]).toBeLessThan(0);
    expect(band(r)).toBe("block");
  });
});
