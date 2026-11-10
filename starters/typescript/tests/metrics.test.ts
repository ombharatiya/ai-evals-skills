import { describe as group, expect, it } from "vitest";
import type { Label } from "../src/data.js";
import { cohenKappa, confusion, judgeReport, wilson } from "../src/metrics.js";

const rep = (label: Label, n: number): Label[] => Array.from({ length: n }, () => label);

group("metrics", () => {
  it("counts the confusion matrix with PASS as positive", () => {
    const c = confusion(["PASS", "PASS", "FAIL", "FAIL", "FAIL"], ["PASS", "FAIL", "PASS", "FAIL", "FAIL"]);
    expect(c).toEqual({ tp: 1, fn: 1, fp: 1, tn: 2 });
  });

  it("shows why agreement is not the headline", () => {
    const r = judgeReport([...rep("PASS", 95), ...rep("FAIL", 5)], rep("PASS", 100));
    expect(r.agreement).toBe(0.95);
    expect(r.fail_recall).toBe(0);
    expect(r.kappa).toBe(0);
  });

  it("matches a hand-computed kappa", () => {
    const human = [...rep("PASS", 6), ...rep("FAIL", 4)];
    const judge: Label[] = [...rep("PASS", 5), "FAIL", "PASS", ...rep("FAIL", 3)];
    expect(cohenKappa(confusion(human, judge))).toBeCloseTo((0.8 - 0.52) / 0.48, 12);
  });

  it("matches the guide's Wilson example and stays in [0, 1]", () => {
    const [lo, hi] = wilson(7, 10);
    expect(lo.toFixed(2)).toBe("0.40");
    expect(hi.toFixed(2)).toBe("0.89");
    expect(wilson(10, 10)[1]).toBe(1);
    expect(wilson(0, 10)[0]).toBe(0);
    expect(wilson(0, 0).every(Number.isNaN)).toBe(true);
  });
});
