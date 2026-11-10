/**
 * Judge-versus-human metrics and confidence intervals.
 *
 * PASS is the positive class, so PASS recall is the TPR and FAIL recall the
 * TNR. Some writing (Hamel Husain's FAQ among it) treats FAIL as positive
 * and calls FAIL recall the TPR, so the code names the rates by class.
 */
import type { Label } from "./data.js";

export interface Confusion {
  tp: number; // human PASS, judge PASS
  fn: number; // human PASS, judge FAIL
  fp: number; // human FAIL, judge PASS: a real failure let through
  tn: number; // human FAIL, judge FAIL
}

export function confusion(human: Label[], judge: Label[]): Confusion {
  if (human.length !== judge.length) throw new Error("human and judge label lists differ in length");
  const c: Confusion = { tp: 0, fn: 0, fp: 0, tn: 0 };
  human.forEach((h, i) => {
    const j = judge[i];
    if (h === "PASS" && j === "PASS") c.tp++;
    else if (h === "PASS") c.fn++;
    else if (j === "PASS") c.fp++;
    else c.tn++;
  });
  return c;
}

const ratio = (num: number, den: number) => (den ? num / den : NaN);

/** Wilson score interval for a proportion (95% by default). */
export function wilson(successes: number, n: number, z = 1.96): [number, number] {
  if (n === 0) return [NaN, NaN];
  const p = successes / n;
  const denom = 1 + (z * z) / n;
  const centre = (p + (z * z) / (2 * n)) / denom;
  const half = (z * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n))) / denom;
  return [Math.max(0, centre - half), Math.min(1, centre + half)];
}

/** Agreement corrected for chance. 1 is perfect, 0 is no better than chance. */
export function cohenKappa(c: Confusion): number {
  const n = c.tp + c.fn + c.fp + c.tn;
  if (n === 0) return NaN;
  const observed = (c.tp + c.tn) / n;
  const humanPass = (c.tp + c.fn) / n;
  const judgePass = (c.tp + c.fp) / n;
  const expected = humanPass * judgePass + (1 - humanPass) * (1 - judgePass);
  return expected === 1 ? NaN : (observed - expected) / (1 - expected);
}

export interface JudgeReport {
  n: number;
  confusion: Confusion;
  pass_recall: number;
  fail_recall: number;
  pass_recall_ci: [number, number];
  fail_recall_ci: [number, number];
  agreement: number;
  kappa: number;
}

export function judgeReport(human: Label[], judge: Label[]): JudgeReport {
  const c = confusion(human, judge);
  const n = c.tp + c.fn + c.fp + c.tn;
  return {
    n,
    confusion: c,
    pass_recall: ratio(c.tp, c.tp + c.fn),
    fail_recall: ratio(c.tn, c.tn + c.fp),
    pass_recall_ci: wilson(c.tp, c.tp + c.fn),
    fail_recall_ci: wilson(c.tn, c.tn + c.fp),
    agreement: ratio(c.tp + c.tn, n),
    kappa: cohenKappa(c),
  };
}
