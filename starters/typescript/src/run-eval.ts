/**
 * Score new outputs and gate a release on them.
 *
 *   npm run eval -- --data data/outputs.example.jsonl --rubric rubrics/refund_policy.md \
 *     --labeled data/labeled.example.jsonl --calibration runs/<calibrate-test>/summary.json \
 *     --baseline runs/<previous-eval>/results.jsonl
 *
 * Code checks run first; a failed check fails the row and skips the judge.
 * The run reports the pass-all rate with a Wilson interval and a per-check
 * drill-down. With --baseline it computes the paired difference and prints
 * ship / hold / investigate / block. Exit code 0 for ship (or a first run
 * with no baseline), 1 otherwise. Rows with "must_pass": true are the
 * regression set: one failure blocks. Rows with no verdict turn a ship into
 * a hold, so nothing passes silently.
 */
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { pathToFileURL } from "node:url";
import { parseArgs } from "node:util";
import { runChecks } from "./checks.js";
import { loadJsonl, stratifiedSplit, writeJsonl, type Row } from "./data.js";
import { band, pairedDelta, type Band } from "./gate.js";
import { claudeJudge, describe, makeConfig, pickFewShots, setupProblem, type Effort, type Judge } from "./judge.js";
import { wilson } from "./metrics.js";
import { Budget, judgeRows } from "./runner.js";

export interface RowResult {
  id: string;
  must_pass: boolean;
  checks: Record<string, boolean>;
  check_failures: string[];
  judge: string | null;
  critique: string | null;
  served_by: string | null;
  fallback_used?: boolean;
  error: string | null;
  passed: boolean | null;
}

export async function scoreRows(rows: Row[], judge: Judge, workers = 4, budget = new Budget()): Promise<RowResult[]> {
  const results = new Map<string, RowResult>();
  const toJudge: Row[] = [];
  for (const row of rows) {
    const checks = runChecks(row);
    const failed = checks.filter((c) => !c.passed);
    results.set(row.id, {
      id: row.id,
      must_pass: Boolean(row.must_pass),
      checks: Object.fromEntries(checks.map((c) => [c.name, c.passed])),
      check_failures: failed.map((c) => c.name + ": " + c.detail),
      judge: null,
      critique: null,
      served_by: null,
      error: null,
      passed: null,
    });
    if (failed.length === 0) toJudge.push(row);
  }
  for (const o of await judgeRows(judge, toJudge, workers, budget)) {
    const r = results.get(o.row.id)!;
    if (o.verdict) {
      r.judge = o.verdict.label;
      r.critique = o.verdict.critique;
      r.served_by = o.verdict.model;
      r.fallback_used = o.verdict.fallbackUsed;
    } else {
      r.error = o.error ?? "no verdict";
    }
  }
  for (const r of results.values()) {
    r.passed = r.error ? null : r.check_failures.length === 0 && r.judge === "PASS";
  }
  return rows.map((row) => results.get(row.id)!);
}

/** Rogan-Gladen correction for judge error (point estimate only). */
export function correctedRate(observed: number, passRecall: number, failRecall: number): number | null {
  const denom = passRecall + failRecall - 1;
  if (denom <= 0) return null;
  return Math.min(1, Math.max(0, (observed + failRecall - 1) / denom));
}

async function main(): Promise<number> {
  const { values } = parseArgs({
    options: {
      data: { type: "string" },
      rubric: { type: "string" },
      labeled: { type: "string" },
      calibration: { type: "string" },
      baseline: { type: "string" },
      window: { type: "string", default: "0.02" },
      workers: { type: "string", default: "4" },
      model: { type: "string" },
      effort: { type: "string" },
      out: { type: "string", default: "runs" },
    },
  });
  if (!values.data || !values.rubric) {
    console.error("usage: npm run eval -- --data <outputs.jsonl> --rubric <rubric.md> [--labeled ...] [--calibration ...] [--baseline ...]");
    return 2;
  }
  const rows = loadJsonl(values.data);
  const config = makeConfig(readFileSync(values.rubric, "utf8"), {
    ...(values.model ? { model: values.model } : {}),
    ...(values.effort ? { effort: values.effort as Effort } : {}),
  });
  if (values.labeled) config.fewShots = pickFewShots(stratifiedSplit(loadJsonl(values.labeled, true))[0]);

  const calibration = values.calibration ? JSON.parse(readFileSync(values.calibration, "utf8")) : null;
  if (calibration && JSON.stringify(calibration.judge) !== JSON.stringify(describe(config))) {
    console.log("WARNING: this judge (model, effort, rubric or few-shots) differs from the calibrated one.");
    console.log("         The calibration numbers do not apply to it. Calibrate again.");
  }

  const budget = Budget.fromEnv();
  const results = await scoreRows(rows, claudeJudge(config), Number(values.workers), budget);
  const known = results.filter((r) => r.passed !== null);
  const passed = known.filter((r) => r.passed).length;
  const rate = known.length ? passed / known.length : NaN;
  const [lo, hi] = wilson(passed, known.length);
  const checkFails: Record<string, number> = {};
  for (const r of results) for (const [name, ok] of Object.entries(r.checks)) if (!ok) checkFails[name] = (checkFails[name] ?? 0) + 1;
  const judgeFails = results.filter((r) => r.judge === "FAIL").length;
  const mustFail = results.filter((r) => r.must_pass && r.passed === false).map((r) => r.id);
  const errors = results.filter((r) => r.error);

  const summary: Record<string, unknown> = {
    date: new Date().toISOString(),
    judge: describe(config),
    n: results.length,
    scored: known.length,
    pass_rate: rate,
    pass_rate_ci: [lo, hi],
    check_failures: checkFails,
    judge_failures: judgeFails,
    must_pass_failures: mustFail,
    errors: errors.length,
    estimated_cost_usd: Number(budget.spentUsd.toFixed(4)),
  };
  if (calibration) {
    summary.pass_rate_corrected = correctedRate(rate, calibration.report.pass_recall, calibration.report.fail_recall);
  }

  // Without a baseline there is nothing to compare: this run becomes the baseline.
  let verdict: Band | "baseline" = "baseline";
  if (values.baseline) {
    const base = new Map<string, boolean>();
    for (const line of readFileSync(values.baseline, "utf8").split("\n")) {
      if (!line.trim()) continue;
      const b = JSON.parse(line) as RowResult;
      if (b.passed !== null) base.set(b.id, b.passed);
    }
    const d = pairedDelta(base, new Map(known.map((r) => [r.id, r.passed as boolean])));
    verdict = band(d, Number(values.window));
    summary.vs_baseline = { ...d, band: verdict };
  }
  if (mustFail.length) verdict = "block";
  if (errors.length && (verdict === "ship" || verdict === "baseline")) verdict = "hold";
  summary.verdict = verdict;

  const stamp = new Date().toISOString().replace(/[-:]/g, "").replace(/\.\d+Z$/, "Z");
  const outDir = join(values.out!, stamp + "-eval");
  mkdirSync(outDir, { recursive: true });
  writeJsonl(join(outDir, "results.jsonl"), results);
  writeFileSync(join(outDir, "summary.json"), JSON.stringify(summary, null, 2) + "\n");

  console.log("pass-all rate " + rate.toFixed(3) + "  95% CI [" + lo.toFixed(2) + ", " + hi.toFixed(2) + "]  n=" + known.length + "  judge=" + config.model + "@" + config.effort);
  if (typeof summary.pass_rate_corrected === "number") console.log("  corrected for judge error: " + summary.pass_rate_corrected.toFixed(3) + " (point estimate)");
  for (const [name, count] of Object.entries(checkFails).sort()) console.log("  code check " + name + ": " + count + " failed");
  console.log("  judge: " + judgeFails + " failed");
  const vs = summary.vs_baseline as (ReturnType<typeof pairedDelta> & { band: Band }) | undefined;
  if (vs) {
    const sign = (x: number) => (x >= 0 ? "+" : "") + x.toFixed(3);
    console.log("  vs baseline: " + vs.baseline_rate.toFixed(3) + " -> " + vs.candidate_rate.toFixed(3) + "  delta " + sign(vs.delta) + "  95% CI [" + sign(vs.ci[0]) + ", " + sign(vs.ci[1]) + "]  n=" + vs.n);
  }
  if (mustFail.length) console.log("  regression set failures: " + mustFail.join(", "));
  if (errors.length) console.log("  " + errors.length + " rows have no verdict (refusal, budget, or bad output); see results.jsonl");
  console.log("verdict: " + verdict.toUpperCase() + "   wrote " + outDir);
  return verdict === "ship" || verdict === "baseline" ? 0 : 1;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().then(
    (code) => process.exit(code),
    (err: unknown) => {
      const problem = setupProblem(err);
      if (!problem) throw err;
      console.error("Can't run the judge: " + problem);
      console.error("Set ANTHROPIC_API_KEY (or run `ant auth login`) and check --model.");
      process.exit(2);
    },
  );
}
