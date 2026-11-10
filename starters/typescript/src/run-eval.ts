/**
 * Score new outputs and gate a release on them.
 *
 *   npm run eval -- --data data/outputs.example.jsonl --rubric rubrics/refund_policy.md \
 *     --labeled data/labeled.example.jsonl --calibration runs/<calibrate-test>/summary.json \
 *     --baseline runs/<previous-eval>/results.jsonl
 *
 * Code checks run first; a failed check fails the row and skips the judge.
 * The run reports the pass-all rate with a Wilson interval and a per-check
 * drill-down, and writes results.jsonl as soon as scoring finishes. With
 * --baseline it pairs the runs on shared ids, computes the difference with
 * Newcombe's paired interval, and prints ship / hold / investigate / block /
 * invalid. By default the exit code is 0 for ship (or a first run with no
 * baseline) and 1 otherwise. In CI, pass --fail-on investigate so a hold (an
 * interval too wide to call either way) is reported for a person to read
 * instead of failing every build; block and invalid always fail. On GitHub
 * Actions the verdict also shows as an annotation on the pull request.
 *
 * Nothing ships silently:
 * - a failing or missing row marked "must_pass": true (the regression set) blocks
 * - a broken measurement gives INVALID, which always fails: a judge that
 *   differs from the calibrated one, a calibration record that isn't a passing
 *   test read, rows with no verdict, rows missing on either side, fewer than
 *   --min-paired paired rows, or verdicts from a fallback model
 */
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { parseArgs } from "node:util";
import { runChecks } from "./checks.js";
import { isMain, numberFlag } from "./cli.js";
import { dumps, fileSha256, loadJsonl, splitRows, writeJsonl, type Row } from "./data.js";
import { band, noChangeHalfwidth, pairedDelta, rowsForIndifference, type Band } from "./gate.js";
import { EFFORTS, claudeJudge, describe, makeConfig, pickFewShots, priceKnown, setupProblem, type Effort, type Judge } from "./judge.js";
import { wilson } from "./metrics.js";
import { Budget, judgeRows } from "./runner.js";

export interface RowResult {
  id: string;
  must_pass: boolean;
  checks: Record<string, boolean>;
  check_failures: string[];
  judged: boolean;
  judge: string | null;
  critique: string | null;
  served_by: string | null;
  fallback_used: boolean | null;
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
      judged: false,
      judge: null,
      critique: null,
      served_by: null,
      fallback_used: null,
      error: null,
      passed: null,
    });
    if (failed.length === 0) toJudge.push(row);
  }
  for (const o of await judgeRows(judge, toJudge, workers, budget)) {
    const r = results.get(o.row.id)!;
    r.judged = true;
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
export function roganGladen(observed: number, passRecall: number | null | undefined, failRecall: number | null | undefined): number | null {
  if (![observed, passRecall, failRecall].every((v) => typeof v === "number" && Number.isFinite(v))) return null;
  const denom = passRecall! + failRecall! - 1;
  if (denom <= 0) return null;
  return Math.min(1, Math.max(0, (observed + failRecall! - 1) / denom));
}

/** Correct only the judged rows, then put the code-check failures back in. */
export function correctedPassRate(results: RowResult[], passRecall?: number | null, failRecall?: number | null): number | null {
  const known = results.filter((r) => r.passed !== null);
  const judged = known.filter((r) => r.judged && r.check_failures.length === 0);
  if (known.length === 0 || judged.length === 0) return null;
  const observed = judged.filter((r) => r.judge === "PASS").length / judged.length;
  const corrected = roganGladen(observed, passRecall, failRecall);
  return corrected === null ? null : (corrected * judged.length) / known.length;
}

/** The baseline's pass/fail per id, and the ids it marks must_pass. */
export function readBaseline(path: string): { base: Map<string, boolean>; mustPass: Set<string> } {
  const base = new Map<string, boolean>();
  const mustPass = new Set<string>();
  readFileSync(path, "utf8")
    .split("\n")
    .forEach((line, index) => {
      if (!line.trim()) return;
      let row: Partial<RowResult>;
      try {
        row = JSON.parse(line) as Partial<RowResult>;
      } catch {
        throw new Error(path + ":" + (index + 1) + ": not valid JSON; --baseline takes a results.jsonl from an earlier run");
      }
      if (!row || typeof row !== "object" || typeof row.id !== "string" || !("passed" in row)) {
        throw new Error(path + ":" + (index + 1) + ": missing 'id' or 'passed'; --baseline takes a results.jsonl from an earlier run");
      }
      if (row.passed !== null && row.passed !== undefined) base.set(row.id, Boolean(row.passed));
      if (row.must_pass) mustPass.add(row.id);
    });
  if (base.size === 0) throw new Error(path + ": no scored rows");
  return { base, mustPass };
}

/** Escape a GitHub Actions annotation message (ids come from your data). */
export function annotationText(text: string): string {
  return text.replace(/%/g, "%25").replace(/\r/g, "%0D").replace(/\n/g, "%0A");
}

interface CalibrationRecord {
  judge?: unknown;
  split?: string;
  bar?: { passed?: boolean };
  label_flips_across_repeats?: number | null;
  report?: { n?: number; pass_recall?: number | null; fail_recall?: number | null };
}

function readCalibration(path: string): CalibrationRecord {
  let data: CalibrationRecord;
  try {
    data = JSON.parse(readFileSync(path, "utf8"));
  } catch {
    throw new Error(path + ": not valid JSON; --calibration takes a calibrate summary.json");
  }
  if (!data || typeof data !== "object" || !("judge" in data) || !("report" in data)) {
    throw new Error(path + ": --calibration takes the summary.json from a calibrate run");
  }
  return data;
}

async function main(): Promise<number> {
  const { values } = parseArgs({
    options: {
      data: { type: "string" },
      rubric: { type: "string" },
      labeled: { type: "string" },
      seed: { type: "string", default: "42" },
      "few-shots": { type: "string", default: "4" },
      calibration: { type: "string" },
      baseline: { type: "string" },
      indifference: { type: "string", default: "0.01" },
      "minor-regression": { type: "string", default: "0.02" },
      "min-paired": { type: "string", default: "30" },
      "fail-on": { type: "string", default: "hold,investigate" },
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

  // Check every input before spending anything on judge calls.
  let setup;
  try {
    if (values.effort && !EFFORTS.includes(values.effort as Effort)) throw new Error("--effort must be one of " + EFFORTS.join(", "));
    const failOn = new Set((values["fail-on"] ?? "").split(",").map((v) => v.trim()).filter(Boolean));
    if ([...failOn].some((v) => !["hold", "investigate", "block", "invalid"].includes(v))) {
      throw new Error("--fail-on takes a comma list of hold and investigate; block and invalid always fail");
    }
    setup = {
      failOn,
      seed: numberFlag("seed", values.seed, { integer: true }),
      fewShots: numberFlag("few-shots", values["few-shots"], { integer: true, min: 0 }),
      indifference: numberFlag("indifference", values.indifference, { min: 0 }),
      minorRegression: numberFlag("minor-regression", values["minor-regression"], { min: 0 }),
      minPaired: numberFlag("min-paired", values["min-paired"], { integer: true, min: 0 }),
      workers: numberFlag("workers", values.workers, { integer: true, min: 1 }),
      budget: Budget.fromEnv(),
      rows: loadJsonl(values.data),
      baseline: values.baseline ? readBaseline(values.baseline) : null,
      calibration: values.calibration ? readCalibration(values.calibration) : null,
      rubric: readFileSync(values.rubric, "utf8"),
      labeled: values.labeled ? loadJsonl(values.labeled, true) : null,
    };
  } catch (err) {
    console.error((err as Error).message);
    return 2;
  }
  const config = makeConfig(setup.rubric, {
    ...(values.model ? { model: values.model } : {}),
    ...(values.effort ? { effort: values.effort as Effort } : {}),
  });
  if (setup.labeled) {
    config.fewShots = pickFewShots(splitRows(setup.labeled, setup.seed)[0], setup.fewShots);
    const labels = new Set(config.fewShots.map((r) => r.label));
    for (const l of ["PASS", "FAIL"]) {
      if (config.fewShots.length && !labels.has(l as "PASS")) console.error("warning: the few-shots have no " + l + " example: train holds none; label more rows");
    }
  }
  if (setup.budget.limitUsd !== null && !priceKnown(config.model)) {
    console.error("EVAL_BUDGET_USD is set but there's no price for " + config.model + " in src/judge.ts PRICES.");
    return 2;
  }

  const problems: string[] = [];
  let calibrationOk = false;
  let flipRate = 0;
  if (setup.calibration) {
    const cal = setup.calibration;
    if (JSON.stringify(cal.judge) !== JSON.stringify(describe(config))) {
      problems.push("this judge (model, effort, prompt or few-shots) differs from the calibrated one; calibrate again");
    } else if (cal.split !== "test" || !cal.bar?.passed) {
      problems.push("the calibration record isn't a passing test read; calibrate on dev until it clears the bar, then read test");
    } else {
      calibrationOk = true;
    }
    const flips = cal.label_flips_across_repeats;
    if (typeof flips === "number" && cal.report?.n) flipRate = flips / cal.report.n;
  }

  const stamp = new Date().toISOString().replace(/[-:]/g, "").replace(/\.\d+Z$/, "Z");
  const outDir = join(values.out!, stamp + "-eval");
  mkdirSync(outDir, { recursive: true });
  writeFileSync(join(outDir, "rubric.md"), setup.rubric);

  const results = await scoreRows(setup.rows, claudeJudge(config), setup.workers, setup.budget);
  writeJsonl(join(outDir, "results.jsonl"), results); // paid for: save before anything else can fail

  const known = results.filter((r) => r.passed !== null);
  const passed = known.filter((r) => r.passed).length;
  const rate = known.length ? passed / known.length : NaN;
  const [lo, hi] = wilson(passed, known.length);
  const checkFails: Record<string, number> = {};
  for (const r of results) for (const [name, ok] of Object.entries(r.checks)) if (!ok) checkFails[name] = (checkFails[name] ?? 0) + 1;
  const judgeFails = results.filter((r) => r.judge === "FAIL").length;
  let mustFail = results.filter((r) => r.must_pass && r.passed === false).map((r) => r.id);
  const errors = results.filter((r) => r.error);
  const fallbacks = results.filter((r) => r.fallback_used).length;
  if (errors.length) problems.push(errors.length + " rows have no verdict (refusal, budget or bad output); see results.jsonl");
  if (fallbacks) problems.push(fallbacks + " verdicts came from a fallback model, not the calibrated judge");

  const summary: Record<string, unknown> = {
    date: new Date().toISOString(),
    judge: describe(config),
    dataset: { path: values.data, sha256: fileSha256(values.data) },
    n: results.length,
    scored: known.length,
    pass_rate: rate,
    pass_rate_ci: [lo, hi],
    check_failures: checkFails,
    judge_failures: judgeFails,
    fallback_verdicts: fallbacks,
    must_pass_failures: mustFail,
    errors: errors.length,
    estimated_cost_usd: Number(setup.budget.spentUsd.toFixed(4)),
  };
  if (setup.calibration && calibrationOk) {
    summary.pass_rate_corrected = correctedPassRate(results, setup.calibration.report?.pass_recall, setup.calibration.report?.fail_recall);
  }

  // Without a baseline there is nothing to compare: this run becomes the baseline.
  let verdict: Band | "baseline" | "invalid" = "baseline";
  if (setup.baseline) {
    const { base, mustPass } = setup.baseline;
    const cand = new Map(known.map((r) => [r.id, r.passed as boolean]));
    const unpaired = [...cand.keys()].filter((id) => !base.has(id)).sort();
    if (unpaired.length) problems.push(unpaired.length + " ids aren't in the baseline and weren't compared: " + unpaired.slice(0, 10).join(", "));
    const produced = new Set(results.map((r) => r.id));
    const absent = [...base.keys()].filter((id) => !produced.has(id)).sort();
    if (absent.length) problems.push(absent.length + " baseline ids are missing from this run: " + absent.slice(0, 10).join(", "));
    const absentMust = absent.filter((id) => mustPass.has(id));
    if (absentMust.length) {
      // A regression row the candidate didn't produce can't count as passing.
      mustFail = [...mustFail, ...absentMust];
      summary.must_pass_failures = mustFail;
    }
    if (![...cand.keys()].some((id) => base.has(id))) {
      problems.push("the baseline shares no scored ids with this run, so nothing was compared");
    } else {
      const d = pairedDelta(base, cand);
      verdict = band(d, setup.indifference, setup.minorRegression);
      if (d.n < setup.minPaired) problems.push("only " + d.n + " paired rows; a verdict needs at least " + setup.minPaired);
      summary.vs_baseline = {
        n: d.n,
        baseline_rate: d.baseline_rate,
        candidate_rate: d.candidate_rate,
        delta: d.delta,
        ci: d.ci,
        band: verdict,
        unpaired_ids: unpaired,
        missing_ids: absent,
        paired_counts: { both_pass: d.both_pass, baseline_only: d.baseline_only, candidate_only: d.candidate_only, both_fail: d.both_fail },
      };
    }
  }
  // The gate set's resolution: how wide the interval is for an unchanged
  // candidate, and how many rows the indifference window needs.
  const vsBase = summary.vs_baseline as { n: number; baseline_rate: number } | undefined;
  const resN = vsBase ? vsBase.n : known.length;
  const resRate = vsBase ? vsBase.baseline_rate : rate;
  const notes: string[] = [];
  if (resN > 0 && Number.isFinite(resRate)) {
    const half = noChangeHalfwidth(resN, resRate, flipRate);
    const needed = rowsForIndifference(resRate, setup.indifference, resN, 1_000_000, flipRate);
    summary.resolution = { n: resN, flip_rate: flipRate, no_change_halfwidth: half, rows_for_indifference: needed };
    const flipsText = flipRate ? " and the judge flipping " + (flipRate * 100).toFixed(1) + "% of labels between runs" : " and a judge that never flips a label";
    if (half > setup.indifference) {
      notes.push(
        "with " + resN + " rows at this pass rate" + flipsText + ", an unchanged candidate's interval is at least +-" + half.toFixed(3) +
          ", wider than the " + setup.indifference.toFixed(3) + " indifference window, so a candidate can't ship by staying the same. " +
          "This is a floor: regenerated outputs and judge noise widen it. Size the window from an A/A run (score the unchanged system twice and compare)" +
          (needed ? ", or grow the gate set past " + needed + " rows" : ""),
      );
    }
  }
  summary.notes = notes;
  // Integrity problems mean the measurement itself can't be trusted, so they
  // always fail, whatever --fail-on says. --fail-on only decides whether an
  // honest hold (an interval too wide to call) or an investigate fails.
  if (mustFail.length) verdict = "block";
  else if (problems.length) verdict = "invalid";
  summary.verdict = verdict;
  summary.problems = problems;
  writeFileSync(join(outDir, "summary.json"), dumps(summary, 2) + "\n");

  console.log("pass-all rate " + (Number.isFinite(rate) ? rate.toFixed(3) : "n/a") + "  95% CI [" + lo.toFixed(2) + ", " + hi.toFixed(2) + "]  n=" + known.length + "  judge=" + config.model + "@" + config.effort);
  if (typeof summary.pass_rate_corrected === "number") console.log("  corrected for judge error: " + summary.pass_rate_corrected.toFixed(3) + " (point estimate)");
  for (const [name, count] of Object.entries(checkFails).sort()) console.log("  code check " + name + ": " + count + " failed");
  console.log("  judge: " + judgeFails + " failed");
  const vs = summary.vs_baseline as { n: number; baseline_rate: number; candidate_rate: number; delta: number; ci: [number, number]; band: Band } | undefined;
  const sign = (x: number) => (x >= 0 ? "+" : "") + x.toFixed(3);
  if (vs) {
    console.log("  vs baseline: " + vs.baseline_rate.toFixed(3) + " -> " + vs.candidate_rate.toFixed(3) + "  delta " + sign(vs.delta) + "  95% CI [" + sign(vs.ci[0]) + ", " + sign(vs.ci[1]) + "]  n=" + vs.n + "  band=" + vs.band);
  }
  if (mustFail.length) console.log("  regression set failures: " + mustFail.join(", "));
  for (const reason of problems) console.log("  problem: " + reason);
  for (const note of notes) console.log("  note: " + note);
  console.log("  estimated cost $" + (summary.estimated_cost_usd as number).toFixed(2));
  console.log("verdict: " + verdict.toUpperCase() + "   wrote " + outDir);
  const failed = verdict === "block" || verdict === "invalid" || setup.failOn.has(verdict);
  if (process.env.GITHUB_ACTIONS === "true" && verdict !== "ship" && verdict !== "baseline") {
    // Show the verdict on the pull request even when it doesn't fail the job.
    const details = [...problems];
    if (mustFail.length) details.push("regression rows failed: " + mustFail.join(", "));
    if (vs) details.push("delta " + sign(vs.delta) + ", 95% CI [" + sign(vs.ci[0]) + ", " + sign(vs.ci[1]) + "], n=" + vs.n);
    details.push("read the rows in results.jsonl (the eval-run artifact in CI)");
    console.log("::" + (failed ? "error" : "warning") + " title=Release gate " + verdict.toUpperCase() + "::" + annotationText(details.join("; ")));
  }
  return failed ? 1 : 0;
}

if (isMain(import.meta.url)) {
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
