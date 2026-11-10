/**
 * Calibrate a judge against human labels.
 *
 *   npm run calibrate -- --data data/labeled.example.jsonl --rubric rubrics/refund_policy.md
 *
 * Splits the labeled set into train / dev / test by a stable hash of each id,
 * builds few-shots from train, runs the judge on dev, and reports PASS recall,
 * FAIL recall, Cohen's kappa and agreement with N. Read the disagreements,
 * edit the rubric, run again. When dev clears the bar, run once with
 * `--split test` and keep that number.
 *
 * The bar: every split row gets a verdict, each class has at least
 * --min-per-class rows, PASS and FAIL recall reach --min-recall, and kappa
 * reaches --min-kappa.
 */
import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { parseArgs } from "node:util";
import { isMain, numberFlag } from "./cli.js";
import { classCounts, dumps, fileSha256, loadJsonl, readText, splitRows, writeJsonl, type Label, type Row } from "./data.js";
import { EFFORTS, claudeJudge, describe, makeConfig, pickFewShots, priceKnown, setupProblem, type Effort, type Judge, type JudgeConfig } from "./judge.js";
import { judgeReport } from "./metrics.js";
import { BUDGET_STOP, Budget, judgeRows } from "./runner.js";

export const TRAIN_FRACTION = 0.15;
export const DEV_FRACTION = 0.4;

export async function calibrate(opts: {
  rows: Row[];
  judge: Judge;
  config: JudgeConfig;
  splitName: "dev" | "test";
  splitRows: Row[];
  outDir: string;
  repeats?: number;
  workers?: number;
  budget?: Budget;
  dataset?: Record<string, unknown>;
}) {
  const { rows, judge, config, splitName, splitRows: part, outDir } = opts;
  const repeats = opts.repeats ?? 1;
  const budget = opts.budget ?? new Budget();
  const runs = [];
  for (let i = 0; i < repeats; i++) runs.push(await judgeRows(judge, part, opts.workers ?? 4, budget));
  const first = runs[0]!;
  const scored = first.filter((o) => o.verdict);
  const report = judgeReport(
    scored.map((o) => o.row.label as Label),
    scored.map((o) => o.verdict!.label),
  );

  let flips: number | null = null;
  if (repeats > 1) {
    const maps = runs.map((run) => new Map(run.filter((o) => o.verdict).map((o) => [o.row.id, o.verdict!.label])));
    const common = [...maps[0]!.keys()].filter((id) => maps.every((m) => m.has(id)));
    flips = common.filter((id) => new Set(maps.map((m) => m.get(id))).size > 1).length;
  }

  const predictions = first.map((o) => ({
    id: o.row.id,
    human: o.row.label,
    judge: o.verdict?.label ?? null,
    critique: o.verdict?.critique ?? null,
    served_by: o.verdict?.model ?? null,
    fallback_used: o.verdict?.fallbackUsed ?? null,
    error: o.error ?? null,
    query: o.row.query,
    response: o.row.response,
    human_critique: o.row.critique ?? null,
  }));
  const disagreements = predictions.filter((p) => p.judge && p.judge !== p.human);

  const summary = {
    date: new Date().toISOString(),
    split: splitName,
    judge: describe(config),
    dataset: opts.dataset ?? {},
    dataset_rows: rows.length,
    split_rows: part.length,
    split_class_counts: classCounts(part),
    report,
    errors: first.filter((o) => o.error).length,
    // Calls the budget kept from running, in any repeat: the bar can't be read then.
    budget_stopped_calls: runs.flat().filter((o) => o.error?.startsWith(BUDGET_STOP)).length,
    fallback_verdicts: scored.filter((o) => o.verdict!.fallbackUsed).length,
    repeats,
    label_flips_across_repeats: flips,
    estimated_cost_usd: Number(budget.spentUsd.toFixed(4)),
  };

  mkdirSync(outDir, { recursive: true });
  writeFileSync(join(outDir, "summary.json"), dumps(summary, 2) + "\n");
  writeFileSync(join(outDir, "rubric.md"), config.rubric);
  writeJsonl(join(outDir, "predictions.jsonl"), predictions);
  // Test disagreements are left out on purpose: reading them invites tuning on test.
  if (splitName === "dev") writeJsonl(join(outDir, "disagreements.jsonl"), disagreements);
  return summary;
}

export function barFailures(summary: Awaited<ReturnType<typeof calibrate>>, minRecall: number, minKappa: number, minPerClass: number): string[] {
  const r = summary.report;
  const reasons: string[] = [];
  if (summary.budget_stopped_calls) reasons.push("the budget ran out: " + summary.budget_stopped_calls + " judge calls didn't run");
  if (summary.errors) reasons.push(summary.errors + " rows got no verdict (see predictions.jsonl)");
  if (summary.fallback_verdicts) reasons.push(summary.fallback_verdicts + " verdicts came from a fallback model, not this judge");
  for (const [label, n] of Object.entries(summary.split_class_counts)) {
    if (n < minPerClass) reasons.push("only " + n + " " + label + " rows in this split; the bar needs " + minPerClass + " per class");
  }
  if (!(r.pass_recall >= minRecall)) reasons.push("pass recall below " + minRecall.toFixed(2));
  if (!(r.fail_recall >= minRecall)) reasons.push("fail recall below " + minRecall.toFixed(2));
  if (!(r.kappa >= minKappa)) reasons.push("kappa below " + minKappa.toFixed(2));
  return reasons;
}

/** The split isn't stratified, so a small train split can miss a class. */
export function fewShotWarnings(fewShots: Row[]): string[] {
  if (fewShots.length === 0) return [];
  const labels = new Set(fewShots.map((r) => r.label));
  return (["PASS", "FAIL"] as const).filter((l) => !labels.has(l)).map((l) => "the few-shots have no " + l + " example: train holds none; label more rows");
}

const f = (x: number) => (Number.isFinite(x) ? x.toFixed(3) : "n/a");
const ci = (pair: [number, number]) => (Number.isFinite(pair[0]) ? "[" + pair[0].toFixed(2) + ", " + pair[1].toFixed(2) + "]" : "[n/a]");

async function main(): Promise<number> {
  const { values } = parseArgs({
    options: {
      data: { type: "string" },
      rubric: { type: "string" },
      split: { type: "string", default: "dev" },
      seed: { type: "string", default: "42" },
      "few-shots": { type: "string", default: "4" },
      repeats: { type: "string", default: "1" },
      workers: { type: "string", default: "4" },
      "min-recall": { type: "string", default: "0.90" },
      "min-kappa": { type: "string", default: "0.60" },
      "min-per-class": { type: "string", default: "30" },
      model: { type: "string" },
      effort: { type: "string" },
      out: { type: "string", default: "runs" },
    },
  });
  if (!values.data || !values.rubric) {
    console.error("usage: npm run calibrate -- --data <labeled.jsonl> --rubric <rubric.md> [--split dev|test]");
    return 2;
  }
  let opts;
  try {
    if (values.split !== "dev" && values.split !== "test") throw new Error("--split must be dev or test, got " + values.split);
    if (values.effort && !EFFORTS.includes(values.effort as Effort)) throw new Error("--effort must be one of " + EFFORTS.join(", "));
    opts = {
      split: values.split as "dev" | "test",
      seed: numberFlag("seed", values.seed, { integer: true }),
      fewShots: numberFlag("few-shots", values["few-shots"], { integer: true, min: 0 }),
      repeats: numberFlag("repeats", values.repeats, { integer: true, min: 1 }),
      workers: numberFlag("workers", values.workers, { integer: true, min: 1 }),
      minRecall: numberFlag("min-recall", values["min-recall"], { min: 0 }),
      minKappa: numberFlag("min-kappa", values["min-kappa"]),
      minPerClass: numberFlag("min-per-class", values["min-per-class"], { integer: true, min: 0 }),
      budget: Budget.fromEnv(),
      rows: loadJsonl(values.data, true),
      rubric: readText(values.rubric),
    };
  } catch (err) {
    console.error((err as Error).message);
    return 2;
  }
  const rows = opts.rows;
  const [train, dev, test] = splitRows(rows, opts.seed, TRAIN_FRACTION, DEV_FRACTION);
  const config = makeConfig(opts.rubric, {
    ...(values.model ? { model: values.model } : {}),
    ...(values.effort ? { effort: values.effort as Effort } : {}),
  });
  config.fewShots = pickFewShots(train, opts.fewShots);
  if (opts.budget.limitUsd !== null && !priceKnown(config.model)) {
    console.error("EVAL_BUDGET_USD is set but there's no price for " + config.model + " in src/judge.ts PRICES.");
    return 2;
  }
  if (opts.split === "test") console.log("Reading the test split. Do this once, after dev clears the bar; keep this number.");

  const stamp = new Date().toISOString().replace(/[-:]/g, "").replace(/\.\d+Z$/, "Z");
  const outDir = join(values.out!, stamp + "-calibrate-" + opts.split);
  const summary = await calibrate({
    rows,
    judge: claudeJudge(config),
    config,
    splitName: opts.split,
    splitRows: opts.split === "dev" ? dev : test,
    outDir,
    repeats: opts.repeats,
    workers: opts.workers,
    budget: opts.budget,
    dataset: {
      path: values.data,
      sha256: fileSha256(values.data),
      seed: opts.seed,
      fractions: { train: TRAIN_FRACTION, dev: DEV_FRACTION, test: Number((1 - TRAIN_FRACTION - DEV_FRACTION).toFixed(2)) },
      few_shots: opts.fewShots,
    },
  });

  const reasons = barFailures(summary, opts.minRecall, opts.minKappa, opts.minPerClass);
  const warnings = fewShotWarnings(config.fewShots);
  // Record the result, so run-eval can refuse a calibration that didn't pass.
  const record = {
    ...summary,
    bar: { passed: reasons.length === 0, reasons, min_recall: opts.minRecall, min_kappa: opts.minKappa, min_per_class: opts.minPerClass },
    warnings,
  };
  writeFileSync(join(outDir, "summary.json"), dumps(record, 2) + "\n");

  const r = summary.report;
  const c = r.confusion;
  const counts = summary.split_class_counts;
  console.log("split=" + opts.split + "  n=" + r.n + "  train/dev/test=" + [train, dev, test].map((x) => x.length).join("/") + "  judge=" + config.model + "@" + config.effort);
  console.log("  PASS recall (TPR) " + f(r.pass_recall) + "  95% CI " + ci(r.pass_recall_ci) + "  n=" + (c.tp + c.fn) + " of " + counts.PASS + " PASS rows");
  console.log("  FAIL recall (TNR) " + f(r.fail_recall) + "  95% CI " + ci(r.fail_recall_ci) + "  n=" + (c.tn + c.fp) + " of " + counts.FAIL + " FAIL rows");
  console.log("  kappa " + f(r.kappa) + "   agreement " + f(r.agreement) + " (not the headline)");
  console.log("  confusion TP=" + c.tp + " FN=" + c.fn + " FP=" + c.fp + " TN=" + c.tn);
  if (summary.fallback_verdicts) console.log("  " + summary.fallback_verdicts + " verdicts came from a fallback model, not the calibrated judge");
  if (summary.label_flips_across_repeats) console.log("  " + summary.label_flips_across_repeats + " labels flipped between repeats: treat small moves as noise");
  for (const warning of warnings) console.log("  warning: " + warning);
  console.log("  estimated cost $" + summary.estimated_cost_usd.toFixed(2) + "   wrote " + outDir);

  if (reasons.length === 0 && opts.split === "dev") console.log("Dev clears the bar. Run once with --split test and keep that number.");
  else if (reasons.length === 0) console.log("Test clears the bar. This is the judge's calibration record; keep summary.json.");
  else {
    console.log("Not there yet:");
    for (const reason of reasons) console.log("  - " + reason);
    if (summary.budget_stopped_calls) {
      console.log("The budget ran out before every call ran, so this isn't a verdict on the judge. Raise EVAL_BUDGET_USD and run the same command again.");
    } else if (opts.split === "dev") {
      console.log("Read disagreements.jsonl, change one thing in the rubric (or label more rows), run again.");
    } else {
      console.log("Don't tune on test. Label a fresh batch, calibrate on dev again, and treat the next test read as a re-read: it includes rows you have already seen.");
    }
  }
  return reasons.length === 0 ? 0 : 1;
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
