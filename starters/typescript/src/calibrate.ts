/**
 * Calibrate a judge against human labels.
 *
 *   npm run calibrate -- --data data/labeled.example.jsonl --rubric rubrics/refund_policy.md
 *
 * Splits the labeled set into train / dev / test (stratified, seeded), builds
 * few-shots from train, runs the judge on dev, and reports PASS recall, FAIL
 * recall, Cohen's kappa and agreement with N. Read the disagreements, edit the
 * rubric, run again. When dev clears the bar, run once with `--split test`
 * and keep that number.
 */
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { pathToFileURL } from "node:url";
import { parseArgs } from "node:util";
import { loadJsonl, stratifiedSplit, writeJsonl, type Label, type Row } from "./data.js";
import { claudeJudge, describe, makeConfig, pickFewShots, setupProblem, type Effort, type Judge, type JudgeConfig } from "./judge.js";
import { judgeReport } from "./metrics.js";
import { Budget, judgeRows } from "./runner.js";

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
}) {
  const { rows, judge, config, splitName, splitRows, outDir } = opts;
  const repeats = opts.repeats ?? 1;
  const budget = opts.budget ?? new Budget();
  const runs = [];
  for (let i = 0; i < repeats; i++) runs.push(await judgeRows(judge, splitRows, opts.workers ?? 4, budget));
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
    dataset_rows: rows.length,
    report,
    errors: first.filter((o) => o.error).length,
    fallback_verdicts: scored.filter((o) => o.verdict!.fallbackUsed).length,
    repeats,
    label_flips_across_repeats: flips,
    estimated_cost_usd: Number(budget.spentUsd.toFixed(4)),
  };

  mkdirSync(outDir, { recursive: true });
  writeFileSync(join(outDir, "summary.json"), JSON.stringify(summary, null, 2) + "\n");
  writeFileSync(join(outDir, "rubric.md"), config.rubric);
  writeJsonl(join(outDir, "predictions.jsonl"), predictions);
  writeJsonl(join(outDir, "disagreements.jsonl"), disagreements);
  return summary;
}

const f = (x: number) => x.toFixed(3);
const ci = (pair: [number, number]) => "[" + pair[0].toFixed(2) + ", " + pair[1].toFixed(2) + "]";

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
      model: { type: "string" },
      effort: { type: "string" },
      out: { type: "string", default: "runs" },
    },
  });
  if (!values.data || !values.rubric) {
    console.error("usage: npm run calibrate -- --data <labeled.jsonl> --rubric <rubric.md> [--split dev|test]");
    return 2;
  }
  const split = values.split === "test" ? "test" : "dev";
  const rows = loadJsonl(values.data, true);
  const [train, dev, test] = stratifiedSplit(rows, Number(values.seed));
  const config = makeConfig(readFileSync(values.rubric, "utf8"), {
    ...(values.model ? { model: values.model } : {}),
    ...(values.effort ? { effort: values.effort as Effort } : {}),
  });
  config.fewShots = pickFewShots(train, Number(values["few-shots"]));
  if (split === "test") console.log("Reading the test split. Do this once, after dev clears the bar; keep this number.");

  const stamp = new Date().toISOString().replace(/[-:]/g, "").replace(/\.\d+Z$/, "Z");
  const outDir = join(values.out!, stamp + "-calibrate-" + split);
  const summary = await calibrate({
    rows,
    judge: claudeJudge(config),
    config,
    splitName: split,
    splitRows: split === "dev" ? dev : test,
    outDir,
    repeats: Number(values.repeats),
    workers: Number(values.workers),
    budget: Budget.fromEnv(),
  });

  const r = summary.report;
  const c = r.confusion;
  console.log("split=" + split + "  n=" + r.n + "  train/dev/test=" + [train, dev, test].map((x) => x.length).join("/") + "  judge=" + config.model + "@" + config.effort);
  console.log("  PASS recall (TPR) " + f(r.pass_recall) + "  95% CI " + ci(r.pass_recall_ci) + "  n=" + (c.tp + c.fn));
  console.log("  FAIL recall (TNR) " + f(r.fail_recall) + "  95% CI " + ci(r.fail_recall_ci) + "  n=" + (c.tn + c.fp));
  console.log("  kappa " + f(r.kappa) + "   agreement " + f(r.agreement) + " (not the headline)");
  console.log("  confusion TP=" + c.tp + " FN=" + c.fn + " FP=" + c.fp + " TN=" + c.tn);
  if (summary.errors) console.log("  " + summary.errors + " rows had no verdict; see predictions.jsonl");
  if (summary.fallback_verdicts) console.log("  " + summary.fallback_verdicts + " verdicts came from a fallback model, not the calibrated judge");
  if (summary.label_flips_across_repeats) console.log("  " + summary.label_flips_across_repeats + " labels flipped between repeats: treat small moves as noise");
  console.log("  wrote " + outDir);

  const ok = r.pass_recall >= Number(values["min-recall"]) && r.fail_recall >= Number(values["min-recall"]) && r.kappa >= Number(values["min-kappa"]);
  if (ok && split === "dev") console.log("Dev clears the bar. Run once with --split test and keep that number.");
  else if (!ok) console.log("Not there yet. Read disagreements.jsonl, change one thing in the rubric, run again.");
  return ok ? 0 : 1;
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
