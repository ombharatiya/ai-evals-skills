# TypeScript starter

The same harness as the Python starter, for Node projects:

1. A binary PASS/FAIL judge on Claude, one failure mode per rubric.
2. Calibration against human labels: PASS recall and FAIL recall, each with N and a Wilson
   interval.
3. A scoring run that does code checks before the judge and gates a release on the paired
   difference against a baseline.

It needs Node 20+ and one runtime dependency, `@anthropic-ai/sdk`. The example data is a
support bot for a fictional store, Brightside Goods, checked for getting the refund policy
wrong.

## Setup

```bash
cd starters/typescript
npm install
export ANTHROPIC_API_KEY=...   # or sign in with `ant auth login`
npm test                       # no API calls
npm run typecheck
```

## 1. Calibrate on dev

```bash
npm run calibrate -- --data data/labeled.example.jsonl --rubric rubrics/refund_policy.md
```

The labeled set is split 15 / 40 / 45 into train, dev and test, stratified and seeded.
The split uses its own seeded shuffle, so it differs from the Python starter's split for
the same seed. Read `runs/<timestamp>-calibrate-dev/disagreements.jsonl`, change one thing
in the rubric, and run again. `--repeats 2` counts labels that flip between identical
runs.

## 2. Read test once

```bash
npm run calibrate -- --data data/labeled.example.jsonl --rubric rubrics/refund_policy.md --split test
```

Do this once, after PASS and FAIL recall both clear the bar on dev (0.90 by default, set
with `--min-recall`).

## 3. Score outputs and gate a release

```bash
npm run eval -- --data data/outputs.example.jsonl --rubric rubrics/refund_policy.md \
  --labeled data/labeled.example.jsonl --calibration runs/<calibrate-test-run>/summary.json \
  --baseline runs/<previous-eval-run>/results.jsonl
```

Without `--baseline`, the run becomes the baseline and exits 0. With it, the verdict reads
the 95% interval of the paired difference: ship (exit 0), or hold, investigate or block
(exit 1). Any failing `"must_pass": true` row blocks. Rows with no verdict turn a ship into
a hold.

## Pinning and cost

- The judge defaults to `claude-opus-5-5` at effort `medium` (`--model`, `--effort`, or
  `JUDGE_MODEL` and `JUDGE_EFFORT`). Calibrate again after changing either.
- Each run folder records the model, effort, rubric SHA-256 and few-shot ids.
- `EVAL_BUDGET_USD` stops new judge calls once the estimated spend reaches it.
- Refusal fallbacks are on (`fallbacks: "default"`). A verdict served by a fallback model
  is flagged and counted, because that model wasn't the one you calibrated.

## Files

```
src/
  data.ts       load JSONL, seeded stratified split
  checks.ts     code-graded checks
  judge.ts      the Claude judge, prompt rendering, config pinning
  metrics.ts    confusion matrix, PASS/FAIL recall, kappa, Wilson interval
  runner.ts     worker pool and budget cap
  calibrate.ts  CLI: calibrate on dev or test
  gate.ts       paired bootstrap difference and the four release bands
  run-eval.ts   CLI: score outputs, compare with baseline, exit code for CI
rubrics/        one rubric per failure mode
data/           example labeled set and outputs
tests/          vitest; the judge test drives the real SDK with a fake fetch
```

Method: https://www.aievals.co/cookbook/llm-judge-calibration and
https://www.aievals.co/learn/production/release-gates.
