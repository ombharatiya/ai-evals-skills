# TypeScript starter

The same harness as the Python starter, for Node projects:

1. A binary PASS/FAIL judge on Claude, one failure mode per rubric.
2. Calibration against human labels: PASS recall and FAIL recall, each with N and a Wilson
   interval.
3. A scoring run that does code checks before the judge and gates a release on the paired
   difference against a baseline.

It needs Node 22.12+ (Vitest 5 requires it) and one runtime dependency,
`@anthropic-ai/sdk`. The example data is a support bot for a fictional store, Brightside
Goods, checked for getting the refund policy wrong.

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

The labeled set is split into train, dev and test (about 15 / 40 / 45) by a hash of each
id: the same hash as the Python starter, so both put the same rows in the same splits, and
adding rows never moves an existing one. Read
`runs/<timestamp>-calibrate-dev/disagreements.jsonl`, change one thing in the rubric, and run
again. `--repeats 2` counts labels that flip between identical runs.

The bar: every split row gets a verdict (none from a fallback model), each class has at least 30 rows in the split
(`--min-per-class`), PASS and FAIL recall reach 0.90 (`--min-recall`), and kappa reaches 0.60
(`--min-kappa`). The 40-row example set shows the loop but can't clear it.

## 2. Read test once

```bash
npm run calibrate -- --data data/labeled.example.jsonl --rubric rubrics/refund_policy.md --split test
```

Do this once, after dev clears the bar, and keep the `summary.json`. A test run writes no
disagreements file; if it misses the bar, label fresh rows instead of tuning on test.

## 3. Score outputs and gate a release

```bash
npm run eval -- --data data/outputs.example.jsonl --rubric rubrics/refund_policy.md \
  --labeled data/labeled.example.jsonl --calibration runs/<calibrate-test-run>/summary.json \
  --baseline runs/<previous-eval-run>/results.jsonl
```

With `--baseline`, the verdict reads the 95% interval of the paired difference (Newcombe's
interval) against a 1-point indifference window and a 2-point minor-regression window:
ship exits 0, and hold, investigate or block exit 1.

Some conditions override the band:

- **Block.** A failing `"must_pass": true` row blocks, and so does a baseline `must_pass` row
  that's missing from this run.
- **Hold.** Any of these turns a ship into a hold:
  - rows with no verdict
  - candidate ids missing from the baseline, or baseline ids missing from this run
  - fewer than 30 paired rows (`--min-paired`)
  - fallback verdicts
  - a judge fingerprint that differs from the calibration record
  - a calibration record that isn't a passing test read
- **Exit code.** Without `--baseline`, the run becomes the baseline and exits 0, unless one
  of the conditions above applies.

Pass the same `--labeled`, `--seed` and `--few-shots` you calibrated with.

Every run prints the gate set's resolution: how wide the interval is for a candidate that
matches the baseline on every row. At a 76% pass rate that's about plus or minus 5.3 points
with 50 rows, so the 1-point default window needs at least 280 rows, and about 830 if the
judge flips 2% of its labels between identical runs. With a `--repeats 2` calibration record
passed as `--calibration`, the resolution uses the measured flip rate.

These figures are floors. When CI regenerates outputs on every pull request, generation
noise and judge noise widen the interval further, so size the window from an A/A run:
generate and score the unchanged system twice, compare the second run with the first using
`--baseline`, and set `--indifference` a little above the interval that prints. In CI, pass
`--fail-on investigate,block`: a hold then shows as a warning on the pull request for a
person to read, and only a confirmed regression or a regression-set failure fails the job.

## Pinning and cost

- The judge defaults to `claude-opus-5-5` at effort `medium` (`--model`, `--effort`, or
  `JUDGE_MODEL` and `JUDGE_EFFORT`). Calibrate again after changing either.
- Each run folder records the judge's fingerprint (model, effort, and a hash of the
  rendered prompt, schema and max_tokens), the few-shot ids, and the dataset's hash, seed
  and split fractions.
- `EVAL_BUDGET_USD` stops new judge calls once the estimated spend reaches it, failed calls
  included. It must be a plain number. With a budget set, a model missing from `PRICES` in
  `src/judge.ts` is refused.
- Refusal fallbacks are on (`fallbacks: "default"`). A verdict served by a fallback model is
  counted in the summary and holds a release, because that model wasn't the one you
  calibrated.

## Files

```
src/
  data.ts       load JSONL, hash-based split, JSON helpers
  checks.ts     code-graded checks
  judge.ts      the Claude judge, prompt rendering, fingerprint, pricing
  metrics.ts    confusion matrix, PASS/FAIL recall, kappa, Wilson interval
  runner.ts     worker pool and budget cap
  calibrate.ts  CLI: calibrate on dev or test
  gate.ts       Newcombe paired interval and the four release bands
  run-eval.ts   CLI: score outputs, compare with baseline, exit code for CI
  cli.ts        flag parsing and the entry-point check
rubrics/        one rubric per failure mode
data/           example labeled set and outputs
tests/          vitest; the judge test drives the real SDK with a fake fetch
```

Method: https://www.aievals.co/cookbook/llm-judge-calibration and
https://www.aievals.co/learn/production/release-gates. With a coding agent:
https://www.aievals.co/agent-pack.
