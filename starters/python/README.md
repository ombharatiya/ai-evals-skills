# Python starter

A small eval harness you can copy into your repo. It does three things:

1. Runs a binary PASS/FAIL judge on Claude, one failure mode per rubric.
2. Calibrates that judge against human labels, with PASS recall and FAIL recall reported
   separately with N and Wilson intervals.
3. Scores new outputs, runs code checks before the judge, and gates a release on the
   paired difference against a baseline.

It needs Python 3.10+ and one runtime dependency, the `anthropic` SDK. The example data is
a support bot for a fictional store, Brightside Goods, checked for one failure mode:
getting the refund policy wrong. Replace it with your own traces and rubric.

## Setup

```bash
cd starters/python
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
export ANTHROPIC_API_KEY=...   # or sign in with `ant auth login`
pytest                         # no API calls
```

## 1. Calibrate the judge on dev

```bash
python -m evals.calibrate --data data/labeled.example.jsonl --rubric rubrics/refund_policy.md
```

This splits the labeled set into train, dev and test (about 15 / 40 / 45), puts four train
examples in the prompt as few-shots, and judges the dev split. Each row's split comes from
a hash of its id, so labeling more rows never moves an existing row from test into train or
dev. The output looks like this:

```
split=dev  n=16  train/dev/test=4/16/20  judge=claude-opus-5-5@medium
  PASS recall (TPR) ...  95% CI [...]  n=8 of 8 PASS rows
  FAIL recall (TNR) ...  95% CI [...]  n=8 of 8 FAIL rows
  kappa ...   agreement ... (not the headline)
```

Open `runs/<timestamp>-calibrate-dev/disagreements.jsonl`, find the pattern the judge
missed, change one thing in the rubric, and run again. Use `--repeats 2` to count how many
labels flip between identical runs before you trust a small change.

The bar has four parts, all set by flags:

- every split row gets a verdict, and none comes from a fallback model
- each class has at least 30 rows in the split (`--min-per-class`)
- PASS and FAIL recall both reach 0.90 (`--min-recall`)
- kappa between the judge and the humans reaches 0.60 (`--min-kappa`)

The 40-row example set shows the loop but can't clear the bar: its dev split has 8 FAIL
rows, and the run says so. The guide's target is 100 to 200 labels per failure mode, with
30 to 50 PASS and 30 to 50 FAIL in each of dev and test. At a low failure rate that means
oversampling failures.

## 2. Read test once

When dev clears the bar:

```bash
python -m evals.calibrate --data data/labeled.example.jsonl --rubric rubrics/refund_policy.md --split test
```

Keep that `summary.json`: it's the judge's calibration record. A test run writes no
disagreements file, and if it misses the bar it tells you to label fresh rows rather than
tune on test. Editing the rubric after reading test turns test into a second dev set.

## 3. Score outputs and gate a release

```bash
# First run: this becomes the baseline.
python -m evals.run_eval --data data/outputs.example.jsonl --rubric rubrics/refund_policy.md \
  --labeled data/labeled.example.jsonl --calibration runs/<calibrate-test-run>/summary.json

# Later runs: compare with it.
python -m evals.run_eval --data data/outputs.example.jsonl --rubric rubrics/refund_policy.md \
  --labeled data/labeled.example.jsonl --calibration runs/<calibrate-test-run>/summary.json \
  --baseline runs/<first-eval-run>/results.jsonl
```

Each row runs the code checks in `evals/checks.py` first (`not_empty`, and `must_include`,
`must_not_include` and `max_words` when the row sets them). A row that fails a code check
fails without a judge call. The rest go to the judge. Pass the same `--labeled`, `--seed`
and `--few-shots` you calibrated with, so the judge's fingerprint matches the calibration
record.

The verdict follows the release-gates chapter on aievals.co. It reads the 95% interval of
the paired difference in pass rate (Newcombe's interval for paired proportions) against two
windows:

| Verdict | Rule | Exit code |
|---|---|---|
| ship | the interval sits above zero, or both ends are inside the 1-point indifference window (`--indifference`) | 0 |
| hold | the interval contains zero and reaches outside the indifference window | 1 |
| investigate | the interval sits below zero and the drop is within the 2-point minor-regression window (`--minor-regression`) | 1 |
| block | the interval sits below zero and the drop is larger | 1 |
| invalid | the measurement can't be trusted (the conditions below) | 1, whatever `--fail-on` says |

Some conditions override the band:

- **Block.** Any failing row marked `"must_pass": true` blocks, and so does a baseline
  `must_pass` row that's missing from this run. Give regression rows code checks where you
  can, so they don't depend on the judge.
- **Invalid.** Any of these makes the run INVALID, whatever the interval says:
  - a row with no verdict (a refusal, a budget stop, malformed output)
  - candidate ids missing from the baseline, or baseline ids missing from this run
  - fewer than 30 paired rows (`--min-paired`)
  - a verdict served by a fallback model
  - a judge whose fingerprint differs from the calibration record
  - a calibration record that isn't a passing test read
- **Exit code.** A first run with no baseline exits 0, unless one of the conditions above
  applies. Block and invalid always exit 1; `--fail-on` decides only whether a hold or an
  investigate does.

`results.jsonl` is written as soon as scoring finishes, and every input file is checked
before the first judge call.

Size the indifference window to your gate set. Every run prints the set's resolution: how
wide the interval is for a candidate that matches the baseline on every row. At a 76% pass
rate that's about plus or minus 5.3 points with 50 rows, so a 1-point window would hold
every unchanged candidate; the 1-point default needs about 280 rows. That's a floor for a
judge that never flips a label: with 2% flips between identical runs it's about 860 rows.
Calibrate with `--repeats 2` and pass that record with `--calibration`, and the resolution
uses the measured flip rate.

These figures are floors. When CI regenerates outputs on every pull request, generation
noise and judge noise widen the interval further, so size the window from an A/A run:
generate and score the unchanged system twice, compare the second run with the first using
`--baseline`, and set `--indifference` a little above the interval that prints. In CI, pass
`--fail-on investigate`: a hold then shows as a warning on the pull request, with the delta
and its interval, for a person to read. The job fails on a confirmed regression
(investigate or block), a regression-set failure, or an invalid run.

## Pinning and cost

- The judge defaults to `claude-opus-5-5` at effort `medium`. Change them with `--model`
  and `--effort`, or with `JUDGE_MODEL` and `JUDGE_EFFORT`. `claude-sonnet-5-5` costs about
  half as much. Calibrate again after any change.
- Every run folder records the judge's fingerprint: the model, the effort, and a SHA-256 of
  the rendered prompt, schema and max_tokens. It also records the rubric hash, the few-shot
  ids, and the dataset's SHA-256, seed and split fractions.
- Set `EVAL_BUDGET_USD` to stop new judge calls once the estimated spend reaches it. Failed
  calls count too, and fallback calls are priced at the model that served them. With a
  budget set, a model missing from `PRICES` in `evals/judge.py` is refused, so you can't
  run uncapped by accident.
- The judge sends `fallbacks: "default"`: if Claude declines a request on a safety
  classifier, the API reruns it on a fallback model in the same call. Those verdicts come
  from a model you didn't calibrate, so they are counted in every summary and make the run
  invalid.

## Files

```
evals/
  data.py       load JSONL, hash-based train/dev/test split, JSON helpers
  checks.py     code-graded checks
  judge.py      the Claude judge, prompt rendering, fingerprint, pricing
  metrics.py    confusion matrix, PASS/FAIL recall, kappa, Wilson interval
  runner.py     worker pool and budget cap
  calibrate.py  CLI: calibrate on dev or test
  gate.py       Newcombe paired interval and the four release bands
  run_eval.py   CLI: score outputs, compare with baseline, exit code for CI
rubrics/        one rubric per failure mode
data/           example labeled set and outputs
tests/          pytest, no network
```

The method behind each step is explained at https://www.aievals.co, starting with
https://www.aievals.co/cookbook/llm-judge-calibration. To have a coding agent do this work
with you, see https://www.aievals.co/agent-pack.
