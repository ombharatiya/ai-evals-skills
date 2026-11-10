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

This splits the labeled set 15 / 40 / 45 into train, dev and test (stratified, seed 42),
puts four train examples in the prompt as few-shots, and judges the dev split:

```
split=dev  n=16  train/dev/test=6/16/18  judge=claude-opus-5-5@medium
  PASS recall (TPR) ...  95% CI [...]  n=10
  FAIL recall (TNR) ...  95% CI [...]  n=6
  kappa ...   agreement ... (not the headline)
```

Open `runs/<timestamp>-calibrate-dev/disagreements.jsonl`, find the pattern the judge
missed, change one thing in the rubric, and run again. Use `--repeats 2` to count how many
labels flip between identical runs before you trust a small change.

The example set is 40 rows so it runs in about a minute. That's enough to see the loop
work, not enough to trust a judge. The guide's target is 100 to 200 labels per failure
mode, with 30 to 50 PASS and 30 to 50 FAIL in each of dev and test.

## 2. Read test once

When PASS and FAIL recall both clear 0.90 on dev (the default bar, set with
`--min-recall`):

```bash
python -m evals.calibrate --data data/labeled.example.jsonl --rubric rubrics/refund_policy.md --split test
```

Keep that number. If you edit the rubric after reading test, test has become a second dev
set and you need fresh labels.

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
fails without a judge call. The rest go to the judge.

The verdict reads the 95% interval of the paired difference in pass rate:

| Verdict | Interval on candidate minus baseline | Exit code |
|---|---|---|
| ship | above zero, or entirely inside the 2-point window | 0 |
| hold | contains zero and reaches outside the window | 1 |
| investigate | below zero, inside the window | 1 |
| block | below zero, past the window | 1 |

Any failing row marked `"must_pass": true` blocks. A row with no verdict (a refusal, a
budget stop, malformed output) turns a ship into a hold.

Pass the same `--labeled` file you calibrated with, so the few-shots match. The run warns
when the judge config differs from the calibrated one.

## Pinning and cost

- The judge defaults to `claude-opus-5-5` at effort `medium`. Change them with `--model`
  and `--effort` or `JUDGE_MODEL` and `JUDGE_EFFORT`. `claude-sonnet-5-5` costs about half
  as much; calibrate again after any change.
- Every run folder records the model, effort, rubric SHA-256 and few-shot ids next to the
  numbers.
- Set `EVAL_BUDGET_USD` to stop new judge calls once the estimated spend reaches it.
- The judge sends `fallbacks: "default"`: if Claude declines a request on a safety
  classifier, the API reruns it on a fallback model in the same call. Those verdicts come
  from a model you didn't calibrate, so they are counted and flagged in every summary.

## Files

```
evals/
  data.py       load JSONL, stratified train/dev/test split
  checks.py     code-graded checks
  judge.py      the Claude judge, prompt rendering, config pinning
  metrics.py    confusion matrix, PASS/FAIL recall, kappa, Wilson interval
  runner.py     worker pool and budget cap
  calibrate.py  CLI: calibrate on dev or test
  gate.py       paired bootstrap difference and the four release bands
  run_eval.py   CLI: score outputs, compare with baseline, exit code for CI
rubrics/        one rubric per failure mode
data/           example labeled set and outputs
tests/          pytest, no network
```

The method behind each step is explained at https://www.aievals.co, starting with
https://www.aievals.co/cookbook/llm-judge-calibration.
