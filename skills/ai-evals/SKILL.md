---
name: ai-evals
description: Build, review and fix evals for LLM apps, RAG pipelines and agents the way the aievals.co guide teaches. Error analysis on real traces before any metric, code checks before LLM judges, one binary PASS/FAIL judge per failure mode, judges calibrated against human labels (PASS and FAIL recall, not agreement), train/dev/test splits, regression sets, Wilson intervals, and paired release gates in CI. Use when asked to add evals or tests to an LLM feature, write or debug an LLM-as-judge prompt, label or review traces, pick metrics or sample sizes, compare two prompts or models, check whether an existing eval can be trusted, or wire evals into CI.
license: CC-BY-4.0
metadata:
  source: https://www.aievals.co
  author: Om Bharatiya
  version: "1.0.0"
---

# AI evals

An eval is a repeatable test that measures whether an AI system does its job. This skill
makes you work the way the aievals.co guide teaches: look at real outputs before you
measure anything, measure the failures you actually found, check every LLM judge against
human labels before you trust it, and put an interval on every number.

The guide is at https://www.aievals.co. Cite its pages when you explain a choice to the
user. The reference files listed at the end carry the details and the sources.

## Ground rules

Follow these in every eval task, whatever the user asked for.

1. **Look at traces before writing metrics.** You can't specify failure modes of a system
   you haven't observed. If the user has no traces, get some first: production logs, or
   the system run on 100 realistic inputs. Then read them.
2. **People label; you propose.** Never write ground-truth labels yourself and present them
   as human judgments. You may draft labels marked `"provisional": true` for a person to
   confirm. Never let a model write both the input and its gold answer, except where code
   computes the answer (SQL results, tests, a calculator).
3. **Code before judges.** Anything a string match, parser, schema check, regex or unit test
   can decide is a code check. Fix what a prompt edit can fix before building any eval.
   Keep the LLM judge for subjective failures that remain.
4. **One binary judge per failure mode.** PASS or FAIL plus a one-sentence critique. No 1-5
   scales. Several quality dimensions means several binary judges, reported as a pass-all
   rate with a per-check breakdown.
5. **Calibrate before trusting a judge.** Compare it with human labels and report PASS
   recall and FAIL recall separately, each with its N. Raw agreement is never the headline:
   a judge that always says PASS agrees 95% of the time when 5% of outputs fail.
6. **Train, dev, test.** Few-shot examples come from train only. Iterate on dev. Read test
   once, at the end. Reading test again after more edits turns it into a second dev set.
7. **Pin the judge.** Model ID, effort, rubric text, output schema and few-shot order are
   part of the instrument. Record them with every result, and calibrate again when any of
   them changes. Don't assume determinism: run dev twice and count label flips.
8. **Every number gets N and an interval.** Use the Wilson interval for rates, never the
   normal approximation. Compare two versions on the same inputs as paired data.
9. **The regression set passes 100% of the time.** It holds reproduced production bugs and
   policy boundaries. One failure blocks the release, and a flaky case is a process bug.
10. **Gate on the difference, not the absolute rate.** Use the four bands in
    [references/ci-and-release-gates.md](references/ci-and-release-gates.md).
11. **Cap spend in the runner.** Judge runs cost money. Enforce a budget in code
    (`EVAL_BUDGET_USD`), not as a printed warning.
12. **Build the smallest thing that works.** Assertions and about 30 labeled examples come
    before a judge, and a calibrated judge comes before dashboards or judge ensembles.

## Pick the workflow

| The user wants | Do | Read |
|---|---|---|
| Evals for a feature that has none | Workflow A | error-analysis, judges, datasets |
| An LLM-as-judge prompt written or fixed | Workflow B | judges, templates |
| To know whether an existing eval can be trusted | Workflow C | judges, statistics |
| To know whether version B beats version A | Workflow D | statistics, ci-and-release-gates |
| Evals running in CI | Workflow E | ci-and-release-gates, datasets |

### Workflow A: add evals to a feature

1. **Collect traces.** Pull at least 100 diverse traces, or run the system on realistic
   inputs if it isn't live. Store them as JSONL with a stable `id` per row. The trace
   fields worth logging are in [references/templates.md](references/templates.md).
2. **Run error analysis with the user** (details in
   [references/error-analysis.md](references/error-analysis.md)). A person reads the traces
   and writes a one-line note on the first thing that went wrong in each. You can build a
   simple viewer and cluster the notes into 4 to 7 named failure modes once the person has
   coded at least 30. Count each mode and rank by frequency times severity.
3. **Fix the cheap ones.** Prompt and code fixes come first. Re-run and re-count.
4. **Write code checks** for every remaining mode a rule can detect.
5. **Write one judge per remaining subjective mode** (Workflow B).
6. **Get human labels**: 100 to 200 per failure mode is the target, with 30 to 50 PASS and
   30 to 50 FAIL in each of dev and test. A 40% dev split of 100 rows can't hold 30 of each
   class, and at a low failure rate you have to oversample failures to fill the FAIL side.
   Start with what you have and say how wide the intervals are.
7. **Calibrate** each judge on dev, then confirm on test once.
8. **Build the regression set** from reproduced failures, and wire the suite into CI
   (Workflow E).

Deliverables in the user's repo, unless they already have a layout:

```
evals/
  data/            labeled.jsonl (human labels), regression.jsonl, traces/
  rubrics/         one file per failure mode
  codebook.md      label, definition, 2 PASS and 2 FAIL examples, last kappa
  checks.*         code checks
  judge.*          the judge call, config pinned
  calibrate.*      split, score, report, disagreements
  run_eval.*       score outputs, compare with baseline, exit code for CI
  runs/            one folder per run: summary, predictions, rubric copy
```

Working starters in Python and TypeScript are at
https://github.com/ombharatiya/ai-evals-skills/tree/main/starters. Copy one rather than
writing the harness from scratch.

### Workflow B: write or fix an LLM judge

1. Confirm the failure mode came from error analysis and that a code check can't catch it.
2. Write the prompt in four parts (template in [references/templates.md](references/templates.md)):
   role and exact definitions; binary criteria ("PASS if X. FAIL if Y, even if the rest of
   the response is correct. When in doubt, FAIL and say why."); 2 or 3 few-shot examples
   from train, including one FAIL that looks like a pass; and a strict output schema,
   `{"label": "PASS" | "FAIL", "critique": "..."}`, enforced by structured output where the
   API supports it. The parser must fail loudly on malformed output.
3. Calibrate: run on dev, compute PASS recall, FAIL recall and Cohen's kappa, and read
   every disagreement. Read false passes and false fails separately. If the human label is
   wrong, fix the label, not the rubric.
4. Change one thing per iteration, in this order of typical impact: rubric specificity,
   few-shot examples, output schema, and model swap last.
5. Expect a plateau around the second or third iteration. Then add few-shots for the edge
   cases: two PASS examples that look like FAIL on a careless read and two FAIL examples
   that look like PASS. If FAIL recall still doesn't move, the judge model may be the
   ceiling: try a stronger one before changing the rubric again, and treat the new model
   as a new instrument to calibrate from scratch.
6. Stop when PASS and FAIL recall both reach the bar on dev (this guide's default is 0.90;
   set it from the cost of a missed failure versus a false alarm), then read test once.

### Workflow C: audit an existing eval

Check each item and report what fails, with file and line:

- The metrics trace back to failure modes someone observed in real traces.
- Code checks cover what code can decide; judges are binary, one per failure mode.
- Each judge has a calibration record: PASS and FAIL recall with N and intervals, on a
  test split the judge never saw. Agreement alone doesn't count.
- Few-shot examples don't appear in dev or test.
- The judge config (model, effort, rubric, schema, few-shots) is pinned and matches the
  config that was calibrated.
- Every reported rate has N and an interval; comparisons are paired.
- A regression set exists, runs on every PR, and passes 100%.
- Release decisions read the paired delta and its interval, not an absolute threshold.
- The dataset is versioned (content hash) and the results name the hash.
- A suite that has sat at 100% for months isn't testing anything. The regression set is
  the one exception.

### Workflow D: compare two versions

Run both versions on the same inputs. Report each pass rate with its Wilson interval and
the paired difference with its interval: McNemar's test or a paired bootstrap for binary
outcomes. Decide the smallest difference worth detecting first, then the sample size
([references/statistics.md](references/statistics.md)). Slice by segment when it matters,
and correct for multiple comparisons when you slice many ways. "No significant difference"
from an underpowered run is not evidence of no difference: say how large a change the run
could have detected.

### Workflow E: evals in CI

Three tiers: a smoke tier of 10 to 50 deterministic assertions on every push (no judge),
a full tier with the calibrated judge on PRs that change prompts or models, and a nightly
tier with the expanded regression and robustness sets. Post a baseline-versus-candidate
table on the PR with the delta, its 95% interval and a ship, hold, investigate or block
verdict. Fail the job on investigate, block or a regression-set failure, and report a hold
for a named person to read: a hold means the interval can't tell the change from no change.
Size the indifference window from an A/A run of the unchanged system, not from the zero-noise
floor. No silent overrides: an override needs a named person and a written reason. Fork
PRs get no API secrets. Details in [references/ci-and-release-gates.md](references/ci-and-release-gates.md).

## How to report results

Report a table, not a sentence:

| Metric | Value | 95% CI | N | Judge | Dataset |
|---|---|---|---|---|---|
| FAIL recall, refund-policy judge (test) | 0.92 | [0.80, 0.97] | 48 | claude-opus-5-5, medium, rubric a1b2c3 | labeled.jsonl @ 9f8e7d |

Say what the numbers can't support. Write "the interval is wide; label more FAIL
examples" rather than rounding a weak result into a claim. Name the positive class when
you say TPR or TNR: some writing, Hamel Husain's FAQ among it, treats FAIL as positive,
so its TPR is the share of real failures caught. Plain "PASS recall" and "FAIL recall"
avoid the confusion.

## Defaults

Use these unless the user's situation calls for others. "Site default" means aievals.co's
own working number, not a published standard.

| Decision | Default | Source |
|---|---|---|
| First error-analysis pass | 100+ traces, first failure only (Hamel and Shankar FAQ); 15 to 45 seconds each | FAQ, and aievals.co for the pace |
| Failure-mode categories | 4 to 7 after axial coding | aievals.co open-coding guide |
| Labels per failure mode | 100 to 200 | Hamel and Shankar FAQ |
| Split | 15% train, 40% dev, 45% test, stratified (the starters split by a hash of each id instead, so a row never changes split; ratios then match only on average) | aievals.co calibration recipe |
| Per class in dev and in test | 30 to 50 PASS and 30 to 50 FAIL | Hamel and Shankar FAQ |
| Judge recall bar | PASS and FAIL recall at least 0.90 on dev, confirmed on test | site default |
| Human agreement before automating | kappa at least 0.6 to proceed, 0.8 for a gold set or gate | team policy, per the guide |
| Interval for a rate | Wilson, 95% | aievals.co statistics chapter |
| Release-gate windows | 1-point indifference window; 2-point minor-regression window on the primary metric | aievals.co release-gates chapter |
| Regression set | 100% pass on every PR | aievals.co regression-sets chapter |

## References

Load the file that matches the task:

- [references/error-analysis.md](references/error-analysis.md): open, axial and
  structured coding, saturation, prioritizing, the trace viewer, dimensional sampling.
- [references/judges.md](references/judges.md): judge design, calibration, biases and how
  to test for them, direct versus pairwise, recalibration triggers.
- [references/datasets.md](references/datasets.md): sizes by purpose, sourcing, synthetic
  data rules, regression sets, coverage grids, versioning.
- [references/statistics.md](references/statistics.md): Wilson intervals, paired tests,
  clustered errors, sample size, multiple comparisons, correcting judge-scored rates.
- [references/ci-and-release-gates.md](references/ci-and-release-gates.md): CI tiers,
  release-gate bands, online evals, sampling, drift.
- [references/templates.md](references/templates.md): judge prompt, codebook entry,
  annotation record, trace schema, PR result comment.
