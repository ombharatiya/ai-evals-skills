# Statistics

Guide pages: https://www.aievals.co/learn/statistics/confidence-intervals,
https://www.aievals.co/learn/statistics/bootstrap-and-paired-tests,
https://www.aievals.co/learn/statistics/clustered-standard-errors,
https://www.aievals.co/learn/statistics/statistical-power,
https://www.aievals.co/learn/statistics/inter-rater-agreement,
https://www.aievals.co/techniques/statistical-methods

Primary source: Evan Miller, "Adding Error Bars to Evals" (https://arxiv.org/abs/2411.00640).

## Report every rate with N and an interval

A bare "72.5%" hides how much it could move. Put the N, the 95% interval, and the judge and
dataset versions next to every pass rate.

## Wilson, not Wald

The Wald (normal approximation) interval collapses at the edges (10 out of 10 gives
[1.0, 1.0]), undercovers (about 88% real coverage at n = 50, p = 0.9) and can leave [0, 1].
Use the Wilson score interval, with z = 1.96 for 95%:

```
centre = (p + z^2 / (2n)) / (1 + z^2 / n)
half   = z * sqrt( p(1 - p)/n + z^2 / (4n^2) ) / (1 + z^2 / n)
```

In Python: `statsmodels.stats.proportion.proportion_confint(k, n, method="wilson")`.
Worked example: 7 out of 10 gives about [0.40, 0.89]. Use Clopper-Pearson (exact, wider)
when an audit or a contract needs guaranteed coverage.

Wilson half-widths around a 70% rate:

| N | 10 | 50 | 100 | 200 | 500 | 1,000 | 2,000 |
|---|---|---|---|---|---|---|---|
| half-width, points | about 25 | about 13 | about 9 | about 6 | about 4 | about 3 | about 2 |

Width shrinks with 1 / sqrt(n): halving it takes four times the examples. N = 200 is the
guide's usual floor for a release gate.

## Non-binary metrics

Use a percentile bootstrap: 1,000 resamples for a headline number, 200 for triage.

## Comparing two versions: pair them

Two versions run on the same inputs are paired data. For binary outcomes use McNemar's
test, or a paired bootstrap that resamples inputs; for continuous scores, a paired
bootstrap. Report an interval on the difference, not just a p-value, and never subtract two
independent intervals to compare paired runs. Where outputs vary run to run, also average
several samples per input.

## Clustered data

When traces group (by user, session, document or topic), use clustered standard errors.
The design effect is `DEFF = 1 + (m - 1) * rho`, and the effective sample size is
`n / DEFF`. In Miller's paper the clustered-to-naive standard error ratio was 3.05 on DROP,
1.88 on MGSM and 1.10 on RACE-H. The fix is a cluster bootstrap: resample whole clusters.

## Sample size

Choose the smallest difference worth detecting first, then derive N.

- Paired binary comparison at 80% power and 5% significance:
  `N = 7.85 * d / delta^2`, where `d` is the share of inputs where the two versions
  disagree and `delta` the difference to detect.
- At a 70% baseline with a conservative 40% disagreement rate:

| Difference to detect | 10 pts | 7 pts | 5 pts | 4 pts | 3 pts | 2 pts |
|---|---|---|---|---|---|---|
| N (paired) | about 320 | about 660 | about 1,300 | about 2,000 | about 3,400 | about 7,800 |

- Unpaired designs need about twice as many. At a more typical 20% disagreement, detecting
  5 points takes about 630.
- Run a pilot of 50 paired inputs to estimate the disagreement rate. Pilot again when the
  baseline moves. Clustering multiplies the required N by the design effect.

## Many slices

Fifteen independent tests at 0.05 give about a 54% chance of at least one false alarm.
Pre-declare one headline metric, and correct the rest with Bonferroni (conservative) or
Benjamini-Hochberg.

## Correcting a judge-scored rate

A pass rate scored by a judge carries the judge's errors. With PASS as positive, the
Rogan-Gladen estimate of the true pass rate is

```
true = (observed + fail_recall - 1) / (pass_recall + fail_recall - 1)
```

Its interval has to include the uncertainty in both the test run and the calibration
(Lee et al., https://arxiv.org/abs/2511.21140): a 90% recall measured on 50 examples has a
Wilson interval of about 79% to 96%.

## Agreement between people

Use Cohen's kappa for two raters and Krippendorff's alpha for more raters, missing data or
ordinal labels; never raw percent agreement. Bands: below 0.4 broken, 0.4 to 0.6
borderline, 0.6 to 0.8 usable, above 0.8 high. Krippendorff's own convention: 0.800 and up
is reliable, 0.667 to 0.800 tentative. Measure agreement again monthly.

## Agents

Log the harness resources and timeouts, and hold them fixed between runs. Anthropic measured
a 6-point gap on Terminal-Bench 2.0 from container setup alone.

## Anti-patterns

- Shipping on a 10-trace eval.
- Picking N before deciding the smallest difference that matters.
- Reading "no significant difference" from an underpowered run as "no difference".
- Naive standard errors on correlated traces.
- Reporting the best of many sub-metrics.
- Comparing single runs while ignoring run-to-run noise.
