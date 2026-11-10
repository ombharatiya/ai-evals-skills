# CI, release gates and production

Guide pages: https://www.aievals.co/learn/production/ci-integration,
https://www.aievals.co/learn/production/release-gates,
https://www.aievals.co/learn/production/online-vs-offline,
https://www.aievals.co/learn/production/trace-schema,
https://www.aievals.co/learn/production/sampling-strategies,
https://www.aievals.co/learn/production/drift-detection

## CI tiers

| Tier | What runs | Budget | When |
|---|---|---|---|
| Smoke | 10 to 50 deterministic assertions, no judge | under 2 minutes, single-digit dollars | every push |
| Full | 100 to 500 labeled examples with the calibrated judge; quality, cost and latency with intervals | 5 to 15 minutes, tens of dollars | PRs that change prompts, models or retrieval, and merge candidates |
| Nightly | expanded regression, contamination probes, robustness sweeps (for example 5 formats x 3 seeds), drift comparison | larger | nightly |

Rules:

- Keep the judge out of the smoke tier. Prefer assertions to judges in CI.
- Shard by dataset row. Pin the judge config and the dataset hash per release.
- Enforce a hard cost cap in the runner (`EVAL_BUDGET_USD`), not a warning.
- Fork PRs don't get API secrets. Never check out untrusted PR code in a job that has them.
- Post a result table on the PR: baseline, candidate, difference, 95% interval, and a
  one-line verdict (ship, hold, investigate or block). Template in [templates.md](templates.md).
- No silent overrides. An override needs a named person and a written reason, logged where
  it can be queried. A pattern of overrides means the bands need retuning.
- A CI eval that never fails isn't testing anything. The regression set is the one suite
  that should sit at 100%.

## Release-gate bands

Read the interval on the paired difference between the candidate and the production
baseline, per metric. Sources: https://www.aievals.co/learn/production/release-gates and
Evan Miller, "Adding Error Bars to Evals" (https://arxiv.org/abs/2411.00640).

Two windows: changes inside the indifference window aren't worth blocking on (the guide's
figure uses 1 point), and the minor-regression window is the largest drop you'd ship with a
written reason (the guide's default for quality is 2 points).

| Band | Rule |
|---|---|
| Ship | the interval sits above zero, or both ends are inside the indifference window |
| Hold | the interval contains zero and reaches outside the indifference window |
| Investigate | the interval sits below zero and the drop (the point difference) is within the minor-regression window. Read 50 traces from the regressed slice; ship only with a written reason |
| Block | the interval sits below zero and the drop is larger than the minor-regression window |

Defaults for the minor-regression window: 2 points on the primary quality metric, 15% of
baseline for p95 latency, 10% of baseline for cost. Any change in refusal rate needs
investigation.

- Compute the difference from per-question paired differences, clustered where questions
  share a passage or a conversation.
- Size the indifference window to the gate set. The guide ties it to the set's statistical
  power: a window narrower than the set can resolve makes every unchanged candidate hold.
  With Newcombe's paired interval at a 76% pass rate, an unchanged candidate's interval is
  about plus or minus 5.3 points at 50 rows, 2.7 at 100 and 1.4 at 200; a 1-point window
  needs about 280 rows at 76%, 360 at 90% and 410 at 97%. The starters print this
  resolution on every run.
- The release decision is the worst band across all axes.
- Compare against the production system, per slice as well as overall.
- Update the baseline only after the candidate has been green online for a sustained
  window, often 2 to 4 days.
- Any failure in the regression set blocks, whatever the bands say.

## Offline and online

- Use the same judge code offline and online, and one results ledger. Each row:
  `(trace_id, eval_name, score, judge_version, dataset_or_window, timestamp)`.
- Guardrails are runtime checks, not evals.
- Online judging is not the offline set run twice. Online samples aren't independent:
  cluster by user or conversation.
- Read production data by hand regularly. Hamel Husain's ladder: level 1 assertions on
  every change, level 2 human review plus a judge on a labeled set on a cadence, level 3
  production A/B tests after significant product changes.

## Sampling production traffic for evals

(https://www.aievals.co/learn/production/sampling-strategies)

- Uniform random sampling at 1% to 5% silently misses the tail.
- Stratify (4 to 8 strata per dimension), oversample negatives (low judge score, thumbs
  down, escalation, refusal, error) by 3x to 5x, and always keep a random slice.
- Push a flagged cluster to 100% sampling until you have enough examples, then drop back.
- Set the per-stratum target high enough for a useful Wilson interval; cut the long tail
  first if the budget binds.
- Report a post-stratified estimate for the overall rate, or per-stratum intervals.

## Drift

(https://www.aievals.co/learn/production/drift-detection)

- Track input shape, output shape and judge score, comparing a rolling window with a frozen
  reference window. The reference window is a content-hashed artifact tied to a release.
- Starting thresholds, to tune against your own false-alarm history: PSI above 0.2 is a
  strong signal, 0.1 to 0.2 worth a look; a rate change whose 95% interval excludes zero is
  a candidate alert.
- Run per stratum, not only on the aggregate. Pin the judge version on every score row,
  and record the model that responded next to the model you requested.
- On large windows, gate on the KS statistic or PSI, not the p-value.
- When an alarm fires: check for input drift, check the judge against held-out labels, then
  read 20 drifted traces. An alarm starts an investigation; it isn't a verdict.

## Anti-patterns

- Running everything on every PR, or no evals in CI at all.
- Gating on an absolute pass rate.
- Silent overrides.
- The offline suite doubling as the "online eval".
- Judging 100% of production traffic when the judge isn't cheap.
- Naive standard errors on clustered online data.
- Alerting on the global mean while strata move in opposite directions.
- Two different judge implementations for offline and online.
