# Datasets

Guide pages: https://www.aievals.co/learn/datasets/construction,
https://www.aievals.co/learn/datasets/regression-sets,
https://www.aievals.co/learn/datasets/synthetic-vs-human,
https://www.aievals.co/learn/datasets/dimensional-coverage,
https://www.aievals.co/techniques/golden-datasets

## How many examples

Start with about 30 examples you label yourself, then 100 or more per failure mode to
calibrate a judge, then about 500 for a CI gate. The 500 comes from the guide's binomial
arithmetic, not from Hamel Husain's writing.

| Purpose | Minimum | Practical | Stop when |
|---|---|---|---|
| Sanity check | 30 | 50 | no obvious failure remains |
| Judge calibration, per failure mode | 100 | 200 | PASS and FAIL recall both above 0.90 on dev |
| Regression set | 50 | 200 | every known failure mode is covered |
| Release gate | 200 | 500 | the smallest change you care about is detectable |
| Error-analysis pool | 100 | 300 | about 20 traces in a row add no new category |
| Production A/B headline | 500 | 2,000+ | the smallest change you care about is detectable |

Go past 500 when a gate has to tell apart changes under 2 points, when a rare slice needs
its own interval, or when examples cluster (many turns from one conversation, many
questions on one document). Clustering can make naive standard errors 3 or more times too
small, which means about 9 times the examples (see [statistics.md](statistics.md)).

## Where inputs come from

1. Sampled production traces. Keep a random slice, and oversample rare but critical
   segments.
2. Synthetic inputs run through the real system.
3. Hand-written adversarial cases.

A mature set is roughly half production samples, a quarter synthetic with human labels and
a quarter adversarial. The guide calls this a rule of thumb from practice, not a measured
result. If pass rates differ by more than 10 points between the synthetic and production
subsets, the synthetic part isn't probing the same surface.

## Synthetic data rules

- Synthesize the input; a person labels the output. Never let a model write both the query
  and the gold answer.
- Exceptions: answers code can compute (a calculator, SQL execution, unit tests, regex), and
  retrieval sets built from a source document.
- Mark bootstrap labels `"provisional": true`, with an owner and a date by which a person
  replaces them. Don't leave provisional labels in for two quarters.
- Propose-and-dispose works (SPADE, Shankar et al., https://arxiv.org/abs/2401.03038): a
  model proposes assertions or labels, and a labeled sample decides which to keep. The
  disposal step has to be able to reject.
- Read every example yourself. A good sign the set is understood: the product owner can
  name the top failure categories from memory.

## Storage and versioning

- JSONL in git, one object per line, with a stable `id` that is never reused.
- Tag each example with its coordinates in a `labels` object (feature, scenario, persona,
  or whatever dimensions you sample on).
- Content-hash the file on every commit and record the hash in every result.
- Treat a dataset update as a release, with notes on what changed.
- Calibrate the judge again when a version bump changes more than 5% of the labeled rows
  (site default).

## One set per job

Keep separate sets, because mixing them is the common mistake:

| Set | Size | Changes |
|---|---|---|
| Calibration | 100 to 300 | only when the rubric changes |
| Regression | 50 to 200 | append-mostly |
| Release gate | 200 to 500 | per release, versioned |
| Error-analysis pool | grows | refreshed continuously |

Outside the regression set, retire evals that stop discriminating: an eval that keeps
passing tells you less and less, and expensive judge-based ones should go sooner than cheap
ones (Hamel Husain and Shreya Shankar, https://hamel.dev/blog/posts/evals-faq/).

## Regression sets

Sources: https://www.aievals.co/learn/datasets/regression-sets and Anthropic, "Demystifying
evals for AI agents" (https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents).

- 100% pass on every PR. A flaky case is a process failure, not noise.
- Admit only reproduced production bugs, policy boundaries and hard-won invariants. Leave
  out aspirational cases, cases that depend on live state, and cases people disagree on.
- Add the case the day the fix ships, with a link to the postmortem or issue.
- Prefer deterministic checks to judges here.
- Tier it: a smoke tier (the top 30 cases) on every PR, the full set nightly and on release
  candidates, an extended set with adversarial probes weekly.
- Move cases that have been green for a year to the nightly tier. A capability case
  graduates into the regression set once it passes consistently.
- Retire a regression case only when the capability it guards is structurally impossible
  to break. Regression cases are supposed to pass every time, so never prune them for
  passing.
- Give regression cases deterministic checks where you can (in the starters,
  `must_include`, `must_not_include` and `max_words` on the row). A judge-graded regression
  case can flip between runs.

## Coverage grids

(https://www.aievals.co/learn/datasets/dimensional-coverage)

- Every cell with real production volume needs at least 2 examples; high-severity cells
  need 10 or more.
- Report per-cell pass rates with N. Grey out cells with N under 5. Flag cells more than
  two interval widths below the overall rate.
- Weight cells by volume, severity and confidence.
- Read the examples in each cell: two typo-only variants don't fill a cell.

## Anti-patterns

- 500 examples drawn from 6 conversations.
- One set doing five jobs.
- Model-written ground truth.
- Aspirational cases in the regression set.
- Provisional labels nobody replaces.
