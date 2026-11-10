# Error analysis

Error analysis is the loop every other eval step depends on: rubrics, judges and datasets
all come out of it. Do it before writing any metric, and again on a schedule.

Guide pages: https://www.aievals.co/learn/error-analysis,
https://www.aievals.co/learn/error-analysis/open-coding-workflow,
https://www.aievals.co/learn/error-analysis/the-60-80-rule,
https://www.aievals.co/learn/error-analysis/failure-mode-taxonomies,
https://www.aievals.co/learn/error-analysis/the-data-viewer,
https://www.aievals.co/learn/error-analysis/dimensional-sampling

## The three coding passes

**Open coding.** A person reads traces and writes notes.

- Pull at least 100 diverse traces for the first pass, and keep going past 100 while new
  problems still appear (Hamel Husain and Shreya Shankar, "AI Evals: Everything You Need to
  Know", https://hamel.dev/blog/posts/evals-faq/).
- Spend 15 to 45 seconds per trace. Write a one-line note only when something is wrong,
  in plain words, with no category labels yet.
- Note only the first failure in each trace. Upstream errors cause downstream ones.
- Don't debate a hard case. Note the ambiguity and move on.
- Expect 40 to 60 notes from 100 traces.

**Axial coding.** Cluster the notes into 4 to 7 categories. Fewer than 4 usually means
under-labeling; more than 7 usually means fitting the batch. Name each label in 2 to 4
specific words, with a definition that includes its pass/fail test.

**Structured coding.** Label a larger set with the categories, then count.

## Who does what

- A person codes at least the first 30 traces. Let an LLM propose clusters only after
  30 to 50 have been coded by hand: early suggestions bias the reader, and open coding can't
  be handed off.
- Treat LLM-proposed clusters as drafts. Run the proposal twice and keep the labels that
  survive both runs. Check every cluster against real traces: a model will happily merge
  "the app crashed" and "the app was slow" into "login issues".
- Name one principal domain expert who has the final say on labels. With two annotators,
  double-label a batch and compute Cohen's kappa
  (https://www.aievals.co/learn/statistics/inter-rater-agreement).

## Prioritize

- Rank failure modes by frequency times severity. The guide's illustrative weights are
  low = 1, medium = 3, high = 10. A domain expert sets severity, not a model.
- Fix the top three per cycle, then measure again. Break ties with the cheaper fix.
- The 60-80 rule: three failure modes usually account for most of the bugs. Treat it as a
  prior to check against your own counts. The 80% end is a rule of thumb, not a published
  figure; the guide's worked case (NurtureBoss) had the top three above 60%.
- Safety failures live in the tail. Send tail failures to the regression set and to
  guardrails; build evals for the head.

## Good labels

Labels should be specific, actionable, distinguishable, counted and revisable.

- Merge two labels when more than 25% of traces could carry either one.
- Split a label when the fix differs by sub-pattern.
- Either change means relabeling the affected traces.
- Build the taxonomy bottom-up from your own traces. Generic labels such as
  "hallucination", "refusal" and "toxicity" are a starting checklist, not a label set.
  Generic labels hide different mechanisms with different fixes (Hamel Husain, "A Field
  Guide to Rapidly Improving AI Products", https://hamel.dev/blog/posts/field-guide/). The
  guide's taxonomy page expects six to nine specific categories after a half-day pass; axial
  coding on a first batch of notes usually lands at four to seven.
- For agents, the MAST taxonomy (Cemri et al. 2025, https://arxiv.org/abs/2503.13657) is a
  useful checklist: 14 failure modes in 3 categories, built from 150 annotated traces at
  kappa 0.88. Use it to prompt your reading, not as your label set.

## Checking an LLM labeler at scale

When an LLM applies the labels to a large set, a person spot-checks about 10% of its
labels, stratified, at least 5 per label. These override-rate bands are the guide's
working heuristics, not published standards:

- under 5% overrides: fine
- 5% to 20%: tighten the definitions and rerun
- over 20%: go back to axial coding

## Saturation and cadence

- After 100 traces, review 25 more and count new categories. Zero new means saturated; one
  or two means top up another 25; five means the distribution moved, so start again from a
  fresh 100. A shorter working rule on this site: stop once about 20 traces in a row add no
  new category.
- Run a full pass of at least 100 fresh traces each review cycle, typically every 2 to 4
  weeks, and 10 to 20 traces a week in between, weighted toward outliers.
- Run a pass after every new feature, prompt change or model switch. New systems need
  weekly passes until the patterns settle; mature ones can go monthly.
- Put the review on the calendar with a named owner.

## The codebook

Keep it in git. Each entry has the label, its definition, 2 PASS and 2 FAIL examples, and
the last measured kappa. Template in [templates.md](templates.md).

## The trace viewer

Annotators shouldn't read raw JSON. Build a small viewer
(https://www.aievals.co/learn/error-analysis/the-data-viewer) with:

- the input, output and tool calls rendered side by side
- copy-whole-trace
- filters on any field; the queue that matters most is "judge said FAIL and no human
  label yet"
- inline annotation that saves on every keystroke
- no charts and no AI-suggested critiques on day one

Annotation records: `trace_id`, `annotator` (one per person, never shared), `pass_fail`,
`label`, `critique`, and optionally `confidence`, `needs_sme_review`, `linked_issue`. Back
up the labels file. It can't be regenerated.

## Dimensional sampling

To build inputs that reach the corners
(https://www.aievals.co/learn/error-analysis/dimensional-sampling):

- Pick dimensions for the product, for example feature, scenario and persona.
- Write about 20 tuples by hand, then let an LLM expand them.
- Generate the tuples and the natural-language queries in separate steps; one prompt
  produces repetitive phrasing.
- Check that each input really lands in its cell.
- Keep random production traces in every review batch.
- When error analysis finds a failure the grid missed, add an axis rather than more
  examples per cell.

## Anti-patterns

- Writing a rubric before reading traces.
- An "other" bucket that swallows a third of the notes.
- A codebook nobody updates.
- Ten categories tracked loosely instead of five tracked well.
- Shared annotator logins, and no backup of the labels.
- A giant matrix of every imaginable query type. Let error analysis decide where to look.
- Treating a failure rate on synthetic inputs as a production rate.
- Generating synthetic inputs once you have 100 or more real traces: read the real ones
  first and spend synthetic inputs on the gaps. Seed specialized documents, low-resource
  languages and high-stakes domains from real examples.
