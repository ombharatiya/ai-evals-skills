# LLM judges

Guide pages: https://www.aievals.co/learn/llm-as-judge,
https://www.aievals.co/learn/llm-as-judge/judge-prompt-structure,
https://www.aievals.co/learn/llm-as-judge/calibration-to-humans,
https://www.aievals.co/learn/llm-as-judge/biases-and-mitigations,
https://www.aievals.co/learn/llm-as-judge/pairwise-vs-direct,
https://www.aievals.co/cookbook/llm-judge-calibration

Primary sources: Hamel Husain, "Creating an LLM-as-a-Judge That Drives Business Results"
(https://hamel.dev/blog/posts/llm-judge/); Hamel Husain and Shreya Shankar, "AI Evals:
Everything You Need to Know" (https://hamel.dev/blog/posts/evals-faq/); Shankar et al.,
"Who Validates the Validators?" (https://arxiv.org/abs/2404.12272).

## When to use a judge at all

1. Fix what a prompt edit can fix.
2. Use a code check wherever a rule can detect the failure: parse the JSON, match the
   schema, run the unit test, assert the tool call.
3. Use an LLM judge only for subjective failures that persist after 1 and 2.

Never make a judge the only verifier when a programmatic check exists. On hard correctness
questions, JudgeBench (Tan et al., https://arxiv.org/abs/2410.12784) measured GPT-4o at
56.6% with the Arena-Hard prompt and 50.9% with a plain prompt, against 50% for a coin
flip. Reasoning judges did better (o3-mini at high effort: 80.9%). Those results predate
current models, so score your own judge on your own data.

## Design

- **Binary PASS/FAIL with a written critique.** Binary needs fewer examples for the same
  power, gets higher agreement between raters, and forces a falsifiable definition. If a
  team insists on a 1-5 scale, make the line between 3 and 4 the pass/fail line.
- **One judge per failure mode.** Report a pass-all rate with a per-check breakdown. Don't
  multiply per-check rates together: failures are rarely independent.
- **Four-part prompt** (template in [templates.md](templates.md)):
  1. Role and exact definitions of every term the criteria use.
  2. Binary criteria: "PASS if X. FAIL if Y, even if the rest of the response is correct.
     When in doubt, FAIL and explain why."
  3. 2 or 3 few-shot examples from the train split only, each with query, response,
     explanation and label. Include one FAIL that looks like a pass.
  4. A strict output schema: `{"label": "PASS" | "FAIL", "critique": "the specific part of
     the response that decided the label"}`.
- **Key order.** Critique-first helps non-reasoning judges; with reasoning models,
  label-first is fine. Pick one and keep it fixed.
- **Parsing.** Use structured output where the API offers it. Constrained decoding removes
  malformed JSON but not wrong labels, and the parser must still fail loudly.
- **No scales, no per-persona rubrics, no chain-of-thought block in the output.** If a
  rubric needs step-by-step reasoning to apply, it's two rubrics. For hard-correctness
  rubrics, sweep the reasoning effort during calibration instead.

## Pin everything

Model ID, effort setting, schema, option labels and few-shot order are all part of the
judge. Meaning-preserving format changes have moved accuracy by up to 76 points on
LLaMA-2-13B (Sclar et al., https://arxiv.org/abs/2310.11324). Record the config hash with
every score, and calibrate again when anything changes. Newer Claude models accept only the
default sampling settings, so you can't make a judge deterministic with `temperature=0`:
run dev twice and count flipped labels before you trust a small change.

## Calibration against human labels

1. Get human labels. Target 100 to 200 per failure mode (Hamel and Shankar FAQ). The guide
   says at least 100, and 200 is more honest; below 100 you're tuning to noise. Hamel
   warns that below roughly 60 the intervals are too wide to support any conclusion.
2. Check the humans first. Two experts double-label about 50 items; compute Cohen's kappa
   (two raters) or Krippendorff's alpha (more than two). Don't proceed below 0.6; use 0.8
   or higher for a gold set or a release gate. The guide calls these cutoffs team policy,
   not a statistical law. Low kappa means the rubric is unclear: fix the rubric, don't
   automate it. Order: rubric, then human agreement, then judge calibration.
3. Split stratified: 10% to 20% train, 40% to 45% dev, 40% to 45% test, with 30 to 50
   PASS and 30 to 50 FAIL in each of dev and test. The starters use 15 / 40 / 45.
4. Iterate on dev. Report PASS recall and FAIL recall separately, with Cohen's kappa
   between judge and humans. Read every disagreement, false passes and false fails
   separately. If the human label is wrong, relabel it.
5. Change one lever per iteration, in this order of typical impact: rubric specificity,
   few-shot examples, output schema, model swap. Hamel's FAQ advises against switching
   judge models by default.
6. Stop when PASS and FAIL recall are both at least 0.90 on dev and judge-human kappa is at
   least 0.6 (this site's default bar, not Hamel's; set yours from the cost of a missed
   failure versus a false alarm). Then read test once and keep that number. If test is
   more than 5 points away from dev in either direction, the dev set isn't representative:
   label more.
7. Expect a plateau around the second or third iteration. Add edge-case few-shots next: two
   PASS examples that look like FAIL on a careless read, and two FAIL examples that look like
   PASS. If FAIL recall still doesn't move about five points, the model may be the ceiling:
   try a stronger judge before changing the rubric again, and calibrate it from scratch
   (https://www.aievals.co/cookbook/llm-judge-calibration).

**Naming.** With PASS as the positive class, PASS recall is the TPR and FAIL recall is the
TNR. Hamel's FAQ treats FAIL as positive, so its TPR is the share of real failures the
judge catches. Say which class is positive whenever you write TPR or TNR.

**Criteria drift.** Grading outputs changes what people think the criteria are (Shankar et
al.). Version the rubric with the calibration set. Have the expert blind-label 50 fresh
traces each quarter; more than 10% disagreement triggers a rubric revision. Don't
over-specify the rubric on day one.

## Recalibrate when

- the generator, the product surface, the judge model or the effort setting changes
- monthly: spot-check 10 to 20 fresh human labels
- quarterly: full recalibration with 100 to 200 new labels

## Judge model choice

The guide's introduction treats "a judge at least as capable as the generator, ideally from
a different model family" as a hard rule; Anthropic's eval guidance asks only for a
different model. The Hamel and Shankar FAQ is softer: start with the most capable model,
validate it, then try cheaper ones; a same-model judge is usually fine for a narrow binary
rubric. Accept a same-family or smaller judge only after PASS and FAIL recall clear the bar
on a test split that includes the generator's own failures.

## Biases and cheap tests

Run these before a judge gates anything (https://www.aievals.co/learn/llm-as-judge/biases-and-mitigations):

| Bias | Mitigation | Test |
|---|---|---|
| Position (pairwise) | Swap the order; count a win only if both orders agree, else a tie. Mandatory for any pairwise gate. | 50 swapped pairs; position 1 winning more than 55% means the swap is mandatory |
| Verbosity | Score conciseness in the rubric, length-normalize, or anchor to a reference answer | 50 traces that differ 2x in length; picking the longer one more than 70% of the time is a problem |
| Self-preference | Judge from a different family, or a larger judge | 100 queries answered by two families; a gap above 10 points means switch family |
| Format sensitivity | Pin the schema, version the rubric, store the prompt hash with results | A meaning-preserving rewrite of the prompt; a recall move above 3 points means the judge is fragile |
| Sycophancy | Neutral framing; strip confidence cues from the candidate | 50 confident but wrong answers; PASS on more than 10% means fix it |

Evidence: position and verbosity bias from Zheng et al., MT-Bench
(https://arxiv.org/abs/2306.05685); self-preference from Panickssery et al.
(https://arxiv.org/abs/2404.13076); sycophancy from Sharma et al.
(https://arxiv.org/abs/2310.13548). A panel of diverse smaller judges can beat one large
judge (Verga et al., https://arxiv.org/abs/2404.18796), but calibrate the panel too.

## Direct versus pairwise

- Direct binary scoring for objective criteria and for release gates. This is the default.
- Pairwise for subjective preference, such as choosing between two prompts. Always run the
  position swap.
- For three or more systems, compare each against one fixed baseline (2n calls rather than
  n(n-1)), aggregate with Bradley-Terry, publish intervals, and don't rank systems whose
  intervals overlap.
- A setup that holds up: direct binary gates on the objective axes plus a small
  fixed-baseline pairwise suite for open-ended quality.

## Anti-patterns

- Agreement percentage as the headline number.
- Likert scales.
- Dev or test examples inside the judge prompt.
- Reading test more than once.
- A vendor's uncalibrated "quality score".
- A judge weaker than the generator, accepted without a test.
- Moving one rubric between model providers without recalibrating.
- Free-form judge output parsed by string matching.
- An ensemble of judges none of which has been calibrated.
