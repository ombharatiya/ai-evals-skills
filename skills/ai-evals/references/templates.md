# Templates

## Judge prompt (four parts)

Put parts 1 to 3 in the system prompt so they can be cached; send the trace as the user
message. Enforce part 4 with structured output where the API supports it.

```
1. ROLE AND DEFINITIONS
You are a <domain expert role>. You check <system> responses for one failure mode:
<failure mode, in plain words>.

Definitions:
- <term>: <exact operative definition>
- <term>: <exact operative definition>

2. CRITERIA
PASS if <the observable condition that makes a response acceptable>.
FAIL if <the observable condition that makes it unacceptable>, even if the rest of the
response is correct.
When in doubt, FAIL and explain why.

3. EXAMPLES (from the train split only)
<example>
<query>...</query>
<response>...</response>
<explanation>why this is a FAIL, naming the exact phrase</explanation>
<label>FAIL</label>
</example>
<example> ... a PASS ... </example>
<example> ... a FAIL that looks like a pass ... </example>

4. OUTPUT
Return a JSON object: {"label": "PASS" or "FAIL", "critique": "..."}.
The critique is one sentence naming the specific part of the response that decided the
label.
```

JSON schema for structured output:

```json
{
  "type": "object",
  "properties": {
    "label": {"type": "string", "enum": ["PASS", "FAIL"]},
    "critique": {"type": "string"}
  },
  "required": ["label", "critique"],
  "additionalProperties": false
}
```

## Labeled dataset row (JSONL)

```json
{"id": "t-0042", "query": "Can I get a refund after 60 days?", "response": "Yes, we refund any order within 90 days.", "label": "FAIL", "critique": "States a 90-day window; the policy allows 30 days from delivery.", "annotator": "anita", "labels": {"feature": "refunds", "persona": "returning customer"}}
```

Optional fields the starters read: `must_include`, `must_not_include` and `max_words` for
code checks, and `"must_pass": true` for regression-set rows.

## Codebook entry

```markdown
### wrong-refund-window
Definition: the response states a refund or return window other than 30 days from
delivery. PASS if no window is stated or the stated window is correct.
PASS examples: t-0012, t-0031
FAIL examples: t-0042, t-0057
Last measured kappa: 0.84 (2 raters, 50 traces, 2026-11-03)
```

## Annotation record

```json
{"trace_id": "t-0042", "annotator": "anita", "pass_fail": "FAIL", "label": "wrong-refund-window", "critique": "Says 90 days.", "confidence": "high", "needs_sme_review": false, "linked_issue": null}
```

## Trace schema (minimum worth logging)

(https://www.aievals.co/learn/production/trace-schema)

```json
{
  "trace_id": "ULID or UUIDv7",
  "session_id": "...",
  "parent_trace_id": null,
  "user_bucket": "salted hash, not the raw user id",
  "ts": "2026-11-10T12:00:00Z",
  "app_version": "...",
  "ab_variant": "...",
  "feature_flags": {},
  "intent": "...",
  "language": "en",
  "input": {
    "user_message": "...",
    "system_prompt_rendered": "...",
    "system_prompt_template_id": "...",
    "messages": [{"role": "user", "content": "..."}],
    "tools_offered": ["..."]
  },
  "retrieval": [{"span_id": "...", "query": "...", "index": "...", "top_k": 5,
                 "results": [{"doc_id": "...", "score": 0.82}], "latency_ms": 41}],
  "tool_calls": [{"span_id": "...", "name": "...", "arguments": "{...json string...}",
                  "raw_response": "...", "ok": true, "latency_ms": 120}],
  "model_call": {"provider": "...", "model": "...", "params": {"effort": "medium"},
                 "input_tokens": 0, "output_tokens": 0, "ttft_ms": 0, "latency_ms": 0,
                 "cost_cents": 0, "stop_reason": "end_turn"},
  "output": {"assistant_text": "...", "refusal": false, "guardrail_actions": []},
  "error": null,
  "user_feedback": {"thumbs": null, "edit_distance": null}
}
```

Keep messages as structured turns, never concatenated. Store the rendered system prompt and
its template id. Record cached and reasoning token counts so cost isn't mispriced. Don't log
raw PII (redact and store a hash), embeddings, or full retrieved text (store document ids
and versions). The test of a good trace: can you replay it tomorrow against a candidate
prompt and make the same model call?

## PR result comment

```markdown
### Evals: refund-policy suite

| Metric | Baseline | Candidate | Difference | 95% CI | N | Band |
|---|---|---|---|---|---|---|
| Pass-all rate | 0.912 | 0.934 | +0.022 | [-0.004, +0.047] | 410 | hold |
| Regression set | 62/62 | 62/62 | 0 | | 62 | ship |
| p95 latency | 2.1 s | 2.2 s | +4.8% | [+1.9%, +7.6%] | 410 | investigate (a sure rise, inside the 15% window) |
| Cost per 1k requests | $4.10 | $4.30 | +4.9% | [+3.1%, +6.7%] | 410 | investigate (a sure rise, inside the 10% window) |

Verdict: **INVESTIGATE**, the worst band across the axes. Latency and cost rose for sure but
stay inside their minor-regression windows, so the release needs 50 traces read from the
slower slice and a written reason to ship. Quality holds: its interval contains zero and
reaches past the 1-point indifference window. Judge:
claude-opus-5-5, effort medium, rubric 3f1c9a; dataset gate-v7 @ 9e2b44.
```
