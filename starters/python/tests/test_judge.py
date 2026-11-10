from types import SimpleNamespace

import pytest

from evals.judge import ClaudeJudge, JudgeConfig, JudgeError, estimate_cost, pick_few_shots, render_system
from tests.fakes import StubClient, message, verdict_json

ROW = {"id": "t-1", "query": "Refund after 60 days?", "response": "Yes, within 90 days."}


def judge_with(response, **config):
    client = StubClient(response)
    return ClaudeJudge(JudgeConfig(rubric="PASS if correct.", **config), client=client), client


def test_request_shape_pins_the_judge():
    judge, client = judge_with(message(verdict_json("FAIL", "Says 90 days.")))
    verdict = judge(ROW)
    assert verdict.label == "FAIL" and verdict.critique == "Says 90 days."
    req = client.requests[0]
    assert req["model"] == "claude-opus-5-5"
    assert req["output_config"]["effort"] == "medium"
    assert req["output_config"]["format"]["type"] == "json_schema"
    assert req["fallbacks"] == "default"
    assert "temperature" not in req and "top_p" not in req and "thinking" not in req
    assert req["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "Refund after 60 days?" in req["messages"][0]["content"]


def test_unusable_answers_raise_and_still_carry_their_cost():
    for response in (message(None, stop_reason="refusal"), message(None, stop_reason="max_tokens"),
                     message("not json"), message(verdict_json("MAYBE")), message("[1, 2]"), message("null")):
        judge, _ = judge_with(response)
        with pytest.raises(JudgeError) as exc:
            judge(ROW)
        assert exc.value.cost_usd > 0


def test_fallback_verdict_is_flagged_and_priced_per_call():
    iterations = [
        SimpleNamespace(type="message", model=None, input_tokens=1000, output_tokens=0,
                        cache_creation_input_tokens=0, cache_read_input_tokens=0),
        SimpleNamespace(type="fallback_message", model="claude-opus-4-8", input_tokens=1000, output_tokens=400,
                        cache_creation_input_tokens=0, cache_read_input_tokens=0),
    ]
    judge, _ = judge_with(message(verdict_json("PASS"), model="claude-opus-4-8", iterations=iterations))
    verdict = judge(ROW)
    assert verdict.fallback_used and verdict.model == "claude-opus-4-8"
    # First call at Opus 5.5 input price, the rescue at Opus 4.8 prices.
    assert verdict.cost_usd == pytest.approx((1000 * 4) / 1e6 + (1000 * 5 + 400 * 25) / 1e6)


def test_cost_uses_the_price_table_and_charges_unknown_models_at_the_top_rate():
    usage = SimpleNamespace(input_tokens=1000, output_tokens=400, cache_creation_input_tokens=0,
                            cache_read_input_tokens=1000, iterations=None)
    assert estimate_cost("claude-opus-5-5", usage) == pytest.approx((1000 * 4 + 1000 * 0.20 + 400 * 20) / 1e6)
    assert estimate_cost("some-new-model", usage) > estimate_cost("claude-opus-5-5", usage)


def test_few_shots_are_balanced_and_prefer_critiques():
    train = [
        {"id": "p1", "query": "q", "response": "r", "label": "PASS"},
        {"id": "p2", "query": "q", "response": "r", "label": "PASS", "critique": "fine"},
        {"id": "f1", "query": "q", "response": "r", "label": "FAIL", "critique": "wrong window"},
        {"id": "f2", "query": "q", "response": "r", "label": "FAIL"},
    ]
    picked = pick_few_shots(train, n=3)
    assert [r["id"] for r in picked] == ["f1", "p2", "f2"]
    system = render_system(JudgeConfig(rubric="R", few_shots=picked))
    assert "<explanation>wrong window</explanation>" in system and system.count("<example>") == 3


def test_fingerprint_changes_with_any_prompt_change():
    shot = {"id": "f1", "query": "q", "response": "r", "label": "FAIL", "critique": "old"}
    a = JudgeConfig(rubric="R", few_shots=[shot]).describe()
    b = JudgeConfig(rubric="R", few_shots=[{**shot, "critique": "new"}]).describe()
    c = JudgeConfig(rubric="R2", few_shots=[shot]).describe()
    assert a["few_shot_ids"] == b["few_shot_ids"]  # same ids...
    assert a["prompt_sha256"] != b["prompt_sha256"] != c["prompt_sha256"]  # ...different prompt
    assert JudgeConfig(rubric="R", few_shots=[shot]).describe() == a


def test_fingerprint_matches_the_typescript_starter():
    # Same canonical JSON in both languages; the TypeScript test checks the same value.
    shot = {"id": "f1", "query": "q", "response": "r", "label": "FAIL", "critique": "naïve x"}
    assert JudgeConfig(rubric="R", few_shots=[shot]).describe()["prompt_sha256"] == (
        "7837062c1b9d6ee52eb994de04b29ecc8a25eb5ab6e1e8e902e4cc12fdac4189"
    )


def test_without_an_iteration_breakdown_the_served_model_is_priced():
    usage = SimpleNamespace(input_tokens=1000, output_tokens=400, cache_creation_input_tokens=0,
                            cache_read_input_tokens=0, iterations=None)
    assert estimate_cost("claude-opus-5-5", usage, "claude-opus-4-8") == pytest.approx((1000 * 5 + 400 * 25) / 1e6)
