import pytest

from evals.judge import ClaudeJudge, JudgeConfig, JudgeError, pick_few_shots, render_system
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


def test_refusal_and_truncation_raise_instead_of_guessing():
    for response in (message(None, stop_reason="refusal"), message(None, stop_reason="max_tokens"),
                     message("not json"), message(verdict_json("MAYBE"))):
        judge, _ = judge_with(response)
        with pytest.raises(JudgeError):
            judge(ROW)


def test_fallback_verdict_is_flagged():
    from types import SimpleNamespace

    judge, _ = judge_with(message(verdict_json("PASS"), model="claude-opus-4-8",
                                  iterations=[SimpleNamespace(type="fallback_message")]))
    verdict = judge(ROW)
    assert verdict.fallback_used and verdict.model == "claude-opus-4-8"


def test_cost_estimate_uses_the_price_table():
    judge, _ = judge_with(message(verdict_json("PASS")))
    # 1000 input at $4/M + 400 output at $20/M
    assert abs(judge(ROW).cost_usd - (1000 * 4 + 400 * 20) / 1e6) < 1e-12


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


def test_describe_changes_when_the_rubric_changes():
    a = JudgeConfig(rubric="one").describe()
    b = JudgeConfig(rubric="two").describe()
    assert a["rubric_sha256"] != b["rubric_sha256"]
