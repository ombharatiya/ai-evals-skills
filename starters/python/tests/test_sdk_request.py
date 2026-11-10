"""Runs the judge through the real Anthropic SDK against a fake HTTP server.

This checks the request the SDK actually sends and that the SDK parses the
response the judge expects. No network, no API key.
"""
import json

import pytest

anthropic = pytest.importorskip("anthropic")
httpx2 = pytest.importorskip("httpx2")

from evals.judge import ClaudeJudge, JudgeConfig  # noqa: E402


def test_real_sdk_round_trip():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["headers"] = dict(request.headers)
        seen["body"] = json.loads(request.content)
        return httpx2.Response(200, json={
            "id": "msg_test", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
            "content": [{"type": "text", "text": json.dumps({"label": "FAIL", "critique": "Says 90 days."})}],
            "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 900, "output_tokens": 300},
        })

    client = anthropic.Anthropic(
        api_key="test-key",
        http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler)),
        max_retries=0,
    )
    judge = ClaudeJudge(JudgeConfig(rubric="PASS if correct."), client=client)
    verdict = judge({"id": "t-1", "query": "Refund after 60 days?", "response": "Yes, within 90 days."})

    assert verdict.label == "FAIL" and not verdict.fallback_used
    body = seen["body"]
    assert seen["url"].endswith("/v1/messages")
    assert "server-side-fallback-2026-07-01" in seen["headers"].get("anthropic-beta", "")
    assert body["fallbacks"] == "default"
    assert body["output_config"]["format"]["schema"]["required"] == ["label", "critique"]
    assert "temperature" not in body


def _client(status, body):
    def handler(request):
        return httpx2.Response(status, json=body)

    return anthropic.Anthropic(
        api_key="test-key",
        http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler)),
        max_retries=0,
    )


def test_rate_limit_fails_the_row_not_the_run():
    from evals.judge import JudgeError

    judge = ClaudeJudge(JudgeConfig(rubric="R"), client=_client(429, {"type": "error", "error": {"type": "rate_limit_error", "message": "slow down"}}))
    with pytest.raises(JudgeError, match="429"):
        judge({"id": "t-1", "query": "q", "response": "r"})


def test_bad_key_stops_the_run():
    judge = ClaudeJudge(JudgeConfig(rubric="R"), client=_client(401, {"type": "error", "error": {"type": "authentication_error", "message": "invalid x-api-key"}}))
    with pytest.raises(anthropic.AuthenticationError):
        judge({"id": "t-1", "query": "q", "response": "r"})
