"""Runs the judge through the real Anthropic SDK against a fake HTTP server.

This checks the request the SDK actually sends and that the SDK parses the
response the judge expects. No network, no API key.
"""
import json

import pytest

anthropic = pytest.importorskip("anthropic")
httpx2 = pytest.importorskip("httpx2")

from evals.judge import ClaudeJudge, JudgeConfig, JudgeError  # noqa: E402

ROW = {"id": "t-1", "query": "Refund after 60 days?", "response": "Yes, within 90 days."}


def _client(handler):
    return anthropic.Anthropic(
        api_key="test-key",
        http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler)),
        max_retries=0,
    )


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

    verdict = ClaudeJudge(JudgeConfig(rubric="PASS if correct."), client=_client(handler))(ROW)
    assert verdict.label == "FAIL" and not verdict.fallback_used
    body = seen["body"]
    assert seen["url"].endswith("/v1/messages")
    assert "server-side-fallback-2026-07-01" in seen["headers"].get("anthropic-beta", "")
    assert body["fallbacks"] == "default"
    assert body["output_config"]["format"]["schema"]["required"] == ["label", "critique"]
    assert "temperature" not in body


def _error(status, kind):
    return lambda request: httpx2.Response(status, json={"type": "error", "error": {"type": kind, "message": "x"}})


@pytest.mark.parametrize("status,kind", [(429, "rate_limit_error"), (408, "timeout_error"),
                                         (413, "request_too_large"), (529, "overloaded_error")])
def test_transient_errors_fail_the_row_not_the_run(status, kind):
    with pytest.raises(JudgeError, match=str(status)):
        ClaudeJudge(JudgeConfig(rubric="R"), client=_client(_error(status, kind)))(ROW)


def test_bad_key_stops_the_run():
    with pytest.raises(anthropic.AuthenticationError):
        ClaudeJudge(JudgeConfig(rubric="R"), client=_client(_error(401, "authentication_error")))(ROW)
