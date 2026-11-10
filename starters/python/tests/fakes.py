"""Test doubles. Nothing in the test suite calls the API."""
from __future__ import annotations

import json
from types import SimpleNamespace

from evals.judge import Verdict


class FakeJudge:
    """Returns a fixed label per id (default PASS) and records what it saw."""

    def __init__(self, labels: dict[str, str] | None = None, default: str = "PASS", cost: float = 0.01):
        self.labels = labels or {}
        self.default = default
        self.cost = cost
        self.seen: list[str] = []

    def __call__(self, row):
        self.seen.append(row["id"])
        label = self.labels.get(row["id"], self.default)
        return Verdict(label=label, critique=f"fake {label}", model="fake-judge", cost_usd=self.cost)


def message(text: str | None = '{"label": "PASS", "critique": "ok"}', stop_reason: str = "end_turn",
            model: str = "claude-opus-5-5", iterations=None, stop_details=None):
    content = [] if text is None else [SimpleNamespace(type="text", text=text)]
    usage = SimpleNamespace(input_tokens=1000, output_tokens=400, cache_creation_input_tokens=0,
                            cache_read_input_tokens=0, iterations=iterations)
    return SimpleNamespace(stop_reason=stop_reason, content=content, usage=usage, model=model,
                           stop_details=stop_details)


class StubClient:
    """Mimics client.beta.messages.create and keeps the last request."""

    def __init__(self, response):
        self.response = response
        self.requests: list[dict] = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        return self.response


def verdict_json(label: str, critique: str = "because") -> str:
    return json.dumps({"label": label, "critique": critique})
