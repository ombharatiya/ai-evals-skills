"""A binary PASS/FAIL judge on Claude, for one failure mode.

The prompt has the four parts the aievals.co judge guide uses: the rubric
(role, definitions, PASS/FAIL criteria) from your rubric file, two to four
few-shot examples drawn from the train split only, the trace under review,
and a strict JSON output schema enforced with structured outputs.

Everything that changes the judge's behavior is pinned in `JudgeConfig` and
written next to every result: model, effort, rubric hash, few-shot ids. Your
calibration numbers belong to that exact configuration. Change any of it and
calibrate again.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Protocol

import anthropic

Row = dict[str, Any]

DEFAULT_MODEL = os.environ.get("JUDGE_MODEL", "claude-opus-5-5")
DEFAULT_EFFORT = os.environ.get("JUDGE_EFFORT", "medium")

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": ["PASS", "FAIL"]},
        "critique": {"type": "string"},
    },
    "required": ["label", "critique"],
    "additionalProperties": False,
}

SYSTEM_PREAMBLE = (
    "You are an evaluator. You grade one assistant response against one rubric "
    "and return a JSON verdict. Judge only what the rubric asks. Do not reward "
    "length, confidence or politeness."
)

OUTPUT_INSTRUCTIONS = (
    'Return a JSON object: {"label": "PASS" or "FAIL", "critique": "..."}. '
    "The critique is one sentence naming the specific part of the response that "
    "decided the label. When in doubt, FAIL and say why."
)

# USD per million tokens, for the per-run budget estimate only.
PRICES = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}


class JudgeError(RuntimeError):
    """The judge returned no usable verdict (refusal, truncation, bad JSON)."""


@dataclass(frozen=True)
class Verdict:
    label: str
    critique: str
    model: str  # the model that actually answered
    fallback_used: bool = False
    cost_usd: float = 0.0


class Judge(Protocol):
    def __call__(self, row: Row) -> Verdict: ...


@dataclass
class JudgeConfig:
    rubric: str
    model: str = DEFAULT_MODEL
    effort: str = DEFAULT_EFFORT
    few_shots: list[Row] = field(default_factory=list)

    @property
    def rubric_sha256(self) -> str:
        return hashlib.sha256(self.rubric.encode("utf-8")).hexdigest()

    def describe(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "effort": self.effort,
            "rubric_sha256": self.rubric_sha256,
            "few_shot_ids": [r["id"] for r in self.few_shots],
        }


def pick_few_shots(train: list[Row], n: int = 4) -> list[Row]:
    """Balanced few-shots from train: alternating FAIL and PASS, FAILs first.

    Rows with a human critique are preferred, since the explanation is what
    teaches the judge where the line is.
    """
    def ranked(label: str) -> list[Row]:
        rows = [r for r in train if r.get("label") == label]
        return sorted(rows, key=lambda r: (not r.get("critique"), r["id"]))

    fails, passes = ranked("FAIL"), ranked("PASS")
    picked: list[Row] = []
    while len(picked) < n and (fails or passes):
        if fails:
            picked.append(fails.pop(0))
        if len(picked) < n and passes:
            picked.append(passes.pop(0))
    return picked


def render_system(config: JudgeConfig) -> str:
    parts = [SYSTEM_PREAMBLE, "<rubric>\n" + config.rubric.strip() + "\n</rubric>"]
    if config.few_shots:
        shots = []
        for r in config.few_shots:
            shot = f"<query>{r['query']}</query>\n<response>{r['response']}</response>\n"
            if r.get("critique"):
                shot += f"<explanation>{r['critique']}</explanation>\n"
            shot += f"<label>{r['label']}</label>"
            shots.append(f"<example>\n{shot}\n</example>")
        parts.append("<examples>\n" + "\n".join(shots) + "\n</examples>")
    parts.append(OUTPUT_INSTRUCTIONS)
    return "\n\n".join(parts)


def render_user(row: Row) -> str:
    return (
        "Grade this response.\n\n"
        f"<query>{row['query']}</query>\n\n"
        f"<response>{row['response']}</response>"
    )


def estimate_cost(model: str, usage: Any) -> float:
    if model not in PRICES or usage is None:
        return 0.0
    price_in, price_out = PRICES[model]
    uncached = getattr(usage, "input_tokens", 0) or 0
    written = getattr(usage, "cache_creation_input_tokens", 0) or 0
    read = getattr(usage, "cache_read_input_tokens", 0) or 0
    out = getattr(usage, "output_tokens", 0) or 0
    return (uncached * price_in + written * price_in * 1.25 + read * price_in * 0.1 + out * price_out) / 1e6


class ClaudeJudge:
    """Calls Claude once per row. Pass `client` to inject a stub in tests."""

    def __init__(self, config: JudgeConfig, client: Any = None) -> None:
        if client is None:
            client = anthropic.Anthropic()  # ANTHROPIC_API_KEY or an `ant auth login` profile
        self.client = client
        self.config = config
        self._system = render_system(config)

    def __call__(self, row: Row) -> Verdict:
        try:
            response = self._create(row)
        except (anthropic.APIConnectionError, anthropic.APITimeoutError) as exc:
            raise JudgeError(f"{row['id']}: {type(exc).__name__} after retries") from exc
        except anthropic.APIStatusError as exc:
            # Rate limits and server errors fail this row after the SDK's retries.
            # Anything else (bad key, unknown model, bad request) is a setup problem
            # that would fail every row, so it stops the run.
            if exc.status_code == 429 or exc.status_code >= 500:
                raise JudgeError(f"{row['id']}: HTTP {exc.status_code} after retries") from exc
            raise
        return self._parse(row, response)

    def _create(self, row: Row) -> Any:
        return self.client.beta.messages.create(
            model=self.config.model,
            max_tokens=16000,  # thinking is on for Opus 5.5 and counts against this
            # If Claude declines the request on a safety classifier, rerun it on a
            # fallback model inside the same call. A rescued verdict comes from a
            # different judge than the one you calibrated, so it is flagged below.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            output_config={
                "effort": self.config.effort,
                "format": {"type": "json_schema", "schema": VERDICT_SCHEMA},
            },
            # The rubric and few-shots are the same on every call, so cache them.
            system=[{"type": "text", "text": self._system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": render_user(row)}],
        )

    def _parse(self, row: Row, response: Any) -> Verdict:
        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None) if details else None
            raise JudgeError(f"{row['id']}: the judge declined (category: {category})")
        if response.stop_reason == "max_tokens":
            raise JudgeError(f"{row['id']}: the judge hit max_tokens before answering")
        text = next((b.text for b in response.content if b.type == "text"), None)
        if text is None:
            raise JudgeError(f"{row['id']}: no text block in the judge response")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise JudgeError(f"{row['id']}: judge output is not JSON: {text[:200]!r}") from exc
        label = str(data.get("label", "")).upper()
        if label not in ("PASS", "FAIL"):
            raise JudgeError(f"{row['id']}: unexpected label {data.get('label')!r}")
        iterations = getattr(response.usage, "iterations", None) or []
        fallback_used = any(getattr(it, "type", None) == "fallback_message" for it in iterations)
        served_by = getattr(response, "model", None) or self.config.model
        return Verdict(
            label=label,
            critique=str(data.get("critique", "")),
            model=served_by,
            fallback_used=fallback_used or served_by != self.config.model,
            cost_usd=estimate_cost(served_by, response.usage),
        )
