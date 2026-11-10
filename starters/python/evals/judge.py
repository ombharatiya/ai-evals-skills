"""A binary PASS/FAIL judge on Claude, for one failure mode.

The prompt has the four parts the aievals.co judge guide uses: the rubric
(role, definitions, PASS/FAIL criteria) from your rubric file, two to four
few-shot examples drawn from the train split only, the trace under review,
and a strict JSON output schema enforced with structured outputs.

Everything that changes the judge's behavior is fingerprinted in
`JudgeConfig.describe()`: the model, the effort, and a hash of the rendered
system prompt (rubric, few-shot text, instructions), the output schema and
max_tokens. Your calibration numbers belong to that exact fingerprint.
Change any of it and calibrate again.
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
MAX_TOKENS = 16000  # thinking is on for Opus 5.5 and counts against this

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

# USD per million tokens: (input, output, cache read). Cache writes bill at
# 1.25x input. Used only for the budget estimate. Includes the models the
# `fallbacks: "default"` setting can route to. Check current prices at
# https://platform.claude.com/docs/en/about-claude/pricing before relying on it.
PRICES: dict[str, tuple[float, float, float]] = {
    "claude-opus-5-5": (4.00, 20.00, 0.20),
    "claude-sonnet-5-5": (2.00, 10.00, 0.10),
    "claude-haiku-5-5": (0.10, 0.50, 0.01),
    "claude-opus-5": (5.00, 25.00, 0.50),
    "claude-opus-4-8": (5.00, 25.00, 0.50),
    "claude-sonnet-5": (2.00, 10.00, 0.20),
    "claude-fable-5-2": (8.00, 40.00, 0.20),
    "claude-fable-5-1": (10.00, 50.00, 0.25),
}
# A model missing from the table is charged at the most expensive known rate,
# so an unknown fallback can't slip under the budget.
_WORST = max(PRICES.values(), key=lambda p: p[1])

# Statuses that fail one row after the SDK's own retries. Anything else (a bad
# key, an unknown model, a malformed request) would fail every row.
ROW_LEVEL_STATUSES = {408, 409, 413, 422, 429}


class JudgeError(RuntimeError):
    """The judge returned no usable verdict (refusal, truncation, bad JSON, transient error)."""

    def __init__(self, message: str, cost_usd: float = 0.0) -> None:
        super().__init__(message)
        self.cost_usd = cost_usd


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

    @property
    def prompt_sha256(self) -> str:
        # Canonical JSON (sorted keys, no spaces, raw UTF-8) so the Python and
        # TypeScript starters compute the same fingerprint for the same judge.
        blob = json.dumps(
            {"max_tokens": MAX_TOKENS, "schema": VERDICT_SCHEMA, "system": render_system(self)},
            sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def describe(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "effort": self.effort,
            "prompt_sha256": self.prompt_sha256,
            "rubric_sha256": self.rubric_sha256,
            "few_shot_ids": [r["id"] for r in self.few_shots],
        }


def price_known(model: str) -> bool:
    return model in PRICES


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
    # Trim only ASCII whitespace: Python's strip() and JavaScript's trim() disagree on the rest.
    parts = [SYSTEM_PREAMBLE, "<rubric>\n" + config.rubric.strip(" \t\r\n") + "\n</rubric>"]
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


def _cost_of(model: str | None, usage: Any) -> float:
    price_in, price_out, price_read = PRICES.get(model or "", _WORST)
    uncached = getattr(usage, "input_tokens", 0) or 0
    written = getattr(usage, "cache_creation_input_tokens", 0) or 0
    read = getattr(usage, "cache_read_input_tokens", 0) or 0
    out = getattr(usage, "output_tokens", 0) or 0
    return (uncached * price_in + written * price_in * 1.25 + read * price_read + out * price_out) / 1e6


def estimate_cost(requested_model: str, usage: Any, served_model: str | None = None) -> float:
    """Price each model call in the response at the model that made it.

    With fallbacks, one response can hold several calls (the declined one
    and the rescue), listed in `usage.iterations`. Without that breakdown,
    the whole response is priced at the model that served it.
    """
    if usage is None:
        return 0.0
    iterations = getattr(usage, "iterations", None) or []
    priced = [it for it in iterations if getattr(it, "input_tokens", None) is not None]
    if priced:
        return sum(_cost_of(getattr(it, "model", None) or requested_model, it) for it in priced)
    return _cost_of(served_model or requested_model, usage)


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
            if exc.status_code in ROW_LEVEL_STATUSES or exc.status_code >= 500:
                raise JudgeError(f"{row['id']}: HTTP {exc.status_code} after retries") from exc
            raise
        return self._parse(row, response)

    def _create(self, row: Row) -> Any:
        return self.client.beta.messages.create(
            model=self.config.model,
            max_tokens=MAX_TOKENS,
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
        # Charge the call before deciding whether it produced a verdict:
        # a refusal or a truncated answer still costs money.
        cost = estimate_cost(self.config.model, getattr(response, "usage", None), getattr(response, "model", None))
        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None) if details else None
            raise JudgeError(f"{row['id']}: the judge declined (category: {category})", cost)
        if response.stop_reason == "max_tokens":
            raise JudgeError(f"{row['id']}: the judge hit max_tokens before answering", cost)
        text = next((b.text for b in response.content if b.type == "text"), None)
        if text is None:
            raise JudgeError(f"{row['id']}: no text block in the judge response", cost)
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise JudgeError(f"{row['id']}: judge output is not JSON: {text[:200]!r}", cost) from exc
        if not isinstance(data, dict):
            raise JudgeError(f"{row['id']}: judge output is not a JSON object: {text[:200]!r}", cost)
        label = str(data.get("label", "")).upper()
        if label not in ("PASS", "FAIL"):
            raise JudgeError(f"{row['id']}: unexpected label {data.get('label')!r}", cost)
        iterations = getattr(response.usage, "iterations", None) or []
        fallback_used = any(getattr(it, "type", None) == "fallback_message" for it in iterations)
        served_by = getattr(response, "model", None) or self.config.model
        return Verdict(
            label=label,
            critique=str(data.get("critique", "")),
            model=served_by,
            fallback_used=fallback_used or served_by != self.config.model,
            cost_usd=cost,
        )
