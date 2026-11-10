/**
 * A binary PASS/FAIL judge on Claude, for one failure mode.
 *
 * The prompt has the four parts the aievals.co judge guide uses: the rubric
 * (role, definitions, PASS/FAIL criteria) from your rubric file, few-shots
 * from the train split only, the trace under review, and a strict JSON
 * schema enforced with structured outputs. Everything that changes the
 * judge is pinned in JudgeConfig and written next to every result.
 */
import { createHash } from "node:crypto";
import Anthropic from "@anthropic-ai/sdk";
import type { Label, Row } from "./data.js";

export type Effort = "low" | "medium" | "high" | "xhigh" | "max";

export const DEFAULT_MODEL = process.env.JUDGE_MODEL ?? "claude-opus-5-5";
export const DEFAULT_EFFORT = (process.env.JUDGE_EFFORT ?? "medium") as Effort;

export const VERDICT_SCHEMA = {
  type: "object",
  properties: {
    label: { type: "string", enum: ["PASS", "FAIL"] },
    critique: { type: "string" },
  },
  required: ["label", "critique"],
  additionalProperties: false,
};

const SYSTEM_PREAMBLE =
  "You are an evaluator. You grade one assistant response against one rubric and return a JSON verdict. " +
  "Judge only what the rubric asks. Do not reward length, confidence or politeness.";

const OUTPUT_INSTRUCTIONS =
  'Return a JSON object: {"label": "PASS" or "FAIL", "critique": "..."}. The critique is one sentence naming ' +
  "the specific part of the response that decided the label. When in doubt, FAIL and say why.";

/** USD per million tokens, for the per-run budget estimate only. */
export const PRICES: Record<string, [number, number]> = {
  "claude-opus-5-5": [4.0, 20.0],
  "claude-sonnet-5-5": [2.0, 10.0],
  "claude-haiku-5-5": [0.1, 0.5],
};

export class JudgeError extends Error {}

export interface Verdict {
  label: Label;
  critique: string;
  /** The model that actually answered. */
  model: string;
  fallbackUsed: boolean;
  costUsd: number;
}

export type Judge = (row: Row) => Promise<Verdict>;

export interface JudgeConfig {
  rubric: string;
  model: string;
  effort: Effort;
  fewShots: Row[];
}

export function makeConfig(rubric: string, overrides: Partial<JudgeConfig> = {}): JudgeConfig {
  return { rubric, model: DEFAULT_MODEL, effort: DEFAULT_EFFORT, fewShots: [], ...overrides };
}

export function describe(config: JudgeConfig) {
  return {
    model: config.model,
    effort: config.effort,
    rubric_sha256: createHash("sha256").update(config.rubric).digest("hex"),
    few_shot_ids: config.fewShots.map((r) => r.id),
  };
}

const byId = (a: Row, b: Row) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** Balanced few-shots from train: alternating FAIL and PASS, rows with a human critique first. */
export function pickFewShots(train: Row[], n = 4): Row[] {
  const ranked = (label: Label) =>
    train.filter((r) => r.label === label).sort((a, b) => Number(!a.critique) - Number(!b.critique) || byId(a, b));
  const fails = ranked("FAIL");
  const passes = ranked("PASS");
  const picked: Row[] = [];
  while (picked.length < n && (fails.length > 0 || passes.length > 0)) {
    const f = fails.shift();
    if (f) picked.push(f);
    if (picked.length < n) {
      const p = passes.shift();
      if (p) picked.push(p);
    }
  }
  return picked;
}

export function renderSystem(config: JudgeConfig): string {
  const parts = [SYSTEM_PREAMBLE, "<rubric>\n" + config.rubric.trim() + "\n</rubric>"];
  if (config.fewShots.length > 0) {
    const shots = config.fewShots.map((r) => {
      let shot = "<query>" + r.query + "</query>\n<response>" + r.response + "</response>\n";
      if (r.critique) shot += "<explanation>" + r.critique + "</explanation>\n";
      return "<example>\n" + shot + "<label>" + r.label + "</label>\n</example>";
    });
    parts.push("<examples>\n" + shots.join("\n") + "\n</examples>");
  }
  parts.push(OUTPUT_INSTRUCTIONS);
  return parts.join("\n\n");
}

export function renderUser(row: Row): string {
  return "Grade this response.\n\n<query>" + row.query + "</query>\n\n<response>" + row.response + "</response>";
}

export function estimateCost(model: string, usage: Anthropic.Beta.Messages.BetaUsage | undefined): number {
  const price = PRICES[model];
  if (!price || !usage) return 0;
  const [pin, pout] = price;
  const written = usage.cache_creation_input_tokens ?? 0;
  const read = usage.cache_read_input_tokens ?? 0;
  return (usage.input_tokens * pin + written * pin * 1.25 + read * pin * 0.1 + usage.output_tokens * pout) / 1e6;
}

/** Calls Claude once per row. Pass `client` to point it at a stub in tests. */
export function claudeJudge(config: JudgeConfig, client: Anthropic = new Anthropic()): Judge {
  const system = renderSystem(config);
  return async (row: Row): Promise<Verdict> => {
    let response: Anthropic.Beta.Messages.BetaMessage;
    try {
      response = await client.beta.messages.create({
      model: config.model,
      max_tokens: 16000, // thinking is on for Opus 5.5 and counts against this
      // If Claude declines on a safety classifier, rerun on a fallback model in
      // the same call. A rescued verdict comes from a different judge than the
      // one you calibrated, so it is flagged below.
      betas: ["server-side-fallback-2026-07-01"],
      fallbacks: "default",
      output_config: { effort: config.effort, format: { type: "json_schema", schema: VERDICT_SCHEMA } },
      // The rubric and few-shots are the same on every call, so cache them.
      system: [{ type: "text", text: system, cache_control: { type: "ephemeral" } }],
      messages: [{ role: "user", content: renderUser(row) }],
      });
    } catch (err) {
      // Rate limits, server errors and lost connections fail this row after the
      // SDK's retries. Anything else (bad key, unknown model, bad request) would
      // fail every row, so it stops the run.
      if (err instanceof Anthropic.APIConnectionError) throw new JudgeError(row.id + ": connection failed after retries");
      if (err instanceof Anthropic.APIError && (err.status === 429 || (err.status ?? 0) >= 500)) {
        throw new JudgeError(row.id + ": HTTP " + err.status + " after retries");
      }
      throw err;
    }
    if (response.stop_reason === "refusal") {
      const category = response.stop_details?.category ?? "none";
      throw new JudgeError(row.id + ": the judge declined (category: " + category + ")");
    }
    if (response.stop_reason === "max_tokens") throw new JudgeError(row.id + ": the judge hit max_tokens");
    const block = response.content.find((b): b is Anthropic.Beta.Messages.BetaTextBlock => b.type === "text");
    if (!block) throw new JudgeError(row.id + ": no text block in the judge response");
    let data: { label?: unknown; critique?: unknown };
    try {
      data = JSON.parse(block.text) as { label?: unknown; critique?: unknown };
    } catch {
      throw new JudgeError(row.id + ": judge output is not JSON: " + block.text.slice(0, 200));
    }
    const label = String(data.label ?? "").toUpperCase();
    if (label !== "PASS" && label !== "FAIL") throw new JudgeError(row.id + ": unexpected label " + String(data.label));
    const servedBy = response.model ?? config.model;
    const fallbackRan = (response.usage.iterations ?? []).some((it) => it.type === "fallback_message");
    return {
      label,
      critique: String(data.critique ?? ""),
      model: servedBy,
      fallbackUsed: fallbackRan || servedBy !== config.model,
      costUsd: estimateCost(servedBy, response.usage),
    };
  };
}

/** A problem with the key or the model, which would fail every row: print it plainly. */
export function setupProblem(err: unknown): string | null {
  if (
    err instanceof Anthropic.AuthenticationError ||
    err instanceof Anthropic.PermissionDeniedError ||
    err instanceof Anthropic.NotFoundError ||
    err instanceof Anthropic.BadRequestError
  ) {
    return err.message;
  }
  if (err instanceof Error && /authentication method/i.test(err.message)) return err.message;
  return null;
}
