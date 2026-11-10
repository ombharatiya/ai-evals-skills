/**
 * A binary PASS/FAIL judge on Claude, for one failure mode.
 *
 * The prompt has the four parts the aievals.co judge guide uses: the rubric
 * (role, definitions, PASS/FAIL criteria) from your rubric file, few-shots
 * from the train split only, the trace under review, and a strict JSON
 * schema enforced with structured outputs.
 *
 * describe() fingerprints everything that changes the judge: the model, the
 * effort, and a hash of the rendered system prompt (rubric, few-shot text,
 * instructions), the output schema and max_tokens. Calibration numbers
 * belong to that exact fingerprint. Change any of it and calibrate again.
 */
import { createHash } from "node:crypto";
import Anthropic from "@anthropic-ai/sdk";
import { trimSpace, type Label, type Row } from "./data.js";

export type Effort = "low" | "medium" | "high" | "xhigh" | "max";
export const EFFORTS: Effort[] = ["low", "medium", "high", "xhigh", "max"];

export const DEFAULT_MODEL = process.env.JUDGE_MODEL ?? "claude-opus-5-5";
export const DEFAULT_EFFORT = (process.env.JUDGE_EFFORT ?? "medium") as Effort;
export const MAX_TOKENS = 16000; // thinking is on for Opus 5.5 and counts against this

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

/**
 * USD per million tokens: [input, output, cache read]. Cache writes bill at
 * 1.25x input. Used only for the budget estimate. Includes the models that
 * `fallbacks: "default"` can route to. Check current prices at
 * https://platform.claude.com/docs/en/about-claude/pricing before relying on it.
 */
export const PRICES: Record<string, [number, number, number]> = {
  "claude-opus-5-5": [4.0, 20.0, 0.2],
  "claude-sonnet-5-5": [2.0, 10.0, 0.1],
  "claude-haiku-5-5": [0.1, 0.5, 0.01],
  "claude-opus-5": [5.0, 25.0, 0.5],
  "claude-opus-4-8": [5.0, 25.0, 0.5],
  "claude-sonnet-5": [2.0, 10.0, 0.2],
  "claude-fable-5-2": [8.0, 40.0, 0.2],
  "claude-fable-5-1": [10.0, 50.0, 0.25],
};
/** A model missing from the table is charged at the most expensive known rate. */
const WORST = Object.values(PRICES).reduce((a, b) => (b[1] > a[1] ? b : a));

/** Statuses that fail one row after the SDK's own retries; anything else would fail every row. */
const ROW_LEVEL_STATUSES = new Set([408, 409, 413, 422, 429]);

export class JudgeError extends Error {
  constructor(
    message: string,
    readonly costUsd = 0,
  ) {
    super(message);
  }
}

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

const sha256 = (text: string) => createHash("sha256").update(text).digest("hex");

/** JSON with sorted keys and no spaces, the same bytes Python's sort_keys=True, separators=(",", ":") writes. */
export function canonicalJson(value: unknown): string {
  if (Array.isArray(value)) return "[" + value.map(canonicalJson).join(",") + "]";
  if (value && typeof value === "object") {
    const entries = Object.keys(value as object)
      .sort()
      .map((k) => JSON.stringify(k) + ":" + canonicalJson((value as Record<string, unknown>)[k]));
    return "{" + entries.join(",") + "}";
  }
  return JSON.stringify(value);
}

export function describe(config: JudgeConfig) {
  return {
    model: config.model,
    effort: config.effort,
    prompt_sha256: sha256(canonicalJson({ max_tokens: MAX_TOKENS, schema: VERDICT_SCHEMA, system: renderSystem(config) })),
    rubric_sha256: sha256(config.rubric),
    few_shot_ids: config.fewShots.map((r) => r.id),
  };
}

export const priceKnown = (model: string) => model in PRICES;

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
  // Trim only ASCII whitespace: JavaScript's trim() and Python's strip() disagree on the rest.
  const parts = [SYSTEM_PREAMBLE, "<rubric>\n" + trimSpace(config.rubric) + "\n</rubric>"];
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

interface TokenCounts {
  input_tokens: number;
  output_tokens: number;
  cache_creation_input_tokens?: number | null;
  cache_read_input_tokens?: number | null;
}

function costOf(model: string | null | undefined, usage: TokenCounts): number {
  const [pin, pout, pread] = PRICES[model ?? ""] ?? WORST;
  const written = usage.cache_creation_input_tokens ?? 0;
  const read = usage.cache_read_input_tokens ?? 0;
  return (usage.input_tokens * pin + written * pin * 1.25 + read * pread + usage.output_tokens * pout) / 1e6;
}

/**
 * Price each model call in the response at the model that made it. With
 * fallbacks, one response can hold several calls, listed in usage.iterations.
 * Without that breakdown, the response is priced at the model that served it.
 */
export function estimateCost(requestedModel: string, usage: Anthropic.Beta.Messages.BetaUsage | undefined, servedModel?: string | null): number {
  if (!usage) return 0;
  const priced = (usage.iterations ?? []).filter(
    (it): it is Extract<typeof it, TokenCounts> => typeof (it as Partial<TokenCounts>).input_tokens === "number",
  );
  if (priced.length > 0) {
    return priced.reduce((sum, it) => sum + costOf(("model" in it ? (it.model as string | null) : null) ?? requestedModel, it), 0);
  }
  return costOf(servedModel ?? requestedModel, usage);
}

/** Calls Claude once per row. Pass `client` to point it at a stub in tests. */
export function claudeJudge(config: JudgeConfig, client: Anthropic = new Anthropic()): Judge {
  const system = renderSystem(config);
  return async (row: Row): Promise<Verdict> => {
    let response: Anthropic.Beta.Messages.BetaMessage;
    try {
      response = await client.beta.messages.create({
        model: config.model,
        max_tokens: MAX_TOKENS,
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
      if (err instanceof Anthropic.APIConnectionError) throw new JudgeError(row.id + ": connection failed after retries");
      if (err instanceof Anthropic.APIError && err.status !== undefined && (ROW_LEVEL_STATUSES.has(err.status) || err.status >= 500)) {
        throw new JudgeError(row.id + ": HTTP " + err.status + " after retries");
      }
      throw err;
    }
    // Charge the call before deciding whether it produced a verdict: a refusal
    // or a truncated answer still costs money.
    const cost = estimateCost(config.model, response.usage, response.model);
    if (response.stop_reason === "refusal") {
      const category = response.stop_details?.category ?? "none";
      throw new JudgeError(row.id + ": the judge declined (category: " + category + ")", cost);
    }
    if (response.stop_reason === "max_tokens") throw new JudgeError(row.id + ": the judge hit max_tokens", cost);
    const block = response.content.find((b): b is Anthropic.Beta.Messages.BetaTextBlock => b.type === "text");
    if (!block) throw new JudgeError(row.id + ": no text block in the judge response", cost);
    let data: unknown;
    try {
      data = JSON.parse(block.text);
    } catch {
      throw new JudgeError(row.id + ": judge output is not JSON: " + block.text.slice(0, 200), cost);
    }
    if (data === null || typeof data !== "object" || Array.isArray(data)) {
      throw new JudgeError(row.id + ": judge output is not a JSON object: " + block.text.slice(0, 200), cost);
    }
    const verdict = data as { label?: unknown; critique?: unknown };
    const label = String(verdict.label ?? "").toUpperCase();
    if (label !== "PASS" && label !== "FAIL") throw new JudgeError(row.id + ": unexpected label " + String(verdict.label), cost);
    const servedBy = response.model ?? config.model;
    const fallbackRan = (response.usage.iterations ?? []).some((it) => it.type === "fallback_message");
    return {
      label,
      critique: String(verdict.critique ?? ""),
      model: servedBy,
      fallbackUsed: fallbackRan || servedBy !== config.model,
      costUsd: cost,
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
