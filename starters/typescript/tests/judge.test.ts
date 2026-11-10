/**
 * Runs the judge through the real Anthropic SDK with a fake fetch, so the
 * test checks the request the SDK sends and the response it parses. No
 * network, no API key.
 */
import { mkdtempSync, readFileSync, existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import Anthropic from "@anthropic-ai/sdk";
import { describe as group, expect, it } from "vitest";
import { barFailures, calibrate } from "../src/calibrate.js";
import { loadJsonl, splitRows, type Row } from "../src/data.js";
import { claudeJudge, describe, estimateCost, JudgeError, makeConfig, pickFewShots, renderSystem, type Verdict } from "../src/judge.js";
import { correctedPassRate, roganGladen, scoreRows, type RowResult } from "../src/run-eval.js";
import { Budget, judgeRows } from "../src/runner.js";

const ROW: Row = { id: "t-1", query: "Refund after 60 days?", response: "Yes, within 90 days." };

function fakeClient(body: Record<string, unknown>, status = 200) {
  const requests: { url: string; headers: Headers; body: Record<string, unknown> }[] = [];
  const fetch = async (input: string | URL | Request, init?: RequestInit) => {
    requests.push({
      url: String(input instanceof Request ? input.url : input),
      headers: new Headers(init?.headers),
      body: JSON.parse(String(init?.body)),
    });
    return new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });
  };
  return { client: new Anthropic({ apiKey: "test-key", fetch, maxRetries: 0 }), requests };
}

function message(text: string | null, extra: Record<string, unknown> = {}) {
  return {
    id: "msg_test",
    type: "message",
    role: "assistant",
    model: "claude-opus-5-5",
    content: text === null ? [] : [{ type: "text", text }],
    stop_reason: "end_turn",
    stop_sequence: null,
    usage: { input_tokens: 1000, output_tokens: 400 },
    ...extra,
  };
}

const verdictJson = (label: string, critique = "because") => JSON.stringify({ label, critique });

group("claude judge", () => {
  it("sends a pinned, schema-constrained request and parses the verdict", async () => {
    const { client, requests } = fakeClient(message(verdictJson("FAIL", "Says 90 days.")));
    const verdict = await claudeJudge(makeConfig("PASS if correct."), client)(ROW);
    expect(verdict.label).toBe("FAIL");
    expect(verdict.costUsd).toBeCloseTo((1000 * 4 + 400 * 20) / 1e6, 12);
    const req = requests[0]!;
    expect(req.url).toMatch(/\/v1\/messages/);
    expect(req.headers.get("anthropic-beta")).toContain("server-side-fallback-2026-07-01");
    expect(req.body.fallbacks).toBe("default");
    expect(req.body).not.toHaveProperty("temperature");
    expect((req.body.output_config as { format: { type: string } }).format.type).toBe("json_schema");
  });

  it("raises on unusable answers and still charges them", async () => {
    for (const body of [
      message(null, { stop_reason: "refusal", stop_details: { type: "refusal", category: "cyber", explanation: null } }),
      message(null, { stop_reason: "max_tokens" }),
      message("not json"),
      message(verdictJson("MAYBE")),
      message("null"),
      message("[1, 2]"),
    ]) {
      const { client } = fakeClient(body);
      const err = await claudeJudge(makeConfig("R"), client)(ROW).catch((e: unknown) => e);
      expect(err).toBeInstanceOf(JudgeError);
      expect((err as JudgeError).costUsd).toBeGreaterThan(0);
    }
  });

  it("flags a fallback verdict and prices each call at its own model", async () => {
    const { client } = fakeClient(
      message(verdictJson("PASS"), {
        model: "claude-opus-4-8",
        usage: {
          input_tokens: 2000,
          output_tokens: 400,
          iterations: [
            { type: "message", model: null, input_tokens: 1000, output_tokens: 0, cache_creation_input_tokens: 0, cache_read_input_tokens: 0, cache_creation: null },
            { type: "fallback_message", model: "claude-opus-4-8", input_tokens: 1000, output_tokens: 400, cache_creation_input_tokens: 0, cache_read_input_tokens: 0, cache_creation: null },
          ],
        },
      }),
    );
    const verdict = await claudeJudge(makeConfig("R"), client)(ROW);
    expect(verdict.fallbackUsed).toBe(true);
    expect(verdict.model).toBe("claude-opus-4-8");
    expect(verdict.costUsd).toBeCloseTo((1000 * 4) / 1e6 + (1000 * 5 + 400 * 25) / 1e6, 12);
  });

  it("charges an unknown model at the top rate", () => {
    const usage = { input_tokens: 1000, output_tokens: 400 } as Anthropic.Beta.Messages.BetaUsage;
    expect(estimateCost("some-new-model", usage)).toBeGreaterThan(estimateCost("claude-opus-5-5", usage));
  });

  it("fingerprints the whole prompt, not just ids", () => {
    const shot: Row = { id: "f1", query: "q", response: "r", label: "FAIL", critique: "old" };
    const a = describe(makeConfig("R", { fewShots: [shot] }));
    const b = describe(makeConfig("R", { fewShots: [{ ...shot, critique: "new" }] }));
    expect(a.few_shot_ids).toEqual(b.few_shot_ids);
    expect(a.prompt_sha256).not.toBe(b.prompt_sha256);
    expect(describe(makeConfig("R", { fewShots: [shot] }))).toEqual(a);
  });

  it("builds balanced few-shots that prefer human critiques", () => {
    const train: Row[] = [
      { id: "p1", query: "q", response: "r", label: "PASS" },
      { id: "p2", query: "q", response: "r", label: "PASS", critique: "fine" },
      { id: "f1", query: "q", response: "r", label: "FAIL", critique: "wrong window" },
      { id: "f2", query: "q", response: "r", label: "FAIL" },
    ];
    const picked = pickFewShots(train, 3);
    expect(picked.map((r) => r.id)).toEqual(["f1", "p2", "f2"]);
    expect(renderSystem(makeConfig("R", { fewShots: picked }))).toContain("<explanation>wrong window</explanation>");
  });
});

group("api errors", () => {
  const errorBody = (type: string) => ({ type: "error", error: { type, message: "x" } });

  for (const [status, type] of [[429, "rate_limit_error"], [408, "timeout_error"], [413, "request_too_large"], [529, "overloaded_error"]] as const) {
    it("fails the row on HTTP " + status, async () => {
      const { client } = fakeClient(errorBody(type), status);
      await expect(claudeJudge(makeConfig("R"), client)(ROW)).rejects.toBeInstanceOf(JudgeError);
    });
  }

  it("stops the run on a bad key", async () => {
    const { client } = fakeClient(errorBody("authentication_error"), 401);
    await expect(claudeJudge(makeConfig("R"), client)(ROW)).rejects.toBeInstanceOf(Anthropic.AuthenticationError);
  });
});

function fakeJudge(labels: Record<string, "PASS" | "FAIL"> = {}, cost = 0.01) {
  const seen: string[] = [];
  const judge = async (row: Row): Promise<Verdict> => {
    seen.push(row.id);
    const label = labels[row.id] ?? "PASS";
    return { label, critique: "fake " + label, model: "fake-judge", fallbackUsed: false, costUsd: cost };
  };
  return { judge, seen };
}

group("pipeline", () => {
  const data = (name: string) => new URL("../data/" + name, import.meta.url).pathname;

  it("calibrates on dev with few-shots from train only, and the small example can't clear the bar", async () => {
    const rows = loadJsonl(data("labeled.example.jsonl"), true);
    const [train, dev] = splitRows(rows);
    const fails = dev.filter((r) => r.label === "FAIL");
    const labels = Object.fromEntries(dev.map((r) => [r.id, r.label!]));
    labels[fails[0]!.id] = "PASS";
    const config = makeConfig("R", { fewShots: pickFewShots(train) });
    const outDir = mkdtempSync(join(tmpdir(), "calib-"));
    const summary = await calibrate({ rows, judge: fakeJudge(labels).judge, config, splitName: "dev", splitRows: dev, outDir, repeats: 2, workers: 2 });
    expect(summary.report.pass_recall).toBe(1);
    expect(summary.report.fail_recall).toBeCloseTo((fails.length - 1) / fails.length, 12);
    expect(summary.label_flips_across_repeats).toBe(0);
    const devIds = new Set(dev.map((r) => r.id));
    expect(summary.judge.few_shot_ids.some((id) => devIds.has(id))).toBe(false);
    const disagreements = readFileSync(join(outDir, "disagreements.jsonl"), "utf8").trim().split("\n");
    expect(disagreements.map((l) => JSON.parse(l).id)).toEqual([fails[0]!.id]);
    expect(barFailures(summary, 0.9, 0.6, 30).some((r) => r.includes("FAIL rows in this split"))).toBe(true);
  });

  it("fails a test read with missing verdicts and writes no test disagreements", async () => {
    const rows = loadJsonl(data("labeled.example.jsonl"), true);
    const test = splitRows(rows)[2];
    const firstFail = test.find((r) => r.label === "FAIL")!.id;
    const judge = async (row: Row): Promise<Verdict> => {
      if (row.label === "FAIL" && row.id !== firstFail) throw new JudgeError("declined", 0.01);
      return { label: row.label!, critique: "", model: "fake", fallbackUsed: false, costUsd: 0.01 };
    };
    const outDir = mkdtempSync(join(tmpdir(), "calib-"));
    const summary = await calibrate({ rows, judge, config: makeConfig("R"), splitName: "test", splitRows: test, outDir });
    expect(existsSync(join(outDir, "disagreements.jsonl"))).toBe(false);
    expect(summary.report.fail_recall).toBe(1);
    expect(barFailures(summary, 0.9, 0.6, 1).some((r) => r.includes("no verdict"))).toBe(true);
  });

  it("runs code checks before the judge", async () => {
    const { judge, seen } = fakeJudge({ "o-008": "FAIL" });
    const results = new Map((await scoreRows(loadJsonl(data("outputs.example.jsonl")), judge, 2)).map((r) => [r.id, r]));
    expect(seen).not.toContain("o-009");
    expect(results.get("o-009")!.passed).toBe(false);
    expect(seen).not.toContain("o-010");
    expect(results.get("o-008")!.passed).toBe(false);
    expect(results.get("o-001")!.passed).toBe(true);
  });

  it("charges failed calls against the budget and stops new ones", async () => {
    const rows: Row[] = Array.from({ length: 10 }, (_, i) => ({ id: String(i), query: "q", response: "r" }));
    const declines = async (): Promise<Verdict> => {
      throw new JudgeError("declined", 0.01);
    };
    const budget = new Budget(0.03);
    const outcomes = await judgeRows(declines, rows, 1, budget);
    expect(budget.spentUsd).toBeCloseTo(0.03, 10);
    expect(outcomes.filter((o) => o.error === "declined")).toHaveLength(3);
    expect(outcomes.slice(3).every((o) => o.error?.includes("budget"))).toBe(true);
  });

  it("rejects a budget that isn't a number", () => {
    process.env.EVAL_BUDGET_USD = "5 USD";
    try {
      expect(() => Budget.fromEnv()).toThrow(/number/);
    } finally {
      delete process.env.EVAL_BUDGET_USD;
    }
  });

  it("stops every worker after a setup error", async () => {
    const rows: Row[] = Array.from({ length: 200 }, (_, i) => ({ id: String(i), query: "q", response: "r" }));
    let calls = 0;
    const badKey = async (): Promise<Verdict> => {
      calls++;
      throw new Error("bad key");
    };
    await expect(judgeRows(badKey, rows, 4)).rejects.toThrow("bad key");
    expect(calls).toBeLessThanOrEqual(4);
  });

  it("corrects only the judged rows", () => {
    const results = [
      ...Array.from({ length: 2 }, () => ({ passed: false, judged: false, check_failures: ["x"], judge: null })),
      ...Array.from({ length: 8 }, (_, i) => ({ passed: i < 6, judged: true, check_failures: [], judge: i < 6 ? "PASS" : "FAIL" })),
    ] as unknown as RowResult[];
    expect(correctedPassRate(results, 0.9, 0.8)).toBeCloseTo((roganGladen(6 / 8, 0.9, 0.8)! * 8) / 10, 12);
    expect(roganGladen(0.5, NaN, 0.8)).toBeNull();
    expect(roganGladen(0.5, 0.5, 0.4)).toBeNull();
  });
});
