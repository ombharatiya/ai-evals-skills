/**
 * Runs the judge through the real Anthropic SDK with a fake fetch, so the
 * test checks the request the SDK sends and the response it parses. No
 * network, no API key.
 */
import Anthropic from "@anthropic-ai/sdk";
import { describe as group, expect, it } from "vitest";
import { loadJsonl } from "../src/data.js";
import { claudeJudge, describe, JudgeError, makeConfig, pickFewShots, renderSystem, type Verdict } from "../src/judge.js";
import { calibrate } from "../src/calibrate.js";
import { correctedRate, scoreRows } from "../src/run-eval.js";
import { Budget, judgeRows } from "../src/runner.js";
import { stratifiedSplit, type Row } from "../src/data.js";
import { mkdtempSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const ROW: Row = { id: "t-1", query: "Refund after 60 days?", response: "Yes, within 90 days." };

function fakeClient(body: Record<string, unknown>) {
  const requests: { url: string; headers: Headers; body: Record<string, unknown> }[] = [];
  const fetch = async (input: string | URL | Request, init?: RequestInit) => {
    requests.push({
      url: String(input instanceof Request ? input.url : input),
      headers: new Headers(init?.headers),
      body: JSON.parse(String(init?.body)),
    });
    return new Response(JSON.stringify(body), { status: 200, headers: { "content-type": "application/json" } });
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
    expect(verdict.critique).toBe("Says 90 days.");
    expect(verdict.costUsd).toBeCloseTo((1000 * 4 + 400 * 20) / 1e6, 12);
    const req = requests[0]!;
    expect(req.url).toMatch(/\/v1\/messages/);
    expect(req.headers.get("anthropic-beta")).toContain("server-side-fallback-2026-07-01");
    expect(req.body.fallbacks).toBe("default");
    expect(req.body).not.toHaveProperty("temperature");
    expect((req.body.output_config as { format: { type: string } }).format.type).toBe("json_schema");
  });

  it("raises on refusal, truncation and malformed output", async () => {
    for (const body of [
      message(null, { stop_reason: "refusal", stop_details: { type: "refusal", category: "cyber", explanation: null } }),
      message(null, { stop_reason: "max_tokens" }),
      message("not json"),
      message(verdictJson("MAYBE")),
    ]) {
      const { client } = fakeClient(body);
      await expect(claudeJudge(makeConfig("R"), client)(ROW)).rejects.toBeInstanceOf(JudgeError);
    }
  });

  it("flags a verdict served by a fallback model", async () => {
    const { client } = fakeClient(
      message(verdictJson("PASS"), {
        model: "claude-opus-4-8",
        usage: { input_tokens: 10, output_tokens: 5, iterations: [{ type: "fallback_message", model: "claude-opus-4-8", input_tokens: 10, output_tokens: 5 }] },
      }),
    );
    const verdict = await claudeJudge(makeConfig("R"), client)(ROW);
    expect(verdict.fallbackUsed).toBe(true);
    expect(verdict.model).toBe("claude-opus-4-8");
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
    const system = renderSystem(makeConfig("R", { fewShots: picked }));
    expect(system).toContain("<explanation>wrong window</explanation>");
    expect(describe(makeConfig("one")).rubric_sha256).not.toBe(describe(makeConfig("two")).rubric_sha256);
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

  it("calibrates on dev with few-shots from train only", async () => {
    const rows = loadJsonl(data("labeled.example.jsonl"), true);
    const [train, dev] = stratifiedSplit(rows);
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

  it("stops new calls at the budget", async () => {
    const rows: Row[] = Array.from({ length: 10 }, (_, i) => ({ id: String(i), query: "q", response: "r" }));
    const outcomes = await judgeRows(fakeJudge({}, 0.01).judge, rows, 1, new Budget(0.03));
    expect(outcomes.filter((o) => o.verdict)).toHaveLength(3);
    expect(outcomes.filter((o) => !o.verdict).every((o) => o.error?.includes("budget"))).toBe(true);
  });

  it("corrects a judge-scored rate", () => {
    expect(correctedRate(0.75, 0.9, 0.8)).toBeCloseTo((0.75 + 0.8 - 1) / 0.7, 12);
    expect(correctedRate(0.5, 0.5, 0.4)).toBeNull();
  });
});

group("api errors", () => {
  function failingClient(status: number, type: string) {
    const fetch = async () =>
      new Response(JSON.stringify({ type: "error", error: { type, message: "x" } }), {
        status,
        headers: { "content-type": "application/json" },
      });
    return new Anthropic({ apiKey: "test-key", fetch, maxRetries: 0 });
  }

  it("fails the row on a rate limit", async () => {
    await expect(claudeJudge(makeConfig("R"), failingClient(429, "rate_limit_error"))(ROW)).rejects.toBeInstanceOf(JudgeError);
  });

  it("stops the run on a bad key", async () => {
    await expect(claudeJudge(makeConfig("R"), failingClient(401, "authentication_error"))(ROW)).rejects.toBeInstanceOf(Anthropic.AuthenticationError);
  });
});
