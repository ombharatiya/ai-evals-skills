/**
 * Runs the real run-eval CLI against a fake Messages API on localhost, so the
 * release-gate decisions are checked end to end. No network, no API key.
 */
import { execFile } from "node:child_process";
import { mkdtempSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { createServer, type Server } from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { promisify } from "node:util";
import { afterAll, beforeAll, describe as group, expect, it } from "vitest";
import { describe, makeConfig } from "../src/judge.js";
import { noChangeHalfwidth, rowsForIndifference } from "../src/gate.js";
import { estimateCost } from "../src/judge.js";
import { annotationText, showIds } from "../src/run-eval.js";
import { readText } from "../src/data.js";
import type Anthropic from "@anthropic-ai/sdk";

const run = promisify(execFile);
const root = new URL("..", import.meta.url).pathname;
let server: Server;
let baseUrl = "";

beforeAll(async () => {
  server = createServer((req, res) => {
    req.resume();
    req.on("end", () => {
      res.setHeader("content-type", "application/json");
      res.end(
        JSON.stringify({
          id: "msg_test", type: "message", role: "assistant", model: "claude-opus-5-5",
          content: [{ type: "text", text: JSON.stringify({ label: "PASS", critique: "ok" }) }],
          stop_reason: "end_turn", stop_sequence: null, usage: { input_tokens: 10, output_tokens: 5 },
        }),
      );
    });
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const address = server.address();
  if (address && typeof address === "object") baseUrl = "http://127.0.0.1:" + address.port;
});

afterAll(() => {
  server.close();
});

function writeLines(path: string, rows: unknown[]) {
  writeFileSync(path, rows.map((r) => JSON.stringify(r) + "\n").join(""));
  return path;
}

async function runEval(dir: string, args: string[], env: Record<string, string> = {}) {
  const out = join(dir, "runs");
  let code = 0;
  let stdout = "";
  try {
    ({ stdout } = await run(join(root, "node_modules/.bin/tsx"), [join(root, "src/run-eval.ts"), ...args, "--out", out], {
      env: { ...process.env, GITHUB_ACTIONS: "", ANTHROPIC_API_KEY: "test-key", ANTHROPIC_BASE_URL: baseUrl, EVAL_BUDGET_USD: "", ...env },
    }));
  } catch (err) {
    code = (err as { code?: number }).code ?? 1;
    stdout = (err as { stdout?: string }).stdout ?? "";
  }
  const runDir = readdirSync(out).filter((d) => d.endsWith("-eval")).sort().pop()!;
  return { code, stdout, summary: JSON.parse(readFileSync(join(out, runDir, "summary.json"), "utf8")) };
}

group("run-eval CLI", () => {
  it("is invalid when the candidate drops baseline rows, and blocks when one was a regression row", async () => {
    const dir = mkdtempSync(join(tmpdir(), "cli-"));
    const rubric = join(dir, "rubric.md");
    writeFileSync(rubric, "R");
    const data = writeLines(join(dir, "data.jsonl"), Array.from({ length: 400 }, (_, i) => ({ id: "r" + i, query: "q", response: "fine" })));
    const base = writeLines(join(dir, "base.jsonl"), Array.from({ length: 500 }, (_, i) => ({ id: "r" + i, passed: true })));
    // A broken comparison fails even when CI only fails on investigate.
    let r = await runEval(dir, ["--data", data, "--rubric", rubric, "--baseline", base, "--indifference", "0.2", "--fail-on", "investigate"]);
    expect(r.code).toBe(1);
    expect(r.summary.verdict).toBe("invalid");
    expect(r.summary.vs_baseline.band).toBe("ship"); // the interval alone would have shipped
    expect(r.summary.vs_baseline.missing_ids).toHaveLength(100);

    const base2 = writeLines(join(dir, "base2.jsonl"), Array.from({ length: 500 }, (_, i) => ({ id: "r" + i, passed: true, must_pass: i === 450 })));
    r = await runEval(dir, ["--data", data, "--rubric", rubric, "--baseline", base2, "--indifference", "0.2"]);
    expect(r.code).toBe(1);
    expect(r.summary.verdict).toBe("block");
    expect(r.summary.must_pass_failures).toContain("r450");
  }, 60_000);

  it("refuses a calibration record that didn't pass", async () => {
    const dir = mkdtempSync(join(tmpdir(), "cli-"));
    const rubric = join(dir, "rubric.md");
    writeFileSync(rubric, "R");
    const data = writeLines(join(dir, "data.jsonl"), [{ id: "a", query: "q", response: "r" }]);
    const cal = join(dir, "summary.json");
    writeFileSync(cal, JSON.stringify({ judge: describe(makeConfig("R")), split: "test", report: { pass_recall: 0.9, fail_recall: 0.143 }, bar: { passed: false } }));
    const r = await runEval(dir, ["--data", data, "--rubric", rubric, "--calibration", cal, "--fail-on", "investigate"]);
    expect(r.code).toBe(1);
    expect(r.summary.verdict).toBe("invalid");
    expect(r.summary.problems.some((p: string) => p.includes("passing test read"))).toBe(true);
    expect(r.summary).not.toHaveProperty("pass_rate_corrected");
  }, 60_000);
});

group("parity with the Python starter", () => {
  it("computes the same judge fingerprint", () => {
    const config = makeConfig("R", { fewShots: [{ id: "f1", query: "q", response: "r", label: "FAIL", critique: "naïve x" }] });
    expect(describe(config).prompt_sha256).toBe("7837062c1b9d6ee52eb994de04b29ecc8a25eb5ab6e1e8e902e4cc12fdac4189");
  });

  it("accounts for judge flips in the gate resolution", () => {
    expect(rowsForIndifference(0.76, 0.01, 1, 1_000_000, 0.02)).toBe(856);
    expect(rowsForIndifference(0.76, 0.01, 1, 1_000_000, 0.05)).toBe(1957);
    expect(noChangeHalfwidth(280, 0.76, 0.02)).toBeCloseTo(0.0192, 3);
    // Expected flips aren't rounded to whole rows, so once a size fits, every larger size does.
    expect(noChangeHalfwidth(855, 0.76, 0.02)).toBeGreaterThan(0.01);
    for (let n = 856; n < 1300; n++) expect(noChangeHalfwidth(n, 0.76, 0.02)).toBeLessThanOrEqual(0.01);
  });

  it("fingerprints a rubric saved on Windows the same way", () => {
    const dir = mkdtempSync(join(tmpdir(), "cli-"));
    const path = join(dir, "rubric.md");
    writeFileSync(path, Buffer.from([0xef, 0xbb, 0xbf, 0x52, 0x0d, 0x0a])); // BOM, "R", CRLF
    const rubric = readText(path);
    expect(rubric).toBe("R\n");
    const config = makeConfig(rubric, { fewShots: [{ id: "f1", query: "q", response: "r", label: "FAIL", critique: "naïve x" }] });
    expect(describe(config).prompt_sha256).toBe("7837062c1b9d6ee52eb994de04b29ecc8a25eb5ab6e1e8e902e4cc12fdac4189");
    expect(describe(config).rubric_sha256).toBe("13b30eab8afdfab285103004686b2e6140ce53aa45a4f917341c740dfdb6e956");
  });

  it("prices a response without a breakdown at the model that served it", () => {
    const usage = { input_tokens: 1000, output_tokens: 400 } as Anthropic.Beta.Messages.BetaUsage;
    expect(estimateCost("claude-opus-5-5", usage, "claude-opus-4-8")).toBeCloseTo((1000 * 5 + 400 * 25) / 1e6, 12);
  });
});

group("run-eval --fail-on", () => {
  it("reports a hold without failing when CI asks only to fail on investigate", async () => {
    // 40 unchanged rows: a clean comparison whose interval is too wide to ship on.
    const dir = mkdtempSync(join(tmpdir(), "cli-"));
    const rubric = join(dir, "rubric.md");
    writeFileSync(rubric, "R");
    const data = writeLines(join(dir, "data.jsonl"), Array.from({ length: 40 }, (_, i) => ({ id: "r" + i, query: "q", response: "fine" })));
    const base = writeLines(join(dir, "base.jsonl"), Array.from({ length: 40 }, (_, i) => ({ id: "r" + i, passed: true })));
    const strict = await runEval(dir, ["--data", data, "--rubric", rubric, "--baseline", base]);
    expect(strict.summary.verdict).toBe("hold");
    expect(strict.summary.problems).toEqual([]);
    expect(strict.code).toBe(1);
    const ci = await runEval(dir, ["--data", data, "--rubric", rubric, "--baseline", base, "--fail-on", "investigate"], { GITHUB_ACTIONS: "true" });
    expect(ci.summary.verdict).toBe("hold");
    expect(ci.code).toBe(0);
    // GitHub decodes %25 back to %.
    expect(ci.stdout.split("\n").filter((l) => l.startsWith("::"))).toEqual([
      "::warning title=Release gate HOLD::delta +0.000, 95%25 CI [-0.088, +0.088], n=40; read the rows in results.jsonl (the eval-run artifact in CI)",
    ]);
  }, 60_000);

  it("escapes annotation text that comes from the data", () => {
    expect(annotationText("id 100%\nnext")).toBe("id 100%25%0Anext");
    expect(showIds(["a\n::notice::x", "b"])).toBe("a\\n::notice::x, b");
    expect(showIds(["a", "b", "c"], 2)).toBe("a, b, ...");
  });

  it("blocks when a regression row with no baseline verdict is dropped, and keeps ids out of workflow commands", async () => {
    const dir = mkdtempSync(join(tmpdir(), "cli-"));
    const rubric = join(dir, "rubric.md");
    writeFileSync(rubric, "R");
    const data = writeLines(join(dir, "data.jsonl"), Array.from({ length: 40 }, (_, i) => ({ id: "r" + i, query: "q", response: "fine" })));
    const rows = Array.from({ length: 40 }, (_, i) => ({ id: "r" + i, passed: true }) as Record<string, unknown>);
    const base = writeLines(join(dir, "base.jsonl"), [...rows, { id: "rx", passed: null, must_pass: true }, { id: "x\n::notice::injected", passed: true }]);
    const r = await runEval(dir, ["--data", data, "--rubric", rubric, "--baseline", base, "--indifference", "0.2"], { GITHUB_ACTIONS: "true" });
    expect(r.code).toBe(1);
    expect(r.summary.verdict).toBe("block");
    expect(r.summary.must_pass_failures).toEqual(["rx"]);
    const commands = r.stdout.split("\n").filter((l) => l.startsWith("::"));
    expect(commands).toHaveLength(1);
    expect(commands[0]).toMatch(/^::error title=Release gate BLOCK::/);
  }, 60_000);
});
