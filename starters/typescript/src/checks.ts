/**
 * Code-graded checks: the cheap, deterministic half of an eval. Run them
 * before the judge. A failed code check fails the row without a model call.
 * Each check reads optional expectation fields from the row.
 */
import type { Row } from "./data.js";

export interface CheckResult {
  name: string;
  passed: boolean;
  detail: string;
}

type Check = (row: Row) => CheckResult | null;

const notEmpty: Check = (row) => {
  const ok = row.response.trim().length > 0;
  return { name: "not_empty", passed: ok, detail: ok ? "" : "response is empty" };
};

const mustInclude: Check = (row) => {
  if (!row.must_include?.length) return null;
  const text = row.response.toLowerCase();
  const missing = row.must_include.filter((s) => !text.includes(s.toLowerCase()));
  return { name: "must_include", passed: missing.length === 0, detail: missing.length ? `missing: ${missing.join(", ")}` : "" };
};

const mustNotInclude: Check = (row) => {
  if (!row.must_not_include?.length) return null;
  const text = row.response.toLowerCase();
  const found = row.must_not_include.filter((s) => text.includes(s.toLowerCase()));
  return { name: "must_not_include", passed: found.length === 0, detail: found.length ? `found: ${found.join(", ")}` : "" };
};

const maxWords: Check = (row) => {
  if (row.max_words === undefined) return null;
  const count = row.response.split(/\s+/).filter(Boolean).length;
  const ok = count <= row.max_words;
  return { name: "max_words", passed: ok, detail: ok ? "" : `${count} words, limit ${row.max_words}` };
};

export const CHECKS: Check[] = [notEmpty, mustInclude, mustNotInclude, maxWords];

export function runChecks(row: Row): CheckResult[] {
  return CHECKS.map((check) => check(row)).filter((r): r is CheckResult => r !== null);
}
