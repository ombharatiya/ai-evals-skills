/** Small helpers shared by the two command-line entry points. */
import { realpathSync } from "node:fs";
import { fileURLToPath } from "node:url";

/** True when this module is the script Node was started with, even through a symlink. */
export function isMain(importMetaUrl: string): boolean {
  const entry = process.argv[1];
  if (!entry) return false;
  try {
    return realpathSync(entry) === realpathSync(fileURLToPath(importMetaUrl));
  } catch {
    return false;
  }
}

/** Parse a numeric flag, or throw a message naming the flag. */
export function numberFlag(name: string, raw: string | undefined, { integer = false, min = -Infinity } = {}): number {
  const value = Number(raw);
  const badInteger = integer && (!/^-?\d+$/.test((raw ?? "").trim()) || !Number.isSafeInteger(value));
  if (raw === undefined || raw.trim() === "" || !Number.isFinite(value) || value < min || badInteger) {
    throw new Error("--" + name + " must be " + (integer ? "an integer" : "a number") + (min > -Infinity ? " of at least " + min : "") + ", got " + String(raw));
  }
  return value;
}
