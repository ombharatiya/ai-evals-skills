/** Run a judge over many rows with a fixed number of workers and a hard budget cap. */
import type { Row } from "./data.js";
import { JudgeError, type Judge, type Verdict } from "./judge.js";

/**
 * Stops new judge calls once the estimated spend reaches the limit. The cap
 * is enforced in the runner, not printed as a warning. Set EVAL_BUDGET_USD.
 * Calls in flight finish, so spend can overshoot by one call per worker.
 */
export class Budget {
  spentUsd = 0;
  constructor(readonly limitUsd: number | null = null) {}

  static fromEnv(): Budget {
    const raw = process.env.EVAL_BUDGET_USD;
    return new Budget(raw ? Number(raw) : null);
  }

  get exceeded(): boolean {
    return this.limitUsd !== null && this.spentUsd >= this.limitUsd;
  }
}

export interface Outcome {
  row: Row;
  verdict?: Verdict;
  error?: string;
}

export async function judgeRows(judge: Judge, rows: Row[], workers = 4, budget = new Budget()): Promise<Outcome[]> {
  const outcomes: Outcome[] = new Array<Outcome>(rows.length);
  let next = 0;
  const worker = async (): Promise<void> => {
    while (next < rows.length) {
      const i = next++;
      const row = rows[i]!;
      if (budget.exceeded) {
        outcomes[i] = { row, error: "not run: budget reached (spent " + budget.spentUsd.toFixed(2) + " USD)" };
        continue;
      }
      try {
        const verdict = await judge(row);
        budget.spentUsd += verdict.costUsd;
        outcomes[i] = { row, verdict };
      } catch (err) {
        if (!(err instanceof JudgeError)) throw err;
        outcomes[i] = { row, error: err.message };
      }
    }
  };
  await Promise.all(Array.from({ length: Math.max(1, workers) }, () => worker()));
  return outcomes;
}
