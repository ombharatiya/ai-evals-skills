# ai-evals-skills

Teach your coding agent to build evals the way [aievals.co](https://www.aievals.co)
teaches them: read real traces before writing any metric, use code checks before LLM
judges, write one binary PASS/FAIL judge per failure mode, calibrate every judge against
human labels, and put an interval on every number.

This repo has three parts:

| | What it is | For |
|---|---|---|
| [`skills/ai-evals`](skills/ai-evals/SKILL.md) | A skill: ground rules, five workflows and six reference files | Claude Code, Codex, and other agents that read Agent Skills |
| [`rules/`](rules/) | The same ground rules, condensed | Cursor (`.mdc`) and any agent that reads `AGENTS.md` |
| [`starters/`](starters/) | Working Python and TypeScript harnesses: judge, calibration, release gate | Copying into your repo |

## Install

### Claude Code

Run these two commands in Claude Code, one at a time. Pasted together, they arrive as one
command and the second doesn't run.

```text
/plugin marketplace add ombharatiya/ai-evals-skills
```

```text
/plugin install ai-evals@ai-evals-skills
```

Claude uses the skill on its own when you ask for evals, a judge prompt, a labeling plan
or an eval step in CI. You can also call it directly with `/ai-evals:ai-evals`.

To install without the plugin system, copy the folder:

```bash
git clone https://github.com/ombharatiya/ai-evals-skills.git
mkdir -p ~/.claude/skills && cp -r ai-evals-skills/skills/ai-evals ~/.claude/skills/
```

Use `.claude/skills/` inside a project instead of `~/.claude/skills/` to share it with
your team through git.

### Codex

The same skill folder works in Codex. Ask Codex's built-in installer:

```text
$skill-installer install https://github.com/ombharatiya/ai-evals-skills/tree/main/skills/ai-evals
```

Or copy it yourself, then restart Codex:

```bash
git clone https://github.com/ombharatiya/ai-evals-skills.git
mkdir -p ~/.agents/skills && cp -r ai-evals-skills/skills/ai-evals ~/.agents/skills/
```

Use `.agents/skills/` at your repo root instead to share it with your team. Codex picks the
skill up on its own when a task matches, or you can call it with `$ai-evals` or from
`/skills`. The folder's [`agents/openai.yaml`](skills/ai-evals/agents/openai.yaml) gives
it a name and description in Codex's skill picker.

### Cursor

Copy [`rules/cursor/ai-evals.mdc`](rules/cursor/ai-evals.mdc) into your project's
`.cursor/rules/` folder.

### Other agents that read AGENTS.md

Paste [`rules/agents/AGENTS.md`](rules/agents/AGENTS.md) into your repo's `AGENTS.md`.
Codex reads it too, if you'd rather have always-on rules than a skill.

## What the skill makes your agent do

- Ask for real traces, and have a person read them, before proposing metrics.
- Refuse to invent ground-truth labels. It drafts labels marked provisional for a person
  to confirm.
- Prefer code checks, and build an LLM judge only for subjective failures that remain.
- Write judges as binary PASS/FAIL with a critique, in the guide's four-part prompt.
- Calibrate judges on a dev split, read the test split once, and report PASS recall and
  FAIL recall with N instead of raw agreement.
- Pin the judge's model, effort, rubric and few-shots, and calibrate again when they
  change.
- Report every rate with a Wilson interval and compare versions as paired data.
- Gate releases on the interval of the paired difference (ship, hold, investigate, block),
  with a regression set that must pass 100%.

The six reference files in [`skills/ai-evals/references`](skills/ai-evals/references) tie
each rule to the guide page that explains it and to its primary sources.

## Your first session

Install the skill, put 50 or more real traces from your feature in the repo as JSON Lines,
and ask your agent:

> We need evals for the refund answers in our support bot. The traces are in
> `evals/data/traces.jsonl`. Start with error analysis.

A good answer builds a viewer for you to read traces in, and holds off on metrics until
you've written notes on at least 30 of them. From there the agent clusters your notes into
failure modes, adds code checks, writes one binary judge, and calibrates it against your
labels with a starter.

The full walkthrough, with the prompts to use and what to push back on, is on aievals.co:

- [How to use the agent pack](https://www.aievals.co/agent-pack)
- [Add evals to a feature with your coding agent](https://www.aievals.co/cookbook/evals-with-a-coding-agent)

## Starters

Both starters run the same loop on an example dataset (a support bot checked for getting
the refund policy wrong). Their tests run without an API key.

Python 3.10+:

```bash
cd starters/python
pip install -e ".[dev]"
pytest
```

Node 22.12+:

```bash
cd starters/typescript
npm install
npm test
```

Then follow [`starters/python/README.md`](starters/python/README.md) or
[`starters/typescript/README.md`](starters/typescript/README.md) to calibrate the judge
and gate a release.

## Learn the method

The skill is a compressed version of the guide. Start here:

- [Why evals matter](https://www.aievals.co/learn/foundations/why-evals-matter)
- [The 60-80 rule of error analysis](https://www.aievals.co/learn/error-analysis/the-60-80-rule)
- [LLM-as-judge](https://www.aievals.co/learn/llm-as-judge)
- [Calibrating a judge](https://www.aievals.co/cookbook/llm-judge-calibration)
- [Release gates](https://www.aievals.co/learn/production/release-gates)

## License

The skill and rules text (`skills/`, `rules/`) is licensed under
[CC BY 4.0](LICENSE-CONTENT), like the aievals.co guide it summarizes. The code
(`starters/`) is licensed under the [MIT License](LICENSE).

Issues and pull requests are welcome. If a rule here disagrees with the guide, the guide
wins: open an issue and link the page.
