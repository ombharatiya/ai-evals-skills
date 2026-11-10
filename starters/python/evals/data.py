"""Load traces and split a labeled set.

A trace file is JSON Lines, one object per line:

    {"id": "t-001", "query": "...", "response": "...", "label": "PASS"}

`label` is the human verdict (PASS or FAIL) and is required for calibration.
For scoring new outputs it is optional. Optional code-check fields:
`must_include`, `must_not_include` (lists of strings) and `max_words` (int).
A row with `"must_pass": true` belongs to the regression set and must pass
for the release gate to pass.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

LABELS = ("PASS", "FAIL")

Row = dict[str, Any]


def load_jsonl(path: str | Path, require_label: bool = False) -> list[Row]:
    rows: list[Row] = []
    seen: set[str] = set()
    for lineno, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{lineno}: not valid JSON ({exc.msg})") from exc
        for field in ("id", "query", "response"):
            if not isinstance(row.get(field), str):
                raise ValueError(f"{path}:{lineno}: '{field}' must be a string")
        if row["id"] in seen:
            raise ValueError(f"{path}:{lineno}: duplicate id {row['id']!r}")
        seen.add(row["id"])
        label = row.get("label")
        if label is not None:
            label = str(label).upper()
            if label not in LABELS:
                raise ValueError(f"{path}:{lineno}: label must be PASS or FAIL, got {row['label']!r}")
            row["label"] = label
        elif require_label:
            raise ValueError(f"{path}:{lineno}: 'label' is required here")
        rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def write_jsonl(path: str | Path, rows: list[Row]) -> None:
    Path(path).write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )


def stratified_split(
    rows: list[Row], seed: int = 42, train: float = 0.15, dev: float = 0.40
) -> tuple[list[Row], list[Row], list[Row]]:
    """Split labeled rows into train / dev / test, keeping the PASS:FAIL ratio.

    Train holds few-shot examples for the judge prompt, dev is where you
    iterate on the rubric, and test is read once at the end. The default
    15 / 40 / 45 split follows the calibration recipe on aievals.co. The same
    seed always gives the same split, so a row never moves between dev and
    test across iterations.
    """
    if not 0 < train < 1 or not 0 < dev < 1 or train + dev >= 1:
        raise ValueError("train and dev must be fractions with train + dev < 1")
    rng = random.Random(seed)
    out: tuple[list[Row], list[Row], list[Row]] = ([], [], [])
    for label in LABELS:
        group = sorted((r for r in rows if r.get("label") == label), key=lambda r: r["id"])
        rng.shuffle(group)
        n_train = round(len(group) * train)
        n_dev = round(len(group) * dev)
        out[0].extend(group[:n_train])
        out[1].extend(group[n_train : n_train + n_dev])
        out[2].extend(group[n_train + n_dev :])
    return out
