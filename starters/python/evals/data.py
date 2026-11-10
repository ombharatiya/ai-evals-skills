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

import hashlib
import json
import math
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
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{lineno}: each line must be a JSON object")
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


def json_safe(value: Any) -> Any:
    """NaN and infinity aren't valid JSON. Write them as null."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def dumps(value: Any, **kwargs: Any) -> str:
    return json.dumps(json_safe(value), ensure_ascii=False, allow_nan=False, **kwargs)


def write_jsonl(path: str | Path, rows: list[Row]) -> None:
    Path(path).write_text("".join(dumps(r) + "\n" for r in rows), encoding="utf-8")


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def split_unit(row_id: str, seed: int) -> float:
    """A number in [0, 1) that depends only on the id and the seed."""
    digest = hashlib.sha256(f"{seed}:{row_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def split_rows(
    rows: list[Row], seed: int = 42, train: float = 0.15, dev: float = 0.40
) -> tuple[list[Row], list[Row], list[Row]]:
    """Split labeled rows into train / dev / test by a stable hash of each id.

    Train holds few-shot examples for the judge prompt, dev is where you
    iterate on the rubric, and test is read once at the end. The default
    15 / 40 / 45 fractions follow the calibration recipe on aievals.co.

    A row's split depends only on its id and the seed, so labeling more rows
    never moves an existing row from test into train or dev, and the PASS:FAIL
    ratio of each split matches the whole set in expectation. Split sizes are
    approximate on small sets; the calibration run prints the per-class
    counts, and the guide's target is 30 to 50 of each class in dev and test.
    """
    if not 0 < train < 1 or not 0 < dev < 1 or train + dev >= 1:
        raise ValueError("train and dev must be fractions with train + dev < 1")
    out: tuple[list[Row], list[Row], list[Row]] = ([], [], [])
    for row in sorted(rows, key=lambda r: r["id"]):
        u = split_unit(row["id"], seed)
        out[0 if u < train else 1 if u < train + dev else 2].append(row)
    return out


def class_counts(rows: list[Row]) -> dict[str, int]:
    return {label: sum(1 for r in rows if r.get("label") == label) for label in LABELS}
