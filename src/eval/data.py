from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Problem:
    problem_id: str
    equation1: str
    equation2: str
    label_true: bool
    difficulty: str  # "normal" or "hard"


def _pick_first(row: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in row:
            return row[key]
    return None


def _parse_label(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().upper()
        if normalized in {"TRUE", "T", "1", "YES"}:
            return True
        if normalized in {"FALSE", "F", "0", "NO"}:
            return False
    raise ValueError(f"Unsupported label value: {value!r}")


def _row_to_problem(row: dict[str, Any], idx: int, difficulty: str) -> Problem:
    equation1 = _pick_first(row, ["equation1", "eq1", "premise", "lhs", "from", "law1"])
    equation2 = _pick_first(row, ["equation2", "eq2", "conclusion", "rhs", "to", "law2"])
    label_raw = _pick_first(
        row,
        ["label", "verdict", "answer", "implies", "truth", "is_true", "is_implication_true"],
    )
    row_id = _pick_first(row, ["id", "problem_id", "uid", "index"]) or f"{difficulty}-{idx}"

    if equation1 is None or equation2 is None:
        raise ValueError(f"Missing equation fields in row #{idx}: keys={sorted(row.keys())}")

    return Problem(
        problem_id=str(row_id),
        equation1=str(equation1),
        equation2=str(equation2),
        label_true=_parse_label(label_raw),
        difficulty=difficulty,
    )


def load_jsonl(path: Path, difficulty: str) -> list[Problem]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    problems: list[Problem] = []
    # Use utf-8-sig so files created by some Windows tools (with BOM) still parse.
    with path.open("r", encoding="utf-8-sig") as fh:
        for idx, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if not isinstance(row, dict):
                raise ValueError(f"Line {idx} in {path} is not an object.")
            problems.append(_row_to_problem(row, idx, difficulty))
    return problems


def load_stage1(normal_path: Path, hard_path: Path) -> tuple[list[Problem], list[Problem]]:
    normal = load_jsonl(normal_path, "normal")
    hard = load_jsonl(hard_path, "hard")
    return normal, hard
