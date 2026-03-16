from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, UTC
import json
from pathlib import Path
import sys
from typing import Any
from urllib import request

from .config import load_config, validate_runtime_requirements
from .data import Problem, load_stage1
from .inference import run_dry_inference, run_inference
from .metrics import EvalMetrics
from .parser import parse_response
from .prompt import build_prompt, clip_cheatsheet
from .reward import compute_reward


def _estimate_cost_usd(
    input_tokens: int,
    output_tokens: int,
    input_cost_per_1m: float,
    output_cost_per_1m: float,
) -> float:
    in_cost = (input_tokens / 1_000_000.0) * input_cost_per_1m
    out_cost = (output_tokens / 1_000_000.0) * output_cost_per_1m
    return in_cost + out_cost


def _evaluate_subset(
    problems: list[Problem],
    cheatsheet: str,
    use_dry_run: bool,
    config: Any,
    max_items: int | None = None,
) -> tuple[int, list[dict[str, Any]], int, float, int, int, float]:
    correct = 0
    valid_count = 0
    per_problem: list[dict[str, Any]] = []

    total_latency = 0.0
    total_input_tokens = 0
    total_output_tokens = 0
    total_cost = 0.0

    iterable = problems if max_items is None else problems[:max_items]

    for item in iterable:
        prompt = build_prompt(item.equation1, item.equation2, cheatsheet)
        if use_dry_run:
            inf = run_dry_inference(item.equation1, item.equation2)
        else:
            inf = run_inference(prompt, config)

        parsed = parse_response(inf.output_text)
        is_correct = parsed.verdict is not None and ((parsed.verdict == "TRUE") == item.label_true)

        if is_correct:
            correct += 1
        if parsed.valid:
            valid_count += 1

        total_latency += inf.latency_ms
        total_input_tokens += inf.input_tokens
        total_output_tokens += inf.output_tokens
        total_cost += _estimate_cost_usd(
            inf.input_tokens,
            inf.output_tokens,
            config.input_cost_per_1m_tokens,
            config.output_cost_per_1m_tokens,
        )

        per_problem.append(
            {
                "problem_id": item.problem_id,
                "difficulty": item.difficulty,
                "label_true": item.label_true,
                "predicted_verdict": parsed.verdict,
                "is_correct": is_correct,
                "format_valid": parsed.valid,
                "parse_errors": parsed.errors,
                "latency_ms": inf.latency_ms,
                "input_tokens": inf.input_tokens,
                "output_tokens": inf.output_tokens,
                "estimated_cost_usd": _estimate_cost_usd(
                    inf.input_tokens,
                    inf.output_tokens,
                    config.input_cost_per_1m_tokens,
                    config.output_cost_per_1m_tokens,
                ),
                "model": inf.model,
                "inference_error": inf.error,
            }
        )

    return (
        correct,
        per_problem,
        valid_count,
        total_latency,
        total_input_tokens,
        total_output_tokens,
        total_cost,
    )


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _command_download_stage1(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    normal_dest = out_dir / "normal.jsonl"
    hard_dest = out_dir / "hard.jsonl"

    request.urlretrieve(args.normal_url, normal_dest)  # noqa: S310
    request.urlretrieve(args.hard_url, hard_dest)  # noqa: S310

    print(f"Downloaded normal set to: {normal_dest}")
    print(f"Downloaded hard set to: {hard_dest}")
    return 0


def _command_evaluate_cheatsheet(args: argparse.Namespace) -> int:
    root_dir = Path(__file__).resolve().parents[2]
    config = load_config(root_dir)

    if args.model:
        config = config.__class__(**{**asdict(config), "model": args.model})
    if args.normal_path:
        config = config.__class__(**{**asdict(config), "normal_path": Path(args.normal_path)})
    if args.hard_path:
        config = config.__class__(**{**asdict(config), "hard_path": Path(args.hard_path)})
    if args.dry_run:
        config = config.__class__(**{**asdict(config), "dry_run": True})

    validate_runtime_requirements(config)

    cheatsheet_path = Path(args.file)
    cheatsheet_raw = cheatsheet_path.read_text(encoding="utf-8")
    cheatsheet, clipped, original_bytes, used_bytes = clip_cheatsheet(
        cheatsheet_raw,
        max_bytes=config.max_cheatsheet_bytes,
    )

    normal, hard = load_stage1(config.normal_path, config.hard_path)

    normal_correct, normal_rows, normal_valid, normal_latency, normal_in, normal_out, normal_cost = _evaluate_subset(
        normal,
        cheatsheet,
        config.dry_run,
        config,
        max_items=args.limit_normal,
    )
    hard_correct, hard_rows, hard_valid, hard_latency, hard_in, hard_out, hard_cost = _evaluate_subset(
        hard,
        cheatsheet,
        config.dry_run,
        config,
        max_items=args.limit_hard,
    )

    used_normal_total = len(normal_rows)
    used_hard_total = len(hard_rows)
    total_count = used_normal_total + used_hard_total

    metrics = EvalMetrics(
        normal_total=used_normal_total,
        hard_total=used_hard_total,
        normal_correct=normal_correct,
        hard_correct=hard_correct,
        valid_format_count=normal_valid + hard_valid,
        total_count=total_count,
        avg_latency_ms=((normal_latency + hard_latency) / total_count) if total_count else 0.0,
        total_input_tokens=normal_in + hard_in,
        total_output_tokens=normal_out + hard_out,
        estimated_total_cost_usd=normal_cost + hard_cost,
    )

    reward = compute_reward(metrics.normal_correct, metrics.hard_correct)

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(args.output_dir) if args.output_dir else (config.output_root / timestamp)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "reward": reward,
        "normal_correct": metrics.normal_correct,
        "hard_correct": metrics.hard_correct,
        "normal_total": metrics.normal_total,
        "hard_total": metrics.hard_total,
        "overall_accuracy": metrics.overall_accuracy,
        "normal_accuracy": metrics.normal_accuracy,
        "hard_accuracy": metrics.hard_accuracy,
        "format_validity_rate": metrics.format_validity_rate,
        "avg_latency_ms": metrics.avg_latency_ms,
        "total_input_tokens": metrics.total_input_tokens,
        "total_output_tokens": metrics.total_output_tokens,
        "estimated_total_cost_usd": metrics.estimated_total_cost_usd,
        "model": config.model if not config.dry_run else "dry-run",
        "dry_run": config.dry_run,
        "cheatsheet_original_bytes": original_bytes,
        "cheatsheet_used_bytes": used_bytes,
        "cheatsheet_clipped": clipped,
        "reward_formula": "reward = normal_correct + 10 * hard_correct",
    }

    _write_json(output_dir / "summary.json", summary)
    _write_json(output_dir / "reward_breakdown.json", {
        "reward": reward,
        "formula": "normal_correct + 10 * hard_correct",
        "normal_term": metrics.normal_correct,
        "hard_term": 10 * metrics.hard_correct,
    })
    _write_jsonl(output_dir / "per_problem.jsonl", normal_rows + hard_rows)
    _write_jsonl(
        output_dir / "errors.jsonl",
        [row for row in (normal_rows + hard_rows) if row["inference_error"] or not row["format_valid"]],
    )

    print(json.dumps(summary, indent=2))
    print(f"Saved evaluation artifacts to: {output_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SAIR Stage 1 cheatsheet evaluator")
    sub = parser.add_subparsers(dest="command", required=True)

    download = sub.add_parser("download-stage1", help="Download normal/hard JSONL datasets")
    download.add_argument("--normal-url", required=True, help="Direct URL to normal.jsonl")
    download.add_argument("--hard-url", required=True, help="Direct URL to hard.jsonl")
    download.add_argument("--out-dir", default="data/stage1", help="Output directory for downloaded files")

    evaluate = sub.add_parser("evaluate-cheatsheet", help="Evaluate a cheatsheet against Stage 1 datasets")
    evaluate.add_argument("--file", required=True, help="Path to cheatsheet text file")
    evaluate.add_argument("--normal-path", default=None, help="Override path to normal.jsonl")
    evaluate.add_argument("--hard-path", default=None, help="Override path to hard.jsonl")
    evaluate.add_argument("--limit-normal", type=int, default=None, help="Evaluate only first N normal problems")
    evaluate.add_argument("--limit-hard", type=int, default=None, help="Evaluate only first N hard problems")
    evaluate.add_argument("--model", default=None, help="Override model name")
    evaluate.add_argument("--dry-run", action="store_true", help="Run without API calls")
    evaluate.add_argument("--output-dir", default=None, help="Override output directory")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "download-stage1":
        return _command_download_stage1(args)
    if args.command == "evaluate-cheatsheet":
        return _command_evaluate_cheatsheet(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
