from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class EvaluatorConfig:
    api_key: str
    model: str
    api_base_url: str
    timeout_seconds: int
    temperature: float
    max_output_tokens: int
    retry_count: int
    retry_backoff_seconds: float
    max_cheatsheet_bytes: int
    normal_path: Path
    hard_path: Path
    output_root: Path
    dry_run: bool
    input_cost_per_1m_tokens: float
    output_cost_per_1m_tokens: float


def _to_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue

        key, value = raw.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key and key not in os.environ:
            os.environ[key] = value


def load_config(root_dir: Path) -> EvaluatorConfig:
    _load_env_file(root_dir / ".env")

    api_key = os.getenv("CHATGPT_KEY", "").strip()
    model = os.getenv("EVAL_MODEL", "gpt-4.1-mini").strip()

    return EvaluatorConfig(
        api_key=api_key,
        model=model,
        api_base_url=os.getenv("EVAL_API_BASE_URL", "https://api.openai.com/v1/responses").strip(),
        timeout_seconds=int(os.getenv("EVAL_TIMEOUT_SECONDS", "120")),
        temperature=float(os.getenv("EVAL_TEMPERATURE", "0")),
        max_output_tokens=int(os.getenv("EVAL_MAX_OUTPUT_TOKENS", "600")),
        retry_count=int(os.getenv("EVAL_RETRY_COUNT", "2")),
        retry_backoff_seconds=float(os.getenv("EVAL_RETRY_BACKOFF_SECONDS", "1.5")),
        max_cheatsheet_bytes=int(os.getenv("EVAL_MAX_CHEATSHEET_BYTES", str(10 * 1024))),
        normal_path=Path(os.getenv("EVAL_NORMAL_PATH", str(root_dir / "data" / "stage1" / "normal.jsonl"))),
        hard_path=Path(os.getenv("EVAL_HARD_PATH", str(root_dir / "data" / "stage1" / "hard.jsonl"))),
        output_root=Path(os.getenv("EVAL_OUTPUT_ROOT", str(root_dir / "outputs" / "eval_runs"))),
        dry_run=_to_bool(os.getenv("EVAL_DRY_RUN"), default=False),
        input_cost_per_1m_tokens=float(os.getenv("EVAL_PRICE_INPUT_PER_1M", "0")),
        output_cost_per_1m_tokens=float(os.getenv("EVAL_PRICE_OUTPUT_PER_1M", "0")),
    )


def validate_runtime_requirements(config: EvaluatorConfig) -> None:
    if not config.dry_run and not config.api_key:
        raise RuntimeError("Missing CHATGPT_KEY. Set it in environment or run with --dry-run.")
