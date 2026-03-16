from __future__ import annotations

from dataclasses import dataclass
import json
import time
from typing import Any
from urllib import request, error

from .config import EvaluatorConfig


@dataclass(frozen=True)
class InferenceResult:
    output_text: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    model: str
    error: str | None = None


def _extract_output_text(payload: dict[str, Any]) -> str:
    direct_text = payload.get("output_text")
    if isinstance(direct_text, str) and direct_text.strip():
        return direct_text

    if isinstance(direct_text, list):
        fragments = [item for item in direct_text if isinstance(item, str)]
        if fragments:
            return "\n".join(fragments)

    output = payload.get("output", [])
    if isinstance(output, list):
        texts: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content", [])
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                text_value = block.get("text")
                if isinstance(text_value, str):
                    texts.append(text_value)
        if texts:
            return "\n".join(texts)

    return ""


def _extract_tokens(payload: dict[str, Any]) -> tuple[int, int]:
    usage = payload.get("usage", {})
    if not isinstance(usage, dict):
        return 0, 0

    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)

    try:
        return int(input_tokens), int(output_tokens)
    except (TypeError, ValueError):
        return 0, 0


def run_inference(prompt: str, config: EvaluatorConfig) -> InferenceResult:
    body = {
        "model": config.model,
        "input": prompt,
        "temperature": config.temperature,
        "max_output_tokens": config.max_output_tokens,
    }

    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
    }

    encoded_body = json.dumps(body).encode("utf-8")

    last_error: str | None = None
    for attempt in range(config.retry_count + 1):
        start = time.perf_counter()
        try:
            req = request.Request(
                url=config.api_base_url,
                data=encoded_body,
                headers=headers,
                method="POST",
            )
            with request.urlopen(req, timeout=config.timeout_seconds) as resp:
                payload = json.loads(resp.read().decode("utf-8"))

            latency_ms = (time.perf_counter() - start) * 1000.0
            output_text = _extract_output_text(payload)
            input_tokens, output_tokens = _extract_tokens(payload)

            return InferenceResult(
                output_text=output_text,
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model=config.model,
                error=None,
            )
        except error.HTTPError as ex:
            latency_ms = (time.perf_counter() - start) * 1000.0
            body_text = ex.read().decode("utf-8", errors="ignore")
            last_error = f"HTTP {ex.code}: {body_text}"
            if attempt >= config.retry_count:
                return InferenceResult(
                    output_text="",
                    latency_ms=latency_ms,
                    input_tokens=0,
                    output_tokens=0,
                    model=config.model,
                    error=last_error,
                )
            time.sleep(config.retry_backoff_seconds)
        except Exception as ex:  # noqa: BLE001
            latency_ms = (time.perf_counter() - start) * 1000.0
            last_error = str(ex)
            if attempt >= config.retry_count:
                return InferenceResult(
                    output_text="",
                    latency_ms=latency_ms,
                    input_tokens=0,
                    output_tokens=0,
                    model=config.model,
                    error=last_error,
                )
            time.sleep(config.retry_backoff_seconds)

    return InferenceResult(
        output_text="",
        latency_ms=0.0,
        input_tokens=0,
        output_tokens=0,
        model=config.model,
        error=last_error or "Unknown inference error.",
    )


def run_dry_inference(equation1: str, equation2: str) -> InferenceResult:
    # Cheap deterministic fallback for local pipeline testing.
    verdict = "TRUE" if equation2 in equation1 else "FALSE"
    if verdict == "TRUE":
        text = (
            "VERDICT: TRUE\n"
            "REASONING: Dry-run heuristic judged implication as likely true.\n"
            "PROOF: Placeholder proof for dry-run mode.\n"
            "COUNTEREXAMPLE:"
        )
    else:
        text = (
            "VERDICT: FALSE\n"
            "REASONING: Dry-run heuristic judged implication as likely false.\n"
            "PROOF:\n"
            "COUNTEREXAMPLE: Placeholder counterexample for dry-run mode."
        )

    return InferenceResult(
        output_text=text,
        latency_ms=1.0,
        input_tokens=0,
        output_tokens=0,
        model="dry-run",
        error=None,
    )
