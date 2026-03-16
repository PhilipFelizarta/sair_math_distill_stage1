from __future__ import annotations

REFERENCE_PROMPT_TEMPLATE = """You are a mathematician specializing in equational theories of magmas.
Your task is to determine whether Equation 1 ({equation1}) implies Equation 2 ({equation2}) over all magmas.
{cheatsheet_block}
Output format (use exact headers without any additional text or formatting):
VERDICT: must be exactly TRUE or FALSE (in the same line).
REASONING: must be non-empty.
PROOF: required if VERDICT is TRUE, empty otherwise.
COUNTEREXAMPLE: required if VERDICT is FALSE, empty otherwise.
"""


def clip_cheatsheet(text: str, max_bytes: int = 10 * 1024) -> tuple[str, bool, int, int]:
    raw = text.encode("utf-8")
    raw_len = len(raw)
    if raw_len <= max_bytes:
        return text, False, raw_len, raw_len

    clipped = raw[:max_bytes]
    # Ignore partial multibyte suffix if clipping lands mid-codepoint.
    clipped_text = clipped.decode("utf-8", errors="ignore")
    return clipped_text, True, raw_len, max_bytes


def build_prompt(equation1: str, equation2: str, cheatsheet: str) -> str:
    cheatsheet_block = cheatsheet.strip() if cheatsheet.strip() else ""
    return REFERENCE_PROMPT_TEMPLATE.format(
        equation1=equation1,
        equation2=equation2,
        cheatsheet_block=cheatsheet_block,
    )
