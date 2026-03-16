from __future__ import annotations

from dataclasses import dataclass


HEADERS = ("VERDICT", "REASONING", "PROOF", "COUNTEREXAMPLE")


@dataclass(frozen=True)
class ParsedResponse:
    raw_text: str
    verdict: str | None
    reasoning: str
    proof: str
    counterexample: str
    valid: bool
    errors: list[str]


def parse_response(text: str) -> ParsedResponse:
    values: dict[str, str] = {key: "" for key in HEADERS}
    current_header: str | None = None

    for line in text.splitlines():
        line_for_match = line.lstrip()
        matched_header = None
        for header in HEADERS:
            prefix = f"{header}:"
            if line_for_match.startswith(prefix):
                matched_header = header
                values[header] = line_for_match[len(prefix) :].strip()
                current_header = header
                break

        if matched_header is None and current_header is not None:
            if values[current_header]:
                values[current_header] += "\n" + line
            else:
                values[current_header] = line

    errors: list[str] = []
    verdict = values["VERDICT"].strip().upper()

    if verdict not in {"TRUE", "FALSE"}:
        errors.append("VERDICT must be exactly TRUE or FALSE.")
        verdict_out: str | None = None
    else:
        verdict_out = verdict

    reasoning = values["REASONING"].strip()
    proof = values["PROOF"].strip()
    counterexample = values["COUNTEREXAMPLE"].strip()

    if not reasoning:
        errors.append("REASONING must be non-empty.")

    if verdict_out == "TRUE":
        if not proof:
            errors.append("PROOF must be non-empty when VERDICT is TRUE.")
        if counterexample:
            errors.append("COUNTEREXAMPLE must be empty when VERDICT is TRUE.")
    elif verdict_out == "FALSE":
        if proof:
            errors.append("PROOF must be empty when VERDICT is FALSE.")
        if not counterexample:
            errors.append("COUNTEREXAMPLE must be non-empty when VERDICT is FALSE.")

    return ParsedResponse(
        raw_text=text,
        verdict=verdict_out,
        reasoning=reasoning,
        proof=proof,
        counterexample=counterexample,
        valid=len(errors) == 0,
        errors=errors,
    )
