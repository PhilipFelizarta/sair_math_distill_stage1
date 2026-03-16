from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvalMetrics:
    normal_total: int
    hard_total: int
    normal_correct: int
    hard_correct: int
    valid_format_count: int
    total_count: int
    avg_latency_ms: float
    total_input_tokens: int
    total_output_tokens: int
    estimated_total_cost_usd: float

    @property
    def overall_accuracy(self) -> float:
        if self.total_count == 0:
            return 0.0
        return (self.normal_correct + self.hard_correct) / self.total_count

    @property
    def normal_accuracy(self) -> float:
        if self.normal_total == 0:
            return 0.0
        return self.normal_correct / self.normal_total

    @property
    def hard_accuracy(self) -> float:
        if self.hard_total == 0:
            return 0.0
        return self.hard_correct / self.hard_total

    @property
    def format_validity_rate(self) -> float:
        if self.total_count == 0:
            return 0.0
        return self.valid_format_count / self.total_count
