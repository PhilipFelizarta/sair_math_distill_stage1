from __future__ import annotations


def compute_reward(normal_correct: int, hard_correct: int) -> int:
    """Simple SAIR Stage 1 reward.

    Reward = 1 * normal_correct + 10 * hard_correct.
    """
    return int(normal_correct) + 10 * int(hard_correct)
