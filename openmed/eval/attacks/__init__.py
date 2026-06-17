"""Adversarial evaluation attacks for benchmark reports."""

from .reid import (
    ReidAttackResult,
    generate_reid_leaderboard,
    render_reid_leaderboard,
    run_reid_attack,
    run_reid_benchmark,
)

__all__ = [
    "ReidAttackResult",
    "generate_reid_leaderboard",
    "render_reid_leaderboard",
    "run_reid_attack",
    "run_reid_benchmark",
]
