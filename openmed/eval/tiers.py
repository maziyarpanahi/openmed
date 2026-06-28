"""Canonical device-tier SLO budgets for evaluation gates."""

from __future__ import annotations

from typing import Final

NANO_SUB_TIER: Final[dict[str, int | str]] = {
    "parent_tier": "Tiny",
    "param_count_min": 10_000_000,
    "param_count_max": 30_000_000,
    "ram_mb_max": 150,
    "p50_ms_max": 25,
    "p95_ms_max": 60,
    "default_format": "INT8",
}

TIERS: Final[dict[str, dict[str, int | str | dict[str, dict[str, int | str]]]]] = {
    "Tiny": {
        "ram_mb_max": 350,
        "p50_ms_max": 60,
        "p95_ms_max": 150,
        "default_format": "INT8 (MLX-8bit / CoreML)",
        "sub_tiers": {"Nano": NANO_SUB_TIER},
    },
    "Base": {
        "ram_mb_max": 900,
        "p50_ms_max": 150,
        "p95_ms_max": 400,
        "default_format": "INT8 (FP fallback)",
    },
    "Large": {
        "ram_mb_max": 4096,
        "p50_ms_max": 250,
        "p95_ms_max": 800,
        "default_format": "FP16 (INT8 if recall holds)",
    },
    "Accurate-XLarge": {
        "ram_mb_max": 8192,
        "p50_ms_max": 400,
        "p95_ms_max": 1200,
        "default_format": "FP16 (INT8 if recall holds)",
    },
}


__all__ = ["NANO_SUB_TIER", "TIERS"]
