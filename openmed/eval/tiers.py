"""Canonical device-tier SLO budgets for evaluation gates."""

from __future__ import annotations

from typing import Final


TIERS: Final[dict[str, dict[str, int | str]]] = {
    "Tiny": {
        "ram_mb_max": 350,
        "p50_ms_max": 60,
        "p95_ms_max": 150,
        "default_format": "INT8 (MLX-8bit / CoreML)",
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


__all__ = ["TIERS"]
