#!/usr/bin/env python3
"""Benchmark synthetic de-identification under the low-resource profile."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Sequence

MIB = 1024 * 1024
GIB = 1024 * MIB
DEFAULT_NOTE_COUNT = 100
DEFAULT_MAX_PEAK_RSS_MIB = 2560.0
DEFAULT_MAX_REGRESSION_PERCENT = 10.0


def synthetic_notes(count: int) -> list[str]:
    """Return deterministic, wholly synthetic clinical notes."""
    if count <= 0:
        raise ValueError("count must be positive")
    return [
        (
            f"Synthetic patient Jordan Example {index:03d}, DOB 1980-01-15, "
            f"MRN LR-{index:06d}, phone 555-010-{index % 10000:04d}, and email "
            f"jordan{index:03d}@example.test attended the clinic on 2026-01-10. "
            "The note contains no real patient information."
        )
        for index in range(count)
    ]


def cgroup_memory_limit_bytes() -> int | None:
    """Return the active cgroup memory limit, if it is finite."""
    for path in (
        Path("/sys/fs/cgroup/memory.max"),
        Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
    ):
        try:
            raw = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if raw == "max":
            return None
        try:
            value = int(raw)
        except ValueError:
            continue
        if 0 < value < 1 << 60:
            return value
    return None


def validate_cgroup_limit(actual: int | None, required_gib: float) -> None:
    """Require a cgroup limit that closely simulates ``required_gib``."""
    expected = required_gib * GIB
    lower = expected * 0.95
    upper = expected * 1.01
    if actual is None or not lower <= actual <= upper:
        actual_text = "unlimited" if actual is None else f"{actual / GIB:.2f} GiB"
        raise RuntimeError(
            f"Expected a {required_gib:.2f} GiB cgroup memory limit; got {actual_text}"
        )


def load_baseline(path: Path) -> dict[str, Any]:
    """Load and validate the committed benchmark baseline."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    peak = payload.get("peak_rss_mib")
    if not isinstance(peak, (int, float)) or peak <= 0:
        raise ValueError(f"{path} must contain a positive peak_rss_mib")
    return payload


def run_benchmark(note_count: int) -> dict[str, Any]:
    """Run end-to-end de-identification and return aggregate metrics only."""
    from openmed.core.config import load_config_with_profile
    from openmed.core.models import ModelLoader
    from openmed.core.pii import deidentify
    from openmed.utils.profiling import measure_peak_rss

    config = load_config_with_profile()
    if config.profile != "low_resource":
        raise RuntimeError("Set OPENMED_PROFILE=low_resource before benchmarking")
    if config.backend != "onnx" or config.onnx_variant != "int8":
        raise RuntimeError("low_resource must resolve to the ONNX INT8 backend")
    if config.device != "cpu":
        raise RuntimeError("low_resource must remain CPU-only")

    notes = synthetic_notes(note_count)
    loader = ModelLoader(config)
    changed_notes = 0
    entity_count = 0
    started = time.perf_counter()
    with measure_peak_rss() as memory:
        for note in notes:
            result = deidentify(note, config=config, loader=loader)
            changed_notes += int(result.deidentified_text != note)
            entity_count += len(result.pii_entities)
    duration = time.perf_counter() - started

    variants = {
        getattr(pipeline, "variant", None) for pipeline in loader._pipelines.values()
    }
    if variants != {"int8"}:
        raise RuntimeError(f"Expected one cached INT8 ONNX pipeline; got {variants}")
    if changed_notes != note_count or entity_count < note_count:
        raise RuntimeError(
            "Synthetic notes did not complete end-to-end de-identification"
        )
    if "torch" in sys.modules:
        raise RuntimeError("low_resource imported torch")

    limit = cgroup_memory_limit_bytes()
    return {
        "schema_version": 1,
        "profile": config.profile,
        "backend": config.backend,
        "onnx_variant": config.onnx_variant,
        "model": config.pii_model,
        "note_count": note_count,
        "entity_count": entity_count,
        "duration_seconds": round(duration, 3),
        "throughput_notes_per_second": round(note_count / duration, 3),
        "peak_rss_mib": round(memory.peak_mib, 3),
        "peak_rss_growth_mib": round(memory.delta_bytes / MIB, 3),
        "cgroup_memory_limit_gib": (None if limit is None else round(limit / GIB, 3)),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch_imported": False,
    }


def enforce_limits(
    result: dict[str, Any],
    *,
    max_peak_rss_mib: float,
    baseline: dict[str, Any] | None,
    max_regression_percent: float,
) -> None:
    """Enforce the absolute memory ceiling and committed regression baseline."""
    peak = float(result["peak_rss_mib"])
    if peak >= max_peak_rss_mib:
        raise RuntimeError(
            f"Peak RSS {peak:.3f} MiB must stay below {max_peak_rss_mib:.3f} MiB"
        )
    if baseline is None:
        return
    allowed = float(baseline["peak_rss_mib"]) * (1.0 + max_regression_percent / 100.0)
    if peak > allowed:
        raise RuntimeError(
            f"Peak RSS {peak:.3f} MiB exceeds the {max_regression_percent:.1f}% "
            f"regression limit of {allowed:.3f} MiB"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notes", type=int, default=DEFAULT_NOTE_COUNT)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument(
        "--max-regression-percent",
        type=float,
        default=DEFAULT_MAX_REGRESSION_PERCENT,
    )
    parser.add_argument(
        "--max-peak-rss-mib",
        type=float,
        default=DEFAULT_MAX_PEAK_RSS_MIB,
    )
    parser.add_argument("--require-cgroup-limit-gib", type=float)
    parser.add_argument("--require-no-gpu", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.require_cgroup_limit_gib is not None:
        validate_cgroup_limit(
            cgroup_memory_limit_bytes(), args.require_cgroup_limit_gib
        )
    if args.require_no_gpu and os.getenv("CUDA_VISIBLE_DEVICES") not in {"", "-1"}:
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES='' for the CPU-only benchmark")

    result = run_benchmark(args.notes)
    baseline = load_baseline(args.baseline) if args.baseline else None
    enforce_limits(
        result,
        max_peak_rss_mib=args.max_peak_rss_mib,
        baseline=baseline,
        max_regression_percent=args.max_regression_percent,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
