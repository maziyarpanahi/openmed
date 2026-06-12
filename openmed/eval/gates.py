"""Quality gate interface seam for the OM-054 ONNX/WebGPU export pipeline.

Both gates are fail-closed: they raise NotImplementedError until OM-032
provides real implementations. This lets OM-054 wire up gate call-sites
and write tests against the interface without blocking on OM-032.

TODO(OM-032): Replace NotImplementedError stubs with real implementations:
  - quant_delta: run run_benchmark on variant + fp_parent, assert recall parity.
  - tier_fit: measure variant size/latency, assert it fits the tier SLO table.
"""

from __future__ import annotations

from typing import Any, Sequence


class GateFailure(Exception):
    """Raised when a quality gate blocks a variant from shipping."""


def quant_delta(
    variant: Any,
    fp_parent: Any,
    fixtures: Sequence[Any],
) -> None:
    """Assert the quantized variant's recall does not drop below the FP parent's floor.

    Args:
        variant: The quantized model variant to evaluate.
        fp_parent: The FP32 parent model to compare against.
        fixtures: Eval fixtures (BenchmarkFixture or compatible sequence).

    Raises:
        GateFailure: If variant recall drops below the parent floor.
        NotImplementedError: Until OM-032 implements this gate (fail-closed default).
    """
    # TODO(OM-032): implement via run_benchmark; assert variant_recall >= parent_recall
    raise NotImplementedError(
        "quant_delta gate not yet implemented (tracking: OM-032). "
        "Fail-closed by default — do not ship until OM-032 lands."
    )


def tier_fit(
    variant: Any,
    tier: str,
) -> None:
    """Assert the variant's size/latency fits the declared tier SLO.

    Args:
        variant: The model variant to check.
        tier: Tier name (e.g. 'edge', 'server').

    Raises:
        GateFailure: If the variant does not fit the tier SLO.
        NotImplementedError: Until OM-032 implements this gate (fail-closed default).
    """
    # TODO(OM-032): implement by measuring size_bytes + p95_latency against SLO table
    raise NotImplementedError(
        "tier_fit gate not yet implemented (tracking: OM-032). "
        "Fail-closed by default — do not ship until OM-032 lands."
    )


__all__ = ["GateFailure", "quant_delta", "tier_fit"]
