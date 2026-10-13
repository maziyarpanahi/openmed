"""Bounded synthetic federation metadata fixtures; never real client records."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone

from openmed.training.federated_metrics import (
    FederatedMetricKind,
    FederatedPrivacyMechanism,
    build_federated_metric_envelope,
)
from openmed.training.federated_schedule import FederatedRoundSchedule
from openmed.training.federated_update_metadata import (
    FederatedParameterMetadata,
    FederatedUpdateMetadata,
    FederatedUpdatePolicy,
)

VARIANTS = frozenset(
    {"valid", "capability", "schedule", "update", "dropout", "aggregate"}
)


def _digest(seed: int, category: str, index: int = 0) -> str:
    payload = f"openmed.synthetic.federation.v1:{seed}:{category}:{index}".encode(
        "ascii"
    )
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def generate_private_learning_fixture(
    seed: int, *, cohort_size: int = 8, variant: str = "valid"
) -> dict[str, object]:
    """Build fresh deterministic fixtures with an explicitly named corruption.

    Seed and cohort bounds are checked before allocation. Capability examples
    are synthetic coarse declarations, not an enrollment protocol or measured
    hardware evidence. Schedule, update and aggregate envelopes use existing
    OpenMed builders. No tensors, clinical examples or client IDs are generated.
    """
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("fixture seed must be an unsigned 64-bit integer")
    if type(cohort_size) is not int or not 1 <= cohort_size <= 128:
        raise ValueError("fixture cohort must contain 1 through 128 entries")
    if type(variant) is not str or variant not in VARIANTS:
        raise ValueError("unsupported fixture corruption category")
    model = _digest(seed, "model")
    parameters = (FederatedParameterMetadata("adapter.weight", (2, 8), "float32"),)
    policy = FederatedUpdatePolicy(model, parameters)
    updates = [
        FederatedUpdateMetadata(
            model_digest=model,
            adapter_format="dense",
            parameters=parameters,
            total_elements=16,
            update_digest=_digest(seed, "update", i),
            clipped=True,
            policy=policy,
        ).to_dict()
        for i in range(cohort_size)
    ]
    # Derive only a bounded synthetic clock offset; never consult the wall clock.
    start = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=seed % 1440)
    schedule = FederatedRoundSchedule(
        *(start + timedelta(minutes=i) for i in range(5))
    ).to_dict()
    aggregate = build_federated_metric_envelope(
        metric_id="synthetic.acceptance_rate",
        metric_kind=FederatedMetricKind.RATE,
        aggregate_value=0.5,
        clipping_lower_bound=0,
        clipping_upper_bound=1,
        privacy_mechanism=FederatedPrivacyMechanism.THRESHOLD_ONLY,
        privacy_mechanism_version="v1",
        participant_count=cohort_size,
    ).to_dict()
    result = {
        "schema_version": "openmed.synthetic.federation_fixture.v1",
        "synthetic": True,
        "variant": variant,
        "seed_commitment": _digest(seed, "seed"),
        "capabilities": [
            {
                "protocol_version": "1.0",
                "backend": ("cpu", "numpy")[i % 2],
                "adapter_format": "dense",
                "resource_class": ("small", "medium")[i % 2],
                "deterministic_kernels": True,
            }
            for i in range(cohort_size)
        ],
        "schedule": schedule,
        "updates": updates,
        "dropout": {
            "submitted_count": cohort_size,
            "dropout_count": seed % (cohort_size + 1),
        },
        "aggregate_result": aggregate,
    }
    if variant == "capability":
        result["capabilities"][0]["backend"] = "unsupported"
    elif variant == "schedule":
        result["schedule"]["boundaries"]["finishes_at"] = result["schedule"][
            "boundaries"
        ]["enrollment_starts_at"]
    elif variant == "update":
        result["updates"][0]["total_elements"] += 1
    elif variant == "dropout":
        result["dropout"]["dropout_count"] = cohort_size + 1
    elif variant == "aggregate":
        result["aggregate_result"]["schema_version"] = "unsupported"
    return result


def render_private_learning_fixture(
    seed: int, *, cohort_size: int = 8, variant: str = "valid"
) -> str:
    """Render byte-stable ASCII JSON, with no wall-clock or random-state input."""
    return (
        json.dumps(
            generate_private_learning_fixture(
                seed, cohort_size=cohort_size, variant=variant
            ),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    )
