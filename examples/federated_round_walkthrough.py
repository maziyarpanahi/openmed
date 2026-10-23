"""Walk through a synthetic metadata-only federated round offline.

Run ``python -m examples.federated_round_walkthrough --format json`` or use
``--format markdown`` for the final status. No tensors, training, files, client
identities, or network providers are used. Metadata and clipping declarations
are not verification of update contents; threshold suppression is not DP.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from typing import Any

from openmed.training import (
    FEDERATED_UPDATE_METADATA_SCHEMA_VERSION,
    FederatedMetricKind,
    FederatedParameterMetadata,
    FederatedPrivacyMechanism,
    FederatedRoundLifecycle,
    FederatedRoundReasonCode,
    FederatedRoundSchedule,
    FederatedRoundState,
    FederatedRoundStatus,
    FederatedUncertaintyMethod,
    FederatedUpdateMetadata,
    FederatedUpdateMetadataError,
    FederatedUpdatePolicy,
    build_federated_metric_envelope,
    build_federated_round_status,
)
from openmed.training.federated_schema_fingerprint import fingerprint_update_schema


def _update_payload(digest_digit: str, *, reverse: bool = False) -> dict[str, Any]:
    """Build incoming synthetic declarations independently of the policy."""
    parameters = [
        {"name": "adapter.lora_A.weight", "shape": [2, 3], "dtype": "float32"},
        {"name": "adapter.lora_B.weight", "shape": [4, 2], "dtype": "float32"},
    ]
    return {
        "schema_version": FEDERATED_UPDATE_METADATA_SCHEMA_VERSION,
        "model_digest": "sha256:" + "a" * 64,
        "adapter_format": "dense",
        "parameters": list(reversed(parameters)) if reverse else parameters,
        "total_elements": 14,
        "update_digest": "sha256:" + digest_digit * 64,
        "clipped": True,
    }


def run_example() -> tuple[dict[str, Any], FederatedRoundStatus]:
    """Build a safe report and its status using fixed synthetic inputs.

    Returns:
        The JSON-ready walkthrough and the same status object for Markdown.

    Raises:
        AssertionError: If the unknown parameter is unexpectedly accepted.
    """
    # Trusted coordinator expectations are fixed before any payload is built.
    policy = FederatedUpdatePolicy(
        model_digest="sha256:" + "a" * 64,
        parameters=(
            FederatedParameterMetadata("adapter.lora_A.weight", (2, 3), "float32"),
            FederatedParameterMetadata("adapter.lora_B.weight", (4, 2), "float32"),
        ),
        require_clipped=True,
        max_total_elements=14,
    )
    start = datetime(2026, 9, 1, tzinfo=timezone.utc)
    schedule = FederatedRoundSchedule(
        enrollment_starts_at=start,
        update_submission_starts_at=start + timedelta(hours=1),
        aggregation_starts_at=start + timedelta(hours=2),
        evaluation_starts_at=start + timedelta(hours=3),
        finishes_at=start + timedelta(hours=4),
        max_enrollment_duration_seconds=3600,
        max_update_submission_duration_seconds=3600,
        max_aggregation_duration_seconds=3600,
        max_evaluation_duration_seconds=3600,
    )
    phases = {}
    for boundary in schedule.to_dict()["boundaries"]:
        phase = schedule.active_phase_at(getattr(schedule, boundary))
        phases[boundary] = phase.value if phase is not None else None

    lifecycle = FederatedRoundLifecycle()
    states = [lifecycle.state.value]
    for state in (FederatedRoundState.PREFLIGHT, FederatedRoundState.COLLECTING):
        lifecycle = lifecycle.transition_to(state)
        states.append(lifecycle.state.value)

    accepted = []
    for index, digit in enumerate("bcdef"):
        payload = _update_payload(digit, reverse=bool(index % 2))
        accepted.append(FederatedUpdateMetadata.from_dict(payload, policy=policy))

    rejected = _update_payload("1")
    rejected["parameters"][1]["name"] = "adapter.unknown.weight"
    try:
        FederatedUpdateMetadata.from_dict(rejected, policy=policy)
    except FederatedUpdateMetadataError:
        rejection_code = "unknown_parameter"
    else:
        raise AssertionError("unknown_parameter_not_rejected")

    # Only successfully validated updates reach fingerprint grouping.
    groups: dict[str, int] = {}
    for metadata in accepted:
        fingerprint = fingerprint_update_schema(metadata)
        groups[fingerprint] = groups.get(fingerprint, 0) + 1

    for state in (FederatedRoundState.AGGREGATING, FederatedRoundState.EVALUATING):
        lifecycle = lifecycle.transition_to(state)
        states.append(lifecycle.state.value)

    suppressed = build_federated_metric_envelope(
        metric_id="synthetic_subset_rate",
        metric_kind=FederatedMetricKind.RATE,
        aggregate_value=0.73,
        clipping_lower_bound=0.0,
        clipping_upper_bound=1.0,
        privacy_mechanism=FederatedPrivacyMechanism.THRESHOLD_ONLY,
        privacy_mechanism_version="v1",
        participant_count=4,
        minimum_group_size=5,
        uncertainty_method=FederatedUncertaintyMethod.CONFIDENCE_INTERVAL,
        uncertainty_lower_bound=0.61,
        uncertainty_upper_bound=0.89,
        confidence_level=0.95,
    )
    released = build_federated_metric_envelope(
        metric_id="synthetic_round_rate",
        metric_kind=FederatedMetricKind.RATE,
        aggregate_value=0.8,
        clipping_lower_bound=0.0,
        clipping_upper_bound=1.0,
        privacy_mechanism=FederatedPrivacyMechanism.THRESHOLD_ONLY,
        privacy_mechanism_version="v1",
        participant_count=len(accepted),
        minimum_group_size=5,
    )

    # A fixed synthetic review decision, not an automated quality gate.
    lifecycle = lifecycle.transition_to(FederatedRoundState.HELD)
    states.append(lifecycle.state.value)
    status = build_federated_round_status(
        state=lifecycle.state,
        participant_count=len(accepted),
        completed_participant_count=len(accepted),
        required_quorum=5,
        minimum_group_size=5,
        reason_code=FederatedRoundReasonCode.QUALITY_REVIEW_REQUIRED,
    )
    report = {
        "synthetic_only": True,
        "schedule": schedule.to_dict(),
        "schedule_phases_at_boundaries": phases,
        "lifecycle_states": states,
        "updates": {
            "accepted_count": len(accepted),
            "rejected_count": 1,
            "rejection_code": rejection_code,
        },
        "schema_groups": [
            {"schema_fingerprint": fingerprint, "update_count": count}
            for fingerprint, count in sorted(groups.items())
        ],
        # Serialize only the public envelopes, never their input arguments.
        "metrics": [suppressed.to_dict(), released.to_dict()],
        "status": status.to_dict(),
    }
    return report, status


def main(argv: list[str] | None = None) -> int:
    """Print the walkthrough JSON or the public Markdown status summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    args = parser.parse_args(argv)
    report, status = run_example()
    if args.format == "markdown":
        print(status.to_markdown(), end="")
    else:
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
