"""Validate synthetic private-training metadata offline.

Run: python examples/private_training_schema_validation.py
Requires the jsonschema and referencing packages included in openmed[dev].
"""

from datetime import datetime, timedelta, timezone

from jsonschema import Draft202012Validator, ValidationError
from referencing import Registry

from openmed.training.federated_metrics import (
    FederatedMetricKind,
    FederatedPrivacyMechanism,
    build_federated_metric_envelope,
)
from openmed.training.federated_round import (
    FederatedRoundLifecycle,
    FederatedRoundState,
)
from openmed.training.federated_schedule import FederatedRoundSchedule
from openmed.training.federated_status import build_federated_round_status
from openmed.training.private_training_schemas import build_schema


def reject_remote_resolution(uri: str):
    """Refuse external schema retrieval without exposing the requested URI."""
    raise ValueError("external schema resolution is disabled")


def main() -> None:
    """Validate synthetic payloads and print only schema names and outcomes."""
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    day = timedelta(days=1)
    payloads = {
        "federated_round_lifecycle": FederatedRoundLifecycle().to_dict(),
        "federated_round_schedule": FederatedRoundSchedule(
            enrollment_starts_at=start,
            update_submission_starts_at=start + day,
            aggregation_starts_at=start + 2 * day,
            evaluation_starts_at=start + 3 * day,
            finishes_at=start + 4 * day,
        ).to_dict(),
        "federated_round_status": build_federated_round_status(
            state=FederatedRoundState.COLLECTING,
            participant_count=10,
            completed_participant_count=6,
            required_quorum=5,
            aggregate_digest_refs=("sha256:" + "a" * 64,),
        ).to_dict(),
        "federated_aggregate_metric": build_federated_metric_envelope(
            metric_id="round.loss",
            metric_kind=FederatedMetricKind.BOUNDED_MEAN,
            aggregate_value=0.5,
            clipping_lower_bound=0.0,
            clipping_upper_bound=1.0,
            privacy_mechanism=FederatedPrivacyMechanism.LAPLACE,
            privacy_mechanism_version="v1",
            participant_count=100,
        ).to_dict(),
    }
    registry = Registry(retrieve=reject_remote_resolution)
    for name, payload in payloads.items():
        validator = Draft202012Validator(
            build_schema(name),
            registry=registry,
            format_checker=Draft202012Validator.FORMAT_CHECKER,
        )
        validator.validate(payload)
        print(f"{name}: valid")

    # Reuse the metric validator to demonstrate rejection of an unknown field.
    try:
        validator.validate({**payload, "unknown_field": True})
    except ValidationError:
        print(f"{name}: rejected")
    else:
        raise AssertionError("unknown field was accepted")


if __name__ == "__main__":
    main()
