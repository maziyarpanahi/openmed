"""Optional cohort service bridge protocol tests."""

from __future__ import annotations

from pathlib import Path

from openmed.interop.bridges import (
    COHORT_SERVICE_COMPATIBILITY_POLICY,
    COHORT_SERVICE_PROTOCOL_VERSION,
    CohortDefinitionServiceBridge,
)
from openmed.structured.cohort import (
    CohortSourceSnapshot,
    PhenotypeDefinition,
    export_cohort_definition,
)
from openmed.structured.store import StoreState

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "cohort"


def _envelope():
    definition = PhenotypeDefinition.load(
        FIXTURES / "phenotypes" / "diabetes_on_metformin.json"
    )
    result = export_cohort_definition(
        definition,
        source_snapshot=CohortSourceSnapshot(
            snapshot_id="snapshot_bridgecohort0001",
            digest="sha256:" + "a" * 64,
            schema_version="synthetic-1",
            license_tags=("synthetic",),
        ),
    )
    assert result.value is not None
    return result.value


def test_injected_adapter_returns_digest_bound_success() -> None:
    def runner(request):
        return {
            "adapter_name": "synthetic-adapter",
            "adapter_version": "1.2.3",
            "compatibility_policy": COHORT_SERVICE_COMPATIBILITY_POLICY,
            "direction": request["direction"],
            "losses": [],
            "protocol_version": COHORT_SERVICE_PROTOCOL_VERSION,
            "source_digest": request["source_digest"],
            "state": "success",
            "target": {"format": request["target_format"], "synthetic": True},
        }

    result = CohortDefinitionServiceBridge(runner=runner).export(
        _envelope(), target_format="open-service.v1"
    )

    assert result.state is StoreState.SUCCESS
    assert result.value is not None
    assert result.value.target == {"format": "open-service.v1", "synthetic": True}


def test_adapter_loss_is_partial_or_strictly_unsupported() -> None:
    def runner(request):
        return {
            "adapter_name": "synthetic-adapter",
            "adapter_version": "1.2.3",
            "compatibility_policy": COHORT_SERVICE_COMPATIBILITY_POLICY,
            "direction": request["direction"],
            "losses": [
                {
                    "path": "expression.children[0]",
                    "reason_code": "operator_unsupported",
                }
            ],
            "protocol_version": COHORT_SERVICE_PROTOCOL_VERSION,
            "source_digest": request["source_digest"],
            "state": "partial",
            "target": {"synthetic": True},
        }

    bridge = CohortDefinitionServiceBridge(runner=runner)
    assert (
        bridge.export(_envelope(), target_format="open-service.v1").state
        is StoreState.PARTIAL
    )
    strict = bridge.export(_envelope(), target_format="open-service.v1", strict=True)
    assert strict.state is StoreState.UNSUPPORTED
    assert strict.value is not None
    assert strict.value.losses[0].reason_code == "operator_unsupported"


def test_unavailable_and_digest_mismatch_fail_closed() -> None:
    unavailable = CohortDefinitionServiceBridge()
    assert (
        unavailable.export(_envelope(), target_format="open-service.v1").state
        is StoreState.UNKNOWN
    )

    def bad_runner(request):
        return {
            "adapter_name": "synthetic-adapter",
            "adapter_version": "1.2.3",
            "compatibility_policy": COHORT_SERVICE_COMPATIBILITY_POLICY,
            "direction": request["direction"],
            "losses": [],
            "protocol_version": COHORT_SERVICE_PROTOCOL_VERSION,
            "source_digest": "sha256:" + "f" * 64,
            "state": "success",
            "target": {"synthetic": True},
        }

    assert (
        CohortDefinitionServiceBridge(runner=bad_runner)
        .export(_envelope(), target_format="open-service.v1")
        .state
        is StoreState.FAILURE
    )

    executable_failure = CohortDefinitionServiceBridge(command=("/usr/bin/false",))
    assert (
        executable_failure.export(_envelope(), target_format="open-service.v1").state
        is StoreState.FAILURE
    )
