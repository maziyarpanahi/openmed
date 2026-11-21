import asyncio
from contextvars import copy_context
from dataclasses import asdict, replace
from datetime import datetime, timezone

import pytest

from openmed import clinical
from openmed.clinical.output_use_policy import (
    DEFAULT_OUTPUT_USE_POLICY_FINGERPRINT,
    OutputUseReasonCode,
    evaluate_output_use,
)
from openmed.core.clinician_mode import (
    ClinicianModeError,
    clinician_mode,
    require_clinician_mode,
)
from openmed.core.disclaimer import OpenMedDisclaimer
from openmed.core.intended_use import load_intended_use_registry
from openmed.structured.decision import (
    DecisionMode,
    DecisionRequest,
    DecisionState,
    decide,
    decide_batch,
)


def ack(tier="restricted"):
    return OpenMedDisclaimer.for_tier(tier).acknowledge(
        at=datetime(2026, 1, 1, tzinfo=timezone.utc)
    )


def test_decision_entry_points_and_policy(monkeypatch):
    monkeypatch.setenv("OPENMED_CLINICIAN_MODE", "true")
    request = DecisionRequest(
        mode=DecisionMode.FIXED_CHOICE,
        input_text="Synthetic review priority is urgent.",
        options=("urgent", "routine"),
    )
    assert decide(request).code == "clinician_mode_required"
    assert decide_batch([request])[0].state == DecisionState.DENIED
    with clinician_mode(purpose="review", acknowledgement=ack()) as receipt:
        assert set(asdict(receipt)) == {"purpose", "notice_id", "notice_digest"}
        assert decide(request).state == DecisionState.SUCCESS
        denied = evaluate_output_use(
            category="decision_support",
            purpose="review",
            audience="clinician",
            review_state="reviewed",
            decision_triggering=True,
            policy_fingerprint=DEFAULT_OUTPUT_USE_POLICY_FINGERPRINT,
        )
        assert (
            not denied.allowed
            and OutputUseReasonCode.DECISION_TRIGGERING_USE.value in denied.reason_codes
        )
    assert decide(request).code == "clinician_mode_required"


def test_all_clinical_restricted_exports_reject_before_processing():
    registry = load_intended_use_registry()
    for alias, code in registry.entry_points.items():
        module, _, name = alias.rpartition(".")
        if (
            module == "openmed.clinical"
            and registry.resolve(code).regulatory_tier == "restricted"
        ):
            with pytest.raises(ClinicianModeError, match="^clinician_mode_required$"):
                getattr(clinical, name)()
    with clinician_mode(purpose="quality_assurance", acknowledgement=ack()):
        assert clinical.parse_unit("mg") is not None
    assert require_clinician_mode("deidentify") is None


def test_explicit_acknowledgement_and_purpose_only():
    for receipt, reason in [
        (None, "acknowledgement_required"),
        (ack("assistive_review"), "invalid_acknowledgement"),
        (replace(ack(), version_digest="stale"), "invalid_acknowledgement"),
        (replace(ack(), acknowledged_at="invalid"), "invalid_acknowledgement"),
    ]:
        with (
            pytest.raises(ClinicianModeError, match=f"^{reason}$"),
            clinician_mode(purpose="review", acknowledgement=receipt),
        ):
            pytest.fail()
    with (
        pytest.raises(ClinicianModeError, match="^invalid_clinician_purpose$"),
        clinician_mode(purpose="clinical_decision", acknowledgement=ack()),
    ):
        pytest.fail()


def test_nested_and_copied_scopes_expire():
    with clinician_mode(purpose="review", acknowledgement=ack()) as outer:
        with (
            pytest.raises(RuntimeError),
            clinician_mode(purpose="research", acknowledgement=ack()),
        ):
            copied = copy_context()
            raise RuntimeError
        assert require_clinician_mode("decisions") is outer
        with pytest.raises(ClinicianModeError):
            copied.run(require_clinician_mode, "decisions")
    with pytest.raises(ClinicianModeError):
        require_clinician_mode("mental_health")


def test_async_task_cannot_reuse_expired_scope():
    async def exercise():
        release = asyncio.Event()

        async def later():
            await release.wait()
            return require_clinician_mode("ecg")

        with clinician_mode(purpose="research", acknowledgement=ack()):
            task = asyncio.create_task(later())
        release.set()
        with pytest.raises(ClinicianModeError):
            await task

    asyncio.run(exercise())
