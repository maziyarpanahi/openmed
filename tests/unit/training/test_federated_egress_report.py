"""Synthetic, offline contracts for federated egress reports and approvals."""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import re
from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from openmed.training.federated_egress_report import (
    FEDERATED_EGRESS_APPROVAL_SCHEMA_VERSION,
    FEDERATED_EGRESS_REPORT_SCHEMA_VERSION,
    FORBIDDEN_EGRESS_REPORT_FIELDS,
    MAX_FEDERATED_EGRESS_APPROVAL_VALIDITY,
    FederatedClippingNorm,
    FederatedClippingScope,
    FederatedEgressApproval,
    FederatedEgressClipping,
    FederatedEgressError,
    FederatedEgressMechanism,
    FederatedEgressPrivacySpend,
    FederatedEgressReasonCode,
    FederatedEgressReport,
    FederatedEgressSubmissionGuard,
    FederatedPrivacyAccountant,
    FederatedPrivacySpendBand,
    FederatedUpdateNoiseMechanism,
    approve_federated_egress,
    render_federated_egress_report,
)
from openmed.training.federated_schema_fingerprint import fingerprint_update_schema
from openmed.training.federated_update_metadata import (
    FEDERATED_UPDATE_METADATA_SCHEMA_VERSION,
    FederatedParameterMetadata,
    FederatedUpdateMetadata,
    FederatedUpdatePolicy,
)
from tests.fixtures.private_learning_forbidden import (
    FORBIDDEN_FIELD_CASES,
    ForbiddenFieldCase,
)

Reason = FederatedEgressReasonCode

MODEL = "sha256:" + "a" * 64
UPDATE = "sha256:" + "b" * 64
ROUND = "sha256:" + "c" * 64
OTHER_ROUND = "sha256:" + "d" * 64
CLIP_CONFIG = "sha256:" + "e" * 64
NOISE_CONFIG = "sha256:" + "f" * 64
LEDGER = "sha256:" + "1" * 64
SEALED = b"synthetic sealed update bytes v1"
OTHER_SEALED = b"synthetic sealed update bytes v2"
START = datetime(2026, 9, 1, 12, 0, 0, tzinfo=timezone.utc)
VALID_FOR = timedelta(hours=1)
SECRET_EPSILON = 0.123456789
SECRET_DELTA = 0.0000031415

GOLDEN_REPORT_DIGEST = (
    "sha256:4ca47eb60d02673f7dcb45ef077423ddd75811b1672c50e822ec21a81be06323"
)

_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")


class FakeClock:
    """Injected UTC clock that the tests advance explicitly."""

    def __init__(self, now: datetime = START) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


class HmacApprover:
    """Synthetic site approver that signs statements with a local HMAC key."""

    def __init__(self, key: bytes = b"synthetic-approver-key", *, approve=True):
        self._key = key
        self._approve = approve
        self.reviewed: list[tuple[str, bytes]] = []

    def _sign(self, statement: bytes) -> str:
        return "hmac-sha256:" + hmac.new(self._key, statement, "sha256").hexdigest()

    def approve(self, report: FederatedEgressReport, statement: bytes) -> str | None:
        self.reviewed.append((report.digest, statement))
        return self._sign(statement) if self._approve else None

    def verify(self, statement: bytes, signature: str) -> bool:
        return hmac.compare_digest(self._sign(statement), signature)


class Transport:
    """Injected sender that records every submission it receives."""

    def __init__(self) -> None:
        self.sent: list[bytes] = []

    def __call__(self, payload: bytes) -> str:
        self.sent.append(payload)
        return "accepted"


def _policy() -> FederatedUpdatePolicy:
    return FederatedUpdatePolicy(
        model_digest=MODEL,
        parameters=(
            FederatedParameterMetadata("adapter.lora_A.weight", (2, 3), "float32"),
            FederatedParameterMetadata("adapter.lora_B.weight", (4, 2), "float32"),
        ),
        max_total_elements=14,
    )


def _metadata(*, reverse: bool = False, clipped: bool = True):
    parameters = [
        FederatedParameterMetadata("adapter.lora_A.weight", (2, 3), "float32"),
        FederatedParameterMetadata("adapter.lora_B.weight", (4, 2), "float32"),
    ]
    if reverse:
        parameters.reverse()
    policy = _policy()
    if not clipped:
        policy = replace(policy, require_clipped=False)
    return FederatedUpdateMetadata(
        model_digest=MODEL,
        adapter_format="dense",
        parameters=tuple(parameters),
        total_elements=14,
        update_digest=UPDATE,
        clipped=clipped,
        policy=policy,
    )


def _clipping() -> FederatedEgressClipping:
    return FederatedEgressClipping(
        scope=FederatedClippingScope.PER_EXAMPLE,
        norm=FederatedClippingNorm.L2,
        config_digest=CLIP_CONFIG,
    )


def _mechanism() -> FederatedEgressMechanism:
    return FederatedEgressMechanism(
        mechanism=FederatedUpdateNoiseMechanism.GAUSSIAN,
        version="v1",
        accountant=FederatedPrivacyAccountant.RDP,
        config_digest=NOISE_CONFIG,
    )


def _spend(**changes: Any) -> FederatedEgressPrivacySpend:
    values: dict[str, Any] = {
        "epsilon_spent": SECRET_EPSILON,
        "epsilon_budget": 1.0,
        "delta_spent": SECRET_DELTA,
        "delta_budget": 1e-5,
        "ledger_digest": LEDGER,
    }
    values.update(changes)
    return FederatedEgressPrivacySpend(**values)


def _report(
    *,
    sealed: bytes = SEALED,
    round_digest: str = ROUND,
    reverse: bool = False,
    spend: FederatedEgressPrivacySpend | None = None,
) -> FederatedEgressReport:
    return render_federated_egress_report(
        _metadata(reverse=reverse),
        sealed_update=sealed,
        round_digest=round_digest,
        clipping=_clipping(),
        mechanism=_mechanism(),
        privacy_spend=spend or _spend(),
    )


def _approve(report, approver=None, clock=None, valid_for=VALID_FOR):
    return approve_federated_egress(
        report,
        approver=approver or HmacApprover(),
        clock=clock or FakeClock(),
        valid_for=valid_for,
    )


def _guard(approver=None, clock=None) -> FederatedEgressSubmissionGuard:
    return FederatedEgressSubmissionGuard(
        approver=approver or HmacApprover(), clock=clock or FakeClock()
    )


def _assert_refused(error: pytest.ExceptionInfo, reason: Reason) -> None:
    assert error.value.reason_code is reason
    assert error.value.__cause__ is None
    assert error.value.__suppress_context__ or error.value.__context__ is None
    rendered = str(error.value)
    for sentinel in ("synthetic sealed", "123456789", "synthetic-approver-key"):
        assert sentinel not in rendered


# ---------------------------------------------------------------------------
# Byte-stable rendering


def test_synthetic_sealed_update_yields_byte_stable_golden_report():
    first = _report()
    second = _report(reverse=True)

    assert first.to_json() == second.to_json()
    assert first.to_json().endswith("\n")
    assert first.to_markdown() == second.to_markdown()
    assert first.digest == second.digest == GOLDEN_REPORT_DIGEST
    expected = "sha256:" + hashlib.sha256(first.to_json().encode()).hexdigest()
    assert first.digest == expected
    assert first.sealed_update_digest == (
        "sha256:" + hashlib.sha256(SEALED).hexdigest()
    )
    assert first.round_digest == ROUND
    assert FederatedEgressReport.from_json(first.to_json()) == first
    assert FederatedEgressReport.from_dict(first.to_dict()) == first


def test_report_discloses_only_names_shapes_bands_codes_and_digests():
    report = _report().to_dict()

    assert report == {
        "schema_version": FEDERATED_EGRESS_REPORT_SCHEMA_VERSION,
        "round_digest": ROUND,
        "sealed_update_digest": "sha256:" + hashlib.sha256(SEALED).hexdigest(),
        "update": {
            "metadata_schema_version": FEDERATED_UPDATE_METADATA_SCHEMA_VERSION,
            "model_digest": MODEL,
            "update_digest": UPDATE,
            "adapter_format": "dense",
            "schema_fingerprint": fingerprint_update_schema(_metadata()),
            "clipped": True,
            "parameter_count": 2,
            "total_elements": 14,
            "parameters": [
                {
                    "name": "adapter.lora_A.weight",
                    "shape": [2, 3],
                    "dtype": "float32",
                    "element_count": 6,
                },
                {
                    "name": "adapter.lora_B.weight",
                    "shape": [4, 2],
                    "dtype": "float32",
                    "element_count": 8,
                },
            ],
        },
        "clipping": {
            "scope": "per_example",
            "norm": "l2",
            "config_digest": CLIP_CONFIG,
        },
        "mechanism": {
            "mechanism": "gaussian",
            "version": "v1",
            "accountant": "rdp",
            "config_digest": NOISE_CONFIG,
        },
        "privacy_spend": {
            "epsilon_band": "under_quarter",
            "delta_band": "quarter_to_under_half",
            "ledger_digest": LEDGER,
        },
    }


def test_report_and_declarations_never_echo_local_values():
    sealed = b"placeholder example text /placeholder/path patient 12345"
    report = _report(sealed=sealed)
    spend = _spend()
    outputs = (
        report.to_json(),
        report.to_markdown(),
        repr(report),
        repr(spend),
        str(spend),
    )
    for output in outputs:
        assert "placeholder" not in output
        assert "12345" not in output
        assert "123456789" not in output
        assert repr(SECRET_EPSILON) not in output
        assert repr(SECRET_DELTA) not in output
    assert repr(report) == f"FederatedEgressReport(digest={report.digest!r})"
    assert report.digest in report.to_markdown()


def test_report_is_immutable_and_exports_fresh_copies():
    report = _report()
    canonical = report.to_json()
    exported = report.to_dict()
    exported["update"]["parameters"][0]["name"] = "changed"
    assert report.to_json() == canonical
    with pytest.raises((FrozenInstanceError, AttributeError)):
        report.digest = "sha256:" + "0" * 64


def test_report_changes_when_any_bound_input_changes():
    reference = _report()
    assert _report(sealed=OTHER_SEALED).digest != reference.digest
    assert _report(round_digest=OTHER_ROUND).digest != reference.digest
    assert _report(spend=_spend(epsilon_spent=0.9)).digest != reference.digest


# ---------------------------------------------------------------------------
# Approval and submission guard


def test_approved_update_passes_submission_guard_and_is_sent():
    approver = HmacApprover()
    clock = FakeClock()
    report = _report()
    approval = _approve(report, approver, clock)
    transport = Transport()

    assert approver.reviewed and approver.reviewed[0][0] == report.digest
    assert approval.report_digest == report.digest
    assert approval.sealed_update_digest == report.sealed_update_digest
    assert approval.round_digest == ROUND
    assert approval.approved_at == START
    assert approval.expires_at == START + VALID_FOR

    clock.now = START + VALID_FOR - timedelta(seconds=1)
    guard = _guard(approver, clock)
    clearance = guard.check(
        SEALED, report=report, approval=approval, round_digest=ROUND
    )
    assert clearance.report_digest == report.digest
    assert clearance.expires_at == approval.expires_at

    result = guard.submit(
        SEALED,
        report=report,
        approval=approval,
        round_digest=ROUND,
        transport=transport,
    )
    assert result == "accepted"
    assert transport.sent == [SEALED]


def test_submission_without_approval_is_refused_and_nothing_is_sent():
    transport = Transport()
    with pytest.raises(FederatedEgressError) as error:
        _guard().submit(
            SEALED,
            report=_report(),
            approval=None,
            round_digest=ROUND,
            transport=transport,
        )
    _assert_refused(error, Reason.APPROVAL_MISSING)
    assert transport.sent == []


def test_approval_for_different_sealed_bytes_is_refused():
    approver = HmacApprover()
    report = _report()
    approval = _approve(report, approver)
    transport = Transport()

    # Same approved report, different bytes on the wire.
    with pytest.raises(FederatedEgressError) as error:
        _guard(approver).submit(
            OTHER_SEALED,
            report=report,
            approval=approval,
            round_digest=ROUND,
            transport=transport,
        )
    _assert_refused(error, Reason.SEALED_UPDATE_MISMATCH)

    # A correct report for the other bytes still lacks a matching approval.
    other_report = _report(sealed=OTHER_SEALED)
    with pytest.raises(FederatedEgressError) as error:
        _guard(approver).submit(
            OTHER_SEALED,
            report=other_report,
            approval=approval,
            round_digest=ROUND,
            transport=transport,
        )
    _assert_refused(error, Reason.APPROVAL_REPORT_MISMATCH)
    assert transport.sent == []


def test_approval_for_another_round_is_refused():
    approver = HmacApprover()
    report = _report()
    approval = _approve(report, approver)
    transport = Transport()

    with pytest.raises(FederatedEgressError) as error:
        _guard(approver).submit(
            SEALED,
            report=report,
            approval=approval,
            round_digest=OTHER_ROUND,
            transport=transport,
        )
    _assert_refused(error, Reason.ROUND_MISMATCH)

    other_round_report = _report(round_digest=OTHER_ROUND)
    with pytest.raises(FederatedEgressError) as error:
        _guard(approver).submit(
            SEALED,
            report=other_round_report,
            approval=approval,
            round_digest=OTHER_ROUND,
            transport=transport,
        )
    _assert_refused(error, Reason.APPROVAL_REPORT_MISMATCH)
    assert transport.sent == []


def test_approval_outside_its_window_is_refused():
    approver = HmacApprover()
    clock = FakeClock()
    report = _report()
    approval = _approve(report, approver, clock)
    transport = Transport()
    guard = _guard(approver, clock)

    clock.now = approval.expires_at
    with pytest.raises(FederatedEgressError) as error:
        guard.submit(
            SEALED,
            report=report,
            approval=approval,
            round_digest=ROUND,
            transport=transport,
        )
    _assert_refused(error, Reason.APPROVAL_EXPIRED)

    clock.now = START - timedelta(seconds=1)
    with pytest.raises(FederatedEgressError) as error:
        guard.submit(
            SEALED,
            report=report,
            approval=approval,
            round_digest=ROUND,
            transport=transport,
        )
    _assert_refused(error, Reason.APPROVAL_NOT_YET_VALID)
    assert transport.sent == []

    # Negative control: the same approval is accepted inside the window.
    clock.now = START
    guard.submit(
        SEALED,
        report=report,
        approval=approval,
        round_digest=ROUND,
        transport=transport,
    )
    assert transport.sent == [SEALED]


@pytest.mark.parametrize(
    "change",
    [
        {"expires_at": START + timedelta(hours=2)},
        {"approved_at": START - timedelta(minutes=5)},
        {"signature": "hmac-sha256:" + "0" * 64},
    ],
    ids=["extended-expiry", "backdated", "forged-signature"],
)
def test_tampered_approval_fails_signature_verification(change):
    approver = HmacApprover()
    report = _report()
    tampered = replace(_approve(report, approver), **change)
    with pytest.raises(FederatedEgressError) as error:
        _guard(approver).check(
            SEALED, report=report, approval=tampered, round_digest=ROUND
        )
    _assert_refused(error, Reason.APPROVAL_SIGNATURE_INVALID)


def test_approval_from_another_approver_key_is_refused():
    report = _report()
    approval = _approve(report, HmacApprover(b"another-synthetic-key"))
    with pytest.raises(FederatedEgressError) as error:
        _guard(HmacApprover()).check(
            SEALED, report=report, approval=approval, round_digest=ROUND
        )
    _assert_refused(error, Reason.APPROVAL_SIGNATURE_INVALID)


def test_declined_and_failing_approvers_fail_closed():
    with pytest.raises(FederatedEgressError) as error:
        _approve(_report(), HmacApprover(approve=False))
    _assert_refused(error, Reason.APPROVAL_DECLINED)

    class Exploding(HmacApprover):
        def approve(self, report, statement):
            raise RuntimeError("synthetic sealed secret 123456789")

        def verify(self, statement, signature):
            raise RuntimeError("synthetic sealed secret 123456789")

    with pytest.raises(FederatedEgressError) as error:
        _approve(_report(), Exploding())
    _assert_refused(error, Reason.APPROVER_FAILED)

    report = _report()
    approval = _approve(report)
    with pytest.raises(FederatedEgressError) as error:
        _guard(Exploding()).check(
            SEALED, report=report, approval=approval, round_digest=ROUND
        )
    _assert_refused(error, Reason.APPROVER_FAILED)


@pytest.mark.parametrize("signature", ["", "has space", "x" * 600, 7, None])
def test_approver_must_return_a_bounded_signature(signature):
    class Odd(HmacApprover):
        def approve(self, report, statement):
            return signature

    with pytest.raises(FederatedEgressError) as error:
        _approve(_report(), Odd())
    expected = (
        Reason.APPROVAL_DECLINED if signature is None else Reason.INVALID_APPROVAL
    )
    _assert_refused(error, expected)


@pytest.mark.parametrize(
    "valid_for",
    [
        timedelta(0),
        timedelta(seconds=-1),
        timedelta(seconds=1.5),
        MAX_FEDERATED_EGRESS_APPROVAL_VALIDITY + timedelta(seconds=1),
        3600,
        None,
    ],
)
def test_approval_window_must_be_bounded_whole_seconds(valid_for):
    with pytest.raises(FederatedEgressError) as error:
        _approve(_report(), valid_for=valid_for)
    _assert_refused(error, Reason.INVALID_APPROVAL_WINDOW)


@pytest.mark.parametrize(
    "now",
    [datetime(2026, 9, 1, 12), "2026-09-01T12:00:00Z", None],
    ids=["naive", "string", "none"],
)
def test_clock_must_return_aware_datetimes(now):
    with pytest.raises(FederatedEgressError) as error:
        _approve(_report(), clock=lambda: now)
    _assert_refused(error, Reason.INVALID_CLOCK)

    report = _report()
    approval = _approve(report)
    with pytest.raises(FederatedEgressError) as error:
        FederatedEgressSubmissionGuard(
            approver=HmacApprover(), clock=lambda: now
        ).check(SEALED, report=report, approval=approval, round_digest=ROUND)
    _assert_refused(error, Reason.INVALID_CLOCK)


def test_non_utc_clock_is_normalized_and_microseconds_truncated():
    offset = timezone(timedelta(hours=2))
    local = datetime(2026, 9, 1, 14, 0, 0, 999_999, tzinfo=offset)
    approval = _approve(_report(), clock=lambda: local)
    assert approval.approved_at == START
    assert approval.approved_at.tzinfo is timezone.utc


def test_failing_transport_is_wrapped_without_leaking():
    approver = HmacApprover()
    report = _report()
    approval = _approve(report, approver)

    def explode(payload: bytes) -> None:
        raise OSError("/placeholder/path synthetic sealed 123456789")

    with pytest.raises(FederatedEgressError) as error:
        _guard(approver).submit(
            SEALED,
            report=report,
            approval=approval,
            round_digest=ROUND,
            transport=explode,
        )
    _assert_refused(error, Reason.TRANSPORT_FAILED)
    assert "placeholder" not in str(error.value)


@pytest.mark.parametrize(
    ("arguments", "reason"),
    [
        ({"sealed_update": "text"}, Reason.INVALID_SEALED_UPDATE),
        ({"sealed_update": b""}, Reason.INVALID_SEALED_UPDATE),
        ({"report": _report().to_dict()}, Reason.INVALID_REPORT),
        ({"approval": {"signature": "x"}}, Reason.INVALID_APPROVAL),
        ({"round_digest": "sha256:XYZ"}, Reason.INVALID_ROUND_DIGEST),
    ],
    ids=["text-bytes", "empty-bytes", "mapping-report", "mapping-approval", "round"],
)
def test_guard_rejects_malformed_arguments(arguments, reason):
    report = _report()
    values: dict[str, Any] = {
        "sealed_update": SEALED,
        "report": report,
        "approval": _approve(report),
        "round_digest": ROUND,
    }
    values.update(arguments)
    sealed = values.pop("sealed_update")
    with pytest.raises(FederatedEgressError) as error:
        _guard().check(sealed, **values)
    _assert_refused(error, reason)


def test_approval_record_round_trips_and_still_passes_guard():
    approver = HmacApprover()
    report = _report()
    approval = _approve(report, approver)
    payload = approval.to_json()

    assert payload.endswith("\n")
    assert json.loads(payload) == {
        "schema_version": FEDERATED_EGRESS_APPROVAL_SCHEMA_VERSION,
        "report_digest": report.digest,
        "sealed_update_digest": report.sealed_update_digest,
        "round_digest": ROUND,
        "approved_at": "2026-09-01T12:00:00Z",
        "expires_at": "2026-09-01T13:00:00Z",
        "signature": approval.signature,
    }
    parsed = FederatedEgressApproval.from_json(payload)
    assert parsed == approval
    assert parsed.statement() == approval.statement()
    _guard(approver).check(SEALED, report=report, approval=parsed, round_digest=ROUND)


@pytest.mark.parametrize(
    "change",
    [
        {"expires_at": "2026-09-01T11:00:00Z"},
        {"expires_at": "2026-09-30T12:00:00Z"},
        {"approved_at": "2026-09-01 12:00:00"},
        {"report_digest": "sha256:short"},
        {"schema_version": "openmed.training.federated_egress_approval.v0"},
        {"signature": 5},
    ],
)
def test_malformed_approval_records_are_rejected(change):
    payload = _approve(_report()).to_dict()
    payload.update(change)
    with pytest.raises(FederatedEgressError) as error:
        FederatedEgressApproval.from_dict(payload)
    _assert_refused(error, Reason.INVALID_APPROVAL)


# ---------------------------------------------------------------------------
# Input declarations


def test_unclipped_update_cannot_be_reported_for_egress():
    with pytest.raises(FederatedEgressError) as error:
        render_federated_egress_report(
            _metadata(clipped=False),
            sealed_update=SEALED,
            round_digest=ROUND,
            clipping=_clipping(),
            mechanism=_mechanism(),
            privacy_spend=_spend(),
        )
    _assert_refused(error, Reason.CLIPPING_NOT_DECLARED)


@pytest.mark.parametrize(
    ("arguments", "reason"),
    [
        ({"metadata": _policy()}, Reason.INVALID_UPDATE_METADATA),
        ({"sealed_update": bytearray(SEALED)}, Reason.INVALID_SEALED_UPDATE),
        ({"round_digest": ROUND.upper()}, Reason.INVALID_ROUND_DIGEST),
        ({"clipping": None}, Reason.INVALID_CLIPPING_DECLARATION),
        ({"mechanism": "gaussian"}, Reason.INVALID_MECHANISM_DECLARATION),
        ({"privacy_spend": {"epsilon": 1}}, Reason.INVALID_PRIVACY_SPEND),
    ],
)
def test_renderer_rejects_untyped_inputs(arguments, reason):
    values: dict[str, Any] = {
        "metadata": _metadata(),
        "sealed_update": SEALED,
        "round_digest": ROUND,
        "clipping": _clipping(),
        "mechanism": _mechanism(),
        "privacy_spend": _spend(),
    }
    values.update(arguments)
    metadata = values.pop("metadata")
    with pytest.raises(FederatedEgressError) as error:
        render_federated_egress_report(metadata, **values)
    _assert_refused(error, reason)


@pytest.mark.parametrize(
    ("factory", "change", "reason"),
    [
        (_clipping, {"scope": "per_site"}, Reason.INVALID_CLIPPING_DECLARATION),
        (_clipping, {"norm": "l1"}, Reason.INVALID_CLIPPING_DECLARATION),
        (_clipping, {"config_digest": "/tmp/x"}, Reason.INVALID_CLIPPING_DECLARATION),
        (_mechanism, {"mechanism": "none"}, Reason.INVALID_MECHANISM_DECLARATION),
        (_mechanism, {"version": "1"}, Reason.INVALID_MECHANISM_DECLARATION),
        (_mechanism, {"accountant": "x"}, Reason.INVALID_MECHANISM_DECLARATION),
        (_mechanism, {"config_digest": None}, Reason.INVALID_MECHANISM_DECLARATION),
    ],
)
def test_declarations_accept_only_closed_codes_and_digests(factory, change, reason):
    values = {
        field: getattr(factory(), field)
        for field in factory().__dataclass_fields__  # type: ignore[attr-defined]
    }
    values.update(change)
    with pytest.raises(FederatedEgressError) as error:
        type(factory())(**values)
    _assert_refused(error, reason)


def test_declarations_accept_code_strings():
    clipping = FederatedEgressClipping(
        scope="per_update", norm="l2", config_digest=CLIP_CONFIG
    )
    assert clipping.scope is FederatedClippingScope.PER_UPDATE
    mechanism = FederatedEgressMechanism(
        mechanism="gaussian", version="v12", accountant="prv", config_digest=LEDGER
    )
    assert mechanism.accountant is FederatedPrivacyAccountant.PRV


@pytest.mark.parametrize(
    ("spent", "band"),
    [
        (0.0, FederatedPrivacySpendBand.NONE),
        (0.1, FederatedPrivacySpendBand.UNDER_QUARTER),
        (0.25, FederatedPrivacySpendBand.QUARTER_TO_UNDER_HALF),
        (0.5, FederatedPrivacySpendBand.HALF_TO_UNDER_THREE_QUARTERS),
        (0.75, FederatedPrivacySpendBand.THREE_QUARTERS_TO_FULL),
        (1.0, FederatedPrivacySpendBand.THREE_QUARTERS_TO_FULL),
    ],
)
def test_privacy_spend_is_reported_as_a_band(spent, band):
    # Power-of-two delta budget keeps the scaled boundaries exact.
    spend = _spend(
        epsilon_spent=spent, delta_spent=spent * 2.0**-17, delta_budget=2.0**-17
    )
    assert spend.epsilon_band is band
    assert spend.delta_band is band


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"epsilon_spent": 1.01}, Reason.PRIVACY_BUDGET_EXCEEDED),
        ({"delta_spent": 2e-5}, Reason.PRIVACY_BUDGET_EXCEEDED),
        ({"epsilon_spent": -0.1}, Reason.INVALID_PRIVACY_SPEND),
        ({"epsilon_spent": float("nan")}, Reason.INVALID_PRIVACY_SPEND),
        ({"epsilon_budget": float("inf")}, Reason.INVALID_PRIVACY_SPEND),
        ({"epsilon_budget": 0.0}, Reason.INVALID_PRIVACY_SPEND),
        ({"delta_budget": 1.0}, Reason.INVALID_PRIVACY_SPEND),
        ({"epsilon_spent": True}, Reason.INVALID_PRIVACY_SPEND),
        ({"delta_spent": "0"}, Reason.INVALID_PRIVACY_SPEND),
        ({"ledger_digest": "ledger"}, Reason.INVALID_PRIVACY_SPEND),
    ],
)
def test_privacy_spend_fails_closed(change, reason):
    with pytest.raises(FederatedEgressError) as error:
        _spend(**change)
    _assert_refused(error, reason)


# ---------------------------------------------------------------------------
# Strict report parsing and forbidden fields

_LEVELS: dict[str, tuple[str | int, ...]] = {
    "top": (),
    "update": ("update",),
    "parameter": ("update", "parameters", 0),
    "clipping": ("clipping",),
    "mechanism": ("mechanism",),
    "privacy-spend": ("privacy_spend",),
}
_CASE_IDS = [case.reason_code for case in FORBIDDEN_FIELD_CASES]


def _target(payload: Any, level: tuple[str | int, ...]) -> dict[str, Any]:
    for key in level:
        payload = payload[key]
    return payload


def test_module_catalog_covers_every_shared_forbidden_field():
    shared = {case.field for case in FORBIDDEN_FIELD_CASES}
    assert shared <= FORBIDDEN_EGRESS_REPORT_FIELDS
    report_fields = set(_report().to_dict())
    assert not report_fields & FORBIDDEN_EGRESS_REPORT_FIELDS


@pytest.mark.parametrize("case", FORBIDDEN_FIELD_CASES, ids=_CASE_IDS)
@pytest.mark.parametrize("level", _LEVELS.values(), ids=_LEVELS.keys())
def test_report_rejects_each_forbidden_field(level, case: ForbiddenFieldCase):
    payload = _report().to_dict()
    original = copy.deepcopy(payload)
    _target(payload, level)[case.field] = copy.deepcopy(case.value)

    for parse in (
        FederatedEgressReport.from_dict,
        lambda p: FederatedEgressReport.from_json(json.dumps(p)),
    ):
        with pytest.raises(FederatedEgressError) as error:
            parse(payload)
        _assert_refused(error, Reason.FORBIDDEN_REPORT_FIELD)
        assert case.field not in str(error.value)
        if case.marker is not None:
            assert case.marker not in str(error.value)

    # Negative control: removing the field restores a valid report.
    del _target(payload, level)[case.field]
    assert payload == original
    assert FederatedEgressReport.from_dict(payload) == _report()


@pytest.mark.parametrize("case", FORBIDDEN_FIELD_CASES, ids=_CASE_IDS)
def test_approval_record_rejects_each_forbidden_field(case: ForbiddenFieldCase):
    payload = _approve(_report()).to_dict()
    payload[case.field] = copy.deepcopy(case.value)
    with pytest.raises(FederatedEgressError) as error:
        FederatedEgressApproval.from_json(json.dumps(payload))
    _assert_refused(error, Reason.FORBIDDEN_REPORT_FIELD)
    if case.marker is not None:
        assert case.marker not in str(error.value)


@pytest.mark.parametrize("level", _LEVELS.values(), ids=_LEVELS.keys())
def test_unknown_non_catalog_fields_are_rejected(level):
    payload = _report().to_dict()
    _target(payload, level)["reviewer_note"] = "fine"
    with pytest.raises(FederatedEgressError) as error:
        FederatedEgressReport.from_dict(payload)
    _assert_refused(error, Reason.INVALID_REPORT)


def _mutate(payload: dict[str, Any], change: str) -> None:
    update = payload["update"]
    if change == "element-count":
        update["parameters"][0]["element_count"] = 7
    elif change == "total":
        update["total_elements"] = 15
    elif change == "parameter-count":
        update["parameter_count"] = 3
    elif change == "unsorted":
        update["parameters"].reverse()
    elif change == "fingerprint":
        update["schema_fingerprint"] = "sha256:" + "9" * 64
    elif change == "unclipped":
        update["clipped"] = False
    elif change == "bool-shape":
        update["parameters"][0]["shape"] = [True, 6]
    elif change == "path-name":
        update["parameters"][0]["name"] = "/placeholder/path"
    elif change == "band":
        payload["privacy_spend"]["epsilon_band"] = "0.12"
    elif change == "sealed-digest":
        payload["sealed_update_digest"] = "/placeholder/path"
    elif change == "schema":
        payload["schema_version"] = "openmed.training.federated_egress_report.v0"
    elif change == "missing":
        del payload["mechanism"]
    else:  # pragma: no cover - parametrization guard
        raise AssertionError(change)


@pytest.mark.parametrize(
    "change",
    [
        "element-count",
        "total",
        "parameter-count",
        "unsorted",
        "fingerprint",
        "unclipped",
        "bool-shape",
        "path-name",
        "band",
        "sealed-digest",
        "schema",
        "missing",
    ],
)
def test_inconsistent_or_value_bearing_reports_are_rejected(change):
    payload = _report().to_dict()
    _mutate(payload, change)
    with pytest.raises(FederatedEgressError) as error:
        FederatedEgressReport.from_dict(payload)
    _assert_refused(error, Reason.INVALID_REPORT)
    assert "placeholder" not in str(error.value)


@pytest.mark.parametrize(
    "text",
    [
        "not json",
        '{"a": 1, "a": 2}',
        '{"schema_version": 1.5}',
        '{"schema_version": NaN}',
        "[]",
        "x" * (1024 * 1024 + 1),
        b"{}",
    ],
    ids=["garbage", "duplicate", "float", "nan", "list", "oversized", "bytes"],
)
def test_report_json_parsing_is_strict(text):
    with pytest.raises(FederatedEgressError) as error:
        FederatedEgressReport.from_json(text)
    _assert_refused(error, Reason.INVALID_REPORT)


def test_report_cannot_be_constructed_directly():
    with pytest.raises((FederatedEgressError, TypeError)):
        FederatedEgressReport("{}")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# End-to-end


@pytest.mark.integration
def test_site_flow_from_metadata_json_to_guarded_submission():
    metadata_json = _metadata().to_json()
    metadata = FederatedUpdateMetadata.from_json(metadata_json, policy=_policy())
    approver = HmacApprover()
    clock = FakeClock()

    report = render_federated_egress_report(
        metadata,
        sealed_update=SEALED,
        round_digest=ROUND,
        clipping=_clipping(),
        mechanism=_mechanism(),
        privacy_spend=_spend(),
    )
    # The coordinator-side report copy is reparsed from the reviewed bytes.
    reviewed = FederatedEgressReport.from_json(report.to_json())
    stored = approve_federated_egress(
        reviewed, approver=approver, clock=clock, valid_for=VALID_FOR
    ).to_json()

    clock.now = START + timedelta(minutes=30)
    transport = Transport()
    guard = FederatedEgressSubmissionGuard(approver=approver, clock=clock)
    guard.submit(
        SEALED,
        report=reviewed,
        approval=FederatedEgressApproval.from_json(stored),
        round_digest=ROUND,
        transport=transport,
    )
    assert transport.sent == [SEALED]
    assert _DIGEST.fullmatch(reviewed.digest)
