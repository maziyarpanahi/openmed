"""Synthetic identifier recall, leakage, span integrity and negative controls."""

import hashlib
import json
import threading

import pytest

from openmed.multimodal.generated_text_guard import (
    AssetIdentifierDigest,
    GeneratedTextGuard,
    GuardPolicy,
    PHILabel,
    PHISpan,
    identifier_digest,
)
from openmed.multimodal.provider_result import (
    ProviderResultEnvelope,
    ProviderResultError,
)

ASSET = "a" * 64


def evidence(value, label=PHILabel.NAME, asset=ASSET):
    return AssetIdentifierDigest(
        asset, label, identifier_digest(value), len("".join(value.split()).casefold())
    )


@pytest.mark.parametrize(
    "value,output,label",
    [
        ("Mira Vale", "MIRA   VALE", PHILabel.NAME),
        ("Mira Vale", "m i r a\tv a l e", PHILabel.NAME),
        ("ZX1234", "z x 1 2 3 4", PHILabel.MRN),
        ("2031-04-19", "2031 - 04 - 19", PHILabel.DATE),
        ("Éva", "É V A", PHILabel.NAME),
        ("李明", "李 明", PHILabel.NAME),
        ("AB１２", "a b 1 2", PHILabel.MRN),
    ],
)
def test_direct_identifier_safety_sweep(value, output, label, caplog):
    guard = GeneratedTextGuard(lambda text: (), detector_version="synthetic-v1")
    result = guard.release(
        f"Label: {output}. Review required.",
        asset_digest=ASSET,
        identifiers=[evidence(value, label)],
    )
    assert result.text == "Label: [REDACTED]. Review required."
    assert result.envelope.guard_metadata["decision"] == "redact"
    assert result.envelope.count_metadata["detection_count"] == 1
    assert output not in result.report_json + repr(result) + caplog.text
    assert value not in result.report_json
    assert (
        result.envelope.output_digest
        == hashlib.sha256(result.text.encode()).hexdigest()
    )


@pytest.mark.parametrize("action", ["redact", "withhold", "abstain"])
def test_policy_actions(action):
    result = GeneratedTextGuard(
        lambda _: [PHISpan(0, 9, PHILabel.NAME)], policy=GuardPolicy(action=action)
    ).release("Mira Vale rest", asset_digest=ASSET)
    assert result.text == ("[REDACTED] rest" if action == "redact" else None)
    assert result.envelope.guard_metadata["decision"] == action
    if action != "redact":
        assert result.envelope.abstention_code.value == "phi_uncertainty"
        assert result.envelope.output_digest is None


def test_clean_output_deterministic_and_negative_control():
    guard = GeneratedTextGuard(lambda _: (), detector_version="fake-v1")
    first = guard.release(
        "Review the label locally.",
        asset_digest=ASSET,
        identifiers=[evidence("Mira Vale")],
    )
    second = guard.release(
        "Review the label locally.",
        asset_digest=ASSET,
        identifiers=[evidence("Mira Vale")],
    )
    assert first == second
    assert first.text == "Review the label locally."
    assert first.envelope.guard_metadata["decision"] == "pass"
    assert ProviderResultEnvelope.from_json(first.envelope.to_json()) == first.envelope
    assert (
        first.envelope.guard_metadata["report_digest"]
        == hashlib.sha256(first.report_json.encode()).hexdigest()
    )


def test_detector_error_never_exposes_message(caplog, capsys):
    def broken(text):
        raise RuntimeError("Mira Vale " + text)

    result = GeneratedTextGuard(broken).release("ZX1234", asset_digest=ASSET)
    assert result.text is None
    assert result.envelope.abstention_code.value == "phi_uncertainty"
    assert "ZX1234" not in repr(result) + caplog.text + capsys.readouterr().err
    assert "Mira Vale" not in result.report_json


def test_timeout_and_bounded_outstanding_detector():
    entered = threading.Event()
    finish = threading.Event()
    calls = []

    def delayed(text):
        calls.append(1)
        entered.set()
        finish.wait(2)
        return ()

    guard = GeneratedTextGuard(delayed, policy=GuardPolicy(timeout_seconds=0.01))
    try:
        first = guard.release("Mira Vale", asset_digest=ASSET)
        assert entered.is_set()
        assert first.text is None
        assert guard.release("Mira Vale", asset_digest=ASSET).text is None
        assert len(calls) == 1
    finally:
        finish.set()


@pytest.mark.parametrize(
    "findings",
    [
        [PHISpan(-1, 5, PHILabel.NAME)],
        [PHISpan(0, 100, PHILabel.NAME)],
        [PHISpan(0, 0, PHILabel.NAME)],
        [PHISpan(True, 2, PHILabel.NAME)],
        [PHISpan(0, 2, "Mira Vale")],
        [PHISpan(0, 2, PHILabel.NAME, False)],
        ["Mira Vale"],
        [PHISpan(0, 1, PHILabel.NAME)] * 4097,
    ],
)
def test_invalid_or_unresolved_findings_abstain(findings):
    result = GeneratedTextGuard(lambda _: findings).release(
        "Mira Vale", asset_digest=ASSET
    )
    assert result.text is None
    assert result.envelope.abstention_code.value == "phi_uncertainty"


def test_merged_span_integrity_and_repeated_surrogate():
    guard = GeneratedTextGuard(
        lambda _: [PHISpan(0, 4, PHILabel.NAME), PHISpan(3, 9, PHILabel.IDENTIFIER)]
    )
    result = guard.release(
        "Mira Vale; MIRA VALE", asset_digest=ASSET, identifiers=[evidence("Mira Vale")]
    )
    assert result.text == "[REDACTED]; [REDACTED]"


@pytest.mark.parametrize(
    "identifiers",
    [
        [evidence("Mira Vale", asset="b" * 64)],
        ["Mira Vale"],
        [evidence("Mira Vale")] * 129,
    ],
)
def test_untrusted_asset_evidence_abstains(identifiers):
    result = GeneratedTextGuard(lambda _: ()).release(
        "Mira Vale", asset_digest=ASSET, identifiers=identifiers
    )
    assert result.text is None
    assert "Mira Vale" not in result.report_json


def test_missing_detector_and_oversized_generation_abstain():
    assert (
        GeneratedTextGuard(None).release("Mira Vale", asset_digest=ASSET).text is None
    )
    assert (
        GeneratedTextGuard(lambda _: ()).release("a" * 8193, asset_digest=ASSET).text
        is None
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"action": "Mira Vale"},
        {"timeout_seconds": float("nan")},
        {"timeout_seconds": 0},
        {"max_characters": True},
    ],
)
def test_policy_rejects_invalid_values_without_echo(changes):
    with pytest.raises(ValueError, match="^guard policy is invalid$"):
        GuardPolicy(**changes)


def test_metadata_rejects_payload_injection_and_conflicting_outcome():
    result = GeneratedTextGuard(lambda _: ()).release("Clean", asset_digest=ASSET)
    payload = result.envelope.to_dict()
    payload["guard_metadata"]["decision"] = "Mira Vale"
    with pytest.raises(ProviderResultError, match="guard decision is invalid"):
        ProviderResultEnvelope.from_dict(payload)
    payload = result.envelope.to_dict()
    payload["guard_metadata"]["decision"] = "withhold"
    with pytest.raises(ProviderResultError, match="conflicts"):
        ProviderResultEnvelope.from_dict(payload)
    counts = json.loads(result.report_json)["label_counts"]
    assert set(counts) == {label.value for label in PHILabel}


def test_finding_limit_and_notice_are_bound_to_release():
    result = GeneratedTextGuard(lambda _: ()).release(
        "a" * 4097, asset_digest=ASSET, identifiers=[evidence("a")]
    )
    assert result.text is None
    assert result.envelope.guard_metadata["decision"] == "abstain"
    assert result.requires_reviewer_confirmation is True
    assert "Not diagnostic" in result.non_diagnostic_notice
    assert (
        json.loads(result.report_json)["use_policy"] == "non_diagnostic_review_required"
    )


def test_detector_base_exception_is_suppressed(caplog, capsys):
    def broken(_):
        raise SystemExit("Mira Vale")

    result = GeneratedTextGuard(broken).release("Mira Vale", asset_digest=ASSET)
    assert result.text is None
    assert "Mira Vale" not in caplog.text + capsys.readouterr().err
