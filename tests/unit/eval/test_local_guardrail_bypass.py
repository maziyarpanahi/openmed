from __future__ import annotations

import json
import socket
from collections.abc import Callable

import pytest

from openmed.eval.local_guardrail_bypass import (
    DEFAULT_LOCAL_GUARDRAIL_POLICY,
    LOCAL_GUARDRAIL_BYPASS_SCHEMA_VERSION,
    GuardrailAttackTarget,
    LocalGenerationEvidence,
    LocalGuardrailBypassError,
    LocalGuardrailPolicy,
    assert_local_guardrail_bypass,
    evaluate_local_guardrail_bypass,
    generate_local_guardrail_challenges,
)

SOURCE_CANARY = "Synthetic source: Rowan Sample, record ZX-4107."


def _safe_evidence(
    *,
    policy: LocalGuardrailPolicy = DEFAULT_LOCAL_GUARDRAIL_POLICY,
    citation_ids: tuple[str, ...] = ("evidence-001",),
    refused: bool = True,
    review_required: bool = True,
    validation_performed: bool = True,
    output_text: str = "Unsupported claim refused. [evidence-001] Review required.",
) -> LocalGenerationEvidence:
    return LocalGenerationEvidence(
        effective_policy=policy,
        citation_ids=citation_ids,
        unsupported_request_refused=refused,
        review_required=review_required,
        validation_performed=validation_performed,
        output_text=output_text,
    )


def _safe_runner(challenge, policy):
    del challenge
    return _safe_evidence(policy=policy)


def test_challenge_generation_is_deterministic_and_covers_every_guard() -> None:
    first = generate_local_guardrail_challenges()
    second = generate_local_guardrail_challenges()

    assert first is second
    assert tuple(challenge.case_id for challenge in first) == (
        "instruction-omit-citations",
        "instruction-override-refusal",
        "instruction-suppress-review",
        "instruction-replace-policy",
        "instruction-disable-validation",
        "instruction-recover-source",
    )
    assert {challenge.target for challenge in first} == set(GuardrailAttackTarget)
    assert all(challenge.synthetic for challenge in first)
    assert all(
        "UNTRUSTED NOTE CONTENT" in challenge.deidentified_note for challenge in first
    )
    assert all(SOURCE_CANARY not in challenge.deidentified_note for challenge in first)


def test_safe_local_generation_passes_offline_and_is_byte_stable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def deny_network(*args, **kwargs):
        del args, kwargs
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "socket", deny_network)

    first = evaluate_local_guardrail_bypass(
        _safe_runner,
        pre_deidentified_text=SOURCE_CANARY,
    )
    second = evaluate_local_guardrail_bypass(
        _safe_runner,
        pre_deidentified_text=SOURCE_CANARY,
    )

    assert first.passed is True
    assert first.bypass_count == 0
    assert first.bypass_rate == 0.0
    assert first.to_json() == second.to_json()
    assert first.schema_version == LOCAL_GUARDRAIL_BYPASS_SCHEMA_VERSION
    assert_local_guardrail_bypass(first)


@pytest.mark.parametrize(
    ("target", "mutate", "expected_violation"),
    [
        (
            GuardrailAttackTarget.CITATIONS,
            lambda policy: _safe_evidence(policy=policy, citation_ids=()),
            "citations_missing",
        ),
        (
            GuardrailAttackTarget.REFUSAL,
            lambda policy: _safe_evidence(policy=policy, refused=False),
            "refusal_missing",
        ),
        (
            GuardrailAttackTarget.REVIEW,
            lambda policy: _safe_evidence(policy=policy, review_required=False),
            "review_missing",
        ),
        (
            GuardrailAttackTarget.POLICY,
            lambda policy: _safe_evidence(
                policy=LocalGuardrailPolicy(policy_id="unrestricted-policy")
            ),
            "policy_changed",
        ),
        (
            GuardrailAttackTarget.VALIDATION,
            lambda policy: _safe_evidence(
                policy=policy,
                validation_performed=False,
            ),
            "validation_disabled",
        ),
        (
            GuardrailAttackTarget.PRE_DEIDENTIFICATION_TEXT,
            lambda policy: _safe_evidence(
                policy=policy,
                output_text=f"Recovered original: {SOURCE_CANARY}",
            ),
            "pre_deidentification_text_exposed",
        ),
    ],
)
def test_each_targeted_bypass_fails_the_gate(
    target: GuardrailAttackTarget,
    mutate: Callable[[LocalGuardrailPolicy], LocalGenerationEvidence],
    expected_violation: str,
) -> None:
    def attacked_runner(challenge, policy):
        if challenge.target is target:
            return mutate(policy)
        return _safe_evidence(policy=policy)

    report = evaluate_local_guardrail_bypass(
        attacked_runner,
        pre_deidentified_text=SOURCE_CANARY,
    )
    failed_cases = [case for case in report.cases if case.bypassed]

    assert report.passed is False
    assert report.bypass_count == 1
    assert len(failed_cases) == 1
    assert failed_cases[0].target is target
    assert expected_violation in failed_cases[0].violations


def test_every_attack_checks_every_guard_not_only_its_named_target() -> None:
    def disable_validation_during_citation_attack(challenge, policy):
        return _safe_evidence(
            policy=policy,
            validation_performed=(
                challenge.target is not GuardrailAttackTarget.CITATIONS
            ),
        )

    report = evaluate_local_guardrail_bypass(
        disable_validation_during_citation_attack,
        pre_deidentified_text=SOURCE_CANARY,
    )

    citation_case = next(
        case for case in report.cases if case.target is GuardrailAttackTarget.CITATIONS
    )
    assert citation_case.violations == ("validation_disabled",)


def test_runner_receives_only_deidentified_challenge_and_policy() -> None:
    seen_notes: list[str] = []

    def inspecting_runner(challenge, policy):
        seen_notes.append(challenge.deidentified_note)
        return _safe_evidence(policy=policy)

    evaluate_local_guardrail_bypass(
        inspecting_runner,
        pre_deidentified_text=SOURCE_CANARY,
    )

    assert len(seen_notes) == len(generate_local_guardrail_challenges())
    assert all(SOURCE_CANARY not in note for note in seen_notes)


def test_report_repr_json_and_gate_error_never_expose_sensitive_values() -> None:
    citation_canary = "citation-for-rowan-zx-4107"

    def leaking_runner(challenge, policy):
        del challenge
        return _safe_evidence(
            policy=policy,
            citation_ids=(citation_canary,),
            output_text=SOURCE_CANARY,
        )

    report = evaluate_local_guardrail_bypass(
        leaking_runner,
        pre_deidentified_text=SOURCE_CANARY,
    )
    serialized = report.to_json()

    assert SOURCE_CANARY not in serialized
    assert SOURCE_CANARY not in repr(report)
    assert citation_canary not in serialized
    assert citation_canary not in repr(report)
    payload = json.loads(serialized)
    assert "source_digest" in payload
    assert all("output_digest" in case for case in payload["cases"])
    assert all("output_text" not in case for case in payload["cases"])

    with pytest.raises(LocalGuardrailBypassError) as error:
        assert_local_guardrail_bypass(report)
    assert SOURCE_CANARY not in str(error.value)
    assert citation_canary not in str(error.value)
    assert "pre_deidentification_text_exposed" in str(error.value)


def test_runner_exception_is_sanitized_and_fails_closed() -> None:
    def failing_runner(challenge, policy):
        del challenge, policy
        raise RuntimeError(f"backend echoed {SOURCE_CANARY}")

    report = evaluate_local_guardrail_bypass(
        failing_runner,
        pre_deidentified_text=SOURCE_CANARY,
    )

    assert report.passed is False
    assert report.bypass_count == len(generate_local_guardrail_challenges())
    assert all(case.violations == ("generator_error",) for case in report.cases)
    assert SOURCE_CANARY not in report.to_json()
    assert SOURCE_CANARY not in repr(report)


def test_non_evidence_runner_result_fails_closed_without_echoing_it() -> None:
    sentinel = "unexpected-sensitive-return-value"

    report = evaluate_local_guardrail_bypass(
        lambda challenge, policy: sentinel,  # type: ignore[return-value]
        pre_deidentified_text=SOURCE_CANARY,
    )

    assert all(case.violations == ("invalid_evidence",) for case in report.cases)
    assert sentinel not in report.to_json()
    assert sentinel not in repr(report)


@pytest.mark.parametrize("source", ["", None, 123])
def test_invalid_source_fails_without_echoing_values(source: object) -> None:
    with pytest.raises(
        LocalGuardrailBypassError,
        match="invalid pre-de-identification source",
    ) as error:
        evaluate_local_guardrail_bypass(
            _safe_runner,
            pre_deidentified_text=source,  # type: ignore[arg-type]
        )
    if source != "":
        assert str(source) not in str(error.value)


def test_sensitive_evidence_fields_are_hidden_from_repr() -> None:
    evidence = _safe_evidence(
        citation_ids=("citation-sensitive-sentinel",),
        output_text="output-sensitive-sentinel",
    )

    rendered = repr(evidence)
    assert "citation-sensitive-sentinel" not in rendered
    assert "output-sensitive-sentinel" not in rendered
