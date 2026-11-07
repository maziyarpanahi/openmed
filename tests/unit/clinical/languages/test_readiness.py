"""Synthetic readiness metadata gates, independent of model availability."""

from dataclasses import FrozenInstanceError

import pytest

from openmed.clinical.languages.readiness import (
    ClinicalLanguageReadiness,
    ClinicalLanguageReadinessRegistry,
    LanguageEvidence,
)


def evidence(task="entities", tier="nano"):
    return tuple(
        LanguageEvidence(kind, task, "a" * 64)
        for kind in ("lexicon", "fixture", "evaluation", "native_review")
    ) + (LanguageEvidence("device", task, "b" * 64, tier),)


def entry(**overrides):
    args = dict(
        language="en-US",
        version="1.0.0",
        scripts=("Latin",),
        tasks=("entities",),
        model_tiers=("nano",),
        reviewer_status="approved",
        evidence=evidence(),
    )
    args.update(overrides)
    return ClinicalLanguageReadiness(**args)


def test_complete_evidence_supports_declaration_only():
    assert entry().state == "supported"
    assert entry().missing_evidence() == ()


@pytest.mark.parametrize(
    "kind,state",
    [
        ("lexicon", "experimental"),
        ("fixture", "experimental"),
        ("evaluation", "experimental"),
        ("native_review", "experimental"),
        ("device", "preview"),
    ],
)
def test_each_evidence_requirement_is_independent(kind, state):
    row = entry(evidence=tuple(e for e in evidence() if e.kind != kind))
    assert row.state == state
    assert kind + "_missing" in row.missing_evidence()


def test_locale_registration_alone_never_implies_support():
    row = ClinicalLanguageReadiness("ne", "1.0.0", ("Devanagari",), ("entities",))
    assert row.state == "experimental"


def test_approved_status_without_review_evidence_is_experimental():
    row = entry(evidence=tuple(e for e in evidence() if e.kind != "native_review"))
    assert row.state == "experimental"


def test_pending_reviewer_blocks_support():
    assert entry(reviewer_status="pending").state == "experimental"


def test_limitations_cap_fully_evidenced_entry_at_preview():
    assert entry(limitations=("limited_case_mix",)).state == "preview"


def test_all_tasks_and_device_tiers_require_evidence():
    row = entry(tasks=("entities", "relations"), model_tiers=("nano", "small"))
    assert row.state == "experimental"
    assert "device_missing" in row.missing_evidence()


def test_aliases_reordering_and_frozen_inventory_are_stable():
    a, b = entry(), entry(language="EN_us", evidence=tuple(reversed(evidence())))
    assert a == b
    assert (
        ClinicalLanguageReadinessRegistry([a]).to_json()
        == ClinicalLanguageReadinessRegistry([b]).to_json()
    )
    with pytest.raises(FrozenInstanceError):
        a.language = "de"
    with pytest.raises(ValueError, match="duplicate"):
        ClinicalLanguageReadinessRegistry([a, b])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"language": "/private/example"},
        {"scripts": ("raw note",)},
        {"reviewer_status": "Dr Synthetic"},
        {"limitations": ("private raw text",)},
        {"evidence": (LanguageEvidence("fixture", "unknown", "a" * 64),)},
        {"evidence": evidence() + evidence()},
        {"tasks": "entities"},
        {"version": "latest"},
    ],
)
def test_invalid_inventory_uses_controlled_errors(kwargs):
    with pytest.raises(ValueError) as exc:
        entry(**kwargs)
    assert "/private/example" not in str(exc.value)
    assert "private raw text" not in str(exc.value)


@pytest.mark.parametrize(
    "args",
    [
        ("fixture", "entities", "raw record"),
        ("device", "entities", "a" * 64),
        ("fixture", "entities", "a" * 64, "nano"),
        ("native_review", "raw record", "a" * 64),
    ],
)
def test_invalid_evidence_never_echoes_values(args):
    with pytest.raises(ValueError) as exc:
        LanguageEvidence(*args)
    assert "raw record" not in str(exc.value)


def test_arbitrary_iterables_are_consumed_with_a_hard_bound():
    consumed = []

    def infinite_codes():
        while True:
            consumed.append(1)
            yield "synthetic"

    with pytest.raises(ValueError):
        entry(tasks=infinite_codes())
    assert len(consumed) == 129
    consumed.clear()

    def infinite_evidence():
        while True:
            consumed.append(1)
            yield LanguageEvidence("fixture", "entities", "a" * 64)

    with pytest.raises(ValueError):
        entry(evidence=infinite_evidence())
    assert len(consumed) == 1025
