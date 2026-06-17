from __future__ import annotations

from datetime import datetime

import pytest

from openmed.core.arbitration import MODE_HIGH_RECALL_UNION
from openmed.core.cascade import R3_ACCURATE, CascadeRouter
from openmed.core.labels import CANONICAL_LABELS
from openmed.core.pipeline import Pipeline
from openmed.core.policy import CANONICAL_POLICY_NAMES, load_policy
from openmed.core.pii import deidentify
from openmed.core.schemas.span import OpenMedSpan, hmac_text_hash
from openmed.processing.outputs import EntityPrediction, PredictionResult


def _prediction(
    text: str,
    entities: list[EntityPrediction] | None = None,
) -> PredictionResult:
    return PredictionResult(
        text=text,
        entities=entities or [],
        model_name="stub",
        timestamp=datetime.now().isoformat(),
    )


def _entity(text: str, surface: str, label: str, score: float) -> EntityPrediction:
    start = text.index(surface)
    return EntityPrediction(
        text=surface,
        label=label,
        start=start,
        end=start + len(surface),
        confidence=score,
    )


def _patch_extract(monkeypatch, entity: EntityPrediction) -> None:
    from openmed.core import pii

    def fake_extract(text: str, *args: object, **kwargs: object) -> PredictionResult:
        return _prediction(text, [entity])

    monkeypatch.setattr(pii, "extract_pii", fake_extract)


def _span(label: str = "PERSON", score: float = 0.95) -> OpenMedSpan:
    return OpenMedSpan(
        doc_id="doc-1",
        start=0,
        end=8,
        text_hash=hmac_text_hash(f"{label}:{score}", "test-secret"),
        entity_type=label,
        canonical_label=label,
        score=score,
        detector="model:tiny",
    )


def test_all_policy_literals_load_and_gdpr_alias_resolves():
    for name in CANONICAL_POLICY_NAMES:
        profile = load_policy(name)

        assert profile.name == name
        assert set(profile.actions) == set(CANONICAL_LABELS)

    assert load_policy("gdpr").name == "gdpr_pseudonymization"


def test_deidentify_without_policy_preserves_default_output(monkeypatch):
    text = "Patient John Doe"
    _patch_extract(monkeypatch, _entity(text, "John Doe", "NAME", 0.95))

    result = deidentify(text, method="mask", use_safety_sweep=False)

    assert result.deidentified_text == "Patient [NAME]"
    assert result.mapping is None
    assert "reversible_id" not in result.to_dict()["pii_entities"][0]


def test_unknown_policy_raises_before_detection():
    with pytest.raises(ValueError, match="unknown policy"):
        deidentify("Patient John Doe", policy="not_a_policy")


def test_strict_no_leak_forces_union_sweep_and_accurate_cascade(monkeypatch):
    text = "Visited Paris"
    _patch_extract(monkeypatch, _entity(text, "Paris", "LOCATION", 0.2))

    result = deidentify(text, policy="strict_no_leak", use_safety_sweep=False)

    assert result.deidentified_text == "Visited [LOCATION]"

    profile = load_policy("strict_no_leak")
    assert "R3" in profile.forced_cascade_tiers

    pipeline_result = Pipeline(
        model_detector=lambda text, **kwargs: _prediction(text),
        policy="strict_no_leak",
        use_safety_sweep=False,
    ).run("No identifiers")
    assert pipeline_result.stage("safety_sweep").metadata["enabled"] is True
    assert (
        pipeline_result.audit_record["policy"]["arbitration_mode"]
        == MODE_HIGH_RECALL_UNION
    )

    router = CascadeRouter(
        tiny_detector=lambda text, **kwargs: [_span(score=0.2)],
        accurate_detector=lambda text, **kwargs: [],
    )
    cascade_result = Pipeline(
        cascade_router=router,
        policy="strict_no_leak",
        use_safety_sweep=False,
    ).run("Patient record")
    routes = [
        route["route"]
        for route in cascade_result.stage("deterministic_detectors").metadata["routes"]
    ]
    assert R3_ACCURATE in routes


def test_clinical_minimal_redaction_keeps_quasi_identifier_that_strict_masks(
    monkeypatch,
):
    text = "Follow up in Paris"
    entity = _entity(text, "Paris", "LOCATION", 0.95)
    _patch_extract(monkeypatch, entity)

    clinical = deidentify(
        text,
        policy="clinical_minimal_redaction",
        use_safety_sweep=False,
    )
    strict = deidentify(text, policy="strict_no_leak", use_safety_sweep=False)

    assert clinical.deidentified_text == text
    assert strict.deidentified_text == "Follow up in [LOCATION]"


def test_gdpr_pseudonymization_retains_mapping_and_reversible_id(monkeypatch):
    text = "Patient John Doe"
    _patch_extract(monkeypatch, _entity(text, "John Doe", "NAME", 0.95))

    result = deidentify(
        text,
        policy="gdpr_pseudonymization",
        use_safety_sweep=False,
        seed=42,
    )

    assert result.mapping is not None
    assert "John Doe" in result.mapping.values()
    assert result.pii_entities[0].reversible_id is not None
    assert result.pii_entities[0].reversible_id.startswith("rev_")
    assert (
        result.to_dict()["pii_entities"][0]["reversible_id"]
        == result.pii_entities[0].reversible_id
    )
    assert result.metadata["policy"]["name"] == "gdpr_pseudonymization"


def test_hipaa_safe_harbor_applies_safe_harbor_action_map(monkeypatch):
    text = "Visit on someday"
    _patch_extract(monkeypatch, _entity(text, "someday", "DATE", 0.95))

    result = deidentify(
        text,
        policy="hipaa_safe_harbor",
        use_safety_sweep=False,
    )

    assert load_policy("hipaa_safe_harbor").action_for("DATE") == "mask"
    assert result.deidentified_text == "Visit on [DATE]"
    assert result.pii_entities[0].metadata["policy_action"]["action"] == "mask"
