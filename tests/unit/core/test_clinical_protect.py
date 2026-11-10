from __future__ import annotations

from datetime import datetime

from openmed.core.clinical_protect import (
    add_protected_terms,
    load_bundled_terms,
    protect_spans,
)
from openmed.core.config import OpenMedConfig
from openmed.core.pipeline import Pipeline
from openmed.processing.outputs import EntityPrediction, PredictionResult


def _prediction(text: str, *entities: EntityPrediction) -> PredictionResult:
    return PredictionResult(
        text=text,
        entities=list(entities),
        model_name="stub",
        timestamp=datetime.now().isoformat(),
    )


def _entity(text: str, surface: str, label: str) -> EntityPrediction:
    start = text.index(surface)
    return EntityPrediction(
        text=surface,
        label=label,
        start=start,
        end=start + len(surface),
        confidence=0.99,
    )


def test_protect_spans_drops_clinical_term_but_keeps_real_name():
    text = "Parkinson was discussed with John Doe."
    spans = [
        _entity(text, "Parkinson", "PERSON"),
        _entity(text, "John Doe", "PERSON"),
    ]

    protected = protect_spans(spans, text)

    assert [(span.text, span.label) for span in protected] == [("John Doe", "PERSON")]


def test_pipeline_leaves_protected_term_unredacted_and_redacts_name():
    text = "Parkinson was discussed with John Doe."

    def model_detector(text: str, **kwargs):
        return _prediction(
            text,
            _entity(text, "Parkinson", "NAME"),
            _entity(text, "John Doe", "NAME"),
        )

    result = Pipeline(model_detector=model_detector, use_safety_sweep=False).run(text)

    assert result.redacted_text == "Parkinson was discussed with [NAME]."
    assert [
        (entity.text, entity.label)
        for entity in result.deidentification_result.pii_entities
    ] == [
        ("John Doe", "NAME"),
    ]
    assert (
        result.stage("span_arbitration").metadata["clinical_protection"][
            "suppressed_spans"
        ]
        == 1
    )


def test_protection_never_suppresses_direct_identifier_span():
    text = "SSN 123-45-6789 was reviewed after Parkinson screening."
    spans = [
        _entity(text, "123-45-6789", "PERSON"),
        _entity(text, "Parkinson", "PERSON"),
    ]

    protected = protect_spans(
        spans,
        text,
        extra_terms=["123-45-6789"],
    )

    assert [(span.text, span.label) for span in protected] == [
        ("123-45-6789", "PERSON")
    ]


def test_add_protected_terms_extends_runtime_list():
    text = "Cardioxel was charted near Jane Doe."
    spans = [
        _entity(text, "Cardioxel", "LOCATION"),
        _entity(text, "Jane Doe", "PERSON"),
    ]

    add_protected_terms(["Cardioxel"])

    protected = protect_spans(spans, text)

    assert [(span.text, span.label) for span in protected] == [("Jane Doe", "PERSON")]


def test_config_can_override_bundled_terms():
    text = "Parkinson and Ward Alpha were flagged."

    def model_detector(text: str, **kwargs):
        return _prediction(
            text,
            _entity(text, "Parkinson", "NAME"),
            _entity(text, "Ward Alpha", "LOCATION"),
        )

    result = Pipeline(
        config=OpenMedConfig(
            clinical_protect_terms=["Ward Alpha"],
            clinical_protect_use_builtin=False,
        ),
        model_detector=model_detector,
        use_safety_sweep=False,
    ).run(text)

    assert result.redacted_text == "[NAME] and Ward Alpha were flagged."


def test_bundled_terms_load_without_restricted_license_content():
    terms = load_bundled_terms()

    assert "parkinson" in terms
    assert "metformin" in terms
    assert "emergency department" in terms
    assert not {"umls", "snomed", "rxnorm", "cpt"} & terms
