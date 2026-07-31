from __future__ import annotations

import json
from datetime import datetime

from openmed.core.custom_recognizer import CUSTOM_DENY_DETECTOR, CustomRecognizer
from openmed.core.pipeline import Pipeline
from openmed.processing.outputs import EntityPrediction, PredictionResult


def _empty_prediction(
    text: str,
    model_name: str = "stub",
    **_kwargs,
) -> PredictionResult:
    return PredictionResult(
        text=text,
        entities=[],
        model_name=model_name,
        timestamp=datetime.now().isoformat(),
    )


def test_custom_recognizer_loads_json_and_yaml_paths(tmp_path):
    payload = {
        "deny": {
            "terms": [{"term": "Ward Phoenix", "label": "LOCATION"}],
            "patterns": [{"pattern": r"\bSTUDY-\d+\b", "label": "ID_NUM"}],
        },
        "allow": {"terms": ["Mercy Trial"]},
    }
    json_path = tmp_path / "custom.json"
    yaml_path = tmp_path / "custom.yaml"
    json_path.write_text(json.dumps(payload), encoding="utf-8")
    yaml_path.write_text(
        r"""
deny:
  terms:
    - term: Ward Phoenix
      label: LOCATION
  patterns:
    - pattern: '\bSTUDY-\d+\b'
      label: ID_NUM
allow:
  terms:
    - Mercy Trial
""",
        encoding="utf-8",
    )

    text = "Ward Phoenix enrolled STUDY-123, not Mercy Trial."

    assert len(CustomRecognizer.from_config(json_path).detect_entities(text)) == 2
    assert len(CustomRecognizer.from_config(yaml_path).detect_entities(text)) == 2


def test_deny_term_and_regex_emit_custom_provenance_without_raw_metadata():
    text = "Ward Phoenix enrolled STUDY-123."
    recognizer = CustomRecognizer.from_config(
        {
            "deny_terms": [{"term": "ward phoenix", "label": "LOCATION"}],
            "deny_patterns": [{"pattern": r"\bSTUDY-\d+\b", "label": "ID_NUM"}],
        }
    )

    entities = recognizer.detect_entities(text)

    assert [(entity.text, entity.label) for entity in entities] == [
        ("Ward Phoenix", "LOCATION"),
        ("STUDY-123", "ID_NUM"),
    ]
    for entity in entities:
        metadata = entity.metadata or {}
        custom = metadata["custom_recognizer"]
        assert metadata["detector"] == CUSTOM_DENY_DETECTOR
        assert custom["text_hash"].startswith("hmac-sha256:")
        assert entity.text not in repr(metadata)


def test_extract_pii_adds_custom_deny_and_allow_suppresses_model(monkeypatch):
    import openmed

    text = "Ward Phoenix enrolled Mercy Trial and STUDY-123."

    def fake_analyze_text(text, **kwargs):
        start = text.index("Mercy Trial")
        return PredictionResult(
            text=text,
            entities=[
                EntityPrediction(
                    text="Mercy Trial",
                    label="ORGANIZATION",
                    start=start,
                    end=start + len("Mercy Trial"),
                    confidence=0.99,
                )
            ],
            model_name=kwargs["model_name"],
            timestamp=datetime.now().isoformat(),
        )

    monkeypatch.setattr(openmed, "analyze_text", fake_analyze_text)

    result = openmed.extract_pii(
        text,
        model_name="stub",
        use_smart_merging=False,
        custom_recognizer={
            "deny_terms": [{"term": "Ward Phoenix", "label": "LOCATION"}],
            "deny_patterns": [{"pattern": r"\bSTUDY-\d+\b", "label": "ID_NUM"}],
            "allow_terms": ["Mercy Trial"],
        },
    )

    assert [(entity.text, entity.label) for entity in result.entities] == [
        ("Ward Phoenix", "LOCATION"),
        ("STUDY-123", "ID_NUM"),
    ]
    assert result.metadata["custom_recognizer"]["spans_suppressed_by_allow"] == 1


def test_deidentify_redacts_custom_denies_and_leaves_allowed_model_span(monkeypatch):
    from openmed.core import pii

    text = "Ward Phoenix enrolled Mercy Trial and STUDY-123."

    def fake_extract_pii(text, *args, **kwargs):
        start = text.index("Mercy Trial")
        model_name = kwargs.get("model_name") or args[0]
        return PredictionResult(
            text=text,
            entities=[
                EntityPrediction(
                    text="Mercy Trial",
                    label="ORGANIZATION",
                    start=start,
                    end=start + len("Mercy Trial"),
                    confidence=0.99,
                )
            ],
            model_name=model_name,
            timestamp=datetime.now().isoformat(),
        )

    monkeypatch.setattr(pii, "extract_pii", fake_extract_pii)

    result = pii.deidentify(
        text,
        model_name="stub",
        use_safety_sweep=False,
        custom_recognizer={
            "deny_terms": [{"term": "Ward Phoenix", "label": "LOCATION"}],
            "deny_patterns": [{"pattern": r"\bSTUDY-\d+\b", "label": "ID_NUM"}],
            "allow_terms": ["Mercy Trial"],
        },
    )

    assert result.deidentified_text == "[LOCATION] enrolled Mercy Trial and [ID_NUM]."
    assert [entity.sources for entity in result.pii_entities] == [
        [CUSTOM_DENY_DETECTOR],
        [CUSTOM_DENY_DETECTOR],
    ]


def test_pipeline_custom_deny_uses_normalized_offsets_and_remaps_to_original():
    text = "Patient John   Doe visited"

    result = Pipeline(
        model_detector=_empty_prediction,
        use_safety_sweep=False,
        custom_recognizer={
            "deny_terms": [{"term": "John Doe", "label": "PERSON"}],
        },
    ).run(text, method="mask")

    deterministic_spans = result.stage("deterministic_detectors").spans
    entity = result.deidentification_result.pii_entities[0]

    assert deterministic_spans[-1].detector == CUSTOM_DENY_DETECTOR
    assert result.normalized_text == "Patient John Doe visited"
    span = deterministic_spans[-1]
    assert result.normalized_text[span.start : span.end] == "John Doe"
    assert entity.text == "John   Doe"
    assert result.redacted_text == "Patient [PERSON] visited"
