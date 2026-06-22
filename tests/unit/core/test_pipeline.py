import unicodedata
from datetime import datetime

from openmed.core.pipeline import Pipeline
from openmed.processing.outputs import EntityPrediction, PredictionResult


def _empty_prediction(text: str, model_name: str = "stub") -> PredictionResult:
    return PredictionResult(
        text=text,
        entities=[],
        model_name=model_name,
        timestamp=datetime.now().isoformat(),
    )


def test_normalized_offsets_remap_combining_characters_to_original_positions():
    text = "Cafe\u0301  MRN"
    document = Pipeline().stage1_normalize(text)

    assert document.normalized_text == "Café MRN"

    original_e = text.index("e")
    original_combining = original_e + 1
    normalized_e = document.normalized_text.index("é")

    assert document.offset_map.original_index_to_normalized(original_e) == normalized_e
    assert (
        document.offset_map.original_index_to_normalized(original_combining)
        == normalized_e
    )
    assert document.offset_map.normalized_index_to_original(normalized_e) == original_e
    assert document.offset_map.normalized_span_to_original_offsets(
        normalized_e,
        normalized_e + 1,
    ) == (original_e, original_combining + 1)


def test_normalized_span_round_trips_nfc_length_change_to_original_surface():
    text = "Patient Jose\u0301 Garcia visited"
    document = Pipeline().stage1_normalize(text)
    normalized_surface = "José Garcia"
    normalized_start = document.normalized_text.index(normalized_surface)
    normalized_end = normalized_start + len(normalized_surface)

    original_start, original_end = (
        document.offset_map.normalized_span_to_original_offsets(
            normalized_start,
            normalized_end,
        )
    )
    original_surface = text[original_start:original_end]

    assert original_surface == "Jose\u0301 Garcia"
    assert unicodedata.normalize("NFC", original_surface) == normalized_surface
    assert document.offset_map.original_span_to_normalized(
        original_start,
        original_end,
    ) == (normalized_start, normalized_end)


def test_normalized_span_round_trips_collapsed_whitespace_to_original_surface():
    text = "Patient John   Doe visited"
    document = Pipeline().stage1_normalize(text)
    normalized_surface = "John Doe"
    normalized_start = document.normalized_text.index(normalized_surface)
    normalized_end = normalized_start + len(normalized_surface)

    original_start, original_end = (
        document.offset_map.normalized_span_to_original_offsets(
            normalized_start,
            normalized_end,
        )
    )
    original_surface = text[original_start:original_end]

    assert original_surface == "John   Doe"
    assert Pipeline().stage1_normalize(original_surface).normalized_text == (
        normalized_surface
    )
    assert document.offset_map.original_span_to_normalized(
        original_start,
        original_end,
    ) == (normalized_start, normalized_end)


def test_deidentification_redacts_original_nfc_changed_entity_surface():
    text = "Patient Jose\u0301 Garcia visited"

    def model_detector(text, **kwargs):
        entity_text = "José Garcia"
        start = text.index(entity_text)
        return PredictionResult(
            text=text,
            entities=[
                EntityPrediction(
                    text=entity_text,
                    label="NAME",
                    start=start,
                    end=start + len(entity_text),
                    confidence=0.95,
                )
            ],
            model_name=kwargs["model_name"],
            timestamp=datetime.now().isoformat(),
        )

    result = Pipeline(
        model_detector=model_detector,
        use_safety_sweep=False,
    ).run(text, method="mask")
    entity = result.deidentification_result.pii_entities[0]

    assert result.deidentification_result.original_text == text
    assert entity.original_text == "Jose\u0301 Garcia"
    assert entity.text == "Jose\u0301 Garcia"
    assert (entity.start, entity.end) == (
        text.index("Jose\u0301 Garcia"),
        text.index("Jose\u0301 Garcia") + len("Jose\u0301 Garcia"),
    )
    assert result.redacted_text == "Patient [NAME] visited"


def test_deidentification_redacts_original_collapsed_whitespace_entity_surface():
    text = "Patient John   Doe visited"

    def model_detector(text, **kwargs):
        entity_text = "John Doe"
        start = text.index(entity_text)
        return PredictionResult(
            text=text,
            entities=[
                EntityPrediction(
                    text=entity_text,
                    label="NAME",
                    start=start,
                    end=start + len(entity_text),
                    confidence=0.95,
                )
            ],
            model_name=kwargs["model_name"],
            timestamp=datetime.now().isoformat(),
        )

    result = Pipeline(
        model_detector=model_detector,
        use_safety_sweep=False,
    ).run(text, method="mask")
    entity = result.deidentification_result.pii_entities[0]

    assert result.deidentification_result.original_text == text
    assert entity.original_text == "John   Doe"
    assert entity.text == "John   Doe"
    assert (entity.start, entity.end) == (
        text.index("John   Doe"),
        text.index("John   Doe") + len("John   Doe"),
    )
    assert result.redacted_text == "Patient [NAME] visited"


def test_luhn_valid_mrn_is_detected_before_model_stage_runs():
    model_calls = []

    def model_detector(text, **kwargs):
        model_calls.append(("model", text, kwargs))
        return _empty_prediction(text, model_name=kwargs["model_name"])

    text = "MRN: 4111111111111111"
    result = Pipeline(
        model_detector=model_detector,
        use_safety_sweep=False,
    ).run(text)

    deterministic_stage = result.stage("deterministic_detectors")
    model_stage = result.stage("fast_pii_model")

    assert len(model_calls) == 1
    assert model_calls[0][1] == text
    assert model_stage.spans == ()
    assert deterministic_stage.spans
    assert deterministic_stage.stage < model_stage.stage
    assert deterministic_stage.spans[0].detector.startswith("rules:")
    assert deterministic_stage.spans[0].detector == "rules:mrn_luhn"


def test_stage9_safety_sweep_only_increases_redacted_character_count():
    text = "Patient John Doe email jane.patient@example.com"

    def model_detector(text, **kwargs):
        return PredictionResult(
            text=text,
            entities=[
                EntityPrediction(
                    text="John Doe",
                    label="NAME",
                    start=8,
                    end=16,
                    confidence=0.95,
                )
            ],
            model_name=kwargs["model_name"],
            timestamp=datetime.now().isoformat(),
        )

    result = Pipeline(model_detector=model_detector).run(text, method="mask")
    stage9 = result.stage("safety_sweep")

    assert (
        stage9.metadata["redacted_chars_after"]
        >= stage9.metadata["redacted_chars_before"]
    )
    assert stage9.metadata["spans_added"] == 1
    assert result.redacted_text == "Patient [NAME] email [email]"
