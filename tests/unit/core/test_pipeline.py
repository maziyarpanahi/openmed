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

    assert stage9.metadata["redacted_chars_after"] >= stage9.metadata[
        "redacted_chars_before"
    ]
    assert stage9.metadata["spans_added"] == 1
    assert result.redacted_text == "Patient [NAME] email [email]"
