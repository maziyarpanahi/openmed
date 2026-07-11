from datetime import datetime

from openmed.core.pipeline import Pipeline
from openmed.processing.outputs import EntityPrediction, PredictionResult


def _model_detector(*surfaces: str):
    def detect(text: str, **kwargs):
        return PredictionResult(
            text=text,
            entities=[
                EntityPrediction(
                    text=surface,
                    label="NAME",
                    start=text.index(surface),
                    end=text.index(surface) + len(surface),
                    confidence=0.95,
                )
                for surface in surfaces
            ],
            model_name=kwargs["model_name"],
            timestamp=datetime.now().isoformat(),
        )

    return detect


def _sections_by_surface(result, text: str) -> dict[str, str | None]:
    return {text[span.start : span.end]: span.section for span in result.spans}


def test_stub_section_detector_stamps_emitted_spans_by_containing_section():
    text = (
        "Preamble Alice.\nHPI: John Doe reported pain.\nAssessment: Jane Roe improved."
    )
    hpi_start = text.index("HPI:")
    assessment_start = text.index("Assessment:")

    def section_detector(text: str):
        return {
            "section_hook": "stub",
            "sections": (
                {"label": "HPI", "start": hpi_start, "end": assessment_start},
                {
                    "label": "Assessment",
                    "start": assessment_start,
                    "end": len(text),
                },
            ),
        }

    result = Pipeline(
        model_detector=_model_detector("Alice", "John Doe", "Jane Roe"),
        section_detector=section_detector,
        use_safety_sweep=False,
    ).run(text, method="mask")

    assert _sections_by_surface(result, text) == {
        "Alice": None,
        "John Doe": "HPI",
        "Jane Roe": "Assessment",
    }
    assert [span.section for span in result.stage("span_arbitration").spans] == [
        None,
        "HPI",
        "Assessment",
    ]


def test_default_section_detector_exposes_canonical_sections_to_pipeline():
    text = (
        "Preamble Alice.\n"
        "Past Medical History: John had childhood asthma.\n"
        "Plan: Jane will call."
    )

    result = Pipeline(
        model_detector=_model_detector("Alice", "John", "Jane"),
        use_safety_sweep=False,
    ).run(text, method="mask")

    assert _sections_by_surface(result, text) == {
        "Alice": None,
        "John": "past_medical_history",
        "Jane": "plan",
    }
    section_stage = result.stage("doc_type_section").metadata
    assert section_stage["section_hook"] == "detect_sections"
    assert [section["label"] for section in section_stage["sections"]] == [
        "unsectioned",
        "past_medical_history",
        "plan",
    ]


def test_unavailable_section_hook_leaves_pipeline_output_unchanged():
    text = "Patient John Doe visited."

    unavailable_result = Pipeline(
        model_detector=_model_detector("John Doe"),
        section_detector=lambda text: {"section_hook": "unavailable"},
        use_safety_sweep=False,
    ).run(text, method="mask")
    empty_sections_result = Pipeline(
        model_detector=_model_detector("John Doe"),
        section_detector=lambda text: {"section_hook": "unavailable", "sections": ()},
        use_safety_sweep=False,
    ).run(text, method="mask")

    assert unavailable_result.spans == empty_sections_result.spans
    assert unavailable_result.redacted_text == empty_sections_result.redacted_text
    assert unavailable_result.spans[0].section is None


def test_section_assignment_uses_span_start_with_half_open_boundaries():
    text = "History details. Jane boundary patient."
    boundary = text.index("Jane")

    def section_detector(text: str):
        return {
            "section_hook": "stub",
            "sections": (
                {"label": "History", "start": 0, "end": boundary},
                {"label": "Plan", "start": boundary, "end": len(text)},
            ),
        }

    crossing_result = Pipeline(
        model_detector=_model_detector("details. Jane"),
        section_detector=section_detector,
        use_safety_sweep=False,
    ).run(text, method="mask")
    boundary_result = Pipeline(
        model_detector=_model_detector("Jane"),
        section_detector=section_detector,
        use_safety_sweep=False,
    ).run(text, method="mask")

    assert crossing_result.spans[0].section == "History"
    assert boundary_result.spans[0].section == "Plan"
