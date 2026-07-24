"""End-to-end pipeline model selection from automatic language routing."""

from __future__ import annotations

from datetime import datetime

import pytest

from openmed.core.pii_i18n import DEFAULT_PII_MODELS
from openmed.core.pipeline import LanguageRoute, Pipeline
from openmed.processing.outputs import PredictionResult


def test_language_route_preserves_existing_positional_metadata_contract():
    route = LanguageRoute("en", "Latin", "local-model", {"existing": True})

    assert route.metadata == {"existing": True}
    assert route.decision is None


@pytest.mark.parametrize(
    ("text", "expected_language", "expected_model"),
    (
        (
            "患者王芳因发热入院。",
            "zh",
            DEFAULT_PII_MODELS["zh"],
        ),
        (
            "रोगी अनिता बुखार के कारण भर्ती हुई।",
            "hi",
            DEFAULT_PII_MODELS["hi"],
        ),
    ),
)
def test_pipeline_auto_route_selects_document_language_pack_and_model(
    text,
    expected_language,
    expected_model,
):
    detector_calls = []

    def detector(routed_text, **kwargs):
        detector_calls.append((routed_text, kwargs))
        return PredictionResult(
            text=routed_text,
            entities=[],
            model_name=kwargs["model_name"],
            timestamp=datetime.now().isoformat(),
        )

    result = Pipeline(
        lang="auto",
        model_detector=detector,
        use_safety_sweep=False,
    ).run(text)

    assert result.route.lang == expected_language
    assert result.route.model_name == expected_model
    assert result.route.decision is not None
    assert result.route.decision.dominant_pack.code == expected_language
    assert detector_calls[0][1]["lang"] == expected_language
    assert detector_calls[0][1]["model_name"] == expected_model
    assert result.stage("language_script").metadata["dominant_pack"] == (
        expected_language
    )
    assert result.stage("language_script").metadata["runs"]
