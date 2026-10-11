"""End-to-end pipeline model selection from automatic language routing."""

from __future__ import annotations

from datetime import datetime

import pytest

from openmed.core.errors import InputError
from openmed.core.language_router import LanguageRouter
from openmed.core.pii_i18n import DEFAULT_PII_MODELS
from openmed.core.pipeline import LanguageRoute, Pipeline
from openmed.processing.outputs import PredictionResult


@pytest.mark.parametrize(
    ("source", "script"),
    [
        ("မြန်မာ", "Myanmar"),
        ("ខ្មែរ", "Khmer"),
        ("ລາວ", "Lao"),
        ("සිංහල", "Sinhala"),
        ("ދިވެހި", "Thaana"),
        ("བོད", "Tibetan"),
    ],
)
def test_auto_pipeline_rejects_unsupported_letters_before_model_inference(
    source, script
):
    calls = []
    router = LanguageRouter(use_optional_lid=False)
    decision = router.route(source)
    assert decision.runs[0].script == script
    assert decision.runs[0].reason == "unsupported_script"
    assert decision.confidence < 0.99
    with pytest.raises(InputError, match="unsupported script") as exc:
        Pipeline(
            lang="auto",
            language_router=router,
            model_detector=lambda *a, **k: calls.append(a),
        ).run(source)
    assert source not in str(exc.value)
    assert not calls
    explicit = LanguageRouter(use_optional_lid=False, fallback_pack="en")
    assert (
        Pipeline(lang="auto", language_router=explicit)
        .stage2_language_script(source)
        .lang
        == "en"
    )


@pytest.mark.parametrize(
    ("source", "lang", "identifier", "phone"),
    [
        (
            "ผู้ป่วย เลขประจำตัวประชาชน 1101700203450 โทร 0812345678 เกิด 15.01.1980",
            "th",
            "1101700203450",
            "0812345678",
        ),
        (
            "תעודת זהות 123456782 טלפון 0501234567 נולד ב-15.01.1980",
            "he",
            "123456782",
            "0501234567",
        ),
    ],
)
def test_thai_hebrew_auto_packs_detect_structured_phi_without_model(
    source, lang, identifier, phone
):
    def empty(text, **kwargs):
        return PredictionResult(
            text=text, entities=[], model_name="synthetic", timestamp="2026-01-01"
        )

    result = Pipeline(lang="auto", model_detector=empty).run(source)
    assert result.route.lang == lang
    for value in (identifier, phone, "15.01.1980"):
        assert value not in result.redacted_text
        assert any(source[span.start : span.end] == value for span in result.spans)
    assert any(span.canonical_label == "PHONE" for span in result.spans)


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
        (
            "रुग्ण वैशाली देशमुख स्थिर आहे.",
            "mr",
            DEFAULT_PII_MODELS["mr"],
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
