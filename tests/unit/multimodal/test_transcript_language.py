"""Synthetic, offline privacy controls for finalized transcript routing."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from openmed.core.language_pack_catalog import LANGUAGE_PACK_ADAPTERS
from openmed.core.language_router import LanguagePrediction
from openmed.multimodal.transcript_language import (
    TranscriptLanguageError,
    TranscriptLanguageRouter,
    TranscriptPHISpan,
    transcript_language_report,
)


class SyntheticIdentifier:
    name = "synthetic"

    def __init__(self, labels=None, confidence=0.99):
        self.labels = labels or {}
        self.confidence = confidence

    def identify(self, text, candidates):
        code = self.labels.get(text)
        return None if code is None else LanguagePrediction(code, self.confidence)


def router(labels, *, installed=("en", "es"), detectors=None, confidence=0.99):
    return TranscriptLanguageRouter(
        installed_packs=[
            LANGUAGE_PACK_ADAPTERS.registry.get(code) for code in installed
        ],
        detectors=detectors or dict.fromkeys(installed, lambda _: ()),
        text_identifier=SyntheticIdentifier(labels, confidence),
        candidate_languages=("en", "es", "fr"),
    )


def route(instance, text, language="en", confidence=0.99, **kwargs):
    return instance.route(
        segment_index=4,
        text=text,
        provider_language=LanguagePrediction(language, confidence),
        **kwargs,
    )


@pytest.mark.parametrize("text,language", [("Hello Ada", "en"), ("Hola Lucía", "es")])
def test_supported_language_uses_only_matching_detector(text, language):
    calls = []

    def detector(value):
        calls.append(value)
        return [TranscriptPHISpan(value.index(" ") + 1, len(value))]

    instance = router(
        dict.fromkeys(text.split(), language),
        detectors={language: detector},
        installed=(language,),
    )
    decision = route(instance, text, language)
    assert decision.status == "supported"
    assert calls == [text]
    assert decision.phi_spans == (TranscriptPHISpan(text.index(" ") + 1, len(text)),)
    with pytest.raises(TranscriptLanguageError, match="^review_required$"):
        decision.reviewed_text()
    assert decision.reviewed_text(reviewer_confirmed=True) == text.split()[
        0
    ] + " " + "█" * len(text.split()[1])


def test_code_switch_tiles_source_and_masks_names_ids_and_neutral_tokens():
    text = "🙂 My name is Ada 123-45-6789. Mi nombre es Lucía 987654321.  "
    labels = {word: "en" for word in ("My", "name", "is", "Ada")}
    labels.update({word: "es" for word in ("Mi", "nombre", "es", "Lucía")})
    calls = []

    def detector(code, identifiers):
        def detect(value):
            calls.append((code, value))
            return [
                TranscriptPHISpan(value.index(token), value.index(token) + len(token))
                for token in identifiers
            ]

        return detect

    instance = router(
        labels,
        detectors={
            "en": detector("en", ("Ada", "123-45-6789")),
            "es": detector("es", ("Lucía", "987654321")),
        },
    )
    decision = route(instance, text, "en-US")
    assert decision.status == "mixed"
    assert decision.provider_tag == "en-US"
    boundary = text.index("Mi")
    assert [(run.start, run.end, run.tag) for run in decision.runs] == [
        (0, boundary, "en"),
        (boundary, len(text), "es"),
    ]
    assert calls == [("en", text[:boundary]), ("es", text[boundary:])]
    assert len(decision.reviewed_text(reviewer_confirmed=True)) == len(text)
    for token in ("Ada", "123-45-6789", "Lucía", "987654321"):
        start = text.index(token)
        assert TranscriptPHISpan(start, start + len(token)) in decision.phi_spans
        assert token not in decision.reviewed_text(reviewer_confirmed=True)
        assert token not in repr(decision)
        assert token not in json.dumps(decision.to_dict())
        assert token not in json.dumps(transcript_language_report([decision]))
    report = transcript_language_report([decision])
    assert report == {
        "status_counts": {"mixed": 1},
        "reason_counts": {"language_routed": 1},
        "tag_counts": {"en": 1, "es": 1},
        "provider_tag_counts": {"en-US": 1},
        "confidence_bucket_counts": {"high": 1},
    }


@pytest.mark.parametrize(
    "language,confidence,labels,installed,reason,status",
    [
        (
            "es",
            0.79,
            {"Hola": "es"},
            ("en", "es"),
            "provider_confidence_low",
            "uncertain",
        ),
        (
            "fr",
            0.99,
            {"Bonjour": "fr"},
            ("en", "es"),
            "phi_pack_unavailable",
            "unsupported",
        ),
        (
            "en",
            0.99,
            {"Hola": "es"},
            ("en", "es"),
            "language_disagreement",
            "uncertain",
        ),
        ("en", 0.99, {}, ("en", "es"), "text_language_uncertain", "uncertain"),
    ],
)
def test_withheld_segments_never_reach_detectors_or_reviewed_drafts(
    language, confidence, labels, installed, reason, status
):
    calls = []
    instance = router(
        labels,
        installed=installed,
        detectors=dict.fromkeys(installed, lambda text: calls.append(text)),
    )
    decision = route(instance, next(iter(labels), "Unknown"), language, confidence)
    assert (decision.reason_code, decision.status) == (reason, status)
    assert calls == []
    with pytest.raises(TranscriptLanguageError, match="^segment_withheld$"):
        decision.reviewed_text(reviewer_confirmed=True)


def test_uninstalled_part_of_mixed_segment_withholds_whole_segment():
    calls = []
    decision = route(
        router(
            {"Hello": "en", "Hola": "es"},
            installed=("en",),
            detectors={"en": lambda value: calls.append(value)},
        ),
        "Hello Hola",
    )
    assert decision.status == "unsupported"
    assert decision.reason_code == "phi_pack_unavailable"
    assert calls == []


@pytest.mark.parametrize(
    "confidence,expected",
    [(0.79, "text_confidence_low"), (0.8, "language_routed"), (0.9, "language_routed")],
)
def test_text_confidence_threshold(confidence, expected):
    decision = route(router({"Hello": "en"}, confidence=confidence), "Hello")
    assert decision.reason_code == expected


def test_dominant_provider_must_match_dominant_text_not_merely_one_minor_token():
    decision = route(
        router({"Hello": "en", "Hola": "es", "nombre": "es"}), "Hello Hola nombre"
    )
    assert decision.reason_code == "language_disagreement"


def test_partial_missing_provider_and_empty_segments_fail_closed():
    instance = router({"Hello": "en"})
    assert (
        route(instance, "Hello", finalized=False).reason_code == "segment_not_finalized"
    )
    assert route(instance, "123").reason_code == "text_language_uncertain"
    assert route(instance, "").reason_code == "text_language_uncertain"
    assert (
        instance.route(
            segment_index=0, text="Hello", provider_language=None
        ).reason_code
        == "provider_language_missing"
    )


@pytest.mark.parametrize(
    "detector,reason",
    [
        (lambda text: [TranscriptPHISpan(0, len(text) + 1)], "detector_result_invalid"),
        (lambda text: ["synthetic-private-payload"], "detector_result_invalid"),
        (lambda text: None, "detector_failed"),
    ],
)
def test_invalid_detector_results_fail_closed(detector, reason):
    decision = route(
        router({"Hello": "en"}, detectors={"en": detector, "es": lambda _: ()}), "Hello"
    )
    assert decision.reason_code == reason
    with pytest.raises(TranscriptLanguageError):
        decision.reviewed_text(reviewer_confirmed=True)


def test_backend_exceptions_are_not_copied_to_decisions(caplog):
    class BrokenIdentifier(SyntheticIdentifier):
        def identify(self, text, candidates):
            raise ValueError("synthetic-private-payload")

    instance = router({})
    instance._identifier = BrokenIdentifier()
    decision = route(instance, "Hello")
    assert decision.reason_code == "text_identifier_failed"
    assert (
        "synthetic-private-payload"
        not in repr(decision) + json.dumps(decision.to_dict()) + caplog.text
    )


def test_detector_failure_after_success_withholds_all_output():
    def broken(text):
        raise RuntimeError("synthetic-private-payload")

    decision = route(
        router(
            {"Hello": "en", "Hola": "es"}, detectors={"en": lambda _: (), "es": broken}
        ),
        "Hello Hola",
    )
    assert decision.reason_code == "detector_failed"
    assert decision.phi_spans == ()
    with pytest.raises(TranscriptLanguageError):
        decision.reviewed_text(reviewer_confirmed=True)


def test_overlapping_identifier_spans_are_merged_without_offset_changes():
    decision = route(
        router(
            {"Hello": "en"},
            detectors={
                "en": lambda _: [TranscriptPHISpan(0, 3), TranscriptPHISpan(2, 5)],
                "es": lambda _: (),
            },
        ),
        "Hello",
    )
    assert decision.phi_spans == (TranscriptPHISpan(0, 5),)
    assert decision.reviewed_text(reviewer_confirmed=True) == "█████"


@pytest.mark.parametrize(
    "tag", ["en-x-private", "synthetic-payload", "en_US", "en\n", "en-John"]
)
def test_invalid_tags_have_value_free_errors(tag):
    with pytest.raises(TranscriptLanguageError, match="^invalid_language_tag$"):
        route(router({}), "Hello", tag)


@pytest.mark.parametrize("confidence", [float("inf"), True, -0.1, 1.1])
def test_thresholds_must_be_valid_probabilities(confidence):
    with pytest.raises(TranscriptLanguageError):
        TranscriptLanguageRouter(
            installed_packs=[],
            detectors={},
            text_identifier=SyntheticIdentifier(),
            candidate_languages=("en",),
            confidence_threshold=confidence,
        )


def test_local_identifier_invalid_confidence_and_tag_fail_closed():
    class InvalidIdentifier(SyntheticIdentifier):
        def identify(self, text, candidates):
            return replace(LanguagePrediction("en", 0.9), language="en-x-secret")

    instance = router({})
    instance._identifier = InvalidIdentifier()
    assert route(instance, "Hello").reason_code == "text_identifier_failed"


def test_pack_catalog_does_not_mean_detector_is_installed():
    with pytest.raises(TranscriptLanguageError, match="^invalid_installed_detectors$"):
        TranscriptLanguageRouter(
            installed_packs=[LANGUAGE_PACK_ADAPTERS.registry.get("es")],
            detectors={},
            text_identifier=SyntheticIdentifier(),
            candidate_languages=("en", "es"),
        )


def test_shared_python_swift_synthetic_language_decisions():
    path = Path(__file__).parents[2] / "fixtures/multimodal/transcript_language.json"
    for case in json.loads(path.read_text()):

        def detector(text):
            return [
                TranscriptPHISpan(text.index(name), text.index(name) + len(name))
                for name in ("Ada", "Lucía")
                if name in text
            ]

        instance = router(
            case["labels"],
            confidence=case["text_confidence"],
            detectors={"en": detector, "es": detector},
        )
        prediction = (
            None
            if case["provider_tag"] is None
            else LanguagePrediction(case["provider_tag"], case["confidence"])
        )
        decision = instance.route(
            segment_index=5, text=case["text"], provider_language=prediction
        )
        assert decision.to_dict() == case["expected"]
        if case["output"] is None:
            with pytest.raises(TranscriptLanguageError):
                decision.reviewed_text(reviewer_confirmed=True)
        else:
            assert decision.reviewed_text(reviewer_confirmed=True) == case["output"]
