from __future__ import annotations

import logging
import socket
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest

from openmed.core.offline import HF_OFFLINE_ENV_VARS
from openmed.interop import presidio_recognizer as pr
from openmed.processing.outputs import EntityPrediction

NOTE = "Patient Jane Roe (age 54) can be reached at 555-0142 after discharge."
NAME = "Jane Roe"
PHONE = "555-0142"

# Entity names Presidio's predefined recognizers emit (presidio-analyzer 2.2.x).
PRESIDIO_ENTITIES = {
    "CREDIT_CARD",
    "CRYPTO",
    "DATE_TIME",
    "EMAIL_ADDRESS",
    "IBAN_CODE",
    "IP_ADDRESS",
    "LOCATION",
    "MEDICAL_LICENSE",
    "NRP",
    "PERSON",
    "PHONE_NUMBER",
    "URL",
    "US_BANK_NUMBER",
    "US_DRIVER_LICENSE",
    "US_ITIN",
    "US_PASSPORT",
    "US_SSN",
}


def _span(label: str, surface: str, confidence: float = 0.95) -> EntityPrediction:
    start = NOTE.index(surface)
    return EntityPrediction(
        text=surface,
        label=label,
        confidence=confidence,
        start=start,
        end=start + len(surface),
    )


def fake_extractor(text: str) -> SimpleNamespace:
    assert text == NOTE
    return SimpleNamespace(
        entities=[
            _span("first_name", NAME),
            _span("phone_number", PHONE),
            _span("age", "54"),  # no Presidio equivalent
            _span("last_name", "Roe", confidence=0.2),  # below threshold
        ]
    )


class FakeBase:
    def __init__(self, supported_entities, name=None, supported_language="en"):
        self.supported_entities = supported_entities
        self.name = name
        self.supported_language = supported_language
        self.load()


@dataclass
class FakeResult:
    entity_type: str
    start: int
    end: int
    score: float
    analysis_explanation: Any = None
    recognition_metadata: dict = field(default_factory=dict)


def _fake_recognizer(**kwargs):
    return pr.create_openmed_recognizer(
        extractor=fake_extractor,
        base_cls=FakeBase,
        result_cls=FakeResult,
        **kwargs,
    )


def test_maps_openmed_spans_to_presidio_entities_with_offsets():
    results = _fake_recognizer().analyze(NOTE, ["PERSON", "PHONE_NUMBER"])

    assert [(r.entity_type, NOTE[r.start : r.end]) for r in results] == [
        ("PERSON", NAME),
        ("PHONE_NUMBER", PHONE),
    ]
    assert all(
        r.recognition_metadata["recognizer_name"] == "OpenMedRecognizer"
        for r in results
    )


def test_only_requested_entities_are_returned():
    recognizer = _fake_recognizer()

    assert [r.entity_type for r in recognizer.analyze(NOTE, ["PHONE_NUMBER"])] == [
        "PHONE_NUMBER"
    ]
    assert recognizer.analyze(NOTE, ["US_SSN"]) == []
    person_only = _fake_recognizer(supported_entities=["PERSON"])
    assert [r.entity_type for r in person_only.analyze(NOTE, ["PHONE_NUMBER"])] == []


def test_supported_entities_are_existing_presidio_names():
    recognizer = _fake_recognizer()

    assert set(recognizer.supported_entities) <= PRESIDIO_ENTITIES
    assert set(pr.SUPPORTED_ENTITIES) <= PRESIDIO_ENTITIES
    with pytest.raises(ValueError, match="OPENMED_PHI"):
        _fake_recognizer(supported_entities=["OPENMED_PHI"])


def test_supported_entities_match_installed_presidio_registry():
    presidio_analyzer = pytest.importorskip("presidio_analyzer")
    registry = presidio_analyzer.RecognizerRegistry()
    registry.load_predefined_recognizers()

    assert set(pr.SUPPORTED_ENTITIES) <= set(registry.get_supported_entities())


def test_default_extractor_is_lazy_local_only_and_works_with_network_blocked(
    monkeypatch,
):
    for name in ("OPENMED_OFFLINE", *HF_OFFLINE_ENV_VARS):
        monkeypatch.delenv(name, raising=False)
    calls: list[dict[str, Any]] = []

    def cached_extract_pii(text, **kwargs):
        calls.append(kwargs)
        return fake_extractor(text)

    monkeypatch.setattr("openmed.core.pii.extract_pii", cached_extract_pii)
    monkeypatch.setattr("openmed.core.models.ModelLoader", lambda config: object())

    def no_network(*args, **kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)

    recognizer = pr.create_openmed_recognizer(base_cls=FakeBase, result_cls=FakeResult)
    assert calls == []  # load() binds the extractor without running the model

    results = recognizer.analyze(NOTE, ["PERSON"])

    assert [r.entity_type for r in results] == ["PERSON"]
    assert calls[0]["model_name"] == pr.DEFAULT_MODEL
    assert calls[0]["config"].local_only is True


def test_analyzer_engine_returns_person_and_phone_without_leaking_text(caplog):
    pytest.importorskip("presidio_analyzer")
    spacy = pytest.importorskip("spacy")
    from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
    from presidio_analyzer.nlp_engine import NlpArtifacts, NlpEngine

    blank = spacy.blank("en")

    class BlankNlpEngine(NlpEngine):
        def load(self):
            pass

        def is_loaded(self):
            return True

        def process_text(self, text, language):
            doc = blank(text)
            return NlpArtifacts(
                entities=[],
                tokens=doc,
                tokens_indices=[token.idx for token in doc],
                lemmas=[token.text for token in doc],
                nlp_engine=self,
                language=language,
            )

        def process_batch(self, texts, language, batch_size=1, n_process=1, **kwargs):
            for text in texts:
                yield text, self.process_text(text, language)

        def is_stopword(self, word, language):
            return False

        def is_punct(self, word, language):
            return False

        def get_supported_entities(self):
            return []

        def get_supported_languages(self):
            return ["en"]

    recognizer = pr.create_openmed_recognizer(extractor=fake_extractor)
    registry = RecognizerRegistry(recognizers=[recognizer])
    analyzer = AnalyzerEngine(nlp_engine=BlankNlpEngine(), registry=registry)

    with caplog.at_level(logging.DEBUG):
        results = analyzer.analyze(
            text=NOTE, language="en", return_decision_process=True
        )

    found = sorted((r.entity_type, NOTE[r.start : r.end]) for r in results)
    assert found == [("PERSON", NAME), ("PHONE_NUMBER", PHONE)]
    for result in results:
        explanation = str(result.analysis_explanation.to_dict())
        assert pr.DEFAULT_MODEL in explanation
        assert NAME not in explanation and PHONE not in explanation
    assert NAME not in caplog.text and PHONE not in caplog.text


def test_openmed_recognizer_attribute_is_presidio_subclass():
    presidio_analyzer = pytest.importorskip("presidio_analyzer")

    assert issubclass(pr.OpenMedRecognizer, presidio_analyzer.EntityRecognizer)
