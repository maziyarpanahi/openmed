"""Shared synthetic speech fields consumed by the native Swift adapter tests."""

import json
from pathlib import Path

import pytest

from openmed.multimodal.streaming.speech.providers import (
    FinalizedSegment,
    LanguageConfidence,
    PartialHypothesis,
    ProviderProvenance,
    SpeechProviderError,
    TokenTime,
)

FIXTURE = (
    Path(__file__).resolve().parents[2] / "fixtures/parity/speech_contract_v1.json"
)


def test_swift_speech_vectors_preserve_shared_units_and_protected_text():
    corpus = json.loads(FIXTURE.read_text(encoding="utf-8"))
    row = corpus["segment"]
    tokens = tuple(TokenTime(**token) for token in row["token_times"])
    segment = FinalizedSegment(**(row | {"token_times": tokens}))
    partial = PartialHypothesis(revision=2, text=row["text"], token_times=tokens)
    assert segment.duration_ms == 400
    assert segment.to_dict()["text_length"] == 6
    assert partial.to_dict()["token_count"] == 2
    assert "Café" not in repr(segment)
    for patch in corpus["invalid_tokens"]:
        with pytest.raises(SpeechProviderError):
            TokenTime(**(row["token_times"][0] | patch))


def test_swift_provenance_vectors_match_python_validation():
    corpus = json.loads(FIXTURE.read_text(encoding="utf-8"))

    def validate(row):
        ProviderProvenance(
            provider_id=row["provider_id"],
            provider_revision=row["provider_revision"],
            model_fingerprint=row["model_fingerprint"],
            requires_network=False,
        )
        LanguageConfidence(language=row["language"], confidence_ppm=0)

    validate(corpus["provenance"])
    for row in corpus["invalid_provenance"]:
        with pytest.raises(SpeechProviderError):
            validate(corpus["provenance"] | {row["field"]: row["value"]})
