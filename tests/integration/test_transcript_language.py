"""Offline integration with existing token LID and registered pack metadata."""

import pytest

from openmed.core.language_pack_catalog import LANGUAGE_PACK_ADAPTERS
from openmed.core.language_router import LanguagePrediction
from openmed.multimodal.transcript_language import (
    TranscriptLanguageRouter,
    TranscriptPHISpan,
)


@pytest.mark.integration
def test_repeated_synthetic_code_switch_preserves_ids_and_review_boundary():
    class LocalIdentifier:
        name = "synthetic-local"

        def identify(self, text, candidates):
            return LanguagePrediction(
                "en" if text in {"Patient", "Ada"} else "es", 0.99
            )

    def detect(value):
        return [
            TranscriptPHISpan(value.index(name), value.index(name) + len(name))
            for name in ("Ada", "Lucía")
            if name in value
        ]

    instance = TranscriptLanguageRouter(
        installed_packs=[
            LANGUAGE_PACK_ADAPTERS.registry.get(code) for code in ("en", "es")
        ],
        detectors={"en": detect, "es": detect},
        text_identifier=LocalIdentifier(),
        candidate_languages=("en", "es", "fr"),
    )
    text = "Patient Ada. Paciente Lucía."
    decisions = [
        instance.route(
            segment_index=0, text=text, provider_language=LanguagePrediction("en", 0.99)
        )
        for _ in range(20)
    ]
    assert len({repr(decision.to_dict()) for decision in decisions}) == 1
    for decision in decisions:
        assert decision.status == "mixed"
        assert (
            decision.reviewed_text(reviewer_confirmed=True)
            == "Patient ███. Paciente █████."
        )
        assert decision.notice.startswith("Non-diagnostic")
