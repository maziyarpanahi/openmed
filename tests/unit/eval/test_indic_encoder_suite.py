from __future__ import annotations

import hashlib

from openmed.eval.metrics import EvalSpan
from openmed.eval.suites.indic_encoder import (
    SYNTHETIC_INDIC_GOLD,
    run_indic_encoder_recall_delta,
)
from openmed.ner.families.base import EncoderOutput


class _SyntheticEncoder:
    tokenizer = object()

    def encode(self, text: str, *, max_length: int = 512) -> EncoderOutput:
        del max_length
        return EncoderOutput(
            tokenizer_outputs={
                "input_ids": [[1]],
                "attention_mask": [[1]],
            },
            offset_mapping=((0, len(text)),),
            last_hidden_state=[[[1.0, 0.0]]],
            text_sha256=hashlib.sha256(text.encode()).hexdigest(),
        )


class _SyntheticPerfectAdapter:
    encoder = _SyntheticEncoder()

    def predict_entities(self, text, *, language, encoding):
        assert encoding.hidden_size == 2
        fixture = next(item for item in SYNTHETIC_INDIC_GOLD if item.text == text)
        assert fixture.language == language
        return [
            EvalSpan(
                start=span.start,
                end=span.end,
                label=span.label,
                text=span.text,
                language=language,
            )
            for span in fixture.gold_spans
        ]


def test_encoder_backed_synthetic_recall_beats_regex_with_zero_leakage():
    report = run_indic_encoder_recall_delta(_SyntheticPerfectAdapter())

    assert report.gold_entities == 6
    assert report.encoder_recall == 1.0
    assert report.encoder_recall > report.regex_recall
    assert report.recall_delta > 0.0
    assert report.entity_leakage == 0
    assert report.encoder_leakage_rate == 0.0
