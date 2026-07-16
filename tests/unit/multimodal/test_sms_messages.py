"""Tests for SMS-scale de-identification and export handling."""

from __future__ import annotations

import csv
import io
import json
import math
from pathlib import Path

import openmed
from openmed.core.pipeline import Pipeline
from openmed.interop import assert_redacted
from openmed.multimodal.sms_messages import (
    SHORT_TEXT,
    deidentify_short_text,
    iter_redacted_sms_records,
    redact_sms_csv,
    redact_sms_json,
)
from openmed.processing import BatchProcessor
from openmed.processing.outputs import PredictionResult

FIXTURES = Path(__file__).parent / "fixtures" / "sms"


def _empty_detector(text: str, **_: object) -> PredictionResult:
    return PredictionResult(
        text=text,
        entities=[],
        model_name="synthetic-empty-detector",
        timestamp="2026-01-01T00:00:00Z",
    )


def _expected_replacement_only_text(result: object) -> str:
    original = result.original_text
    cursor = 0
    parts: list[str] = []
    for entity in sorted(result.pii_entities, key=lambda item: item.start):
        parts.append(original[cursor : entity.start])
        parts.append(entity.redacted_text or "")
        cursor = entity.end
    parts.append(original[cursor:])
    return "".join(parts)


def _synthetic_corpus(
    size: int = 600,
) -> tuple[list[dict[str, str]], list[tuple[str, ...]]]:
    names = (
        "Amina Njeri",
        "Nomsa Dlamini",
        "Achieng Otieno",
        "Nabirye Namusoke",
        "Thandi Mokoena",
    )
    languages = (
        ("HELLO", "MAMA"),
        ("HABARI", "MAMA"),
        ("SAWUBONA", "SISI"),
    )
    rows: list[dict[str, str]] = []
    seeded: list[tuple[str, ...]] = []
    for index in range(size):
        greeting, honorific = languages[index % len(languages)]
        name = names[index % len(names)]
        country = ("254", "256", "27")[index % 3]
        subscriber_length = 9 if country != "27" else 9
        subscriber = str(700000000 + index).zfill(subscriber_length)
        phone = f"+{country}{subscriber}"
        identifier = f"ID: OM{index:06d}"
        message_date = f"2026-07-{(index % 28) + 1:02d}"
        text = (
            f"{greeting} {honorific} {name} {identifier} "
            f"CALL {phone} ANC {message_date} PLS"
        )
        assert len(text) <= 320
        rows.append(
            {
                "urn": f"tel:{phone}",
                "contact": f"tel:{phone}",
                "direction": "in" if index % 2 == 0 else "out",
                "text": text,
                "sent_on": f"{message_date}T10:11:12Z",
            }
        )
        seeded.append((name, identifier, phone, message_date))
    return rows, seeded


def _recall(outputs: list[str], seeded: list[tuple[str, ...]]) -> float:
    total = sum(len(values) for values in seeded)
    redacted = sum(
        value not in output
        for output, values in zip(outputs, seeded)
        for value in values
    )
    return redacted / total


def test_short_text_preset_preserves_every_non_replacement_character():
    text = "\tSAWUBONA SISI Nomsa Dlamini\ncall 0712345678 ANC pls\n"

    result = deidentify_short_text(text, model_detector=_empty_detector)

    assert result.original_text == text
    assert result.deidentified_text == _expected_replacement_only_text(result)
    assert result.deidentified_text.startswith("\t")
    assert result.deidentified_text.endswith(" ANC pls\n")
    assert "Nomsa Dlamini" not in result.deidentified_text
    assert "0712345678" not in result.deidentified_text
    assert result.metadata["pipeline_preset"] == SHORT_TEXT


def test_default_pipeline_clinical_note_behavior_remains_unchanged():
    note = "Patient is stable with no identifiers."

    result = Pipeline(model_detector=_empty_detector).run(note)

    assert result.original_text == note
    assert result.redacted_text == note


def test_short_text_corpus_has_higher_recall_and_no_seeded_phi_leaks(monkeypatch):
    rows, seeded = _synthetic_corpus()
    monkeypatch.setattr(openmed, "analyze_text", _empty_detector)

    iterator = iter_redacted_sms_records(
        rows,
        batch_size=128,
        contact_hash_key="synthetic-test-key",
        loader=object(),
    )
    short_rows = list(iterator)
    short_outputs = [row["text"] for row in short_rows]

    default_processor = BatchProcessor(
        operation="deidentify",
        batch_size=100,
        continue_on_error=False,
        loader=object(),
        method="mask",
        use_safety_sweep=True,
    )
    default_result = default_processor.process_texts([row["text"] for row in rows])
    default_outputs = [item.result.deidentified_text for item in default_result.items]

    short_recall = _recall(short_outputs, seeded)
    default_recall = _recall(default_outputs, seeded)
    assert short_recall == 1.0
    assert short_recall > default_recall
    assert iterator.summary.row_count == 600
    assert iterator.summary.batch_count == math.ceil(600 / 128)

    corpus_text = "\n".join(short_outputs)
    original_mapping = {
        f"seed_{row_index}_{value_index}": value
        for row_index, values in enumerate(seeded)
        for value_index, value in enumerate(values[:3])
    }
    assert assert_redacted(corpus_text, original_mapping) == corpus_text


def test_rapidpro_json_round_trip_preserves_schema_and_flow_fields(monkeypatch):
    monkeypatch.setattr(openmed, "analyze_text", _empty_detector)

    result = redact_sms_json(
        FIXTURES / "rapidpro_messages.json",
        batch_size=2,
        contact_hash_key="round-trip-key",
        loader=object(),
    )
    payload = json.loads(result.text or "")
    original = json.loads((FIXTURES / "rapidpro_messages.json").read_text())

    assert set(payload) == set(original)
    assert len(payload["messages"]) == len(original["messages"])
    assert payload["flow"] == original["flow"]
    assert [row["direction"] for row in payload["messages"]] == ["in", "out"]
    assert [row["uuid"] for row in payload["messages"]] == [
        row["uuid"] for row in original["messages"]
    ]
    assert payload["messages"][0]["urn"] == payload["messages"][1]["urn"]
    assert payload["messages"][0]["contact"]["urn"] == payload["messages"][0]["urn"]
    assert payload["messages"][0]["contact"]["uuid"] == "contact-0001"
    assert payload["messages"][0]["sent_on"] == "2026-07-17"
    assert "Amina Njeri" not in result.text
    assert "+254712345678" not in result.text
    assert result.summary.row_count == 2
    assert result.summary.message_count == 2


def test_flow_result_input_text_is_redacted_without_changing_flow_uuid(monkeypatch):
    monkeypatch.setattr(openmed, "analyze_text", _empty_detector)
    payload = {
        "results": [
            {
                "flow": {"uuid": "flow-result-uuid"},
                "contact": {"uuid": "contact-uuid", "urn": "tel:+27821234567"},
                "input": {"text": "MAMA Thandi Mokoena ID: ZA123456"},
                "created_on": "2026-07-17T12:00:00+02:00",
            }
        ]
    }

    result = redact_sms_json(
        payload,
        contact_hash_key="flow-result-key",
        loader=object(),
    )
    reparsed = json.loads(result.text or "")

    row = reparsed["results"][0]
    assert row["flow"]["uuid"] == "flow-result-uuid"
    assert row["contact"]["uuid"] == "contact-uuid"
    assert row["created_on"] == "2026-07-17"
    assert "Thandi Mokoena" not in row["input"]["text"]
    assert "ZA123456" not in row["input"]["text"]


def test_generic_csv_round_trip_preserves_columns_rows_and_non_text_fields(monkeypatch):
    monkeypatch.setattr(openmed, "analyze_text", _empty_detector)

    result = redact_sms_csv(
        FIXTURES / "generic_messages.csv",
        batch_size=1,
        contact_hash_key="csv-round-trip-key",
        loader=object(),
    )
    rows = list(csv.DictReader(io.StringIO(result.text or "")))

    assert len(rows) == 2
    assert list(rows[0]) == [
        "urn",
        "contact",
        "direction",
        "text",
        "sent_on",
        "flow_uuid",
    ]
    assert [row["direction"] for row in rows] == ["in", "out"]
    assert rows[0]["flow_uuid"] == rows[1]["flow_uuid"]
    assert rows[0]["urn"] == rows[1]["urn"]
    assert rows[0]["contact"] == rows[1]["contact"]
    assert rows[0]["sent_on"] == "2026-07-17"
    assert "Nomsa Dlamini" not in rows[0]["text"]
    assert "0712345678" not in rows[0]["text"]
    assert "12345" not in rows[1]["text"]
    assert result.summary.row_count == 2
    assert result.summary.batch_count == 2
