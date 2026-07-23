from __future__ import annotations

import re
from types import SimpleNamespace

import pytest

from openmed.core.pii import PIIEntity
from openmed.interop import get_adapter, scrubadub

GOLDEN_TEXT = "John Doe visited City Hospital on 2024-01-02. Call 555-123-4567."


def _span(text: str, scrubadub_type: str) -> dict[str, object]:
    start = GOLDEN_TEXT.index(text)
    return {
        "beg": start,
        "end": start + len(text),
        "text": text,
        "type": scrubadub_type,
        "detector_name": "golden",
        "document_name": "note.txt",
    }


GOLDEN_RESULTS = [
    _span("John Doe", "name"),
    _span("City Hospital", "organization"),
    _span("2024-01-02", "date_of_birth"),
    _span("555-123-4567", "phone"),
]


def test_registry_loads_scrubadub_adapter_lazily():
    adapter = get_adapter("scrubadub")

    assert adapter is scrubadub
    assert hasattr(adapter, "to_canonical")


def test_to_canonical_preserves_offsets_and_uses_canonical_labels():
    entities = scrubadub.to_canonical(GOLDEN_RESULTS, text=GOLDEN_TEXT)

    assert [entity.label for entity in entities] == [
        "PERSON",
        "ORGANIZATION",
        "DATE_OF_BIRTH",
        "PHONE",
    ]
    assert [entity.text for entity in entities] == [
        "John Doe",
        "City Hospital",
        "2024-01-02",
        "555-123-4567",
    ]
    assert [(entity.start, entity.end) for entity in entities] == [
        (item["beg"], item["end"]) for item in GOLDEN_RESULTS
    ]
    assert all(entity.metadata["adapter"] == "scrubadub" for entity in entities)


def test_from_canonical_round_trip_preserves_scrubadub_types():
    entities = scrubadub.to_canonical(GOLDEN_RESULTS, text=GOLDEN_TEXT)

    round_tripped = scrubadub.from_canonical(entities)

    assert [item["type"] for item in round_tripped] == [
        item["type"] for item in GOLDEN_RESULTS
    ]
    assert [
        entity.label
        for entity in scrubadub.to_canonical(round_tripped, text=GOLDEN_TEXT)
    ] == [entity.label for entity in entities]


def test_round_trip_preserves_filth_metadata():
    record = {
        **_span("555-123-4567", "phone"),
        "locale": "en_US",
        "replacement_string": "{{PHONE-1}}",
    }

    [round_tripped] = scrubadub.from_canonical(
        scrubadub.to_canonical([record], text=GOLDEN_TEXT)
    )

    assert round_tripped["locale"] == "en_US"
    assert round_tripped["replacement_string"] == "{{PHONE-1}}"


def test_merged_filth_is_flattened_without_losing_source_labels():
    merged = SimpleNamespace(
        filths=[
            _span("John Doe", "name"),
            _span("John Doe", "organization"),
        ]
    )

    entities = scrubadub.to_canonical([merged], text=GOLDEN_TEXT)

    assert [(entity.text, entity.label) for entity in entities] == [
        ("John Doe", "PERSON"),
        ("John Doe", "ORGANIZATION"),
    ]


def test_national_insurance_number_maps_to_generic_id():
    entity = scrubadub.to_canonical(
        [_span("555-123-4567", "national_insurance_number")],
        text=GOLDEN_TEXT,
    )[0]

    assert entity.label == "ID_NUM"


def test_invalid_span_fails_closed():
    with pytest.raises(ValueError, match="exceeds source text length"):
        scrubadub.to_canonical(
            [{"beg": 0, "end": len(GOLDEN_TEXT) + 1, "type": "name"}],
            text=GOLDEN_TEXT,
        )


def test_credential_filth_with_match_splits_into_username_and_password():
    text = "Login with username: alice password: hunter2 to continue."
    pattern = re.compile(
        r"username:\s*(?P<username>\S+)\s+password:\s*(?P<password>\S+)"
    )
    match = pattern.search(text)
    assert match is not None
    record = {
        "beg": match.start(),
        "end": match.end(),
        "text": match.group(),
        "type": "credential",
        "match": match,
        "detector_name": "credential",
        "document_name": "note.txt",
    }

    entities = scrubadub.to_canonical([record], text=text)

    assert [(entity.label, entity.text) for entity in entities] == [
        ("USERNAME", "alice"),
        ("PASSWORD", "hunter2"),
    ]
    assert [(entity.start, entity.end) for entity in entities] == [
        (match.start("username"), match.end("username")),
        (match.start("password"), match.end("password")),
    ]
    assert entities[0].metadata["scrubadub_credential_field"] == "username"
    assert entities[1].metadata["scrubadub_credential_field"] == "password"


def test_credential_filth_without_match_falls_back_to_single_password_entity():
    record = _span("555-123-4567", "credential")

    entities = scrubadub.to_canonical([record], text=GOLDEN_TEXT)

    assert len(entities) == 1
    assert entities[0].label == "PASSWORD"
    assert entities[0].text == "555-123-4567"


def test_credential_pair_recombines_losslessly_on_round_trip():
    text = "Login with username: alice password: hunter2 to continue."
    pattern = re.compile(
        r"username:\s*(?P<username>\S+)\s+password:\s*(?P<password>\S+)"
    )
    match = pattern.search(text)
    assert match is not None
    record = {
        "beg": match.start(),
        "end": match.end(),
        "text": match.group(),
        "type": "credential",
        "match": match,
        "detector_name": "credential",
        "document_name": "note.txt",
    }

    entities = scrubadub.to_canonical([record], text=text)
    round_tripped = scrubadub.from_canonical(entities)

    assert round_tripped == [
        {
            "beg": match.start(),
            "end": match.end(),
            "text": match.group(),
            "type": "credential",
            "detector_name": "credential",
            "document_name": "note.txt",
            "replacement_string": None,
        }
    ]


def test_credential_pair_lone_survivor_degrades_to_standalone_record():
    text = "Login with username: alice password: hunter2 to continue."
    pattern = re.compile(
        r"username:\s*(?P<username>\S+)\s+password:\s*(?P<password>\S+)"
    )
    match = pattern.search(text)
    assert match is not None
    record = {
        "beg": match.start(),
        "end": match.end(),
        "text": match.group(),
        "type": "credential",
        "match": match,
        "detector_name": "credential",
        "document_name": "note.txt",
    }

    entities = scrubadub.to_canonical([record], text=text)
    username_only = [e for e in entities if e.label == "USERNAME"]

    round_tripped = scrubadub.from_canonical(username_only)

    assert len(round_tripped) == 1
    assert round_tripped[0]["type"] == "credential"
    assert round_tripped[0]["text"] == "alice"


def test_merge_with_openmed_routes_through_merger_and_adds_adapter_spans(monkeypatch):
    calls = []

    def fake_merge(entities, text, **kwargs):
        calls.append((entities, text, kwargs))
        return entities

    monkeypatch.setattr(scrubadub, "merge_entities_with_semantic_units", fake_merge)
    openmed_entity = PIIEntity(
        text="John Doe",
        label="PERSON",
        entity_type="PERSON",
        start=0,
        end=8,
        confidence=0.70,
    )
    scrubadub_entity = _span("555-123-4567", "phone")

    merged = scrubadub.merge_with_openmed(
        [openmed_entity],
        [scrubadub_entity],
        text=GOLDEN_TEXT,
    )

    assert calls
    assert [(entity.text, entity.label) for entity in merged] == [
        ("John Doe", "PERSON"),
        ("555-123-4567", "PHONE"),
    ]
    assert calls[0][2]["prefer_model_labels"] is True


def test_scoreless_scrubadub_overlap_defaults_to_fallback_priority():
    openmed_entity = PIIEntity(
        text="John Doe",
        label="PERSON",
        entity_type="PERSON",
        start=0,
        end=8,
        confidence=0.70,
    )

    merged = scrubadub.merge_with_openmed(
        [openmed_entity],
        [_span("John Doe", "organization")],
        text=GOLDEN_TEXT,
        use_semantic_patterns=False,
    )

    assert [(entity.text, entity.label, entity.confidence) for entity in merged] == [
        ("John Doe", "PERSON", 0.70),
    ]


def test_actual_scrubadub_filth_objects_when_extra_is_installed():
    scrubadub_package = pytest.importorskip("scrubadub")
    text = "Email jane@example.com or call 212-555-0123."

    entities = scrubadub.to_canonical(
        scrubadub_package.list_filth(text),
        text=text,
    )

    assert ("jane@example.com", "EMAIL") in {
        (entity.text, entity.label) for entity in entities
    }
    assert ("212-555-0123", "PHONE") in {
        (entity.text, entity.label) for entity in entities
    }
