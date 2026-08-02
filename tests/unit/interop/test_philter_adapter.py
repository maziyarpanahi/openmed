from __future__ import annotations

from openmed.core.pii import PIIEntity
from openmed.interop import get_adapter, philter

GOLDEN_TEXT = "John Doe visited City Hospital on 2024-01-02. Call 555-123-4567."


def _span(text: str, phi_type: str) -> dict[str, object]:
    start = GOLDEN_TEXT.index(text)
    return {
        "start": start,
        "stop": start + len(text),
        "word": text,
        "phi_type": phi_type,
        "filepath": "note.txt",
    }


GOLDEN_RESULTS = [
    _span("John Doe", "NAME"),
    _span("City Hospital", "HOSPITAL"),
    _span("2024-01-02", "DATE"),
    _span("555-123-4567", "PHONE"),
]


def test_registry_loads_philter_adapter_lazily():
    adapter = get_adapter("philter")

    assert adapter is philter
    assert hasattr(adapter, "to_canonical")


def test_to_canonical_preserves_offsets_and_uses_canonical_labels():
    entities = philter.to_canonical({"phi": GOLDEN_RESULTS}, text=GOLDEN_TEXT)

    assert [entity.label for entity in entities] == [
        "PERSON",
        "ORGANIZATION",
        "DATE",
        "PHONE",
    ]
    assert [entity.text for entity in entities] == [
        "John Doe",
        "City Hospital",
        "2024-01-02",
        "555-123-4567",
    ]
    assert [(entity.start, entity.end) for entity in entities] == [
        (item["start"], item["stop"]) for item in GOLDEN_RESULTS
    ]
    assert all(entity.metadata["adapter"] == "philter" for entity in entities)


def test_from_canonical_round_trip_preserves_philter_labels():
    entities = philter.to_canonical(GOLDEN_RESULTS, text=GOLDEN_TEXT)

    round_tripped = philter.from_canonical(entities)

    assert [item["phi_type"] for item in round_tripped] == [
        item["phi_type"] for item in GOLDEN_RESULTS
    ]
    assert [
        entity.label for entity in philter.to_canonical(round_tripped, text=GOLDEN_TEXT)
    ] == [entity.label for entity in entities]


def test_merge_with_openmed_routes_through_merger_and_adds_adapter_spans(monkeypatch):
    calls = []

    def fake_merge(entities, text, **kwargs):
        calls.append((entities, text, kwargs))
        return entities

    monkeypatch.setattr(philter, "merge_entities_with_semantic_units", fake_merge)
    openmed_entity = PIIEntity(
        text="John Doe",
        label="PERSON",
        entity_type="PERSON",
        start=0,
        end=8,
        confidence=0.70,
    )
    philter_entity = _span("2024-01-02", "DATE")

    merged = philter.merge_with_openmed(
        [openmed_entity],
        [philter_entity],
        text=GOLDEN_TEXT,
    )

    assert calls
    assert [(entity.text, entity.label) for entity in merged] == [
        ("John Doe", "PERSON"),
        ("2024-01-02", "DATE"),
    ]
    assert calls[0][2]["prefer_model_labels"] is True


def test_merge_with_openmed_preserves_provenance_on_real_semantic_merge():
    date_start = GOLDEN_TEXT.index("2024-01-02")
    openmed_entity = PIIEntity(
        text="2024",
        label="DATE",
        entity_type="DATE",
        start=date_start,
        end=date_start + len("2024"),
        confidence=0.70,
        metadata={"adapter": "openmed", "source": "openmed"},
        sources=["openmed"],
    )

    merged = philter.merge_with_openmed(
        [openmed_entity],
        [_span("2024-01-02", "DATE")],
        text=GOLDEN_TEXT,
    )

    date_entity = next(entity for entity in merged if entity.text == "2024-01-02")
    assert date_entity.metadata["adapter"] == "merged"
    assert date_entity.metadata["source_adapters"] == ["openmed", "philter"]
    assert date_entity.metadata["philter_phi_type"] == "DATE"
    assert date_entity.sources == ["openmed", "philter"]
