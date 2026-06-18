from __future__ import annotations

from dataclasses import dataclass

from openmed.core.pii import PIIEntity
from openmed.interop import get_adapter
from openmed.interop import presidio


@dataclass
class RecognizerResultLike:
    entity_type: str
    start: int
    end: int
    score: float
    recognizer_name: str = "fake-recognizer"


GOLDEN_TEXT = (
    "John Doe emailed jane@example.com from 123 Main Street on 2024-01-02."
)
GOLDEN_RESULTS = [
    RecognizerResultLike("PERSON", 0, 8, 0.98),
    RecognizerResultLike("EMAIL_ADDRESS", 17, 33, 0.99),
    RecognizerResultLike("LOCATION", 39, 54, 0.91),
    RecognizerResultLike("DATE_TIME", 58, 68, 0.88),
]


def test_registry_loads_presidio_adapter_lazily():
    adapter = get_adapter("presidio")

    assert adapter is presidio
    assert hasattr(adapter, "to_canonical")


def test_to_canonical_preserves_offsets_and_uses_canonical_labels():
    entities = presidio.to_canonical(GOLDEN_RESULTS, text=GOLDEN_TEXT)

    assert [entity.label for entity in entities] == [
        "PERSON",
        "EMAIL",
        "LOCATION",
        "DATE",
    ]
    assert [entity.text for entity in entities] == [
        "John Doe",
        "jane@example.com",
        "123 Main Street",
        "2024-01-02",
    ]
    assert [(entity.start, entity.end) for entity in entities] == [
        (item.start, item.end)
        for item in GOLDEN_RESULTS
    ]
    assert all(entity.metadata["adapter"] == "presidio" for entity in entities)


def test_from_canonical_round_trip_preserves_presidio_labels():
    entities = presidio.to_canonical(GOLDEN_RESULTS, text=GOLDEN_TEXT)

    round_tripped = presidio.from_canonical(
        entities,
        result_cls=RecognizerResultLike,
    )

    assert [item.entity_type for item in round_tripped] == [
        item.entity_type
        for item in GOLDEN_RESULTS
    ]
    assert [
        entity.label
        for entity in presidio.to_canonical(round_tripped, text=GOLDEN_TEXT)
    ] == [
        entity.label
        for entity in entities
    ]


def test_merge_with_openmed_routes_through_merger_and_resolves_overlaps(monkeypatch):
    calls = []

    def fake_merge(entities, text, **kwargs):
        calls.append((entities, text, kwargs))
        return entities

    monkeypatch.setattr(presidio, "merge_entities_with_semantic_units", fake_merge)
    openmed_entity = PIIEntity(
        text="John",
        label="PERSON",
        entity_type="PERSON",
        start=0,
        end=4,
        confidence=0.70,
    )
    presidio_entity = RecognizerResultLike("PERSON", 0, 8, 0.95)

    merged = presidio.merge_with_openmed(
        [openmed_entity],
        [presidio_entity],
        text=GOLDEN_TEXT,
    )

    assert calls
    assert len(merged) == 1
    assert merged[0].text == "John Doe"
    assert merged[0].label == "PERSON"
    assert merged[0].start == 0
    assert merged[0].end == 8
