from __future__ import annotations

from openmed.core.pii import PIIEntity
from openmed.interop import get_adapter
from openmed.interop import pydeid


GOLDEN_TEXT = (
    "John Doe visited City Hospital on 2024-01-02. Call 555-123-4567."
)


def _span(text: str, types: list[str]) -> dict[str, object]:
    start = GOLDEN_TEXT.index(text)
    return {
        "phi_start": start,
        "phi_end": start + len(text),
        "phi": text,
        "surrogate_start": start,
        "surrogate_end": start + len("<PHI>"),
        "surrogate": "<PHI>",
        "types": types,
    }


GOLDEN_RESULTS = [
    _span("John Doe", ["First Name", "Last Name"]),
    _span("City Hospital", ["Hospital Name"]),
    _span("2024-01-02", ["Month Day Year [yyyy-mm-dd]"]),
    _span("555-123-4567", ["Telephone/Fax"]),
]


def test_registry_loads_pydeid_adapter_lazily():
    adapter = get_adapter("pydeid")

    assert adapter is pydeid
    assert hasattr(adapter, "to_canonical")


def test_to_canonical_preserves_offsets_and_uses_canonical_labels():
    entities = pydeid.to_canonical(GOLDEN_RESULTS, text=GOLDEN_TEXT)

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
        (item["phi_start"], item["phi_end"])
        for item in GOLDEN_RESULTS
    ]
    assert all(entity.metadata["adapter"] == "pydeid" for entity in entities)


def test_from_canonical_round_trip_preserves_pydeid_labels():
    entities = pydeid.to_canonical(GOLDEN_RESULTS, text=GOLDEN_TEXT)

    round_tripped = pydeid.from_canonical(entities)

    assert [item["types"] for item in round_tripped] == [
        item["types"]
        for item in GOLDEN_RESULTS
    ]
    assert [
        entity.label
        for entity in pydeid.to_canonical(round_tripped, text=GOLDEN_TEXT)
    ] == [
        entity.label
        for entity in entities
    ]


def test_merge_with_openmed_routes_through_merger_and_adds_adapter_spans(monkeypatch):
    calls = []

    def fake_merge(entities, text, **kwargs):
        calls.append((entities, text, kwargs))
        return entities

    monkeypatch.setattr(pydeid, "merge_entities_with_semantic_units", fake_merge)
    openmed_entity = PIIEntity(
        text="John Doe",
        label="PERSON",
        entity_type="PERSON",
        start=0,
        end=8,
        confidence=0.70,
    )
    pydeid_entity = _span("555-123-4567", ["Telephone/Fax"])

    merged = pydeid.merge_with_openmed(
        [openmed_entity],
        [pydeid_entity],
        text=GOLDEN_TEXT,
    )

    assert calls
    assert [(entity.text, entity.label) for entity in merged] == [
        ("John Doe", "PERSON"),
        ("555-123-4567", "PHONE"),
    ]
    assert calls[0][2]["prefer_model_labels"] is True
