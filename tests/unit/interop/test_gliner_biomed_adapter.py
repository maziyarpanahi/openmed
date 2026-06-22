from __future__ import annotations

import pytest

from openmed.core.pii import PIIEntity
from openmed.interop import get_adapter, gliner_biomed

GOLDEN_TEXT = "John Doe visited City Hospital on 2024-01-02. Call 555-123-4567."


def _span(text: str, label: str, score: float = 0.91) -> dict[str, object]:
    start = GOLDEN_TEXT.index(text)
    return {
        "text": text,
        "label": label,
        "start": start,
        "end": start + len(text),
        "score": score,
    }


GOLDEN_RESULTS = [
    _span("John Doe", "patient name", 0.98),
    _span("City Hospital", "hospital", 0.93),
    _span("2024-01-02", "date", 0.89),
    _span("555-123-4567", "phone number", 0.95),
]


def test_registry_loads_gliner_biomed_adapter_lazily():
    adapter = get_adapter("gliner-biomed")

    assert adapter is gliner_biomed
    assert hasattr(adapter, "to_canonical")


def test_to_canonical_preserves_offsets_and_uses_canonical_labels():
    entities = gliner_biomed.to_canonical(
        {"entities": GOLDEN_RESULTS},
        text=GOLDEN_TEXT,
    )

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
        (item["start"], item["end"]) for item in GOLDEN_RESULTS
    ]
    assert all(entity.metadata["adapter"] == "gliner_biomed" for entity in entities)


def test_to_canonical_can_locate_gliner_spans_without_offsets():
    entities = gliner_biomed.to_canonical(
        [
            {"text": "John Doe", "label": "patient name", "score": 0.98},
            {"text": "City Hospital", "label": "hospital", "score": 0.93},
        ],
        text=GOLDEN_TEXT,
    )

    assert [(entity.start, entity.end) for entity in entities] == [
        (0, 8),
        (17, 30),
    ]


def test_from_canonical_round_trip_preserves_gliner_labels():
    entities = gliner_biomed.to_canonical(GOLDEN_RESULTS, text=GOLDEN_TEXT)

    round_tripped = gliner_biomed.from_canonical(entities)

    assert [item["label"] for item in round_tripped] == [
        item["label"] for item in GOLDEN_RESULTS
    ]
    assert [
        entity.label
        for entity in gliner_biomed.to_canonical(round_tripped, text=GOLDEN_TEXT)
    ] == [entity.label for entity in entities]


def test_merge_with_openmed_routes_through_merger_and_adds_adapter_spans(monkeypatch):
    calls = []

    def fake_merge(entities, text, **kwargs):
        calls.append((entities, text, kwargs))
        return entities

    monkeypatch.setattr(gliner_biomed, "merge_entities_with_semantic_units", fake_merge)
    openmed_entity = PIIEntity(
        text="John Doe",
        label="PERSON",
        entity_type="PERSON",
        start=0,
        end=8,
        confidence=0.70,
    )
    gliner_entity = _span("City Hospital", "hospital")

    merged = gliner_biomed.merge_with_openmed(
        [openmed_entity],
        [gliner_entity],
        text=GOLDEN_TEXT,
    )

    assert calls
    assert [(entity.text, entity.label) for entity in merged] == [
        ("John Doe", "PERSON"),
        ("City Hospital", "ORGANIZATION"),
    ]
    assert calls[0][2]["prefer_model_labels"] is True


def test_predict_to_canonical_missing_extra_raises_clear_import_error(monkeypatch):
    def missing_dependency(name: str):
        raise ImportError(name)

    monkeypatch.setattr(gliner_biomed, "_import_module", missing_dependency)

    with pytest.raises(ImportError, match=r"openmed\[gliner\]"):
        gliner_biomed.predict_to_canonical(GOLDEN_TEXT)
