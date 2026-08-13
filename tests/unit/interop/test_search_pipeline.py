from __future__ import annotations

import socket
import sys
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
from jsonschema.validators import validator_for

from openmed.core.schemas import OpenMedSpan
from openmed.interop import adapter_spec, get_adapter
from openmed.interop import search_pipeline as search_adapter

SOURCE_TEXT = (
    "Patient Jane Roe, born 1970-01-02, has MRN-0042; call 555-0100 or "
    "jane.roe@example.test."
)
SURFACES = (
    ("Jane Roe", "PERSON"),
    ("1970-01-02", "DATE_OF_BIRTH"),
    ("MRN-0042", "ID_NUM"),
    ("555-0100", "PHONE"),
    ("jane.roe@example.test", "EMAIL"),
)


@dataclass
class FixtureDocument:
    content: str
    meta: dict[str, object] = field(default_factory=dict)
    id: str = "synthetic-document"
    score: float | None = None

    def to_dict(self, *, flatten: bool = True) -> dict[str, object]:
        assert flatten is False
        return {
            "content": self.content,
            "meta": dict(self.meta),
            "id": self.id,
            "score": self.score,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]):
        return cls(**payload)


class ComponentDecorator:
    def __call__(self, cls):
        cls.__haystack_component__ = True
        return cls

    @staticmethod
    def output_types(**output_types):
        def decorate(func):
            func.__haystack_output_types__ = output_types
            return func

        return decorate


def fake_deidentify(text: str, **kwargs):
    assert kwargs["keep_mapping"] is True
    assert kwargs["audit"] is False
    redacted = text
    mapping = {}
    entities = []
    for index, (surface, canonical_label) in enumerate(SURFACES):
        start = text.index(surface)
        placeholder = f"[PHI_{index}]"
        mapping[placeholder] = surface
        entities.append(
            SimpleNamespace(
                label=canonical_label,
                entity_type=canonical_label,
                canonical_label=canonical_label,
                start=start,
                end=start + len(surface),
                confidence=0.98,
                redacted_text=placeholder,
                sources=["synthetic-detector"],
            )
        )
    for entity in reversed(entities):
        redacted = (
            redacted[: entity.start] + entity.redacted_text + redacted[entity.end :]
        )
    return SimpleNamespace(
        deidentified_text=redacted,
        mapping=mapping,
        pii_entities=entities,
    )


def fake_haystack_import(name: str):
    assert name == "haystack"
    return SimpleNamespace(
        component=ComponentDecorator(),
        Document=FixtureDocument,
    )


def test_search_component_round_trips_golden_labels_and_runnable_shapes() -> None:
    component = search_adapter.RedactionComponent(deidentifier=fake_deidentify)
    original = FixtureDocument(
        content=SOURCE_TEXT,
        meta={"source": "synthetic"},
        score=0.91,
    )

    redaction = component.redact(original, doc_id="synthetic-note")
    invoked = component.invoke(original)
    batched = component.batch([original])
    streamed = list(component.transform([original]))
    direct = fake_deidentify(SOURCE_TEXT, **component.deidentify_kwargs)
    direct_labels = {entity.canonical_label for entity in direct.pii_entities}
    restored = redaction.value.content
    for placeholder, surface in redaction.mapping.items():
        restored = restored.replace(placeholder, surface)

    assert len(redaction.spans) == len(SURFACES)
    assert all(isinstance(span, OpenMedSpan) for span in redaction.spans)
    assert {span.canonical_label for span in redaction.spans} == direct_labels
    assert all(surface not in redaction.value.content for surface, _ in SURFACES)
    assert len(redaction.mapping) == len(SURFACES)
    assert redaction.value.meta == original.meta
    assert redaction.value.score == original.score
    assert original.content == SOURCE_TEXT
    assert restored == SOURCE_TEXT
    assert invoked.content == redaction.value.content
    assert batched[0].content == redaction.value.content
    assert streamed[0].content == redaction.value.content


def test_retrieval_filter_redacts_documents_and_preserves_payload() -> None:
    retrieval_filter = search_adapter.RetrievalRedactionFilter(
        deidentifier=fake_deidentify
    )
    payload = {
        "query": "Summarize the synthetic follow-up.",
        "documents": [FixtureDocument(content=SOURCE_TEXT, score=0.82)],
    }

    result = retrieval_filter.invoke(payload)

    assert result["query"] == payload["query"]
    assert result["documents"][0].score == 0.82
    assert all(surface not in result["documents"][0].content for surface, _ in SURFACES)
    assert len(result["spans"]) == len(SURFACES)
    assert all(isinstance(span, OpenMedSpan) for span in result["spans"])


def test_search_components_have_zero_socket_egress(monkeypatch) -> None:
    def blocked(*args, **kwargs):
        del args, kwargs
        raise AssertionError("unexpected socket egress")

    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket.socket, "connect", blocked)

    output = search_adapter.RedactionComponent(deidentifier=fake_deidentify).run(
        [FixtureDocument(content=SOURCE_TEXT)]
    )

    assert len(output["spans"]) == len(SURFACES)


def test_haystack_binding_is_lazy_and_emits_canonical_spans(monkeypatch) -> None:
    monkeypatch.setattr(search_adapter, "_import_module", fake_haystack_import)

    component = search_adapter.create_haystack_component(deidentifier=fake_deidentify)
    result = component.run([FixtureDocument(content=SOURCE_TEXT)])

    assert component.__haystack_component__ is True
    assert component.run.__haystack_output_types__ == {
        "documents": list[FixtureDocument],
        "spans": list[OpenMedSpan],
    }
    assert len(result["spans"]) == len(SURFACES)
    assert all(surface not in result["documents"][0].content for surface, _ in SURFACES)


def test_search_adapter_is_lazy_and_uses_registry_tool_schemas() -> None:
    sys.modules.pop("haystack", None)

    assert get_adapter("search_pipeline") is search_adapter
    assert adapter_spec("search-pipeline").extra == "haystack"
    assert "haystack" not in sys.modules

    definitions = search_adapter.create_tool_definitions()
    assert definitions
    for definition in definitions:
        assert definition["adapter"] == "search_pipeline"
        validator_for(definition["input_schema"]).check_schema(
            definition["input_schema"]
        )
        validator_for(definition["output_schema"]).check_schema(
            definition["output_schema"]
        )


def test_real_haystack_pipeline_round_trip_when_extra_is_installed() -> None:
    haystack = pytest.importorskip("haystack")
    pipeline = haystack.Pipeline()
    search_adapter.bind_search_pipeline(
        pipeline,
        deidentifier=fake_deidentify,
    )

    result = pipeline.run(
        {"openmed_redaction": {"documents": [haystack.Document(content=SOURCE_TEXT)]}}
    )["openmed_redaction"]

    assert len(result["spans"]) == len(SURFACES)
    assert {span.canonical_label for span in result["spans"]} == {
        label for _, label in SURFACES
    }
    assert all(surface not in result["documents"][0].content for surface, _ in SURFACES)
