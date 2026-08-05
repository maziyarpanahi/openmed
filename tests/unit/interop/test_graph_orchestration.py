from __future__ import annotations

import socket
import sys
from types import SimpleNamespace
from typing import Any, TypedDict

import pytest
from jsonschema.validators import validator_for

from openmed.core.labels import normalize_label
from openmed.core.schemas import OpenMedSpan
from openmed.interop import adapter_spec, get_adapter
from openmed.interop import graph_orchestration as graph_adapter
from openmed.interop.gateway import PrivacyGateway

SOURCE_TEXT = (
    "Patient Jane Roe, born 1970-01-02, has MRN-0042; call 555-0100 or "
    "jane.roe@example.test."
)
SURFACES = (
    ("Jane Roe", "NAME", "PERSON"),
    ("1970-01-02", "DATE_OF_BIRTH", "DATE_OF_BIRTH"),
    ("MRN-0042", "MEDICAL_RECORD_NUMBER", "ID_NUM"),
    ("555-0100", "PHONE", "PHONE"),
    ("jane.roe@example.test", "EMAIL", "EMAIL"),
)


def fake_deidentify(text: str, **kwargs):
    assert text == SOURCE_TEXT
    assert kwargs["keep_mapping"] is True
    assert kwargs["audit"] is False
    redacted = text
    mapping = {}
    entities = []
    for index, (surface, label, canonical_label) in enumerate(SURFACES):
        start = text.index(surface)
        placeholder = f"[PHI_{index}]"
        mapping[placeholder] = surface
        entities.append(
            SimpleNamespace(
                label=label,
                entity_type=label,
                canonical_label=canonical_label,
                start=start,
                end=start + len(surface),
                confidence=0.99,
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


def fake_reidentify(text: str, mapping) -> str:
    for placeholder, original in mapping.items():
        text = text.replace(placeholder, original)
    return text


def test_graph_nodes_round_trip_golden_spans_through_gateway() -> None:
    gateway = PrivacyGateway(
        deidentifier=fake_deidentify,
        reidentifier=fake_reidentify,
    )
    deidentify_node = graph_adapter.DeidentifyNode(deidentifier=fake_deidentify)
    seen_by_external = []

    def mock_external_llm(text: str) -> str:
        seen_by_external.append(text)
        assert all(surface not in text for surface, _, _ in SURFACES)
        return f"Mock summary: {text}"

    external_node = graph_adapter.GatewayBoundLLMNode(
        mock_external_llm,
        gateway=gateway,
    )
    reidentify_node = graph_adapter.ReidentifyNode(gateway=gateway)

    state = {"text": SOURCE_TEXT, "doc_id": "synthetic-note"}
    state.update(deidentify_node(state))
    state.update(external_node(state))
    state.update(reidentify_node(state))

    direct = fake_deidentify(
        SOURCE_TEXT,
        **deidentify_node.config.to_deidentify_kwargs(),
    )
    direct_labels = {
        entity.canonical_label or normalize_label(entity.entity_type)
        for entity in direct.pii_entities
    }
    graph_labels = {span.canonical_label for span in state["spans"]}

    assert len(state["spans"]) == len(SURFACES)
    assert all(isinstance(span, OpenMedSpan) for span in state["spans"])
    assert graph_labels == direct_labels
    assert seen_by_external == [state["redacted_text"]]
    assert state["response"] == f"Mock summary: {SOURCE_TEXT}"


def test_redaction_and_reidentification_nodes_have_zero_socket_egress(
    monkeypatch,
) -> None:
    def blocked(*args, **kwargs):
        del args, kwargs
        raise AssertionError("unexpected socket egress")

    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket.socket, "connect", blocked)
    gateway = PrivacyGateway(reidentifier=fake_reidentify)
    deidentify_node = graph_adapter.DeidentifyNode(deidentifier=fake_deidentify)

    state = {"text": SOURCE_TEXT}
    state.update(deidentify_node(state))
    state["redacted_response"] = state["redacted_text"]
    state.update(graph_adapter.ReidentifyNode(gateway=gateway)(state))

    assert state["response"] == SOURCE_TEXT


def test_binding_adds_safe_flow_without_importing_langgraph() -> None:
    class GraphBuilder:
        def __init__(self):
            self.nodes = {}
            self.edges = []

        def add_node(self, name, node):
            self.nodes[name] = node

        def add_edge(self, left, right):
            self.edges.append((left, right))

    sys.modules.pop("langgraph", None)
    graph = GraphBuilder()

    result = graph_adapter.bind_state_graph(
        graph,
        external_call=lambda text: text,
        deidentifier=fake_deidentify,
        reidentifier=fake_reidentify,
        entry_point="START",
        finish_point="END",
    )

    assert result is graph
    assert set(graph.nodes) == {
        "openmed_deidentify",
        "openmed_external_llm",
        "openmed_reidentify",
    }
    assert graph.edges == [
        ("START", "openmed_deidentify"),
        ("openmed_deidentify", "openmed_external_llm"),
        ("openmed_external_llm", "openmed_reidentify"),
        ("openmed_reidentify", "END"),
    ]
    assert "langgraph" not in sys.modules


def test_state_graph_binding_has_clear_missing_extra_error(monkeypatch) -> None:
    monkeypatch.setattr(
        graph_adapter,
        "_import_module",
        lambda name: (_ for _ in ()).throw(ImportError(name)),
    )

    with pytest.raises(ImportError, match=r"openmed\[langgraph\]"):
        graph_adapter.create_state_graph(
            dict,
            external_call=lambda text: text,
        )


def test_graph_adapter_is_lazy_and_uses_registry_tool_schemas() -> None:
    sys.modules.pop("langgraph", None)

    assert get_adapter("graph_orchestration") is graph_adapter
    assert adapter_spec("graph-orchestration").extra == "langgraph"
    assert "langgraph" not in sys.modules

    definitions = graph_adapter.create_tool_definitions()
    assert definitions
    for definition in definitions:
        assert definition["adapter"] == "graph_orchestration"
        validator_for(definition["input_schema"]).check_schema(
            definition["input_schema"]
        )
        validator_for(definition["output_schema"]).check_schema(
            definition["output_schema"]
        )


def test_real_state_graph_round_trip_when_extra_is_installed() -> None:
    pytest.importorskip("langgraph.graph")

    class GraphState(TypedDict, total=False):
        text: str
        redacted_text: str
        redaction_mapping: dict[str, str]
        spans: tuple[OpenMedSpan, ...]
        redacted_response: str
        response: str
        doc_id: str

    gateway = PrivacyGateway(
        deidentifier=fake_deidentify,
        reidentifier=fake_reidentify,
    )
    graph = graph_adapter.create_state_graph(
        GraphState,
        external_call=lambda text: f"Mock summary: {text}",
        gateway=gateway,
        deidentifier=fake_deidentify,
        reidentifier=fake_reidentify,
    ).compile()

    result: dict[str, Any] = graph.invoke(
        {"text": SOURCE_TEXT, "doc_id": "synthetic-note"}
    )

    assert len(result["spans"]) == len(SURFACES)
    assert {span.canonical_label for span in result["spans"]} == {
        canonical_label for _, _, canonical_label in SURFACES
    }
    assert result["response"] == f"Mock summary: {SOURCE_TEXT}"


def test_real_state_graph_uses_safe_default_state_schema() -> None:
    pytest.importorskip("langgraph.graph")
    gateway = PrivacyGateway(
        deidentifier=fake_deidentify,
        reidentifier=fake_reidentify,
    )
    graph = graph_adapter.create_state_graph(
        external_call=lambda text: text,
        gateway=gateway,
        deidentifier=fake_deidentify,
        reidentifier=fake_reidentify,
    ).compile()

    result = graph.invoke({"text": SOURCE_TEXT})

    assert result["response"] == SOURCE_TEXT
    assert len(result["spans"]) == len(SURFACES)
