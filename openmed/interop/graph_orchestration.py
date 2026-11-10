"""Redaction-safe callables and bindings for stateful graph orchestration."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from importlib import import_module as _import_module
from typing import Any, TypedDict

from openmed.core.schemas import OpenMedSpan
from openmed.interop._pii import canonical_redaction
from openmed.interop.gateway import (
    PrivacyGateway,
    PrivacyGatewayConfig,
    RedactionMapping,
)
from openmed.mcp.tool_registry import (
    render_graph_orchestration_tool_definitions,
)

Deidentifier = Callable[..., Any]
Reidentifier = Callable[[str, Mapping[str, str]], str]
ExternalTextCall = Callable[[str], str]


class GraphState(TypedDict, total=False):
    """Default state contract for the bundled redaction-safe graph flow."""

    text: str
    doc_id: str
    redacted_text: str
    redaction_mapping: RedactionMapping
    spans: tuple[OpenMedSpan, ...]
    redacted_response: str
    response: str


class DeidentifyNode:
    """Create a graph-state update containing redacted text and safe spans."""

    def __init__(
        self,
        *,
        config: PrivacyGatewayConfig | None = None,
        deidentifier: Deidentifier | None = None,
        input_key: str = "text",
        output_key: str = "redacted_text",
        mapping_key: str = "redaction_mapping",
        spans_key: str = "spans",
        doc_id_key: str = "doc_id",
        default_doc_id: str = "graph-orchestration",
    ) -> None:
        self.config = config or PrivacyGatewayConfig()
        self._deidentifier = deidentifier
        self.input_key = input_key
        self.output_key = output_key
        self.mapping_key = mapping_key
        self.spans_key = spans_key
        self.doc_id_key = doc_id_key
        self.default_doc_id = default_doc_id

    def __call__(
        self,
        state: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Return a LangGraph-style partial state update without egress."""

        if self.input_key not in state:
            raise KeyError(f"graph state is missing {self.input_key!r}")
        text = state[self.input_key]
        if not isinstance(text, str):
            raise TypeError(f"graph state {self.input_key!r} must be a string")
        doc_id = str(state.get(self.doc_id_key) or self.default_doc_id)
        kwargs = self.config.to_deidentify_kwargs()
        kwargs["cache_results"] = False
        result = self._deidentifier_or_default()(text, **kwargs)
        artifact = canonical_redaction(
            result,
            source_text=text,
            doc_id=doc_id,
            lang=self.config.lang,
            method=self.config.method,
        )
        return {
            self.output_key: artifact.redacted_text,
            self.mapping_key: RedactionMapping(artifact.mapping),
            self.spans_key: artifact.spans,
        }

    def _deidentifier_or_default(self) -> Deidentifier:
        if self._deidentifier is not None:
            return self._deidentifier
        from openmed.core.pii import deidentify

        return deidentify


class GatewayBoundLLMNode:
    """Invoke an external text callable only after a privacy-gateway guard."""

    def __init__(
        self,
        external_call: ExternalTextCall | Any,
        *,
        gateway: PrivacyGateway,
        input_key: str = "redacted_text",
        mapping_key: str = "redaction_mapping",
        output_key: str = "redacted_response",
    ) -> None:
        self.external_call = external_call
        self.gateway = gateway
        self.input_key = input_key
        self.mapping_key = mapping_key
        self.output_key = output_key

    def __call__(
        self,
        state: Mapping[str, Any],
    ) -> dict[str, str]:
        """Send only gateway-validated redacted text to the external callable."""

        text = state.get(self.input_key)
        mapping = state.get(self.mapping_key)
        if not isinstance(text, str):
            raise TypeError(f"graph state {self.input_key!r} must be a string")
        if not isinstance(mapping, Mapping):
            raise TypeError(f"graph state {self.mapping_key!r} must be a mapping")
        protected_text = self.gateway.input_guardrail(mapping)(text)
        response = _invoke_external(self.external_call, protected_text)
        return {self.output_key: response}


class ReidentifyNode:
    """Restore a graph response locally from its in-memory mapping."""

    def __init__(
        self,
        *,
        gateway: PrivacyGateway | None = None,
        reidentifier: Reidentifier | None = None,
        input_key: str = "redacted_response",
        mapping_key: str = "redaction_mapping",
        output_key: str = "response",
    ) -> None:
        self.gateway = gateway or PrivacyGateway(reidentifier=reidentifier)
        self.input_key = input_key
        self.mapping_key = mapping_key
        self.output_key = output_key

    def __call__(
        self,
        state: Mapping[str, Any],
    ) -> dict[str, str]:
        """Return a partial state update produced entirely on device."""

        text = state.get(self.input_key)
        mapping = state.get(self.mapping_key)
        if not isinstance(text, str):
            raise TypeError(f"graph state {self.input_key!r} must be a string")
        if not isinstance(mapping, Mapping):
            raise TypeError(f"graph state {self.mapping_key!r} must be a mapping")
        return {self.output_key: self.gateway.output_guardrail(mapping)(text)}


def bind_state_graph(
    graph: Any,
    *,
    external_call: ExternalTextCall | Any,
    gateway: PrivacyGateway | None = None,
    deidentifier: Deidentifier | None = None,
    reidentifier: Reidentifier | None = None,
    gateway_config: PrivacyGatewayConfig | None = None,
    entry_point: Any | None = None,
    finish_point: Any | None = None,
) -> Any:
    """Add the safe three-node flow to a state-graph builder.

    The builder is duck typed, so importing this module never imports a graph
    framework. Pass the host framework's start/end sentinels when this helper
    should also connect the flow to graph boundaries.
    """

    resolved_config = gateway_config or (
        gateway.config if gateway is not None else PrivacyGatewayConfig()
    )
    shared_gateway = gateway or PrivacyGateway(
        config=resolved_config,
        deidentifier=deidentifier,
        reidentifier=reidentifier,
    )
    graph.add_node(
        "openmed_deidentify",
        DeidentifyNode(config=resolved_config, deidentifier=deidentifier),
    )
    graph.add_node(
        "openmed_external_llm",
        GatewayBoundLLMNode(external_call, gateway=shared_gateway),
    )
    graph.add_node(
        "openmed_reidentify",
        ReidentifyNode(gateway=shared_gateway),
    )
    if entry_point is not None:
        graph.add_edge(entry_point, "openmed_deidentify")
    graph.add_edge("openmed_deidentify", "openmed_external_llm")
    graph.add_edge("openmed_external_llm", "openmed_reidentify")
    if finish_point is not None:
        graph.add_edge("openmed_reidentify", finish_point)
    return graph


def create_state_graph(
    state_schema: Any = GraphState,
    *,
    external_call: ExternalTextCall | Any,
    gateway: PrivacyGateway | None = None,
    deidentifier: Deidentifier | None = None,
    reidentifier: Reidentifier | None = None,
    gateway_config: PrivacyGatewayConfig | None = None,
) -> Any:
    """Return a LangGraph builder with the safe OpenMed flow installed."""

    try:
        graph_module = _import_module("langgraph.graph")
        state_graph = graph_module.StateGraph
        start = graph_module.START
        end = graph_module.END
    except (ImportError, AttributeError) as exc:
        raise ImportError(
            "Graph orchestration requires the 'langgraph' extra. "
            "Install with `pip install openmed[langgraph]`."
        ) from exc

    graph = state_graph(state_schema)
    return bind_state_graph(
        graph,
        external_call=external_call,
        gateway=gateway,
        deidentifier=deidentifier,
        reidentifier=reidentifier,
        gateway_config=gateway_config,
        entry_point=start,
        finish_point=end,
    )


def create_tool_definitions() -> tuple[dict[str, Any], ...]:
    """Return graph-facing tool definitions from the shared registry."""

    return render_graph_orchestration_tool_definitions()


def _invoke_external(external_call: Any, text: str) -> str:
    invoke = getattr(external_call, "invoke", None)
    response = invoke(text) if callable(invoke) else external_call(text)
    if isinstance(response, str):
        return response
    if isinstance(response, Mapping):
        for key in ("content", "text", "response", "output"):
            value = response.get(key)
            if isinstance(value, str):
                return value
    for attribute in ("content", "text"):
        value = getattr(response, attribute, None)
        if isinstance(value, str):
            return value
    raise TypeError("external graph callable must return text content")


OpenMedDeidentifyNode = DeidentifyNode
OpenMedReidentifyNode = ReidentifyNode

__all__ = [
    "Deidentifier",
    "DeidentifyNode",
    "ExternalTextCall",
    "GatewayBoundLLMNode",
    "GraphState",
    "OpenMedDeidentifyNode",
    "OpenMedReidentifyNode",
    "OpenMedSpan",
    "Reidentifier",
    "ReidentifyNode",
    "bind_state_graph",
    "create_state_graph",
    "create_tool_definitions",
]
