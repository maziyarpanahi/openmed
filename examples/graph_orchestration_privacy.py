"""Synthetic LangGraph flow with an OpenMed privacy-gateway boundary."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, TypedDict

from openmed.interop.gateway import PrivacyGateway, RedactionMapping
from openmed.interop.graph_orchestration import create_state_graph


@dataclass(frozen=True)
class _SyntheticEntity:
    label: str
    canonical_label: str
    start: int
    end: int
    confidence: float = 1.0
    redacted_text: str = "[PERSON]"
    entity_type: str = "PERSON"


class _GraphState(TypedDict, total=False):
    text: str
    doc_id: str
    redacted_text: str
    redaction_mapping: RedactionMapping
    spans: tuple[Any, ...]
    redacted_response: str
    response: str


class _MockExternalLLM:
    def invoke(self, redacted_text: str) -> str:
        if "Avery Patient" in redacted_text:
            raise RuntimeError("mock external model received raw PHI")
        return f"Mock external summary: {redacted_text}"


def _synthetic_deidentify(text: str, **_: Any) -> SimpleNamespace:
    surface = "Avery Patient"
    start = text.index(surface)
    placeholder = "[PERSON]"
    return SimpleNamespace(
        deidentified_text=text.replace(surface, placeholder),
        mapping={placeholder: surface},
        pii_entities=[
            _SyntheticEntity(
                label="PERSON",
                canonical_label="PERSON",
                start=start,
                end=start + len(surface),
            )
        ],
    )


def _local_reidentify(text: str, mapping: dict[str, str]) -> str:
    for placeholder, original in mapping.items():
        text = text.replace(placeholder, original)
    return text


def main() -> None:
    """Run redact -> mock external model -> local restore in LangGraph."""

    gateway = PrivacyGateway(
        deidentifier=_synthetic_deidentify,
        reidentifier=_local_reidentify,
    )
    graph = create_state_graph(
        _GraphState,
        external_call=_MockExternalLLM(),
        gateway=gateway,
        deidentifier=_synthetic_deidentify,
        reidentifier=_local_reidentify,
    ).compile()
    result = graph.invoke(
        {
            "text": "Avery Patient started metformin.",
            "doc_id": "synthetic-note",
        }
    )
    print(result["response"])
    print([span.canonical_label for span in result["spans"]])


if __name__ == "__main__":
    main()
