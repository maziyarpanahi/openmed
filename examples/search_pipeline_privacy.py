"""Synthetic Haystack pipeline with local OpenMed document redaction."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from openmed.interop.gateway import PrivacyGateway
from openmed.interop.search_pipeline import bind_search_pipeline


@dataclass(frozen=True)
class _SyntheticEntity:
    label: str
    canonical_label: str
    start: int
    end: int
    confidence: float = 1.0
    redacted_text: str = "[PERSON]"
    entity_type: str = "PERSON"


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


def _mock_external_llm(redacted_text: str) -> str:
    if "Avery Patient" in redacted_text:
        raise RuntimeError("mock external model received raw PHI")
    return f"Mock external summary: {redacted_text}"


def main() -> None:
    """Run a Haystack redactor and a gateway-guarded mock external model."""

    from haystack import Document, Pipeline

    note = "Avery Patient started metformin."
    pipeline = bind_search_pipeline(
        Pipeline(),
        deidentifier=_synthetic_deidentify,
    )
    result = pipeline.run(
        {"openmed_redaction": {"documents": [Document(content=note)]}}
    )
    redacted = result["openmed_redaction"]["documents"][0].content

    gateway = PrivacyGateway(
        deidentifier=_synthetic_deidentify,
        reidentifier=_local_reidentify,
    )
    clean_text, mapping = gateway.redact(note)
    assert redacted == clean_text
    external_response = _mock_external_llm(gateway.input_guardrail(mapping)(redacted))
    restored = gateway.output_guardrail(mapping)(external_response)

    print(restored)
    print([span.canonical_label for span in result["openmed_redaction"]["spans"]])


if __name__ == "__main__":
    main()
