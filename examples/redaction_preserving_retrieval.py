"""Synthetic, offline redaction-preserving retrieval walkthrough."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from openmed.interop.langchain import create_retrieval_chain
from openmed.interop.retrieval import (
    AuthorizedReidentifier,
    EncryptedMappingVault,
    GatewayBoundExternalLLM,
    InMemoryVectorStore,
    RedactedIndex,
    RedactedRetriever,
    UnauthorizedPrincipalError,
)
from openmed.service.privacy_gateway import reidentify_placeholders


@dataclass(frozen=True)
class _SyntheticEntity:
    label: str
    start: int
    end: int


class _MockPrivacyGateway:
    """Offline stand-in for both gateway operations used by the example."""

    def complete_redacted(self, payload: dict[str, Any]) -> str:
        passages = payload["passages"]
        context = passages[0]["text"] if passages else "No matching context."
        return f"Mock external summary: {context}"

    def reidentify(
        self,
        text: str,
        *,
        mapping: dict[str, str],
        principal: str,
        request_id: str,
    ) -> str:
        del principal, request_id
        return reidentify_placeholders(text, mapping)


def _synthetic_deidentifier(text: str, **_: Any) -> SimpleNamespace:
    protected = {
        "Avery Patient": "PERSON",
        "MRN-000042": "MEDICAL_RECORD_NUMBER",
    }
    entities = [
        _SyntheticEntity(label, start, start + len(surface))
        for surface, label in protected.items()
        if (start := text.find(surface)) >= 0
    ]
    entities.sort(key=lambda entity: entity.start)
    redacted = text
    for entity in reversed(entities):
        redacted = redacted[: entity.start] + "[PHI]" + redacted[entity.end :]
    return SimpleNamespace(deidentified_text=redacted, pii_entities=entities)


def main() -> None:
    """Run a local index, mock external model, and two-role authorization demo."""

    note = "Avery Patient (MRN-000042) started metformin and will return for follow-up."
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        vault = EncryptedMappingVault.from_file(
            root / "mapping-vault.json",
            secret="synthetic-example-secret",
        )
        index = RedactedIndex.from_file(root / "redacted-index.json")
        local_vector_store = InMemoryVectorStore()
        retriever = RedactedRetriever(index, vector_store=local_vector_store)
        gateway = _MockPrivacyGateway()
        audit_trail_reidentifier = AuthorizedReidentifier(
            vault=vault,
            gateway_proxy=gateway,
            allowed_principals=("clinician",),
        )
        chain = create_retrieval_chain(
            index=index,
            vault=vault,
            retriever=retriever,
            external_llm=GatewayBoundExternalLLM(gateway),
            reidentifier=audit_trail_reidentifier,
            deidentifier=_synthetic_deidentifier,
        )
        chain.index_document("synthetic-note-1", note, chunk_size=256)

        try:
            chain.invoke({"query": "metformin follow-up", "principal": "researcher"})
        except UnauthorizedPrincipalError:
            print("researcher: re-identification denied")

        restored = chain.invoke(
            {"query": "metformin follow-up", "principal": "clinician"}
        )
        print(f"clinician: {restored}")
        print(
            "audit statuses:",
            [record.status for record in audit_trail_reidentifier.audit_trail.records],
        )


if __name__ == "__main__":
    main()
