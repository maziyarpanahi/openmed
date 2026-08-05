"""Security and round-trip tests for redaction-preserving retrieval."""

from __future__ import annotations

import json
import logging
import socket
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from openmed.interop.langchain import create_retrieval_chain
from openmed.interop.retrieval import (
    AuthorizedReidentifier,
    EncryptedMappingVault,
    GatewayBoundExternalLLM,
    InMemoryVectorStore,
    InvalidDocumentKeyError,
    MappingVaultIntegrityError,
    RedactedIndex,
    RedactedRetriever,
    UnauthorizedPrincipalError,
)
from openmed.service.privacy_gateway import reidentify_placeholders


@dataclass(frozen=True)
class _Entity:
    label: str
    start: int
    end: int


class _MockGateway:
    def __init__(self) -> None:
        self.external_payloads: list[dict[str, Any]] = []
        self.reidentification_calls = 0

    def complete_redacted(self, payload: dict[str, Any]) -> str:
        self.external_payloads.append(payload)
        passages = payload["passages"]
        return passages[0]["text"] if passages else "No result"

    def reidentify(
        self,
        text: str,
        *,
        mapping: dict[str, str],
        principal: str,
        request_id: str,
    ) -> str:
        del principal, request_id
        self.reidentification_calls += 1
        return reidentify_placeholders(text, mapping)


def _detector(protected: dict[str, str]):
    def detect(text: str, **kwargs: Any) -> SimpleNamespace:
        if kwargs:
            assert kwargs["keep_mapping"] is False
            assert kwargs.get("audit", False) is False
            assert kwargs.get("cache_results", False) is False
        entities: list[_Entity] = []
        for surface, label in protected.items():
            cursor = 0
            while (start := text.find(surface, cursor)) >= 0:
                entities.append(_Entity(label, start, start + len(surface)))
                cursor = start + len(surface)
        entities.sort(key=lambda entity: entity.start)
        redacted = text
        for entity in reversed(entities):
            redacted = redacted[: entity.start] + "[PHI]" + redacted[entity.end :]
        return SimpleNamespace(
            deidentified_text=redacted,
            pii_entities=entities,
        )

    return detect


def _paths(tmp_path: Path) -> tuple[Path, Path]:
    return tmp_path / "redacted-index.json", tmp_path / "mapping-vault.json"


def test_fuzz_corpus_has_zero_phi_in_index_retrieval_and_external_boundary(
    tmp_path,
):
    protected = {
        "Avery Patient": "PERSON",
        "Morgan Example": "PERSON",
        "avery@example.test": "EMAIL",
        "555-010-0042": "PHONE",
        "MRN-000042": "MEDICAL_RECORD_NUMBER",
        "1980-02-03": "DATE",
        "SECRET_IDENTIFIER": "SECRET_IDENTIFIER",
    }
    notes = (
        "Avery Patient has MRN-000042 and starts metformin.",
        "Morgan Example called 555-010-0042 about follow-up.",
        "Email avery@example.test after the 1980-02-03 visit.",
        "Token SECRET_IDENTIFIER requires local review.",
    )
    index_path, vault_path = _paths(tmp_path)
    vault = EncryptedMappingVault.from_file(
        vault_path,
        secret="synthetic-retrieval-secret",
    )
    index = RedactedIndex.from_file(index_path)
    detector = _detector(protected)

    results = [
        index.index_document(
            f"synthetic-{ordinal}",
            note,
            vault=vault,
            chunk_size=47,
            deidentifier=detector,
        )
        for ordinal, note in enumerate(notes)
    ]
    passages = RedactedRetriever(
        index,
        vector_store=InMemoryVectorStore(),
    ).retrieve("metformin follow-up visit", k=8)
    gateway = _MockGateway()
    GatewayBoundExternalLLM(gateway).invoke("summarize follow-up", passages)

    external_json = json.dumps(gateway.external_payloads, sort_keys=True)
    indexed_json = index_path.read_text(encoding="utf-8")
    vault_json = vault_path.read_text(encoding="utf-8")
    retrieved_text = "".join(passage.text for passage in passages)
    all_redacted_text = "".join(result.redacted_text for result in results)
    for raw_phi in protected:
        assert raw_phi not in indexed_json
        assert raw_phi not in vault_json
        assert raw_phi not in retrieved_text
        assert raw_phi not in all_redacted_text
        assert raw_phi not in external_json
    assert all(result.placeholders for result in results)
    assert all(isinstance(passage.placeholders, tuple) for passage in passages)
    assert all(
        "placeholders" in passage
        for payload in gateway.external_payloads
        for passage in payload["passages"]
    )

    reloaded_index = RedactedIndex.from_file(index_path)
    reloaded_vault = EncryptedMappingVault.from_file(
        vault_path,
        secret="synthetic-retrieval-secret",
    )
    assert reloaded_index.chunks == index.chunks
    assert len(reloaded_vault) == len(notes)


def test_authorization_audits_once_and_round_trip_uses_only_vault(
    tmp_path,
    monkeypatch,
):
    note = "Avery Patient (MRN-000042) takes metformin twice daily."
    protected = {
        "Avery Patient": "PERSON",
        "MRN-000042": "MEDICAL_RECORD_NUMBER",
    }
    vault = EncryptedMappingVault.in_memory("synthetic-retrieval-secret")
    index = RedactedIndex()
    indexed = index.index_document(
        "synthetic-note",
        note,
        vault=vault,
        deidentifier=_detector(protected),
    )
    gateway = _MockGateway()
    reidentifier = AuthorizedReidentifier(
        vault=vault,
        gateway_proxy=gateway,
        allowed_principals=("clinician-1",),
    )

    def fail_if_index_is_touched(*args, **kwargs):
        del args, kwargs
        raise AssertionError("round-trip re-identification touched the index")

    monkeypatch.setattr(index, "chunks_for_document", fail_if_index_is_touched)
    with pytest.raises(UnauthorizedPrincipalError):
        reidentifier.reidentify(
            indexed.redacted_text,
            document_keys=(indexed.document_key,),
            principal="researcher-1",
        )

    assert len(reidentifier.audit_trail.records) == 1
    assert reidentifier.audit_trail.records[0].status == "denied"
    assert gateway.reidentification_calls == 0

    restored = reidentifier.reidentify(
        indexed.redacted_text,
        document_keys=(indexed.document_key,),
        principal="clinician-1",
    )

    assert restored.encode("utf-8") == note.encode("utf-8")
    assert len(reidentifier.audit_trail.records) == 2
    assert reidentifier.audit_trail.records[1].status == "succeeded"
    assert reidentifier.audit_trail.verify() is True
    assert gateway.reidentification_calls == 1


def test_index_retriever_are_socket_free_and_chain_uses_explicit_gateway(
    monkeypatch,
):
    note = "Avery Patient needs a local follow-up summary."
    protected = {"Avery Patient": "PERSON"}
    vault = EncryptedMappingVault.in_memory("synthetic-retrieval-secret")
    index = RedactedIndex()

    def network_forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("unexpected socket access")

    monkeypatch.setattr(socket, "socket", network_forbidden)
    monkeypatch.setattr(socket, "create_connection", network_forbidden)
    indexed = index.index_document(
        "synthetic-note",
        note,
        vault=vault,
        deidentifier=_detector(protected),
    )
    retriever = RedactedRetriever(index)
    passages = retriever.retrieve("local follow-up")

    gateway = _MockGateway()
    response = GatewayBoundExternalLLM(gateway).invoke("local follow-up", passages)

    assert indexed.document_key in response.document_keys
    assert len(gateway.external_payloads) == 1
    assert "Avery Patient" not in json.dumps(gateway.external_payloads)


def test_langchain_composition_and_phi_free_index_logs_vault_and_audit(
    tmp_path,
    caplog,
):
    note = "Avery Patient (MRN-000042) needs metformin follow-up."
    protected = {
        "Avery Patient": "PERSON",
        "MRN-000042": "MEDICAL_RECORD_NUMBER",
    }
    index_path, vault_path = _paths(tmp_path)
    vault = EncryptedMappingVault.from_file(
        vault_path,
        secret="synthetic-retrieval-secret",
    )
    index = RedactedIndex.from_file(index_path)
    retriever = RedactedRetriever(index)
    gateway = _MockGateway()
    reidentifier = AuthorizedReidentifier(
        vault=vault,
        gateway_proxy=gateway,
        allowed_principals=("clinician-1",),
    )
    chain = create_retrieval_chain(
        index=index,
        vault=vault,
        retriever=retriever,
        external_llm=GatewayBoundExternalLLM(gateway),
        reidentifier=reidentifier,
        deidentifier=_detector(protected),
    )

    with caplog.at_level(logging.DEBUG):
        chain.index_document("synthetic-note", note, chunk_size=256)
        restored = chain.invoke(
            {"query": "metformin follow-up", "principal": "clinician-1"}
        )

    assert restored == note
    persisted = index_path.read_text(encoding="utf-8") + vault_path.read_text(
        encoding="utf-8"
    )
    audit_json = reidentifier.audit_trail.to_json()
    logs = "\n".join(record.getMessage() for record in caplog.records)
    for raw_phi in protected:
        assert raw_phi not in persisted
        assert raw_phi not in audit_json
        assert raw_phi not in logs
    assert "clinician-1" not in audit_json
    assert (
        reidentifier.audit_trail.contains_plaintext([*protected, "clinician-1"])
        is False
    )


def test_encrypted_vault_rejects_wrong_key_and_tampering(tmp_path):
    vault_path = tmp_path / "mapping-vault.json"
    vault = EncryptedMappingVault.from_file(
        vault_path,
        secret="synthetic-retrieval-secret",
    )
    index = RedactedIndex()
    index.index_document(
        "synthetic-note",
        "Avery Patient needs follow-up.",
        vault=vault,
        deidentifier=_detector({"Avery Patient": "PERSON"}),
    )

    with pytest.raises(MappingVaultIntegrityError):
        EncryptedMappingVault.from_file(
            vault_path,
            secret="different-synthetic-secret",
        )

    payload = json.loads(vault_path.read_text(encoding="utf-8"))
    payload["entries"][0]["ciphertext"] = "AA=="
    vault_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(MappingVaultIntegrityError):
        EncryptedMappingVault.from_file(
            vault_path,
            secret="synthetic-retrieval-secret",
        )


def test_invalid_document_reference_cannot_enter_audit_records():
    raw_value = "Avery Patient"
    vault = EncryptedMappingVault.in_memory("synthetic-retrieval-secret")
    gateway = _MockGateway()
    reidentifier = AuthorizedReidentifier(
        vault=vault,
        gateway_proxy=gateway,
        allowed_principals=("clinician-1",),
    )

    with pytest.raises(InvalidDocumentKeyError):
        reidentifier.reidentify(
            "redacted text",
            document_keys=(raw_value,),
            principal="clinician-1",
        )

    assert len(reidentifier.audit_trail.records) == 1
    assert reidentifier.audit_trail.records[0].status == "rejected"
    assert raw_value not in reidentifier.audit_trail.to_json()
    assert gateway.reidentification_calls == 0


def test_chain_refuses_a_direct_external_model_boundary():
    vault = EncryptedMappingVault.in_memory("synthetic-retrieval-secret")
    index = RedactedIndex()
    gateway = _MockGateway()
    reidentifier = AuthorizedReidentifier(
        vault=vault,
        gateway_proxy=gateway,
        allowed_principals=("clinician-1",),
    )

    with pytest.raises(TypeError, match="gateway-bound"):
        create_retrieval_chain(
            index=index,
            vault=vault,
            retriever=RedactedRetriever(index),
            external_llm=gateway,
            reidentifier=reidentifier,
            deidentifier=_detector({}),
        )
