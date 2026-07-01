"""Unit tests for the OpenMed gRPC service."""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import grpc
from google.protobuf.json_format import MessageToDict

import openmed
from openmed.core.pii import DeidentificationResult, PIIEntity
from openmed.core.streaming import StreamingDeidentificationEvent
from openmed.service import runtime as service_runtime
from openmed.service.app import _pii_deidentify_payload
from openmed.service.grpc_server import create_grpc_server
from openmed.service.proto.generated import openmed_pb2, openmed_pb2_grpc
from openmed.service.schemas import PIIDeidentifyRequest


class FakeLoader:
    """Loader double used to prove the shared warm-pool is passed through."""

    instances: list["FakeLoader"] = []

    def __init__(self, config):
        self.config = config
        self.pipelines = {}
        FakeLoader.instances.append(self)

    @classmethod
    def reset(cls) -> None:
        cls.instances = []

    def resolve_model_name(self, model_name: str) -> str:
        return model_name

    def create_pipeline(self, model_name: str, **kwargs: Any):
        key = (model_name, tuple(sorted(kwargs.items())))
        self.pipelines.setdefault(key, object())
        return self.pipelines[key]

    def loaded_models(self) -> dict[str, Any]:
        return {}


def _sample_deid_result() -> DeidentificationResult:
    entity = PIIEntity(
        text="Maria Garcia",
        label="NAME",
        confidence=0.98,
        start=9,
        end=21,
        entity_type="NAME",
        redacted_text="[NAME]",
        canonical_label="PERSON",
        sources=["rules"],
        evidence={"detector": "fixture"},
        threshold=0.7,
        action="mask",
        metadata={"source": "unit"},
    )
    return DeidentificationResult(
        original_text="Patient Maria Garcia visited cardiology.",
        deidentified_text="Patient [NAME] visited cardiology.",
        pii_entities=[entity],
        method="mask",
        timestamp=datetime(2026, 1, 1, 12, 0, 0),
        metadata={"lang": "en"},
    )


def _runtime(monkeypatch):
    FakeLoader.reset()
    monkeypatch.setenv("OPENMED_PROFILE", "test")
    monkeypatch.delenv("OPENMED_SERVICE_PRELOAD_MODELS", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_KEEP_ALIVE", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_MAX_RESIDENT_MODELS", raising=False)
    monkeypatch.setattr(service_runtime, "ModelLoader", FakeLoader)
    return service_runtime.ServiceRuntime.from_env()


def _grpc_stub(runtime):
    server = create_grpc_server(runtime=runtime, max_workers=2)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    channel = grpc.insecure_channel(f"127.0.0.1:{port}")
    stub = openmed_pb2_grpc.OpenMedServiceStub(channel)
    return server, channel, stub


def test_grpc_deidentify_matches_rest_payload_entities(monkeypatch):
    runtime = _runtime(monkeypatch)
    seen_loaders: list[Any] = []

    def fake_deidentify(*args, **kwargs):
        seen_loaders.append(kwargs["loader"])
        return _sample_deid_result()

    monkeypatch.setattr(openmed, "deidentify", fake_deidentify)
    rest_request = PIIDeidentifyRequest(
        text="Patient Maria Garcia visited cardiology.",
        method="mask",
        lang="en",
    )
    rest_payload = runtime.run_model_request(
        rest_request.model_name,
        rest_request.keep_alive,
        lambda: _pii_deidentify_payload(rest_request, runtime),
    )

    server, channel, stub = _grpc_stub(runtime)
    try:
        response = stub.Deidentify(
            openmed_pb2.DeidentifyRequest(
                text="Patient Maria Garcia visited cardiology.",
                method="mask",
                lang="en",
            )
        )
    finally:
        channel.close()
        server.stop(0).wait()

    assert response.deidentified_text == rest_payload["deidentified_text"]
    assert [_pii_entity_to_rest(entity) for entity in response.pii_entities] == (
        rest_payload["pii_entities"]
    )
    assert len({id(loader) for loader in seen_loaders}) == 1
    assert seen_loaders[0] is runtime.get_loader()


def test_grpc_stream_deidentify_yields_chunked_results_offline(monkeypatch):
    runtime = _runtime(monkeypatch)
    observed: dict[str, Any] = {}

    def fake_deidentify_stream(chunks, **kwargs):
        observed["chunks"] = tuple(chunks)
        observed["loader"] = kwargs["loader"]
        yield StreamingDeidentificationEvent(redacted_text="Patient [NAME] ")
        yield StreamingDeidentificationEvent(redacted_text="was discharged.")
        yield StreamingDeidentificationEvent(
            redacted_text="",
            final=True,
            audit_record={"chunks": 2},
        )

    monkeypatch.setattr(
        "openmed.service.grpc_server.deidentify_stream",
        fake_deidentify_stream,
    )
    server, channel, stub = _grpc_stub(runtime)
    try:
        responses = list(
            stub.StreamDeidentify(
                openmed_pb2.DeidentifyStreamRequest(
                    request=openmed_pb2.DeidentifyRequest(method="mask", lang="en"),
                    chunks=["Patient Maria ", "was discharged."],
                    max_buffer=128,
                )
            )
        )
    finally:
        channel.close()
        server.stop(0).wait()

    assert [event.redacted_text for event in responses] == [
        "Patient [NAME] ",
        "was discharged.",
        "",
    ]
    assert responses[-1].final is True
    assert MessageToDict(responses[-1].audit_record) == {"chunks": 2}
    assert observed["chunks"] == ("Patient Maria ", "was discharged.")
    assert observed["loader"] is runtime.get_loader()


def test_grpc_generated_stubs_are_current():
    repo_root = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "generate_grpc_stubs.py"),
            "--check",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def _pii_entity_to_rest(entity) -> dict[str, Any]:
    return {
        "text": entity.text,
        "label": entity.label,
        "entity_type": entity.entity_type,
        "start": entity.start if entity.HasField("start") else None,
        "end": entity.end if entity.HasField("end") else None,
        "confidence": entity.confidence,
        "redacted_text": (
            entity.redacted_text if entity.HasField("redacted_text") else None
        ),
        "canonical_label": (
            entity.canonical_label if entity.HasField("canonical_label") else None
        ),
        "sources": list(entity.sources),
        "evidence": MessageToDict(
            entity.evidence,
            preserving_proto_field_name=True,
        ),
        "threshold": entity.threshold if entity.HasField("threshold") else None,
        "action": entity.action if entity.HasField("action") else None,
        "surrogate": entity.surrogate if entity.HasField("surrogate") else None,
        "metadata": MessageToDict(
            entity.metadata,
            preserving_proto_field_name=True,
        ),
    }
