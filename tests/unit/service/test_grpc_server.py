"""Unit tests for the OpenMed gRPC service."""

from __future__ import annotations

import importlib.util
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


def test_grpc_denies_request_selected_paths_before_loading(monkeypatch):
    import pytest

    from openmed.service.grpc_server import OpenMedGrpcServicer

    runtime = _runtime(monkeypatch)

    class Aborted(Exception):
        pass

    class Context:
        def set_trailing_metadata(self, metadata):
            assert tuple(metadata) == (("openmed-reason", "model_not_served"),)

        def abort(self, code, details):
            assert code == grpc.StatusCode.INVALID_ARGUMENT
            assert details == "model_not_served"
            raise Aborted()

    for model in ("/etc", "~/.ssh", "../synthetic", "unknown/model"):
        with pytest.raises(Aborted):
            OpenMedGrpcServicer(runtime).Analyze(
                openmed_pb2.AnalyzeRequest(text="synthetic", model_name=model),
                Context(),
            )
    assert runtime._loader is None


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


def test_grpc_generated_check_ignores_protobuf_patch_version(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location(
        "generate_grpc_stubs",
        repo_root / "scripts" / "generate_grpc_stubs.py",
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    expected = tmp_path / "expected_pb2.py"
    actual = tmp_path / "actual_pb2.py"
    expected.write_text(
        """# Protobuf Python Version: 6.33.5
_runtime_version.ValidateProtobufRuntimeVersion(
    _runtime_version.Domain.PUBLIC,
    6,
    33,
    5,
    '',
    'openmed.proto'
)
DESCRIPTOR = b'same-proto'
""",
        encoding="utf-8",
    )
    actual.write_text(
        """# Protobuf Python Version: 6.33.6
_runtime_version.ValidateProtobufRuntimeVersion(
    _runtime_version.Domain.PUBLIC,
    6,
    33,
    6,
    '',
    'openmed.proto'
)
DESCRIPTOR = b'same-proto'
""",
        encoding="utf-8",
    )

    assert module._generated_files_match(expected, actual, "openmed_pb2.py")


def _capabilities_to_descriptor(response) -> dict[str, Any]:
    """Rebuild the descriptor JSON from a ``CapabilitiesResponse``."""
    return {
        "schema": response.schema,
        "schema_version": response.schema_version,
        "backends": [
            {"name": item.name, "extra": item.extra, "available": item.available}
            for item in response.backends
        ],
        "surfaces": [
            {"name": item.name, "available": item.available}
            for item in response.surfaces
        ],
        "served_models": list(response.served_models),
        "schema_versions": [
            {"name": item.name, "version": item.version}
            for item in response.schema_versions
        ],
        "policy_profiles": list(response.policy_profiles),
        "digest": response.digest,
    }


def test_grpc_get_capabilities_returns_the_python_descriptor_digest(monkeypatch):
    from openmed.core.capabilities import (
        capability_descriptor,
        capability_descriptor_digest,
        validate_capability_descriptor,
    )

    monkeypatch.setenv(
        "OPENMED_SERVICE_SERVED_MODELS", "synthetic-org/synthetic-model,synthetic-a"
    )
    runtime = _runtime(monkeypatch)
    server, channel, stub = _grpc_stub(runtime)
    try:
        response = stub.GetCapabilities(
            openmed_pb2.GetCapabilitiesRequest(), timeout=10
        )
    finally:
        channel.close()
        server.stop(0).wait()

    descriptor = _capabilities_to_descriptor(response)
    expected = capability_descriptor(served_models=runtime.served_model_ids())
    assert descriptor == expected
    assert response.digest == expected["digest"]
    assert capability_descriptor_digest(descriptor) == response.digest
    validate_capability_descriptor(descriptor)
    assert descriptor["served_models"] == [
        "synthetic-a",
        "synthetic-org/synthetic-model",
    ]
    assert runtime._loader is None


def test_grpc_get_capabilities_carries_no_paths_or_environment_values(
    monkeypatch, tmp_path: Path
):
    from openmed.service.grpc_server import OpenMedGrpcServicer

    sentinel = "synthetic-grpc-capability-sentinel"
    monkeypatch.setenv("HF_TOKEN", sentinel)
    monkeypatch.setenv("OPENMED_CHINESE_USER_DICT", str(tmp_path / sentinel))
    monkeypatch.setenv("OPENMED_CLINICAL_PROTECT_TERMS", sentinel)
    runtime = _runtime(monkeypatch)

    class Context:
        def invocation_metadata(self):
            return ()

        def abort(self, code, details):
            raise AssertionError(f"unexpected abort: {code}")

    response = OpenMedGrpcServicer(runtime).GetCapabilities(
        openmed_pb2.GetCapabilitiesRequest(), Context()
    )

    rendered = response.SerializeToString() + str(response).encode("utf-8")
    assert sentinel.encode("utf-8") not in rendered
    assert str(tmp_path).encode("utf-8") not in rendered
    assert str(Path.home()).encode("utf-8") not in rendered
    assert response.digest.startswith("sha256:")


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


class HealthRuntime:
    """A private-valued runtime double; only preload lifecycle is observable."""

    def __init__(self, block=False, fail=False):
        import threading

        self.started = threading.Event()
        self.finished = threading.Event()
        self.block = block
        self.fail = fail
        self.calls = 0
        self.private_model = "synthetic-private-model-and-configuration"
        # The gRPC security policy reads throttle settings and the RPC timeout.
        from types import SimpleNamespace

        from openmed.service.runtime import parse_service_throttle_config

        self.throttle = parse_service_throttle_config()
        self.config = SimpleNamespace(timeout=0)

    def preload(self):
        self.calls += 1
        self.started.set()
        if self.block:
            assert self.finished.wait(3)
        if self.fail:
            raise ValueError(self.private_model)


def _health_stub(server, secure=False):
    from grpc_health.v1 import health_pb2_grpc

    port = server.add_insecure_port("127.0.0.1:0")
    assert port > 0
    server.start()
    channel = grpc.insecure_channel(f"127.0.0.1:{port}")
    grpc.channel_ready_future(channel).result(timeout=3)
    return channel, health_pb2_grpc.HealthStub(channel)


def _health_status(stub, service=""):
    from grpc_health.v1 import health_pb2

    response = stub.Check(health_pb2.HealthCheckRequest(service=service), timeout=3)
    assert [field.name for field, _value in response.ListFields()] == ["status"]
    assert "synthetic-private" not in str(response)
    return response.status


def test_standard_health_waits_for_actual_preload_and_stays_not_serving_after_drain():
    import threading

    from grpc_health.v1 import health_pb2

    runtime = HealthRuntime(block=True)
    server = create_grpc_server(runtime=runtime, max_workers=2)
    channel, stub = _health_stub(server)
    thread = threading.Thread(target=server.preload_runtime)
    try:
        service = openmed_pb2.DESCRIPTOR.services_by_name["OpenMedService"].full_name
        for name in ("", service):
            assert (
                _health_status(stub, name) == health_pb2.HealthCheckResponse.NOT_SERVING
            )
        thread.start()
        assert runtime.started.wait(3)
        assert _health_status(stub) == health_pb2.HealthCheckResponse.NOT_SERVING
        runtime.finished.set()
        thread.join(3)
        assert not thread.is_alive()
        for name in ("", service):
            assert _health_status(stub, name) == health_pb2.HealthCheckResponse.SERVING
        server.begin_drain()
        for name in ("", service):
            assert (
                _health_status(stub, name) == health_pb2.HealthCheckResponse.NOT_SERVING
            )
    finally:
        runtime.finished.set()
        if thread.is_alive():
            thread.join(3)
        channel.close()
        server.stop(0).wait()


def test_drain_during_preload_cannot_publish_serving_later():
    import threading

    from grpc_health.v1 import health_pb2

    runtime = HealthRuntime(block=True)
    server = create_grpc_server(runtime=runtime, max_workers=2)
    channel, stub = _health_stub(server)
    thread = threading.Thread(target=server.preload_runtime)
    try:
        thread.start()
        assert runtime.started.wait(3)
        server.begin_drain()
        runtime.finished.set()
        thread.join(3)
        assert not thread.is_alive()
        assert _health_status(stub) == health_pb2.HealthCheckResponse.NOT_SERVING
        with __import__("pytest").raises(RuntimeError, match="grpc_runtime_draining"):
            server.preload_runtime()
        assert runtime.calls == 1
    finally:
        runtime.finished.set()
        if thread.is_alive():
            thread.join(3)
        channel.close()
        server.stop(0).wait()


def test_failed_preload_health_and_error_are_value_free():
    import pytest
    from grpc_health.v1 import health_pb2

    runtime = HealthRuntime(fail=True)
    server = create_grpc_server(runtime=runtime, max_workers=2)
    channel, stub = _health_stub(server)
    try:
        with pytest.raises(RuntimeError, match="^grpc_preload_failed$") as failure:
            server.preload_runtime()
        assert failure.value.__cause__ is None
        assert failure.value.__context__ is None
        assert runtime.private_model not in str(failure.value)
        assert _health_status(stub) == health_pb2.HealthCheckResponse.NOT_SERVING
    finally:
        channel.close()
        server.stop(0).wait()


def test_standard_health_unknown_service_and_reflection_are_not_exposed():
    import pytest
    from grpc_health.v1 import health_pb2

    server = create_grpc_server(runtime=HealthRuntime(), max_workers=2)
    channel, stub = _health_stub(server)
    try:
        with pytest.raises(grpc.RpcError) as unknown:
            stub.Check(health_pb2.HealthCheckRequest(service="unknown"), timeout=3)
        assert unknown.value.code() == grpc.StatusCode.NOT_FOUND
        reflection = channel.unary_unary(
            "/grpc.reflection.v1alpha.ServerReflection/ServerReflectionInfo"
        )
        with pytest.raises(grpc.RpcError) as missing:
            reflection(b"", timeout=3)
        assert missing.value.code() == grpc.StatusCode.UNIMPLEMENTED
    finally:
        channel.close()
        server.stop(0).wait()


def test_standard_health_watch_receives_preload_and_drain_transitions():
    from grpc_health.v1 import health_pb2

    server = create_grpc_server(runtime=HealthRuntime(), max_workers=2)
    channel, stub = _health_stub(server)
    stream = stub.Watch(health_pb2.HealthCheckRequest(service=""), timeout=3)
    try:
        assert next(stream).status == health_pb2.HealthCheckResponse.NOT_SERVING
        server.preload_runtime()
        assert next(stream).status == health_pb2.HealthCheckResponse.SERVING
        server.begin_drain()
        assert next(stream).status == health_pb2.HealthCheckResponse.NOT_SERVING
    finally:
        stream.cancel()
        channel.close()
        server.stop(0).wait()


def test_tls_listener_delegates_credentials_and_serves_the_same_health_protocol():
    import ipaddress
    from datetime import timedelta, timezone

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID
    from grpc_health.v1 import health_pb2, health_pb2_grpc

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
    now = datetime.now(timezone.utc)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(days=1))
        .not_valid_after(now + timedelta(days=1))
        .add_extension(
            x509.SubjectAlternativeName(
                [
                    x509.DNSName("localhost"),
                    x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
                ]
            ),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    cert = certificate.public_bytes(serialization.Encoding.PEM)
    secret_key = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    server = create_grpc_server(runtime=HealthRuntime(), max_workers=2)
    port = server.add_secure_port(
        "127.0.0.1:0", grpc.ssl_server_credentials(((secret_key, cert),))
    )
    assert port > 0
    server.start()
    channel = grpc.secure_channel(
        f"127.0.0.1:{port}", grpc.ssl_channel_credentials(root_certificates=cert)
    )
    try:
        grpc.channel_ready_future(channel).result(timeout=3)
        stub = health_pb2_grpc.HealthStub(channel)
        assert _health_status(stub) == health_pb2.HealthCheckResponse.NOT_SERVING
        server.preload_runtime()
        assert _health_status(stub) == health_pb2.HealthCheckResponse.SERVING
        server.begin_drain()
        assert _health_status(stub) == health_pb2.HealthCheckResponse.NOT_SERVING
    finally:
        channel.close()
        server.stop(0).wait()
