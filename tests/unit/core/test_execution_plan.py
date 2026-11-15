"""Tests for the execution-plan resolver and its shared golden vectors."""

from __future__ import annotations

import ast
import base64
import dataclasses
import hashlib
import importlib.util
import json
import socket
import sys
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from openmed.core import device_capabilities as dc
from openmed.core import execution_plan as ep
from openmed.core import portable_bundle as pb

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "device_capabilities"
VECTORS = json.loads((FIXTURES / "vectors.json").read_text(encoding="utf-8"))
CASES = VECTORS["cases"]
CASES_BY_ID = {case["id"]: case for case in CASES}
PLAN_SCHEMA = Draft202012Validator(ep.load_execution_plan_schema("execution_plan"))
MODULES = [
    Path(ep.__file__),
    Path(dc.__file__),
]


def _generator():
    spec = importlib.util.spec_from_file_location(
        "device_capability_vector_generator", FIXTURES / "generate_vectors.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _manifest(name: str) -> bytes:
    return VECTORS["manifests"][name]["text"].encode("utf-8")


def _verification(raw: dict[str, Any]) -> pb.PortableBundleVerification:
    return pb.PortableBundleVerification(
        ok=raw["ok"],
        reason=raw["reason"],
        content_id=raw["content_id"],
        valid_signatures=raw["valid_signatures"],
        eligible_runtimes=tuple(raw["eligible_runtimes"]),
        files_checked=raw["files_checked"],
        files_verified=raw["files_verified"],
    )


def _inputs(case: dict[str, Any]) -> tuple[Any, ...]:
    return (
        _manifest(case["manifest"]),
        _verification(case["verification"]),
        dc.DeviceCapabilityProfile.from_mapping(
            VECTORS["profiles"][case["profile"]]["document"]
        ),
        ep.ExecutionConstraints.from_mapping(
            VECTORS["constraints"][case["constraints"]]["document"]
        ),
    )


def _resolve(case: dict[str, Any]) -> ep.ExecutionPlan:
    return ep.resolve_execution_plan(*_inputs(case))


def test_committed_vectors_match_the_generator() -> None:
    assert (FIXTURES / "vectors.json").read_text(encoding="utf-8") == (
        _generator().render()
    )


def test_vectors_are_synthetic() -> None:
    assert VECTORS["synthetic_only"] is True
    generator = (FIXTURES / "generate_vectors.py").read_text(encoding="utf-8")
    assert "synthetic" in generator.lower()
    assert "SENTINEL" in VECTORS["notice"]


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_golden_case_matches_reference_plan(case: dict[str, Any]) -> None:
    plan = _resolve(case)
    assert plan.to_dict() == case["expected"]
    PLAN_SCHEMA.validate(case["expected"])


def test_every_reason_code_and_assumption_has_a_golden_vector() -> None:
    covered = {case["expected"]["reason"] for case in CASES} - {None}
    for case in CASES:
        covered |= {item["reason"] for item in case["expected"]["skipped"]}
    assert covered == set(ep.REASON_CODES)
    assumed = {item for case in CASES for item in case["expected"]["assumptions"]}
    assert assumed == set(ep.ASSUMPTIONS)
    assert VECTORS["vocabulary"]["reason_codes"] == list(ep.REASON_CODES)


def test_vocabulary_block_matches_resolver_constants() -> None:
    vocabulary = VECTORS["vocabulary"]
    assert vocabulary["format_providers"] == {
        key: sorted(value) for key, value in ep.FORMAT_PROVIDERS.items()
    }
    assert vocabulary["default_providers"] == {
        key: list(value) for key, value in ep.DEFAULT_PROVIDERS.items()
    }
    assert vocabulary["step_roles"] == list(ep.STEP_ROLES)
    assert vocabulary["assumptions"] == list(ep.ASSUMPTIONS)
    assert vocabulary["invalidation_components"] == list(ep.INVALIDATION_COMPONENTS)
    assert vocabulary["execution_extension"] == ep.EXECUTION_EXTENSION
    assert vocabulary["resolver_version"] == ep.RESOLVER_VERSION
    assert set(ep.FORMAT_PROVIDERS) == set(pb.FORMAT_PLATFORMS)
    for runtime_format, providers in ep.DEFAULT_PROVIDERS.items():
        assert set(providers) <= ep.FORMAT_PROVIDERS[runtime_format]


def test_reason_codes_are_unique_and_value_free() -> None:
    assert len(set(ep.REASON_CODES)) == len(ep.REASON_CODES)
    for code in ep.REASON_CODES:
        assert code == code.lower()
        assert all(char.isalpha() or char == "_" for char in code)


def test_schema_enums_match_constants() -> None:
    plan = ep.load_execution_plan_schema("execution_plan")["properties"]
    assert plan["reason"]["oneOf"][1]["enum"] == list(ep.REASON_CODES)
    assert plan["assumptions"]["items"]["enum"] == list(ep.ASSUMPTIONS)
    step = plan["steps"]["items"]["properties"]
    assert step["role"]["enum"] == list(ep.STEP_ROLES)
    assert step["provider"]["enum"] == sorted(dc.PROVIDERS)
    for name in ep.SCHEMA_NAMES:
        Draft202012Validator.check_schema(ep.load_execution_plan_schema(name))
    with pytest.raises(KeyError):
        ep.load_execution_plan_schema("portable_bundle")


@pytest.mark.parametrize("name", sorted(VECTORS["constraints"]))
def test_constraint_documents_round_trip(name: str) -> None:
    entry = VECTORS["constraints"][name]
    constraints = ep.ExecutionConstraints.from_mapping(entry["document"])
    assert constraints.to_dict() == entry["document"]
    assert constraints.digest() == entry["digest"]
    Draft202012Validator(
        ep.load_execution_plan_schema("execution_constraints")
    ).validate(entry["document"])


def test_default_constraints_are_local_and_require_verified_files() -> None:
    constraints = ep.ExecutionConstraints()
    assert constraints.local_only is True
    assert constraints.require_verified_files is True
    assert constraints.max_steps == ep.DEFAULT_MAX_STEPS
    assert constraints.to_dict() == VECTORS["constraints"]["default"]["document"]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda doc: doc.update(remote_endpoint="https://example.invalid"),
        lambda doc: doc.pop("energy"),
        lambda doc: doc.update(schema="openmed.other"),
        lambda doc: doc.update(local_only="yes"),
        lambda doc: doc.update(max_steps=0),
        lambda doc: doc.update(max_steps=ep.MAX_STEPS_LIMIT + 1),
        lambda doc: doc.update(deadline_ms=0),
        lambda doc: doc.update(energy="turbo"),
        lambda doc: doc.update(allowed_precisions=["fp64"]),
        lambda doc: doc.update(disallowed_providers=["cloud"]),
        lambda doc: doc.update(revoked_artifacts=["md5:abc"]),
        lambda doc: doc.update(required_evidence=["Clinical Validation"]),
        lambda doc: doc.update(max_memory_bytes=-1),
        lambda doc: doc.update(min_sequence_length=True),
    ],
)
def test_constraint_documents_are_closed(mutate: Any) -> None:
    document = json.loads(json.dumps(VECTORS["constraints"]["default"]["document"]))
    mutate(document)
    with pytest.raises(ep.ExecutionConstraintsError, match="^constraints_invalid$"):
        ep.ExecutionConstraints.from_mapping(document)


@pytest.mark.parametrize(
    "item",
    VECTORS["invalidations"],
    ids=[f"{i['plan']}->{i['inputs']}" for i in VECTORS["invalidations"]],
)
def test_invalidation_vectors(item: dict[str, Any]) -> None:
    plan = _resolve(CASES_BY_ID[item["plan"]])
    current = _inputs(CASES_BY_ID[item["inputs"]])
    assert list(ep.plan_invalidation(plan, *current)) == item["expected"]


def test_resolver_version_change_invalidates_cached_plans() -> None:
    plan = _resolve(CASES_BY_ID["web_webgpu_then_wasm"])
    old_key = ep.compute_plan_key(
        plan.bundle_content_id,
        plan.profile_digest,
        plan.constraints_digest,
        plan.verification_digest,
        resolver_version=0,
    )
    stale = dataclasses.replace(plan, plan_key=old_key)
    current = _inputs(CASES_BY_ID["web_webgpu_then_wasm"])
    assert ep.plan_invalidation(stale, *current) == ("resolver",)
    assert ep.plan_invalidation(plan, *current) == ()


def test_plan_key_binds_every_input_digest() -> None:
    plan = _resolve(CASES_BY_ID["web_webgpu_then_wasm"])
    assert plan.plan_key == ep.compute_plan_key(
        plan.bundle_content_id,
        plan.profile_digest,
        plan.constraints_digest,
        plan.verification_digest,
    )
    keys = {
        _resolve(case).plan_key
        for case in CASES
        if case["profile"] != "web_full_offline"
    }
    assert len(keys) == len(CASES) - 1


def test_resolution_is_deterministic() -> None:
    for case in CASES:
        first = _resolve(case).canonical_bytes()
        assert first == _resolve(case).canonical_bytes()
        assert first == pb.canonical_json_bytes(case["expected"])


def test_offline_and_online_profiles_resolve_identically() -> None:
    online = CASES_BY_ID["web_webgpu_then_wasm"]["expected"]
    offline = CASES_BY_ID["web_offline_same_plan"]["expected"]
    assert online == offline


def test_steps_follow_only_the_declared_fallback_graph() -> None:
    for case in CASES:
        manifest = json.loads(VECTORS["manifests"][case["manifest"]]["text"])
        by_id = {item["id"]: item for item in manifest["runtimes"]}
        steps = case["expected"]["steps"]
        for previous, step in zip(steps, steps[1:], strict=False):
            if step["role"] == "provider_fallback":
                assert step["runtime_id"] == previous["runtime_id"]
                continue
            assert step["role"] == "artifact_fallback"
            chain = [previous["runtime_id"]]
            while by_id[chain[-1]]["fallback"] is not None:
                chain.append(by_id[chain[-1]]["fallback"])
            assert step["runtime_id"] in chain[1:]
        assert all(step["provider"] in dc.PROVIDERS for step in steps)
        assert case["expected"]["local_only"] is True


def test_unsupported_results_never_carry_steps() -> None:
    for case in CASES:
        expected = case["expected"]
        if expected["status"] == "unsupported":
            assert expected["steps"] == [] and expected["root_runtime"] is None
            assert expected["reason"] in ep.REASON_CODES


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [*value.keys(), *(s for item in value.values() for s in _strings(item))]
    if isinstance(value, list):
        return [s for item in value for s in _strings(item)]
    return []


def test_plans_carry_no_identity_values_paths_or_bundle_text() -> None:
    sentinels = VECTORS["identity_sentinels"]
    for case in CASES:
        text = json.dumps(case["expected"])
        assert not any(sentinel in text for sentinel in sentinels)
        for value in _strings(case["expected"]):
            assert "/" not in value, value
            assert len(value) <= 71


def test_resolver_never_opens_sockets(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    for case in CASES:
        assert _resolve(case).to_dict() == case["expected"]
    for item in VECTORS["normalization"]:
        normalizer = getattr(dc, f"profile_from_{item['kit']}_probe")
        assert normalizer(item["probe"]).to_dict() == item["expected"]


def test_resolver_imports_no_network_or_inference_modules() -> None:
    heavy = ("onnxruntime", "torch", "coremltools", "mlx", "transformers")
    network = ("urllib", "http", "requests", "socket", "subprocess", "httpx")
    for path in MODULES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported |= {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                imported.add(node.module.split(".")[0])
        assert not imported & set(heavy + network), (path.name, imported)
    before = set(sys.modules)
    for case in CASES:
        _resolve(case)
    loaded = {name.split(".")[0] for name in set(sys.modules) - before}
    assert not loaded & set(heavy + network)


def test_resolver_rejects_wrong_argument_types() -> None:
    manifest, verification, profile, constraints = _inputs(CASES[0])
    with pytest.raises(TypeError):
        ep.resolve_execution_plan(manifest, verification.to_dict(), profile)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        ep.resolve_execution_plan(manifest, verification, profile.to_dict())  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        ep.resolve_execution_plan(manifest, verification, profile, {})  # type: ignore[arg-type]


def test_tampered_manifest_bytes_are_unverified() -> None:
    manifest, verification, profile, constraints = _inputs(
        CASES_BY_ID["web_webgpu_then_wasm"]
    )
    for data in (
        b"{}",
        manifest + b" ",
        manifest.replace(b'"schema_version":1', b'"schema_version":2'),
    ):
        forged = dataclasses.replace(verification, content_id=pb.content_id(data))
        plan = ep.resolve_execution_plan(data, forged, profile, constraints)
        assert plan.reason == "bundle_unverified"


def test_extension_documents_validate_against_the_schema() -> None:
    validator = Draft202012Validator(
        ep.load_execution_plan_schema("execution_extension")
    )
    manifest = json.loads(VECTORS["manifests"]["matrix"]["text"])
    validator.validate(manifest["extensions"][ep.EXECUTION_EXTENSION])


def test_signed_bundle_resolves_end_to_end() -> None:
    pytest.importorskip("cryptography")
    generator = _generator()
    manifest = _manifest("matrix")
    seed = hashlib.sha256(b"openmed.device_capabilities.TEST-ONLY.never-trust").digest()
    envelope = pb.sign_portable_manifest(manifest, [("openmed-test-capability", seed)])
    policy = pb.TrustPolicy.from_mapping(
        {
            "keys": [
                {
                    "crv": "Ed25519",
                    "kid": "openmed-test-capability",
                    "kty": "OKP",
                    "not_after": "2027-11-01T00:00:00Z",
                    "not_before": "2026-01-01T00:00:00Z",
                    "status": "active",
                    "x": base64.urlsafe_b64encode(pb.ed25519_public_key(seed))
                    .decode("ascii")
                    .rstrip("="),
                }
            ],
            "revoked_content_ids": [],
            "schema": pb.TRUST_POLICY_SCHEMA,
            "schema_version": pb.TRUST_POLICY_SCHEMA_VERSION,
            "threshold": 1,
        }
    )
    files = {path: text.encode("utf-8") for path, text in generator.TREE.items()}
    verification = pb.verify_portable_bundle(
        manifest,
        envelope,
        trust_policy=policy,
        context=pb.VerificationContext(now="2026-11-07T00:00:00Z"),
        files=files,
    )
    assert verification.eligible_for_activation
    profile = dc.DeviceCapabilityProfile.from_mapping(
        VECTORS["profiles"]["web_full"]["document"]
    )
    plan = ep.resolve_execution_plan(manifest, verification, profile)
    assert plan.to_dict() == CASES_BY_ID["web_webgpu_then_wasm"]["expected"]
