"""Tests for the signed portable model-bundle contract and reference verifier."""

from __future__ import annotations

import builtins
import copy
import hashlib
import importlib.util
import json
import socket
import sys
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from openmed.core import portable_bundle as pb

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "portable_bundle"
VECTORS = json.loads((FIXTURES / "vectors.json").read_text(encoding="utf-8"))
CASES = VECTORS["cases"]
GOLDEN = VECTORS["manifests"]["golden"]["text"].encode("utf-8")
GOLDEN_ENVELOPE = VECTORS["envelopes"]["golden"].encode("utf-8")

# Codes raised for malformed caller input or missing crypto rather than
# returned for a bundle; every other verifier code has a golden vector.
NON_VECTOR_CODES = {
    "signature_verifier_unavailable",
    "trust_policy_invalid",
    "context_invalid",
    "signing_key_invalid",
}


def _generator():
    spec = importlib.util.spec_from_file_location(
        "portable_bundle_vector_generator", FIXTURES / "generate_vectors.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _manifest(key: str) -> bytes:
    entry = VECTORS["manifests"][key]
    if "text" in entry:
        return entry["text"].encode("utf-8")
    base = VECTORS["manifests"][entry["base"]]["text"]
    text = base[: entry["at"]] + entry["insert"] + base[entry["at"] + entry["delete"] :]
    return text.encode("utf-8")


def _context(key: str) -> pb.VerificationContext:
    raw = VECTORS["contexts"][key]
    target = None if raw["target"] is None else pb.VerificationTarget(**raw["target"])
    return pb.VerificationContext(
        now=raw["now"],
        target=target,
        sequence_floors=raw["sequence_floors"],
        understood_extensions=frozenset(raw["understood_extensions"]),
        max_manifest_bytes=raw["max_manifest_bytes"],
        max_envelope_bytes=raw["max_envelope_bytes"],
        max_total_bytes=raw["max_total_bytes"],
        max_files=raw["max_files"],
    )


def _tree(case: dict[str, Any]) -> dict[str, bytes] | None:
    if case["tree"] is None:
        return None
    files = {
        entry["path"]: entry["text"].encode("utf-8")
        for entry in VECTORS["trees"][case["tree"]]
    }
    for path in case["tree_remove"]:
        files.pop(path)
    for entry in case["tree_set"]:
        files[entry["path"]] = entry["text"].encode("utf-8")
    return files


def _policy(name: str = "default") -> pb.TrustPolicy:
    return pb.TrustPolicy.from_mapping(VECTORS["trust_policies"][name])


def _run(case: dict[str, Any]) -> pb.PortableBundleVerification:
    return pb.verify_portable_bundle(
        _manifest(case["manifest"]),
        VECTORS["envelopes"][case["envelope"]].encode("utf-8"),
        trust_policy=_policy(case["trust_policy"]),
        context=_context(case["context"]),
        files=_tree(case),
    )


def _result_payload(result: pb.PortableBundleVerification) -> dict[str, Any]:
    payload = result.to_dict()
    assert payload.pop("schema") == pb.VERIFICATION_SCHEMA
    assert payload.pop("schema_version") == pb.VERIFICATION_SCHEMA_VERSION
    return payload


def test_committed_vectors_match_the_generator() -> None:
    assert (FIXTURES / "vectors.json").read_text(encoding="utf-8") == (
        _generator().render()
    )


def test_vectors_are_synthetic_and_marked_test_only() -> None:
    assert VECTORS["synthetic_only"] is True
    assert VECTORS["test_only_keys"] is True
    assert "TEST-ONLY" in VECTORS["notice"]
    generator = (FIXTURES / "generate_vectors.py").read_text(encoding="utf-8")
    assert "TEST-ONLY KEYS" in generator
    for kid in {
        key["kid"]
        for policy in VECTORS["trust_policies"].values()
        for key in policy["keys"]
    }:
        assert kid.startswith("openmed-test-")


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_golden_case_matches_reference_result(case: dict[str, Any]) -> None:
    assert _result_payload(_run(case)) == case["expected"]


def test_every_bundle_reason_has_a_golden_vector() -> None:
    covered = {case["expected"]["reason"] for case in CASES} - {None}
    assert covered == set(pb.REASON_CODES) - NON_VECTOR_CODES
    assert VECTORS["reason_codes"] == list(pb.REASON_CODES)


def test_reason_codes_are_unique_and_value_free() -> None:
    assert len(set(pb.REASON_CODES)) == len(pb.REASON_CODES)
    for code in pb.REASON_CODES:
        assert code == code.lower()
        assert all(char.isalpha() or char == "_" for char in code)


def test_golden_identity_and_signing_input() -> None:
    golden = VECTORS["golden"]
    assert pb.content_id(GOLDEN) == golden["content_id"]
    assert (
        hashlib.sha256(pb.signing_input(GOLDEN)).hexdigest()
        == golden["signing_input_sha256"]
    )
    prefix = pb.signing_input(b"{}")
    assert prefix == (b"DSSEv1 47 application/vnd.openmed.portable-bundle.v1+json 2 {}")
    assert VECTORS["payload_type"] == pb.PAYLOAD_TYPE


@pytest.mark.parametrize(
    "vector",
    VECTORS["canonicalization"],
    ids=[item["id"] for item in VECTORS["canonicalization"]],
)
def test_canonicalization_vectors(vector: dict[str, str]) -> None:
    value = json.loads(vector["input"])
    assert pb.canonical_json_bytes(value) == vector["canonical"].encode("utf-8")


@pytest.mark.parametrize(
    "vector",
    VECTORS["canonicalization_rejects"],
    ids=[item["id"] for item in VECTORS["canonicalization_rejects"]],
)
def test_canonicalization_rejects(vector: dict[str, str]) -> None:
    value = json.loads(vector["input"])
    with pytest.raises(pb.PortableBundleError) as error:
        pb.canonical_json_bytes(value)
    assert str(error.value) == "manifest_not_canonical"


def test_canonicalization_rejects_unsupported_python_values() -> None:
    for value in (1.0, float("nan"), {1: "x"}, b"bytes", {"a": object()}):
        with pytest.raises(pb.PortableBundleError):
            pb.canonical_json_bytes(value)
    deep: Any = []
    for _ in range(64):
        deep = [deep]
    with pytest.raises(pb.PortableBundleError):
        pb.canonical_json_bytes(deep)


@pytest.mark.parametrize(
    "vector",
    VECTORS["ed25519_rfc8032"],
    ids=[item["name"] for item in VECTORS["ed25519_rfc8032"]],
)
def test_rfc8032_vectors_verify(vector: dict[str, str]) -> None:
    public = bytes.fromhex(vector["public_hex"])
    signature = bytes.fromhex(vector["signature_hex"])
    message = bytes.fromhex(vector["message_hex"])
    assert pb._ed25519_verify(public, signature, message)
    assert not pb._ed25519_verify(public, signature, message + b"x")


def test_manifest_round_trips_unknown_noncritical_extensions() -> None:
    manifest = pb.PortableBundleManifest.from_bytes(GOLDEN)
    assert manifest.canonical_bytes == GOLDEN
    assert manifest.content_id == VECTORS["golden"]["content_id"]
    value = manifest.to_dict()
    assert value["extensions"] == {
        "org.openmed.test.note": {"kept": True, "revision": 1}
    }
    assert pb.PortableBundleManifest.from_mapping(value).canonical_bytes == GOLDEN
    assert pb.encode_portable_manifest(value) == GOLDEN
    assert manifest.bundle_id == "openmed.synthetic.tiny-pii"
    assert manifest.sequence == 7
    assert manifest.runtime_ids == ("coreml-fp16", "onnx-int8", "onnx-fp32")
    assert manifest.fallback_chain("coreml-fp16") == ("coreml-fp16", "onnx-fp32")
    with pytest.raises(KeyError):
        manifest.fallback_chain("missing")


def test_manifest_parse_reports_structural_reasons() -> None:
    critical = _manifest("valid_critical_extension_understood")
    with pytest.raises(pb.PortableBundleError) as error:
        pb.PortableBundleManifest.from_bytes(critical)
    assert error.value.reason is pb.PortableBundleReason.UNKNOWN_CRITICAL_EXTENSION
    parsed = pb.PortableBundleManifest.from_bytes(
        critical, understood_extensions={"org.openmed.test.note"}
    )
    assert parsed.canonical_bytes == critical
    with pytest.raises(pb.PortableBundleError) as error:
        pb.PortableBundleManifest.from_bytes(GOLDEN + b" ")
    assert str(error.value) == "manifest_not_canonical"


def test_schemas_accept_vector_documents() -> None:
    manifest_schema = Draft202012Validator(pb.load_portable_bundle_schema())
    envelope_schema = Draft202012Validator(
        pb.load_portable_bundle_schema("portable_bundle_envelope")
    )
    trust_schema = Draft202012Validator(
        pb.load_portable_bundle_schema("portable_bundle_trust_policy")
    )
    result_schema = Draft202012Validator(
        pb.load_portable_bundle_schema("portable_bundle_verification")
    )
    for schema in pb.SCHEMA_NAMES:
        Draft202012Validator.check_schema(pb.load_portable_bundle_schema(schema))
        assert pb.load_portable_bundle_schema(schema)["schema_version"] == 1
    manifest_schema.validate(json.loads(GOLDEN))
    manifest_schema.validate(
        json.loads(_manifest("valid_critical_extension_understood"))
    )
    for envelope in VECTORS["envelopes"].values():
        value = json.loads(envelope)
        if value.get("schema_version") == 1 and "note" not in value:
            envelope_schema.validate(value)
    for policy in VECTORS["trust_policies"].values():
        trust_schema.validate(policy)
    for case in CASES:
        result_schema.validate(
            {
                **case["expected"],
                "schema": pb.VERIFICATION_SCHEMA,
                "schema_version": pb.VERIFICATION_SCHEMA_VERSION,
            }
        )
    with pytest.raises(KeyError):
        pb.load_portable_bundle_schema("span")


def test_schema_rejects_documented_structural_faults() -> None:
    validator = Draft202012Validator(pb.load_portable_bundle_schema())
    for key in (
        "unknown_critical_top_level_field",
        "unknown_critical_nested_field",
        "missing_required_section",
        "wrong_field_type",
        "path_traversal",
        "path_absolute",
        "path_backslash",
        "path_dot_segment",
        "executable_file",
        "schema_version_unsupported",
    ):
        assert not validator.is_valid(json.loads(_manifest(key))), key


def test_schema_and_verifier_share_vocabularies() -> None:
    schema = pb.load_portable_bundle_schema()
    runtime = schema["$defs"]["runtime"]["properties"]
    assert set(runtime["format"]["enum"]) == set(pb.FORMAT_PLATFORMS)
    assert set(runtime["precision"]["enum"]) == set(pb.PRECISIONS)
    platform = schema["$defs"]["platform"]["properties"]["platform"]
    assert set(platform["enum"]) == set(pb.PLATFORMS)
    assert set(schema["$defs"]["file"]["properties"]["role"]["enum"]) == set(
        pb.FILE_ROLES
    )
    assert set(schema["required"]) == set(pb._TOP_KEYS)
    for platforms in pb.FORMAT_PLATFORMS.values():
        assert platforms <= pb.PLATFORMS
    for precisions in pb.FORMAT_PRECISIONS.values():
        assert precisions <= pb.PRECISIONS
    result = pb.load_portable_bundle_schema("portable_bundle_verification")
    reasons = result["properties"]["reason"]["oneOf"][1]["enum"]
    assert reasons == list(pb.REASON_CODES)


def test_trust_policy_round_trip_and_validation() -> None:
    raw = VECTORS["trust_policies"]["rotation"]
    policy = pb.TrustPolicy.from_mapping(raw)
    assert policy.to_dict() == raw
    assert policy.key("openmed-test-root-b") is not None
    assert policy.key("absent") is None
    assert "public_key" not in repr(policy.keys[0])
    for mutate in (
        lambda value: value.update(schema="other"),
        lambda value: value.update(schema_version=2),
        lambda value: value.update(extra=True),
        lambda value: value.update(threshold=0),
        lambda value: value.update(threshold=True),
        lambda value: value["keys"][0].update(crv="P-256"),
        lambda value: value["keys"][0].update(x=value["keys"][0]["x"] + "A"),
        lambda value: value["keys"][0].update(status="paused"),
        lambda value: value["keys"][0].update(not_after="2025-01-01T00:00:00Z"),
        lambda value: value["keys"][1].update(kid=value["keys"][0]["kid"]),
        lambda value: value.update(keys=[]),
        lambda value: value.update(revoked_content_ids=["sha256:abc"]),
    ):
        candidate = copy.deepcopy(raw)
        mutate(candidate)
        with pytest.raises(pb.PortableBundleError) as error:
            pb.TrustPolicy.from_mapping(candidate)
        assert str(error.value) == "trust_policy_invalid"


def test_context_and_target_validation() -> None:
    for kwargs in (
        {"now": "2026-11-07"},
        {"now": "2026-13-01T00:00:00Z"},
        {"max_files": 0},
        {"sequence_floors": {"bundle": -1}},
        {"understood_extensions": frozenset({"no-dot"})},
        {"target": "android"},
    ):
        with pytest.raises(pb.PortableBundleError) as error:
            pb.VerificationContext(**kwargs)
        assert str(error.value) == "context_invalid"
    from datetime import datetime, timezone

    aware = pb.VerificationContext(now=datetime(2026, 11, 7, tzinfo=timezone.utc))
    assert aware.now == "2026-11-07T00:00:00Z"
    with pytest.raises(pb.PortableBundleError):
        pb.VerificationContext(now=datetime(2026, 11, 7))
    assert pb._is_timestamp(pb.VerificationContext().now)
    for kwargs in (
        {"platform": "amiga"},
        {"sdk_version": "four"},
        {"runtimes": ("tflite",)},
        {"os_version": "17.0.0.0.0"},
        {"memory_bytes": -1},
    ):
        base = {
            "platform": "ios",
            "sdk_name": "openmedkit-swift",
            "sdk_version": "4.0.0",
            "runtimes": ("coreml",),
        }
        with pytest.raises(pb.PortableBundleError):
            pb.VerificationTarget(**{**base, **kwargs})
    with pytest.raises(pb.PortableBundleError) as error:
        pb.verify_portable_bundle(GOLDEN, GOLDEN_ENVELOPE, trust_policy="policy")
    assert str(error.value) == "trust_policy_invalid"
    with pytest.raises(pb.PortableBundleError) as error:
        pb.verify_portable_bundle(
            GOLDEN, GOLDEN_ENVELOPE, trust_policy=_policy(), files={"a": "text"}
        )
    assert str(error.value) == "context_invalid"


def _seed(label: str) -> bytes:
    return hashlib.sha256(label.encode("ascii")).digest()


def _local_policy(*kids: str) -> pb.TrustPolicy:
    return pb.TrustPolicy(
        keys=tuple(
            pb.TrustedKey(
                kid=kid,
                public_key=pb.ed25519_public_key(_seed(kid)),
                not_before="2026-01-01T00:00:00Z",
                not_after="2027-07-01T00:00:00Z",
            )
            for kid in kids
        )
    )


def _context_now() -> pb.VerificationContext:
    return pb.VerificationContext(now="2026-11-07T00:00:00Z")


def test_signing_round_trip_and_rotation_envelope() -> None:
    envelope = pb.sign_portable_manifest(
        GOLDEN, [("unit-old", _seed("unit-old")), ("unit-new", _seed("unit-new"))]
    )
    value = json.loads(envelope)
    assert pb.canonical_json_bytes(value) == envelope
    assert value["content_id"] == pb.content_id(GOLDEN)
    assert [entry["kid"] for entry in value["signatures"]] == ["unit-old", "unit-new"]
    for policy in (_local_policy("unit-old"), _local_policy("unit-new")):
        result = pb.verify_portable_bundle(
            GOLDEN, envelope, trust_policy=policy, context=_context_now()
        )
        assert result.ok and result.valid_signatures == 1
        assert not result.eligible_for_activation
    both = pb.TrustPolicy(keys=_local_policy("unit-old", "unit-new").keys, threshold=2)
    result = pb.verify_portable_bundle(
        GOLDEN, envelope, trust_policy=both, context=_context_now()
    )
    assert result.ok and result.valid_signatures == 2
    assert str(_local_policy("unit-old").keys[0]).count("public_key") == 0


def test_signing_refuses_bad_inputs() -> None:
    with pytest.raises(pb.PortableBundleError) as error:
        pb.sign_portable_manifest(GOLDEN + b"\n", [("unit", _seed("unit"))])
    assert str(error.value) == "manifest_not_canonical"
    for signers in ([], [("bad kid", _seed("x"))], [("unit", b"short")]):
        with pytest.raises(pb.PortableBundleError) as error:
            pb.sign_portable_manifest(GOLDEN, signers)
        assert str(error.value) == "signing_key_invalid"
    with pytest.raises(pb.PortableBundleError):
        pb.ed25519_public_key(b"short")


def test_missing_crypto_dependency_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def blocked(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.startswith("cryptography"):
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    result = pb.verify_portable_bundle(
        GOLDEN, GOLDEN_ENVELOPE, trust_policy=_policy(), context=_context_now()
    )
    assert not result.ok
    assert result.reason == "signature_verifier_unavailable"
    with pytest.raises(pb.PortableBundleError) as error:
        pb.sign_portable_manifest(GOLDEN, [("unit", _seed("unit"))])
    assert str(error.value) == "signature_verifier_unavailable"


def _materialize(root: Path) -> None:
    for entry in VECTORS["trees"]["golden"]:
        path = root / entry["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(entry["text"].encode("utf-8"))


def test_directory_verification_matches_memory_tree(tmp_path: Path) -> None:
    _materialize(tmp_path)
    pb.write_portable_bundle(tmp_path, GOLDEN, GOLDEN_ENVELOPE)
    result = pb.verify_portable_bundle_dir(
        tmp_path, trust_policy=_policy(), context=_context("default")
    )
    assert result.ok and result.files_verified and result.files_checked == 14
    assert result.eligible_for_activation
    (tmp_path / "notes.txt").write_text("extra\n", encoding="utf-8")
    result = pb.verify_portable_bundle_dir(
        tmp_path, trust_policy=_policy(), context=_context("default")
    )
    assert result.reason == "file_undeclared"
    (tmp_path / "notes.txt").unlink()
    (tmp_path / "link.txt").symlink_to(tmp_path / "LICENSE")
    result = pb.verify_portable_bundle_dir(
        tmp_path, trust_policy=_policy(), context=_context("default")
    )
    assert result.reason == "path_unsafe"


def test_directory_verification_handles_missing_and_oversized_documents(
    tmp_path: Path,
) -> None:
    result = pb.verify_portable_bundle_dir(tmp_path, trust_policy=_policy())
    assert result.reason == "file_missing" and result.content_id is None
    (tmp_path / pb.MANIFEST_FILENAME).write_bytes(GOLDEN)
    (tmp_path / pb.ENVELOPE_FILENAME).write_bytes(GOLDEN_ENVELOPE)
    small = pb.VerificationContext(now="2026-11-07T00:00:00Z", max_manifest_bytes=32)
    result = pb.verify_portable_bundle_dir(
        tmp_path, trust_policy=_policy(), context=small
    )
    assert result.reason == "manifest_too_large"


def test_write_refuses_overwrite_and_mismatched_envelope(tmp_path: Path) -> None:
    pb.write_portable_bundle(tmp_path, GOLDEN, GOLDEN_ENVELOPE)
    with pytest.raises(pb.PortableBundleError) as error:
        pb.write_portable_bundle(tmp_path, GOLDEN, GOLDEN_ENVELOPE)
    assert str(error.value) == "path_conflict"
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(pb.PortableBundleError) as error:
        pb.write_portable_bundle(
            other, _manifest("schema_unsupported"), GOLDEN_ENVELOPE
        )
    assert str(error.value) == "content_id_mismatch"


def test_inventory_cross_checks_precomputed_records(tmp_path: Path) -> None:
    _materialize(tmp_path)
    pb.write_portable_bundle(tmp_path, GOLDEN, GOLDEN_ENVELOPE)
    golden_files = json.loads(GOLDEN)["files"]
    files = pb.inventory_files(
        tmp_path,
        roles=_generator().ROLES,
        media_types={item["path"]: item["media_type"] for item in golden_files},
    )
    assert files == golden_files
    records = [
        {
            "path": item["path"],
            "size_bytes": item["size_bytes"],
            "sha256": item["sha256"],
        }
        for item in files
    ]
    assert pb.inventory_files(tmp_path, records=records[:3])
    for mutate, reason in (
        (
            lambda rows: rows[0].update(sha256="sha256:" + "0" * 64),
            "file_digest_mismatch",
        ),
        (lambda rows: rows[0].update(size_bytes=1), "file_size_mismatch"),
        (lambda rows: rows[0].update(path="absent.txt"), "file_missing"),
    ):
        candidate = copy.deepcopy(records)
        mutate(candidate)
        with pytest.raises(pb.PortableBundleError) as error:
            pb.inventory_files(tmp_path, records=candidate)
        assert str(error.value) == reason
    (tmp_path / ".hidden").write_text("x", encoding="utf-8")
    with pytest.raises(pb.PortableBundleError) as error:
        pb.inventory_files(tmp_path)
    assert str(error.value) == "path_unsafe"


def test_inventory_infers_roles_and_media_types(tmp_path: Path) -> None:
    for name in (
        "LICENSE.txt",
        "config.json",
        "id2label.json",
        "model.onnx",
        "weights.safetensors",
        "tokenizer.json",
        "README.md",
    ):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    roles = {
        item["path"]: (item["role"], item["media_type"])
        for item in pb.inventory_files(tmp_path)
    }
    assert roles["LICENSE.txt"] == ("license", "text/plain")
    assert roles["config.json"] == ("config", "application/json")
    assert roles["id2label.json"] == ("labels", "application/json")
    assert roles["model.onnx"] == ("graph", "application/onnx")
    assert roles["weights.safetensors"] == ("weights", "application/vnd.safetensors")
    assert roles["tokenizer.json"] == ("tokenizer", "application/json")
    assert roles["README.md"] == ("metadata", "text/markdown")


def test_verification_never_opens_sockets(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    for case in CASES:
        assert _result_payload(_run(case)) == case["expected"]


def test_results_and_errors_are_value_free() -> None:
    sensitive = {"openmed.synthetic.tiny-pii", "OpenMed/synthetic-tiny-pii"}
    sensitive |= {entry["path"] for entry in VECTORS["trees"]["golden"]}
    sensitive |= {
        key["kid"]
        for policy in VECTORS["trust_policies"].values()
        for key in policy["keys"]
    }
    for case in CASES:
        result = _run(case)
        rendered = json.dumps(result.to_dict()) + repr(result)
        assert not any(token in rendered for token in sensitive), case["id"]
    with pytest.raises(pb.PortableBundleError) as error:
        pb.PortableBundleManifest.from_bytes(_manifest("path_traversal"))
    assert str(error.value) == "path_unsafe"


def test_signature_check_counts_each_key_once() -> None:
    value = json.loads(GOLDEN_ENVELOPE)
    value["signatures"] = value["signatures"] * 3
    envelope = pb.canonical_json_bytes(value)
    result = pb.verify_portable_bundle(
        GOLDEN, envelope, trust_policy=_policy(), context=_context("default")
    )
    assert result.ok and result.valid_signatures == 1


def test_too_many_signatures_is_malformed() -> None:
    value = json.loads(GOLDEN_ENVELOPE)
    value["signatures"] = value["signatures"] * (pb.MAX_SIGNATURES + 1)
    result = pb.verify_portable_bundle(
        GOLDEN,
        pb.canonical_json_bytes(value),
        trust_policy=_policy(),
        context=_context("default"),
    )
    assert result.reason == "envelope_malformed"


def test_module_has_no_network_imports() -> None:
    source = Path(pb.__file__).read_text(encoding="utf-8")
    for token in ("urllib", "http.client", "requests", "socket", "subprocess"):
        assert f"import {token}" not in source
        assert f"from {token}" not in source
    assert "logging" not in source
    assert "openmed.core.portable_bundle" in sys.modules
