"""Portable policy subset: canonical IR, generated kit resources, and vectors."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import socket
import sys
from pathlib import Path

import pytest

from openmed.core import kit_parity
from openmed.core import portable_policy as portable
from openmed.core.labels import (
    _ALIAS_MAP,
    CANONICAL_LABELS,
    is_recognized_label,
    normalize_label,
)
from openmed.core.pii import PIIEntity, _mask_placeholder
from openmed.core.policy import CANONICAL_POLICY_NAMES, load_policy
from openmed.core.thresholds import lookup_threshold

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "parity" / "generate_portable_policies.py"
VECTORS_FILE = ROOT / "tests" / "fixtures" / "parity" / "portable_policy_vectors.json"
SWIFT_DIR = ROOT / "swift/OpenMedKit/Sources/OpenMedKit/Resources/policies"
ANDROID_DIR = ROOT / "android/openmedkit/src/main/assets/policies"
WEB_MODULE = ROOT / "js/openmedkit-web/src/portable-policy-resources.ts"

SPEC = importlib.util.spec_from_file_location("generate_portable_policies", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
generator = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = generator
SPEC.loader.exec_module(generator)

VECTORS = json.loads(VECTORS_FILE.read_text(encoding="utf-8"))
RESOURCES = portable.build_resources()


def _detections(case: dict) -> list[portable.PortableDetection]:
    return generator._detections(case["detections"])


def _key(case: dict) -> bytes | None:
    name = case.get("surrogate_key")
    return None if name is None else bytes.fromhex(VECTORS["surrogate_test_keys"][name])


def test_generated_kit_resources_and_vectors_are_current() -> None:
    assert generator.check() == []
    assert generator.main(["--check"]) == 0


def test_every_kit_ships_byte_identical_resources() -> None:
    assert set(RESOURCES) == {
        *(f"{name}.json" for name in portable.PORTABLE_PROFILES),
        portable.MANIFEST_FILE,
        portable.TAXONOMY_FILE,
    }
    web = WEB_MODULE.read_text(encoding="utf-8")
    for name, data in RESOURCES.items():
        assert (SWIFT_DIR / name).read_bytes() == data, name
        assert (ANDROID_DIR / name).read_bytes() == data, name
        assert f"`{data.decode('ascii')}`" in web, name
    assert sorted(path.name for path in SWIFT_DIR.glob("*.json")) == sorted(RESOURCES)
    assert sorted(path.name for path in ANDROID_DIR.glob("*.json")) == sorted(RESOURCES)


def test_stale_kit_copy_fails_the_staleness_check(tmp_path: Path, monkeypatch) -> None:
    stale = tmp_path / "policies"
    stale.mkdir()
    for name, data in RESOURCES.items():
        (stale / name).write_bytes(data)
    tampered = bytearray(RESOURCES["hipaa_safe_harbor.json"])
    tampered[-2] ^= 0x01
    (stale / "hipaa_safe_harbor.json").write_bytes(bytes(tampered))
    (stale / "china_pipl.json").write_bytes(b"{}\n")
    monkeypatch.setattr(generator, "ANDROID_DIR", stale)

    problems = generator.check()

    assert any(
        problem.startswith("resource_stale") and "hipaa_safe_harbor.json" in problem
        for problem in problems
    )
    assert any(
        problem.startswith("resource_unexpected") and "china_pipl.json" in problem
        for problem in problems
    )


def test_manifest_declares_the_portable_subset_and_every_exclusion() -> None:
    manifest = json.loads(RESOURCES[portable.MANIFEST_FILE])
    policy_files = {
        path.name[: -len(".json")]
        for path in (ROOT / "openmed" / "core" / "policies").glob("*.json")
    }

    assert manifest["default_profile"] == portable.DEFAULT_PROFILE
    assert sorted(manifest["profiles"]) == sorted(portable.PORTABLE_PROFILES)
    assert set(portable.PORTABLE_PROFILES) <= set(CANONICAL_POLICY_NAMES)
    assert set(manifest["profiles"]) | set(manifest["excluded_profiles"]) == (
        policy_files | set(CANONICAL_POLICY_NAMES)
    )
    assert not set(manifest["profiles"]) & set(manifest["excluded_profiles"])
    assert set(manifest["excluded_profiles"].values()) == {
        portable.EXCLUDED_PYTHON_ONLY,
        portable.EXCLUDED_STRUCTURED_SCHEMA,
    }
    assert manifest["excluded_profiles"]["fhir_hipaa_safe_harbor"] == (
        portable.EXCLUDED_STRUCTURED_SCHEMA
    )
    for name, digest in manifest["profiles"].items():
        assert digest == portable.digest_bytes(RESOURCES[f"{name}.json"])
    assert manifest["taxonomy_digest"] == portable.digest_bytes(
        RESOURCES[portable.TAXONOMY_FILE]
    )
    assert VECTORS["profiles"] == manifest["profiles"]
    assert VECTORS["excluded_profiles"] == manifest["excluded_profiles"]
    assert (
        VECTORS["manifest_sha256"]
        == hashlib.sha256(RESOURCES[portable.MANIFEST_FILE]).hexdigest()
    )


@pytest.mark.parametrize("name", portable.PORTABLE_PROFILES)
def test_ir_preserves_python_actions_thresholds_and_metadata(name: str) -> None:
    profile = load_policy(name)
    policy = portable.load_portable_policy(name)
    document = json.loads(RESOURCES[f"{name}.json"])

    assert dict(policy.actions) == dict(profile.actions)
    assert set(policy.actions) == set(CANONICAL_LABELS)
    assert policy.strict_no_leak is profile.strict_no_leak
    assert policy.arbitration_mode == profile.arbitration_mode
    assert policy.threshold_profile == profile.threshold_profile
    assert policy.forced_cascade_tiers == profile.forced_cascade_tiers
    assert document["policy_label_actions"] == dict(profile.policy_label_actions)
    assert (
        document["source_sha256"]
        == hashlib.sha256(
            (ROOT / "openmed" / "core" / "policies" / f"{name}.json").read_bytes()
        ).hexdigest()
    )
    for label in sorted(CANONICAL_LABELS):
        threshold = lookup_threshold(label, "*", profile.threshold_profile)
        entry = policy.threshold_for(label)
        assert entry.keep_floor_ppm == kit_parity.confidence_to_ppm(
            float(threshold["keep_floor"])
        ), label
        assert entry.escalate_below_ppm == kit_parity.confidence_to_ppm(
            float(threshold["escalate_below"])
        ), label
    if profile.strict_no_leak:
        assert "strict_no_leak.v1" in policy.required_capabilities
    if "replace" in profile.actions.values():
        assert "surrogate.hmac_sha256.v1" in policy.required_capabilities
    if profile.safety_sweep_mandatory:
        assert "safety_sweep" in policy.excluded_features
    assert set(policy.required_capabilities) <= set(portable.SUPPORTED_CAPABILITIES)


def test_taxonomy_normalization_equals_python_normalize_label() -> None:
    taxonomy = portable.load_portable_policy("hipaa_safe_harbor").taxonomy
    inputs = set(_ALIAS_MAP) | set(CANONICAL_LABELS)
    variants: set[str] = set()
    for value in inputs:
        variants.update(
            {value, value.upper(), f"B-{value}", f"I-{value.upper()}", f" {value} "}
        )
    variants.update({"season_hint", "x", "OTHER", "B-", "I-other", "zz9"})

    for label in sorted(variants):
        canonical, recognized = taxonomy.normalize(label)
        assert canonical == normalize_label(label), label
        assert recognized == is_recognized_label(label), label
    assert taxonomy.labels == tuple(sorted(CANONICAL_LABELS))
    assert tuple(kit_parity.KIT_CANONICAL_LABELS) == taxonomy.labels


def test_patient_name_and_identifier_labels_never_fall_through_to_other() -> None:
    for label in (
        "patient_name",
        "PATIENT_NAME",
        "B-patient_name",
        "doctor_name",
        "provider_name",
        "PersonName",
    ):
        assert normalize_label(label) == "PERSON", label
    for label in ("passport_number", "member_id", "policy_number", "document_id"):
        assert normalize_label(label) == "ID_NUM", label
    assert normalize_label("routing_number") == "ACCOUNT_NUMBER"
    assert normalize_label("ipv6") == "IP_ADDRESS"
    assert normalize_label("diet_type") == "DIET_TYPE"
    clinical = portable.load_portable_policy("clinical_minimal_redaction")
    assert clinical.action_for("patient_name") == "mask"
    assert clinical.action_for("season_hint") == "keep"


def test_python_mask_placeholders_use_the_canonical_taxonomy() -> None:
    def entity(label: str, canonical: str | None = None) -> PIIEntity:
        return PIIEntity(
            text="x",
            label=label,
            start=0,
            end=1,
            confidence=0.9,
            canonical_label=canonical,
        )

    assert _mask_placeholder(entity("patient_name")) == "[PERSON]"
    assert _mask_placeholder(entity("first_name")) == "[FIRST_NAME]"
    assert _mask_placeholder(entity("season_hint")) == "[OTHER]"
    assert _mask_placeholder(entity("NAME", canonical="PERSON")) == "[PERSON]"


@pytest.mark.parametrize("case", VECTORS["decision_cases"], ids=lambda item: item["id"])
def test_decision_vectors(case: dict) -> None:
    policy = portable.load_portable_policy(case["profile"])
    outcome = portable.decide(
        policy,
        case["text"],
        _detections(case),
        overrides=case.get("overrides"),
        surrogate_key=_key(case),
    )
    serialized = kit_parity.serialize_result(outcome.result)

    assert serialized == case["expected"]["result_json"]
    assert (
        hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        == (case["expected"]["result_sha256"])
    )
    assert outcome.result.deidentified_text.encode("utf-8") == (
        case["expected"]["deidentified_text"].encode("utf-8")
    )
    assert [item.to_dict() for item in outcome.decisions] == (
        case["expected"]["decisions"]
    )
    assert outcome.policy_digest == case["expected"]["policy_digest"]
    kit_parity.verify_result(kit_parity.parse_result(serialized), case["text"])


def test_vectors_cover_every_required_category() -> None:
    categories = {
        category
        for case in VECTORS["decision_cases"]
        for category in case["categories"]
    }
    assert {
        "boundary",
        "duplicate",
        "keep",
        "labels",
        "merge",
        "overlap",
        "override",
        "regression",
        "semantic",
        "semantic_only",
        "strict_no_leak",
        "surrogate",
        "threshold",
        "tie",
        "unknown_label",
    } <= categories
    profiles = {case["profile"] for case in VECTORS["decision_cases"]}
    assert {"strict_no_leak", "gdpr_art9_health", "clinical_minimal_redaction"} <= (
        profiles
    )
    load_ids = {case["id"] for case in VECTORS["load_cases"]}
    assert {f"load_{name}" for name in portable.PORTABLE_PROFILES} <= load_ids


@pytest.mark.parametrize("case", VECTORS["error_cases"], ids=lambda item: item["id"])
def test_error_vectors_fail_with_value_free_reasons(case: dict) -> None:
    policy = portable.load_portable_policy(case["profile"])
    with pytest.raises(portable.PortablePolicyError) as error:
        portable.decide(
            policy,
            case["text"],
            _detections(case),
            overrides=case.get("overrides"),
            surrogate_key=_key(case),
        )

    assert error.value.code == case["expected"]["code"]
    assert error.value.field_path == case["expected"]["field_path"]
    assert case["text"] not in str(error.value)
    for label in case.get("overrides") or {}:
        assert label not in str(error.value)


@pytest.mark.parametrize("case", VECTORS["load_cases"], ids=lambda item: item["id"])
def test_load_vectors(case: dict) -> None:
    expected = case["expected"]
    if "code" in expected:
        with pytest.raises(portable.PortablePolicyError) as error:
            generator.run_load_case(case, RESOURCES)
        assert error.value.code == expected["code"]
        assert error.value.field_path == expected["field_path"]
        assert case["profile"].strip() == "" or case["profile"] not in str(error.value)
        return
    policy = generator.run_load_case(case, RESOURCES)
    assert policy.name == expected["name"]
    assert policy.digest == expected["digest"]
    assert list(policy.required_capabilities) == expected["required_capabilities"]


def test_missing_runtime_capability_refuses_strict_no_leak() -> None:
    supported = [
        item for item in portable.SUPPORTED_CAPABILITIES if "strict" not in item
    ]
    with pytest.raises(portable.PortablePolicyError) as error:
        portable.load_portable_policy(
            "strict_no_leak", supported_capabilities=supported
        )
    assert error.value.code == "capability_missing"
    assert portable.load_portable_policy(
        "hipaa_safe_harbor", supported_capabilities=supported
    ).name == ("hipaa_safe_harbor")


def test_surrogates_are_keyed_deterministic_and_value_free() -> None:
    key_a = bytes.fromhex(VECTORS["surrogate_test_keys"]["synthetic_test_key_a"])
    key_b = bytes.fromhex(VECTORS["surrogate_test_keys"]["synthetic_test_key_b"])
    token = portable.surrogate_token(key_a, "PERSON", "Ada Quill")

    assert token == portable.surrogate_token(key_a, "PERSON", "Ada Quill")
    assert token != portable.surrogate_token(key_b, "PERSON", "Ada Quill")
    assert token != portable.surrogate_token(key_a, "FIRST_NAME", "Ada Quill")
    assert len(token) == 16 and "Ada" not in token
    assert all(
        name.startswith("synthetic_test_key_")
        for name in VECTORS["surrogate_test_keys"]
    )
    serialized = json.dumps(VECTORS["decision_cases"], ensure_ascii=False)
    for name, hex_key in VECTORS["surrogate_test_keys"].items():
        assert hex_key not in serialized, name
    for data in RESOURCES.values():
        for hex_key in VECTORS["surrogate_test_keys"].values():
            assert hex_key.encode("ascii") not in data


def test_decisions_and_resources_carry_no_source_text() -> None:
    for case in VECTORS["decision_cases"]:
        decisions = json.dumps(case["expected"]["decisions"], ensure_ascii=False)
        for detection in case["detections"]:
            surface = case["text"][detection["start"] : detection["end"]]
            if len(surface) >= 3:
                assert surface not in decisions, case["id"]
    for data in RESOURCES.values():
        text = data.decode("ascii")
        for case in VECTORS["decision_cases"]:
            for detection in case["detections"]:
                surface = case["text"][detection["start"] : detection["end"]]
                if len(surface) >= 5:
                    assert surface not in text, case["id"]


def test_portable_policy_runs_without_network(monkeypatch) -> None:
    def refuse(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(portable, "_RESOURCE_CACHE", None)
    for case in VECTORS["decision_cases"]:
        policy = portable.load_portable_policy(case["profile"])
        portable.decide(
            policy,
            case["text"],
            _detections(case),
            overrides=case.get("overrides"),
            surrogate_key=_key(case),
        )


def test_kit_test_consumers_and_ci_gates_are_registered() -> None:
    consumers = {item["surface"]: item for item in VECTORS["consumers"]}
    assert set(consumers) == {"python", "swift", "kotlin", "web"}
    for item in consumers.values():
        assert (ROOT / item["test"]).is_file(), item["surface"]
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    swift = (ROOT / ".github" / "workflows" / "swift-test.yml").read_text("utf-8")
    android = (ROOT / ".github" / "workflows" / "android-ci.yml").read_text("utf-8")
    web = json.loads(
        (ROOT / "js" / "openmedkit-web" / "package.json").read_text(encoding="utf-8")
    )

    assert "scripts/parity/generate_portable_policies.py --check" in ci
    assert "swift test --filter PortablePolicyVectorTests" in swift
    assert "--tests '*PortablePolicyVectorTest'" in android
    for workflow in (swift, android):
        assert "tests/fixtures/parity/portable_policy_vectors.json" in workflow
        assert "openmed/core/policies/**" in workflow
    assert "test_portable_policy.spec.ts" in web["scripts"]["test:node"]
