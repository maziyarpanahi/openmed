"""Checks for the shared managed bundle import vectors.

OpenMedKit for Swift and OpenMedKit for Android replay these vectors; this
test keeps the corpus in sync with its generator, synthetic, value-free, and
consistent with the signed portable-bundle contract, the documentation, and the
reason codes both kits declare.
"""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import io
import json
import re
import zipfile
from pathlib import Path

import pytest

from openmed.core import portable_bundle as pb

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / "tests" / "fixtures" / "managed_import"
VECTORS = json.loads((FIXTURES / "vectors.json").read_text(encoding="utf-8"))
DOC = ROOT / "docs" / "mobile" / "managed-model-bundles.md"
SWIFT = ROOT / "swift/OpenMedKit/Sources/OpenMedKit/OpenMedManagedBundleImport.swift"
KOTLIN = (
    ROOT
    / "android/openmedkit/src/main/kotlin/com/openmed/openmedkit/managed"
    / "ManagedBundleImporter.kt"
)
HOSTILE = {
    "corrupt_deflate",
    "encrypted",
    "local_mismatch",
    "not_zip",
    "symlink",
    "truncated",
    "unsupported_method",
    "zip64",
}
OPS = {"confirm", "import", "recover", "restart", "state"}


def _generator():
    spec = importlib.util.spec_from_file_location(
        "managed_import_vector_generator", FIXTURES / "generate_vectors.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_vectors_match_their_generator():
    pytest.importorskip("cryptography")
    generator = _generator()
    assert (FIXTURES / "vectors.json").read_text(encoding="utf-8") == generator.render()


def test_vectors_are_synthetic_and_test_only():
    assert VECTORS["synthetic_only"] is True
    assert VECTORS["test_only_keys"] is True
    assert "TEST-ONLY" in VECTORS["notice"]
    assert len(VECTORS["scenarios"]) >= 20


def test_archives_decode_and_carry_their_digests():
    for alias, item in VECTORS["archives"].items():
        data = base64.b64decode(item["base64"], validate=True)
        assert len(data) == item["size_bytes"], alias
        assert item["sha256"] == "sha256:" + hashlib.sha256(data).hexdigest(), alias
        if alias in HOSTILE:
            continue
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            # Every non-hostile archive inflates and passes its CRC checks.
            assert archive.testzip() is None, alias


def test_bundles_are_signed_portable_bundles_with_exact_inventories():
    pytest.importorskip("cryptography")
    policy = pb.TrustPolicy.from_mapping(VECTORS["trust_policies"]["default"])
    context = pb.VerificationContext(now="2026-11-07T00:00:00Z", target=None)
    for alias, bundle in VECTORS["bundles"].items():
        manifest = bundle["manifest"].encode("utf-8")
        content_id = pb.content_id(manifest)
        assert bundle["content_id"] == content_id, alias
        assert bundle["generation"] == "g-" + content_id[7:39], alias
        files = {item["path"]: item["text"].encode("utf-8") for item in bundle["files"]}
        assert bundle["total_bytes"] == sum(len(data) for data in files.values())
        result = pb.verify_portable_bundle(
            manifest,
            bundle["envelope"].encode("utf-8"),
            trust_policy=policy,
            context=context,
            files=files,
        )
        if alias == "v2_rogue":
            assert result.reason == "untrusted_key"
        else:
            assert result.eligible_for_activation, (alias, result.reason)


def test_key_pins_follow_the_documented_derivation():
    pins = VECTORS["identities"]["key_pins"]
    for key in VECTORS["trust_policies"]["rotation"]["keys"]:
        raw = base64.urlsafe_b64decode(key["x"] + "=" * (-len(key["x"]) % 4))
        assert pins[key["kid"]] == "sha256:" + hashlib.sha256(raw).hexdigest()


def test_scenarios_use_only_the_shared_vocabulary():
    reasons = set(VECTORS["reason_codes"])
    bundle_reasons = set(pb.REASON_CODES)
    for scenario in VECTORS["scenarios"]:
        assert set(scenario["limits"]) == set(VECTORS["default_limits"])
        assert scenario["target"] in VECTORS["targets"]["android"]
        for step in scenario["steps"]:
            assert step["op"] in OPS, scenario["id"]
            if step["op"] != "import":
                continue
            assert step["config"] is None or step["config"] in VECTORS["configs"]
            if step["handoff"] == "directory":
                assert step["bundle"] in VECTORS["bundles"]
            else:
                assert step["archive"] in VECTORS["archives"]
            for point in (step["crash_at"], step["cancel_at"]):
                if point:
                    assert point["checkpoint"] in VECTORS["checkpoints"]
            expect = step["expect"]
            if expect.get("crashed"):
                continue
            assert expect["outcome"] in VECTORS["outcomes"]
            assert expect["reason"] is None or expect["reason"] in reasons
            assert expect["bundle_reason"] is None or expect["bundle_reason"] in (
                bundle_reasons
            )
            for field in ("generation", "previous_active", "active", "last_green"):
                assert expect[field] is None or expect[field] in VECTORS["bundles"]


def test_managed_configuration_keys_are_namespaced():
    keys = VECTORS["config_keys"].values()
    assert all(key.startswith("openmed_bundle_") for key in keys)
    for alias, values in VECTORS["configs"].items():
        for key in values:
            if key.startswith("openmed_bundle_") and alias != "unknown_key":
                assert key in keys, (alias, key)


def test_both_kits_declare_every_reason_outcome_and_key():
    swift = SWIFT.read_text(encoding="utf-8")
    kotlin = KOTLIN.read_text(encoding="utf-8")
    vocabulary = (
        list(VECTORS["reason_codes"])
        + list(VECTORS["outcomes"])
        + list(VECTORS["config_keys"].values())
    )
    for code in vocabulary:
        assert re.search(rf'"{re.escape(code)}"|case {re.escape(code)}\b', swift), code
        assert f'"{code}"' in kotlin, code


def test_documentation_lists_every_reason_outcome_and_key():
    text = DOC.read_text(encoding="utf-8")
    vocabulary = (
        list(VECTORS["reason_codes"])
        + list(VECTORS["outcomes"])
        + list(VECTORS["config_keys"].values())
    )
    for code in vocabulary:
        assert f"`{code}`" in text, code
    assert "com.apple.configuration.managed" in text
    assert "RestrictionsManager" in text


def test_vectors_carry_no_paths_urls_or_credentials():
    text = (FIXTURES / "vectors.json").read_text(encoding="utf-8")
    for pattern in (
        r"https?://",
        r"content://",
        r"file:",
        r"/Users/",
        r"/private/",
        r"/home/",
        r"token=",
        r"BEGIN [A-Z ]*KEY",
    ):
        assert re.search(pattern, text) is None, pattern
