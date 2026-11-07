"""Checks for the shared cross-kit bundle-cache vectors.

The Android, Swift, and web generation caches replay these vectors; this test
keeps the corpus in sync with its generator, synthetic, value-free, and
consistent with the signed portable-bundle contract it builds on.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from pathlib import Path

import pytest

from openmed.core import portable_bundle as pb

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "bundle_cache"
VECTORS = json.loads((FIXTURES / "vectors.json").read_text(encoding="utf-8"))
OPS = {
    "acquire",
    "activate",
    "advance",
    "confirm",
    "evict",
    "lease",
    "open",
    "pin",
    "recover",
    "release",
    "restart",
    "rollback",
    "snapshot",
    "tamper",
    "unpin",
}


def _generator():
    spec = importlib.util.spec_from_file_location(
        "bundle_cache_vector_generator", FIXTURES / "generate_vectors.py"
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
    assert len(VECTORS["scenarios"]) >= 30


def test_bundles_are_signed_portable_bundles_with_exact_inventories():
    policy = pb.TrustPolicy.from_mapping(VECTORS["trust_policies"]["default"])
    context = pb.VerificationContext(now=VECTORS["context"]["now"], target=None)
    for alias, bundle in VECTORS["bundles"].items():
        manifest = bundle["manifest"].encode("utf-8")
        content_id = pb.content_id(manifest)
        assert bundle["content_id"] == content_id, alias
        assert bundle["generation"] == "g-" + content_id[7:39], alias
        files = {item["path"]: item["text"].encode("utf-8") for item in bundle["files"]}
        assert bundle["total_bytes"] == sum(len(data) for data in files.values())
        if alias in {"v2_untrusted", "v2_schema"}:
            continue
        pytest.importorskip("cryptography")
        result = pb.verify_portable_bundle(
            manifest,
            bundle["envelope"].encode("utf-8"),
            trust_policy=policy,
            context=context,
            files=files,
        )
        assert result.eligible_for_activation, (alias, result.reason)


def test_identities_follow_the_documented_derivations():
    identities = VECTORS["identities"]
    digest = hashlib.sha256(identities["bundle_id"].encode("utf-8")).hexdigest()
    assert identities["bundle_key"] == "b-" + digest[:32]
    for name, value in identities["source_digests"].items():
        assert value == "sha256:" + hashlib.sha256(name.encode("utf-8")).hexdigest()
    generator = _generator()
    for name, value in identities["trust_digests"].items():
        assert generator.trust_digest(VECTORS["trust_policies"][name]) == value


def test_scenarios_use_only_the_shared_vocabulary():
    aliases = set(VECTORS["bundles"])
    reasons = set(VECTORS["reason_codes"]) | set(pb.REASON_CODES)
    actions = set(VECTORS["recovery_actions"])
    states = set(VECTORS["states"])
    checkpoints = set(VECTORS["checkpoints"])
    quarantine = set(VECTORS["quarantine_reasons"])
    for scenario in VECTORS["scenarios"]:
        for step in scenario["steps"]:
            assert step["op"] in OPS, scenario["id"]
            crash = step.get("crash_at")
            if crash:
                assert crash["checkpoint"] in checkpoints
            expect = step.get("expect", {})
            if step["op"] == "acquire":
                assert step["bundle"] in aliases and step["source"]["serve"] in aliases
            if expect.get("ok") is False:
                assert expect["reason"] in reasons
                assert expect["bundle_reason"] in reasons | {None}
            if step["op"] == "acquire" and expect.get("ok"):
                assert expect["state"] in states and expect["generation"] in aliases
            if step["op"] == "recover":
                assert {item["action"] for item in expect["actions"]} <= actions
            if step["op"] == "snapshot":
                for entry in expect["generations"]:
                    assert entry["state"] in states and entry["generation"] in aliases
                for entry in expect["quarantined"]:
                    assert entry["reason"] in quarantine


def test_vectors_carry_no_paths_urls_or_credentials():
    text = (FIXTURES / "vectors.json").read_text(encoding="utf-8")
    for pattern in (
        r"https?://",
        r"/Users/",
        r"/private/",
        r"/home/",
        r"token=",
        r"BEGIN [A-Z ]*KEY",
    ):
        assert re.search(pattern, text) is None, pattern
