"""Activate a signed synthetic bundle into the live process registry offline."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Runs in a child process so the global registry and its live adapters are
# exercised without leaking an activated pack into other tests.
_RUNNER = r"""
import hashlib
import hmac
import importlib.metadata
import json
import socket
import sys
import tempfile
from pathlib import Path
from datetime import datetime, timedelta, timezone


def forbidden(*args, **kwargs):
    raise AssertionError("network, discovery, or model loading is forbidden")


socket.socket.connect = forbidden
socket.socket.connect_ex = forbidden
socket.create_connection = forbidden
socket.getaddrinfo = forbidden
importlib.metadata.entry_points = forbidden

from openmed.core.language_pack import LANGUAGE_PACK_REGISTRY
from openmed.core.language_pack_activation import (
    LanguageBundleActivationError,
    LanguageBundleActivator,
    LanguageReadinessEvidence,
    LocalModelHandle,
    language_bundle_signing_payload,
)
from openmed.core.language_pack_catalog import (
    DEFAULT_PII_MODELS,
    LANG_TO_LOCALE,
    SUPPORTED_LANGUAGES,
)
from openmed.core.models import ModelLoader

ModelLoader.load_model = forbidden

KEY = b"synthetic-integration-key"
NOW = datetime(2026, 9, 1, tzinfo=timezone.utc)
MODEL = "sha256:" + hashlib.sha256(b"synthetic model").hexdigest()
REFERENCE = "/synthetic-models/sl-pii"


class Verifier:
    def verify(self, payload, *, scheme, key_id, signature):
        expected = hmac.new(KEY, payload, hashlib.sha256).hexdigest()
        return scheme == "hmac-sha256" and key_id == "synthetic" and hmac.compare_digest(expected, signature)


manifest = {
    "schema_version": "1.0",
    "bundle_id": "org.example.slovene-clinical",
    "bundle_version": "1.0.0",
    "packs": [
        {
            "language": "sl",
            "model": "sl-pii",
            "tasks": ["deidentification", "summarization"],
            "scripts": ["Latin"],
            "segmenter_id": "pysbd",
            "recognizers": ["builtin-patterns", "model"],
            "surrogate_locale": "sl_SI",
            "policy_overrides": {"profile": "strict_no_leak"},
        }
    ],
    "artifacts": {"sl-pii": {"kind": "model", "sha256": MODEL}},
}
manifest["signature"] = {
    "scheme": "hmac-sha256",
    "key_id": "synthetic",
    "value": hmac.new(
        KEY, language_bundle_signing_payload(manifest), hashlib.sha256
    ).hexdigest(),
}


def readiness(age):
    return LanguageReadinessEvidence(
        language="sl",
        state="supported",
        qualified_tasks=("deidentification",),
        issued_at=NOW - age,
        evidence_digest="sha256:" + hashlib.sha256(b"synthetic evidence").hexdigest(),
    )


# Carry the signed activation manifest as policy data in the signed archive.
# The two schemas have separate purposes and each verifies its own signature.
from openmed.clinical.languages.offline_bundle import (
    HostResources, OfflineBundleComponent, OfflineBundleDescriptor,
    ResourceRequirements, package_offline_bundle, install_offline_bundle,
)
from openmed.core.model_integrity import sha256_file

workspace = tempfile.TemporaryDirectory()
build_root = Path(workspace.name)
payloads = {
    "model/model.onnx": b"synthetic model",
    "lexicons/public.tsv": b"homa\tCONDITION\n",
    "policy/activation.json": json.dumps(manifest).encode(),
    "evidence/readiness.json": json.dumps([readiness(timedelta(days=1)).to_dict()]).encode(),
    "selftests/cases.jsonl": json.dumps({
        "case_id": "sl-001", "synthetic": True, "text": "homa",
        "expected": [{"start": 0, "end": 4, "label": "CONDITION"}],
    }).encode() + b"\n",
}
sources = {}
for relative, payload in payloads.items():
    path = build_root / "sources" / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    sources[relative] = path
components = tuple(
    OfflineBundleComponent(
        component_id=kind.replace("_", "-"), kind=kind, files=(relative,),
        license="Apache-2.0", source="synthetic:openmed/activation",
        data_classification="synthetic" if kind == "self_test" else "public",
    )
    for kind, relative in (
        ("model", "model/model.onnx"), ("lexicon", "lexicons/public.tsv"),
        ("policy", "policy/activation.json"), ("evidence", "evidence/readiness.json"),
        ("self_test", "selftests/cases.jsonl"),
    )
)
descriptor = OfflineBundleDescriptor(
    bundle_id=manifest["bundle_id"], bundle_version=manifest["bundle_version"],
    language="sl", scripts=("Latin",),
    capabilities=tuple(manifest["packs"][0]["tasks"]),
    resources=ResourceRequirements(1, 1024 * 1024), components=components,
)
packaged = package_offline_bundle(
    descriptor, sources, build_root / "portable", signature_key=KEY,
)
installation = install_offline_bundle(
    packaged.path, build_root / "installed", host=HostResources(1024 * 1024, 8 * 1024 * 1024),
    signature_key=KEY, self_test_runner=lambda root, text: [(0, 4, "CONDITION")],
)
assert installation.installed and installation.install_path is not None
installed_root = installation.install_path
documents = {}
for relative in ("policy/activation.json", "evidence/readiness.json"):
    data = (installed_root / relative).read_bytes()
    assert "sha256:" + hashlib.sha256(data).hexdigest() == installation.verification.file_digests[relative]
    documents[relative] = json.loads(data)
manifest = documents["policy/activation.json"]
readiness_rows = documents["evidence/readiness.json"]
assert installation.verification.bundle_id == manifest["bundle_id"]
assert installation.verification.bundle_version == manifest["bundle_version"]
assert installation.verification.language == manifest["packs"][0]["language"]
model_file = installed_root / "model/model.onnx"
REFERENCE = str(model_file)
MODEL = sha256_file(model_file)
assert MODEL == installation.verification.file_digests["model/model.onnx"]


def view():
    return {
        "supported": "sl" in SUPPORTED_LANGUAGES,
        "model": DEFAULT_PII_MODELS.get("sl") == REFERENCE,
        "locale": LANG_TO_LOCALE.get("sl"),
        "languages": len(SUPPORTED_LANGUAGES),
    }


builtin_en = LANGUAGE_PACK_REGISTRY.get("en")
handles = {"sl-pii": LocalModelHandle(reference=REFERENCE, sha256=MODEL)}
activator = LanguageBundleActivator(verifier=Verifier(), clock=lambda: NOW)
result = {"before": view()}

try:
    activator.activate(manifest, [readiness(timedelta(days=90))], handles)
except LanguageBundleActivationError as error:
    result["stale"] = error.reason.value
result["after_stale"] = view()

record = activator.activate(manifest, readiness_rows, handles)
result["active"] = view()
result["tasks"] = list(activator.qualified_tasks("sl"))
result["claim"] = activator.clinical_claim("sl")
result["record"] = json.loads(record.to_json())

activator.rollback(record.activation_id)
result["rolled_back"] = view()
result["builtin_preserved"] = LANGUAGE_PACK_REGISTRY.get("en") is builtin_en
result["installed"] = installation.installed
result["paths_hidden"] = REFERENCE not in record.to_json()
model_file.write_bytes(b"tampered model")
try:
    activator.activate(manifest, readiness_rows, {
        "sl-pii": LocalModelHandle(REFERENCE, sha256_file(model_file)),
    })
except LanguageBundleActivationError as error:
    result["tampered"] = error.reason.value
result["after_tamper"] = view()
print(json.dumps(result, sort_keys=True))
"""


def test_signed_bundle_activates_and_rolls_back_through_live_adapters() -> None:
    pythonpath = os.pathsep.join(
        filter(None, (str(_REPO_ROOT), os.environ.get("PYTHONPATH")))
    )
    env = {**os.environ, "PYTHONPATH": pythonpath, "HF_HUB_OFFLINE": "1"}
    completed = subprocess.run(
        [sys.executable, "-c", _RUNNER],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr[-2000:]
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    before = result["before"]

    assert before["supported"] is False
    assert result["stale"] == "readiness_stale"
    assert result["after_stale"] == before
    assert result["active"] == {
        "supported": True,
        "model": True,
        "locale": "sl_SI",
        "languages": before["languages"] + 1,
    }
    assert result["tasks"] == ["deidentification"]
    assert result["claim"] == "supported"
    assert result["record"]["languages"][0]["withheld_tasks"] == ["summarization"]
    assert "/synthetic-models" not in json.dumps(result["record"])
    assert result["rolled_back"] == before
    assert result["builtin_preserved"] is True
    assert result["installed"] is True
    assert result["paths_hidden"] is True
    assert result["tampered"] == "artifact_digest_mismatch"
    assert result["after_tamper"] == before
