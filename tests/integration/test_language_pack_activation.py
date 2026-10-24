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
        return key_id == "synthetic" and hmac.compare_digest(expected, signature)


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

record = activator.activate(manifest, [readiness(timedelta(days=1))], handles)
result["active"] = view()
result["tasks"] = list(activator.qualified_tasks("sl"))
result["claim"] = activator.clinical_claim("sl")
result["record"] = json.loads(record.to_json())

activator.rollback(record.activation_id)
result["rolled_back"] = view()
result["builtin_preserved"] = LANGUAGE_PACK_REGISTRY.get("en") is builtin_en
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
