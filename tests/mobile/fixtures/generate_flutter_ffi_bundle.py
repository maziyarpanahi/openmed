"""Regenerate the signed portable bundle used by the Flutter FFI tests.

TEST-ONLY KEY. The signing seed is derived from a public label with SHA-256,
so anyone can reproduce it. It exists only to make the Flutter fixture
deterministic. Never add it to a production trust policy, and never sign real
bundles with it.

Usage::

    python tests/mobile/fixtures/generate_flutter_ffi_bundle.py          # write
    python tests/mobile/fixtures/generate_flutter_ffi_bundle.py --check  # verify

Every bundle file is a tiny synthetic text placeholder. The Flutter fixture
session hashes the model placeholders but never interprets them; no model
weights, patient data, credentials, or restricted assets are involved.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openmed.core import portable_bundle as pb  # noqa: E402

FIXTURE_PATH = Path(__file__).with_name("flutter_ffi_bundle.json")

TEST_ONLY_KID = "openmed-test-flutter-ffi"
TEST_ONLY_LABEL = "openmed.portable_bundle.TEST-ONLY.never-trust.flutter-ffi"
"""Public derivation label for the TEST-ONLY Ed25519 seed."""

NOW = "2026-11-07T00:00:00Z"

TREE = {
    "LICENSE": "Synthetic Flutter FFI bundle fixture.\nLicensed Apache-2.0.\n",
    "config.json": '{"model_type":"bert","num_labels":6,"synthetic":true}\n',
    "id2label.json": (
        '{"0":"O","1":"B-PERSON","2":"B-DATE_OF_BIRTH","3":"B-EMAIL",'
        '"4":"B-NAME","5":"I-NAME"}\n'
    ),
    "onnx/model.onnx": "synthetic fp32 onnx graph placeholder for flutter\n",
    "onnx/model_int8.onnx": "synthetic int8 onnx graph placeholder for flutter\n",
    "tokenizer.json": '{"synthetic":true,"version":"1.0"}\n',
}

ROLES = {
    "LICENSE": "license",
    "config.json": "config",
    "id2label.json": "labels",
    "onnx/model.onnx": "graph",
    "onnx/model_int8.onnx": "graph",
    "tokenizer.json": "tokenizer",
}

PLATFORMS = [
    {"max_os_exclusive": None, "min_os": "26", "platform": "android"},
    {"max_os_exclusive": None, "min_os": "13.0", "platform": "ios"},
    {"max_os_exclusive": None, "min_os": None, "platform": "linux"},
    {"max_os_exclusive": None, "min_os": None, "platform": "macos"},
    {"max_os_exclusive": None, "min_os": None, "platform": "windows"},
]


def seed() -> bytes:
    """Return the TEST-ONLY Ed25519 seed."""

    return hashlib.sha256(TEST_ONLY_LABEL.encode("ascii")).digest()


def b64url(data: bytes) -> str:
    """Return unpadded base64url text."""

    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def file_entry(path: str) -> dict[str, Any]:
    """Return a manifest ``files`` entry for a synthetic text file."""

    data = TREE[path].encode("utf-8")
    media = "application/json" if path.endswith(".json") else "text/plain"
    if path.endswith(".onnx"):
        media = "application/onnx"
    return {
        "executable": False,
        "media_type": media,
        "path": path,
        "role": ROLES[path],
        "sha256": "sha256:" + hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
    }


def manifest() -> dict[str, Any]:
    """Return the Flutter fixture manifest value."""

    return {
        "bundle": {
            "expires_at": "2031-01-01T00:00:00Z",
            "id": "openmed.synthetic.flutter-ffi",
            "issued_at": "2026-10-01T00:00:00Z",
            "sequence": 3,
            "version": "1.0.0",
        },
        "calibration": None,
        "compatibility": {
            "sdks": [
                {"max_exclusive": "5.0.0", "min": "4.0.0", "name": "openmedkit-flutter"}
            ]
        },
        "critical_extensions": [],
        "decoder": {"contract_version": 1, "kind": "bio"},
        "evidence": [],
        "extensions": {},
        "files": [
            file_entry(path)
            for path in sorted(TREE, key=lambda item: item.encode("utf-8"))
        ],
        "labels": {"count": 6, "path": "id2label.json", "scheme": "bio"},
        "licenses": [{"files": ["LICENSE"], "id": "apache-2.0"}],
        "model": {
            "architecture": "bert",
            "family": "pii",
            "id": "OpenMed/synthetic-flutter-ffi",
            "languages": ["en"],
            "revision": None,
            "task": "token-classification",
        },
        "offsets": {"contract_version": 1, "unit": "unicode_scalar"},
        "policy": {"max_version_exclusive": 2, "min_version": 1, "path": None},
        "provenance": {"export_record": None, "source_digest": None},
        "runtimes": [
            {
                "entrypoint": "onnx/model_int8.onnx",
                "fallback": "onnx-fp32",
                "files": ["config.json", "onnx/model_int8.onnx"],
                "format": "onnx",
                "id": "onnx-int8",
                "min_memory_bytes": 134217728,
                "min_storage_bytes": 1048576,
                "platforms": PLATFORMS,
                "precision": "int8",
            },
            {
                "entrypoint": "onnx/model.onnx",
                "fallback": None,
                "files": ["config.json", "onnx/model.onnx"],
                "format": "onnx",
                "id": "onnx-fp32",
                "min_memory_bytes": 268435456,
                "min_storage_bytes": 2097152,
                "platforms": PLATFORMS,
                "precision": "fp32",
            },
        ],
        "schema": pb.PORTABLE_BUNDLE_SCHEMA,
        "schema_version": pb.PORTABLE_BUNDLE_SCHEMA_VERSION,
        "segmenter": None,
        "tokenizer": {
            "contract_version": 1,
            "files": ["tokenizer.json"],
            "kind": "wordpiece",
        },
    }


def trust_policy() -> dict[str, Any]:
    """Return the TEST-ONLY trust policy for the fixture key."""

    return {
        "keys": [
            {
                "crv": "Ed25519",
                "kid": TEST_ONLY_KID,
                "kty": "OKP",
                "not_after": "2031-01-01T00:00:00Z",
                "not_before": "2026-01-01T00:00:00Z",
                "status": "active",
                "x": b64url(pb.ed25519_public_key(seed())),
            }
        ],
        "revoked_content_ids": [],
        "schema": pb.TRUST_POLICY_SCHEMA,
        "schema_version": pb.TRUST_POLICY_SCHEMA_VERSION,
        "threshold": 1,
    }


def build() -> dict[str, Any]:
    """Return the fixture document."""

    manifest_bytes = pb.encode_portable_manifest(manifest())
    envelope_bytes = pb.sign_portable_manifest(
        manifest_bytes, [(TEST_ONLY_KID, seed())]
    )
    return {
        "content_id": pb.content_id(manifest_bytes),
        "envelope": envelope_bytes.decode("utf-8"),
        "manifest": manifest_bytes.decode("utf-8"),
        "notice": (
            "TEST-ONLY bundle. The key derives from a public label in "
            "generate_flutter_ffi_bundle.py; never trust it in production."
        ),
        "now": NOW,
        "schema": "openmed.flutter_ffi.bundle_fixture",
        "synthetic_only": True,
        "test_only_keys": True,
        "tree": [
            {"path": path, "text": TREE[path]}
            for path in sorted(TREE, key=lambda item: item.encode("utf-8"))
        ],
        "trust_policy": trust_policy(),
        "version": 1,
    }


def render() -> str:
    """Return the fixture file text."""

    return json.dumps(build(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Write or check the committed fixture."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="fail on drift")
    args = parser.parse_args(argv)
    text = render()
    if args.check:
        current = FIXTURE_PATH.read_text(encoding="utf-8")
        if current != text:
            print("flutter_ffi_bundle.json is out of date", file=sys.stderr)
            return 1
        return 0
    FIXTURE_PATH.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
