"""Regenerate the portable-bundle golden vectors.

TEST-ONLY KEYS. Every signing key used here is derived from a public label with
SHA-256, so anyone can reproduce it. These keys exist only to make the
cross-language vectors deterministic. Never add them to a production trust
policy, and never sign real bundles with them.

Usage::

    python tests/fixtures/portable_bundle/generate_vectors.py          # write
    python tests/fixtures/portable_bundle/generate_vectors.py --check  # verify

All bundle files are tiny synthetic text placeholders. No model weights,
patient data, credentials, or restricted assets are involved.
"""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openmed.core import portable_bundle as pb  # noqa: E402

VECTORS_PATH = Path(__file__).with_name("vectors.json")

TEST_ONLY_KEY_LABELS = {
    "openmed-test-root-a": "openmed.portable_bundle.TEST-ONLY.never-trust.root-a",
    "openmed-test-root-b": "openmed.portable_bundle.TEST-ONLY.never-trust.root-b",
    "openmed-test-rogue": "openmed.portable_bundle.TEST-ONLY.never-trust.rogue",
}
"""Public derivation labels for the TEST-ONLY Ed25519 seeds."""

# RFC 8032 section 7.1 TEST 1-3 public keys, messages, and signatures.
RFC8032_VECTORS = (
    (
        "TEST 1",
        "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a",
        "",
        "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e06522490155"
        "5fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b",
    ),
    (
        "TEST 2",
        "3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c",
        "72",
        "92a009a9f0d4cab8720e820b5f642540a2b27b5416503f8fb3762223ebdb69da"
        "085ac1e43e15996e458f3613d0f11d8c387b2eaeb4302aeeb00d291612bb0c00",
    ),
    (
        "TEST 3",
        "fc51cd8e6218a1a38da47ed00230f0580816ed13ba3303ac5deb911548908025",
        "af82",
        "6291d657deec24024827e69c3abe01a30ce548a284743a445e3680d7db5ac3ac"
        "18ff9b538d16f290ae67f760984dc6594a7c15e9716ed28dc027beceea1ec40a",
    ),
)

ED25519_ORDER = (1 << 252) + 27742317777372353535851937790883648493

NOW = "2026-11-07T00:00:00Z"

GOLDEN_TREE = {
    "LICENSE": "Synthetic portable-bundle test fixture.\nLicensed Apache-2.0.\n",
    "calibration/profile.json": '{"profile_version":1,"synthetic":true}\n',
    "config.json": '{"model_type":"bert","num_labels":3,"synthetic":true}\n',
    "coreml/model.mlpackage/Data/com.apple.CoreML/model.mlmodel": (
        "synthetic coreml graph placeholder\n"
    ),
    "coreml/model.mlpackage/Manifest.json": '{"synthetic":true}\n',
    "evidence/parity.json": '{"cases":0,"synthetic":true}\n',
    "id2label.json": '{"0":"O","1":"B-NAME","2":"I-NAME"}\n',
    "onnx/model.onnx": "synthetic fp32 onnx graph placeholder\n",
    "onnx/model_int8.onnx": "synthetic int8 onnx graph placeholder\n",
    "policy/default.json": '{"policy_version":1,"synthetic":true}\n',
    "provenance/export.json": '{"schema_version":1,"synthetic":true}\n',
    "segmenter/rules.txt": "synthetic segmenter rules\n",
    "tokenizer.json": '{"synthetic":true,"version":"1.0"}\n',
    "vocab.txt": "[PAD]\n[UNK]\nsynthetic\n",
}

ROLES = {
    "LICENSE": "license",
    "calibration/profile.json": "calibration",
    "config.json": "config",
    "coreml/model.mlpackage/Data/com.apple.CoreML/model.mlmodel": "graph",
    "coreml/model.mlpackage/Manifest.json": "graph",
    "evidence/parity.json": "evidence",
    "id2label.json": "labels",
    "onnx/model.onnx": "graph",
    "onnx/model_int8.onnx": "graph",
    "policy/default.json": "policy",
    "provenance/export.json": "provenance",
    "segmenter/rules.txt": "segmenter",
    "tokenizer.json": "tokenizer",
    "vocab.txt": "tokenizer",
}


def seed(kid: str) -> bytes:
    """Return the TEST-ONLY Ed25519 seed for ``kid``."""

    return hashlib.sha256(TEST_ONLY_KEY_LABELS[kid].encode("ascii")).digest()


def b64url(data: bytes) -> str:
    """Return unpadded base64url text."""

    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def jwk(kid: str, not_before: str, not_after: str, status: str = "active") -> dict:
    """Return a TEST-ONLY trust-policy key entry."""

    return {
        "crv": "Ed25519",
        "kid": kid,
        "kty": "OKP",
        "not_after": not_after,
        "not_before": not_before,
        "status": status,
        "x": b64url(pb.ed25519_public_key(seed(kid))),
    }


def policy(keys: list[dict], threshold: int = 1, revoked: list[str] = ()) -> dict:
    """Return a trust-policy document."""

    return {
        "keys": keys,
        "revoked_content_ids": list(revoked),
        "schema": pb.TRUST_POLICY_SCHEMA,
        "schema_version": pb.TRUST_POLICY_SCHEMA_VERSION,
        "threshold": threshold,
    }


def file_entry(path: str, text: str) -> dict:
    """Return a manifest ``files`` entry for a synthetic text file."""

    data = text.encode("utf-8")
    media = "application/json" if path.endswith(".json") else "text/plain"
    if path.endswith(".onnx"):
        media = "application/onnx"
    elif path.endswith(".mlmodel"):
        media = "application/vnd.apple.coreml"
    return {
        "executable": False,
        "media_type": media,
        "path": path,
        "role": ROLES[path],
        "sha256": "sha256:" + hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
    }


def golden_manifest() -> dict:
    """Return the golden manifest value covering every section."""

    coreml = "coreml/model.mlpackage"
    return {
        "bundle": {
            "expires_at": "2027-05-01T00:00:00Z",
            "id": "openmed.synthetic.tiny-pii",
            "issued_at": "2026-10-01T00:00:00Z",
            "sequence": 7,
            "version": "1.2.0",
        },
        "calibration": {"path": "calibration/profile.json", "profile_version": 1},
        "compatibility": {
            "sdks": [
                {"max_exclusive": "5.0.0", "min": "4.0.0", "name": name}
                for name in (
                    "openmed-python",
                    "openmedkit-android",
                    "openmedkit-swift",
                    "openmedkit-web",
                )
            ]
        },
        "critical_extensions": [],
        "decoder": {"contract_version": 1, "kind": "bio"},
        "evidence": [{"kind": "parity", "path": "evidence/parity.json"}],
        "extensions": {"org.openmed.test.note": {"kept": True, "revision": 1}},
        "files": [
            file_entry(path, GOLDEN_TREE[path])
            for path in sorted(GOLDEN_TREE, key=lambda item: item.encode("utf-8"))
        ],
        "labels": {"count": 3, "path": "id2label.json", "scheme": "bio"},
        "licenses": [{"files": ["LICENSE"], "id": "apache-2.0"}],
        "model": {
            "architecture": "bert",
            "family": "pii",
            "id": "OpenMed/synthetic-tiny-pii",
            "languages": ["en"],
            "revision": "0123456789abcdef0123456789abcdef01234567",
            "task": "token-classification",
        },
        "offsets": {"contract_version": 1, "unit": "unicode_scalar"},
        "policy": {
            "max_version_exclusive": 2,
            "min_version": 1,
            "path": "policy/default.json",
        },
        "provenance": {
            "export_record": "provenance/export.json",
            "source_digest": "sha256:"
            + hashlib.sha256(b"synthetic-source").hexdigest(),
        },
        "runtimes": [
            {
                "entrypoint": f"{coreml}/Manifest.json",
                "fallback": "onnx-fp32",
                "files": [
                    f"{coreml}/Data/com.apple.CoreML/model.mlmodel",
                    f"{coreml}/Manifest.json",
                    "config.json",
                ],
                "format": "coreml",
                "id": "coreml-fp16",
                "min_memory_bytes": 268435456,
                "min_storage_bytes": 1048576,
                "platforms": [
                    {"max_os_exclusive": None, "min_os": "17.0", "platform": "ios"},
                    {"max_os_exclusive": None, "min_os": "14.0", "platform": "macos"},
                ],
                "precision": "fp16",
            },
            {
                "entrypoint": "onnx/model_int8.onnx",
                "fallback": "onnx-fp32",
                "files": ["config.json", "onnx/model_int8.onnx"],
                "format": "onnx",
                "id": "onnx-int8",
                "min_memory_bytes": 134217728,
                "min_storage_bytes": 1048576,
                "platforms": [
                    {"max_os_exclusive": None, "min_os": "26", "platform": "android"},
                    {"max_os_exclusive": None, "min_os": None, "platform": "web"},
                    {"max_os_exclusive": None, "min_os": None, "platform": "linux"},
                ],
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
                "platforms": [
                    {"max_os_exclusive": None, "min_os": "26", "platform": "android"},
                    {"max_os_exclusive": None, "min_os": "17.0", "platform": "ios"},
                    {"max_os_exclusive": None, "min_os": "14.0", "platform": "macos"},
                    {"max_os_exclusive": None, "min_os": None, "platform": "linux"},
                    {"max_os_exclusive": None, "min_os": None, "platform": "web"},
                ],
                "precision": "fp32",
            },
        ],
        "schema": pb.PORTABLE_BUNDLE_SCHEMA,
        "schema_version": pb.PORTABLE_BUNDLE_SCHEMA_VERSION,
        "segmenter": {"files": ["segmenter/rules.txt"], "id": "synthetic-rules"},
        "tokenizer": {
            "contract_version": 1,
            "files": ["tokenizer.json", "vocab.txt"],
            "kind": "wordpiece",
        },
    }


ANDROID = {
    "memory_bytes": None,
    "os_version": "26",
    "platform": "android",
    "policy_version": 1,
    "runtimes": ["onnx", "ort"],
    "sdk_name": "openmedkit-android",
    "sdk_version": "4.0.0",
    "storage_bytes": None,
}
IOS = {
    "memory_bytes": 4294967296,
    "os_version": "17.2",
    "platform": "ios",
    "policy_version": 1,
    "runtimes": ["coreml", "mlx", "onnx"],
    "sdk_name": "openmedkit-swift",
    "sdk_version": "4.1.0",
    "storage_bytes": None,
}
WEB = {
    "memory_bytes": None,
    "os_version": None,
    "platform": "web",
    "policy_version": 1,
    "runtimes": ["onnx", "transformersjs"],
    "sdk_name": "openmedkit-web",
    "sdk_version": "4.0.0",
    "storage_bytes": None,
}
PYTHON = {
    "memory_bytes": None,
    "os_version": "15.1",
    "platform": "macos",
    "policy_version": 1,
    "runtimes": ["coreml", "mlx", "onnx", "pytorch"],
    "sdk_name": "openmed-python",
    "sdk_version": "4.0.0",
    "storage_bytes": None,
}


def context(**overrides: Any) -> dict:
    """Return a fully explicit verification context."""

    value = {
        "max_envelope_bytes": pb.DEFAULT_MAX_ENVELOPE_BYTES,
        "max_files": pb.DEFAULT_MAX_FILES,
        "max_manifest_bytes": pb.DEFAULT_MAX_MANIFEST_BYTES,
        "max_total_bytes": pb.DEFAULT_MAX_TOTAL_BYTES,
        "now": NOW,
        "sequence_floors": {},
        "target": copy.deepcopy(ANDROID),
        "understood_extensions": [],
    }
    value.update(overrides)
    return value


def target(base: dict, **overrides: Any) -> dict:
    """Return a copy of a target with overrides."""

    value = copy.deepcopy(base)
    value.update(overrides)
    return value


def canonical_text(value: Any) -> str:
    """Return canonical JSON text for any JSON value."""

    return pb.canonical_json_bytes(value).decode("utf-8")


def sign(manifest_text: str, kids: list[str]) -> str:
    """Return a canonical envelope signed by the TEST-ONLY ``kids``.

    Malformed-manifest cases must still carry valid signatures so verifiers
    reach the structural checks, so this signs raw bytes directly instead of
    using :func:`openmed.core.portable_bundle.sign_portable_manifest`, which
    refuses non-canonical input.
    """

    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    data = manifest_text.encode("utf-8")
    message = pb.signing_input(data)
    envelope = {
        "content_id": pb.content_id(data),
        "schema": pb.ENVELOPE_SCHEMA,
        "schema_version": pb.ENVELOPE_SCHEMA_VERSION,
        "signatures": [
            {
                "alg": pb.SIGNATURE_ALGORITHM,
                "kid": kid,
                "sig": b64url(
                    Ed25519PrivateKey.from_private_bytes(seed(kid)).sign(message)
                ),
            }
            for kid in kids
        ],
    }
    return canonical_text(envelope)


def edit_envelope(envelope_text: str, edit: Callable[[dict], None]) -> str:
    """Return canonical envelope text after ``edit``."""

    value = json.loads(envelope_text)
    edit(value)
    return canonical_text(value)


def mutated(edit: Callable[[dict], None]) -> str:
    """Return canonical text of the golden manifest after ``edit``."""

    value = golden_manifest()
    edit(value)
    return canonical_text(value)


def _runtime(value: dict, runtime_id: str) -> dict:
    return next(item for item in value["runtimes"] if item["id"] == runtime_id)


def _file(value: dict, path: str) -> dict:
    return next(item for item in value["files"] if item["path"] == path)


def _resort(value: dict) -> None:
    value["files"].sort(key=lambda item: item["path"].encode("utf-8"))


def _add_file(value: dict, path: str, text: str = "synthetic\n") -> None:
    data = text.encode("utf-8")
    value["files"].append(
        {
            "executable": False,
            "media_type": "text/plain",
            "path": path,
            "role": "metadata",
            "sha256": "sha256:" + hashlib.sha256(data).hexdigest(),
            "size_bytes": len(data),
        }
    )
    _resort(value)


def malleable_signature(envelope_text: str) -> str:
    """Return the envelope with S replaced by S + L (non-canonical scalar)."""

    def edit(value: dict) -> None:
        raw = base64.urlsafe_b64decode(value["signatures"][0]["sig"] + "==")
        scalar = int.from_bytes(raw[32:], "little") + ED25519_ORDER
        value["signatures"][0]["sig"] = b64url(raw[:32] + scalar.to_bytes(32, "little"))

    return edit_envelope(envelope_text, edit)


def build_cases() -> list[dict]:
    """Return every golden case before expected results are attached."""

    golden = canonical_text(golden_manifest())
    golden_envelope = sign(golden, ["openmed-test-root-a"])
    golden_id = pb.content_id(golden.encode("utf-8"))
    cases: list[dict] = []

    def case(
        case_id: str,
        reason: str | None,
        *,
        manifest: str = golden,
        envelope: str | None = None,
        signers: tuple[str, ...] = ("openmed-test-root-a",),
        trust_policy: str = "default",
        tree: str | None = "golden",
        tree_remove: tuple[str, ...] = (),
        tree_set: dict[str, str] | None = None,
        ctx: dict | None = None,
    ) -> None:
        cases.append(
            {
                "context": ctx or context(),
                "envelope": envelope
                if envelope is not None
                else sign(manifest, list(signers)),
                "id": case_id,
                "intended_reason": reason,
                "manifest": manifest,
                "tree": tree,
                "tree_remove": sorted(tree_remove),
                "tree_set": [
                    {"path": path, "text": text}
                    for path, text in sorted((tree_set or {}).items())
                ],
                "trust_policy": trust_policy,
            }
        )

    # Accepted bundles.
    case("valid_android", None, envelope=golden_envelope)
    case("valid_manifest_only", None, envelope=golden_envelope, tree=None)
    case("valid_ios", None, envelope=golden_envelope, ctx=context(target=IOS))
    case("valid_web", None, envelope=golden_envelope, ctx=context(target=WEB))
    case("valid_python", None, envelope=golden_envelope, ctx=context(target=PYTHON))
    case("valid_no_target", None, envelope=golden_envelope, ctx=context(target=None))
    case(
        "valid_rotation_new_key",
        None,
        signers=("openmed-test-root-b",),
        trust_policy="rotation",
    )
    case(
        "valid_rotation_old_key_expired_dual_signed",
        None,
        signers=("openmed-test-root-a", "openmed-test-root-b"),
        trust_policy="rotation",
        ctx=context(now="2026-12-15T00:00:00Z"),
    )
    case(
        "valid_threshold_two",
        None,
        signers=("openmed-test-root-a", "openmed-test-root-b"),
        trust_policy="threshold_two",
    )
    case(
        "valid_sequence_at_floor",
        None,
        envelope=golden_envelope,
        ctx=context(sequence_floors={"openmed.synthetic.tiny-pii": 7}),
    )
    critical = mutated(
        lambda value: value["critical_extensions"].append("org.openmed.test.note")
    )
    case(
        "valid_critical_extension_understood",
        None,
        manifest=critical,
        ctx=context(understood_extensions=["org.openmed.test.note"]),
    )

    # Size limits and envelope shape.
    case(
        "manifest_too_large",
        "manifest_too_large",
        envelope=golden_envelope,
        ctx=context(max_manifest_bytes=256),
    )
    case(
        "envelope_too_large",
        "envelope_too_large",
        envelope=golden_envelope,
        ctx=context(max_envelope_bytes=64),
    )
    case(
        "envelope_not_canonical",
        "envelope_malformed",
        envelope=json.dumps(json.loads(golden_envelope), indent=2),
    )
    case(
        "envelope_unknown_member",
        "envelope_malformed",
        envelope=edit_envelope(golden_envelope, lambda value: value.update(note=1)),
    )
    case(
        "envelope_version_unsupported",
        "envelope_version_unsupported",
        envelope=edit_envelope(
            golden_envelope, lambda value: value.update(schema_version=2)
        ),
    )

    # Identity, revocation, and signatures.
    modified = golden.replace('"sequence":7', '"sequence":8')
    case(
        "modified_manifest",
        "content_id_mismatch",
        manifest=modified,
        envelope=golden_envelope,
    )
    case(
        "modified_manifest_relabelled",
        "signature_invalid",
        manifest=modified,
        envelope=edit_envelope(
            golden_envelope,
            lambda value: value.update(
                content_id=pb.content_id(modified.encode("utf-8"))
            ),
        ),
    )
    case(
        "bundle_revoked",
        "bundle_revoked",
        envelope=golden_envelope,
        trust_policy="revoked_bundle",
    )
    case("signature_missing", "signature_missing", signers=())
    case(
        "signature_algorithm_unsupported",
        "signature_algorithm_unsupported",
        envelope=edit_envelope(
            golden_envelope,
            lambda value: value["signatures"][0].update(alg="RS256"),
        ),
    )
    case("wrong_signer", "untrusted_key", signers=("openmed-test-rogue",))
    case(
        "key_revoked",
        "key_revoked",
        envelope=golden_envelope,
        trust_policy="key_revoked",
    )
    case(
        "key_not_yet_valid",
        "key_not_yet_valid",
        envelope=golden_envelope,
        trust_policy="key_future",
    )
    case(
        "key_expired",
        "key_expired",
        envelope=golden_envelope,
        ctx=context(now="2027-08-01T00:00:00Z"),
    )
    case(
        "rotation_old_key_after_window",
        "key_expired",
        envelope=golden_envelope,
        trust_policy="rotation",
        ctx=context(now="2026-12-15T00:00:00Z"),
    )

    def flip(value: dict) -> None:
        raw = bytearray(base64.urlsafe_b64decode(value["signatures"][0]["sig"] + "=="))
        raw[10] ^= 0x01
        value["signatures"][0]["sig"] = b64url(bytes(raw))

    case(
        "signature_bit_flip",
        "signature_invalid",
        envelope=edit_envelope(golden_envelope, flip),
    )
    case(
        "signature_padded_base64",
        "signature_invalid",
        envelope=edit_envelope(
            golden_envelope,
            lambda value: value["signatures"][0].update(
                sig=value["signatures"][0]["sig"] + "=="
            ),
        ),
    )
    case(
        "signature_scalar_not_reduced",
        "signature_invalid",
        envelope=malleable_signature(golden_envelope),
    )
    case(
        "signature_threshold_not_met",
        "signature_threshold_not_met",
        envelope=golden_envelope,
        trust_policy="threshold_two",
    )
    case(
        "signature_duplicate_key_counts_once",
        "signature_threshold_not_met",
        envelope=edit_envelope(
            golden_envelope,
            lambda value: value["signatures"].append(dict(value["signatures"][0])),
        ),
        trust_policy="threshold_two",
    )

    # Canonical form and schema.
    case(
        "manifest_whitespace",
        "manifest_not_canonical",
        manifest=json.dumps(golden_manifest(), indent=2, sort_keys=True),
    )
    case(
        "manifest_reordered_members",
        "manifest_not_canonical",
        manifest=json.dumps(
            dict(reversed(list(golden_manifest().items()))),
            separators=(",", ":"),
            sort_keys=False,
        ),
    )
    case(
        "manifest_duplicate_member",
        "manifest_not_canonical",
        manifest=golden[:-1] + ',"tokenizer":null}',
    )
    case(
        "manifest_non_integer_number",
        "manifest_not_canonical",
        manifest=golden.replace('"count":3', '"count":3.0'),
    )
    case(
        "manifest_escaped_ascii",
        "manifest_not_canonical",
        manifest=golden.replace('"family":"pii"', '"family":"p\\u0069i"'),
    )
    case(
        "schema_unsupported",
        "schema_unsupported",
        manifest=mutated(lambda value: value.update(schema="openmed.other")),
    )
    case(
        "schema_version_unsupported",
        "schema_version_unsupported",
        manifest=mutated(lambda value: value.update(schema_version=2)),
    )
    case(
        "unknown_critical_top_level_field",
        "unknown_critical_field",
        manifest=mutated(lambda value: value.update(weights_url="https")),
    )
    case(
        "unknown_critical_nested_field",
        "unknown_critical_field",
        manifest=mutated(
            lambda value: _runtime(value, "onnx-int8").update(execution_hint="gpu")
        ),
    )
    case(
        "missing_required_section",
        "manifest_malformed",
        manifest=mutated(lambda value: value.pop("decoder")),
    )
    case(
        "wrong_field_type",
        "manifest_malformed",
        manifest=mutated(lambda value: value["bundle"].update(sequence="7")),
    )
    case(
        "unknown_critical_extension",
        "unknown_critical_extension",
        manifest=critical,
    )

    # Paths and file inventory structure.
    def bad_path(path: str) -> Callable[[dict], None]:
        def edit(value: dict) -> None:
            value["files"][0]["path"] = path

        return edit

    case("path_traversal", "path_unsafe", manifest=mutated(bad_path("../escape.txt")))
    case("path_absolute", "path_unsafe", manifest=mutated(bad_path("/etc/escape.txt")))
    case("path_backslash", "path_unsafe", manifest=mutated(bad_path("onnx\\x.onnx")))
    case("path_dot_segment", "path_unsafe", manifest=mutated(bad_path("a/./b.txt")))
    case(
        "path_reserved_name",
        "path_unsafe",
        manifest=mutated(bad_path(pb.MANIFEST_FILENAME)),
    )
    case(
        "executable_file",
        "executable_file_forbidden",
        manifest=mutated(lambda value: value["files"][0].update(executable=True)),
    )
    case(
        "duplicate_path",
        "duplicate_path",
        manifest=mutated(
            lambda value: (
                value["files"].append(dict(_file(value, "vocab.txt"))),
                _resort(value),
            )
        ),
    )
    case(
        "duplicate_path_case_folded",
        "duplicate_path",
        manifest=mutated(lambda value: _add_file(value, "license")),
    )
    case(
        "path_file_directory_conflict",
        "path_conflict",
        manifest=mutated(lambda value: _add_file(value, "onnx")),
    )
    case(
        "files_reordered",
        "file_order_invalid",
        manifest=mutated(lambda value: value["files"].insert(0, value["files"].pop(1))),
    )
    case(
        "reference_undeclared",
        "file_reference_undeclared",
        manifest=mutated(
            lambda value: value["tokenizer"]["files"].append("merges.txt")
        ),
    )
    case(
        "runtime_wrong_platform_combination",
        "runtime_combination_unsupported",
        manifest=mutated(
            lambda value: _runtime(value, "coreml-fp16")["platforms"].append(
                {"max_os_exclusive": None, "min_os": "26", "platform": "android"}
            )
        ),
    )
    case(
        "runtime_wrong_precision_combination",
        "runtime_combination_unsupported",
        manifest=mutated(
            lambda value: _runtime(value, "coreml-fp16").update(precision="bf16")
        ),
    )
    case(
        "fallback_cycle",
        "fallback_invalid",
        manifest=mutated(
            lambda value: _runtime(value, "onnx-fp32").update(fallback="onnx-int8")
        ),
    )
    case(
        "fallback_unknown",
        "fallback_invalid",
        manifest=mutated(
            lambda value: _runtime(value, "onnx-int8").update(fallback="tflite-fp32")
        ),
    )
    case(
        "too_many_files",
        "size_limit_exceeded",
        envelope=golden_envelope,
        ctx=context(max_files=4),
    )
    case(
        "declared_bytes_over_limit",
        "size_limit_exceeded",
        envelope=golden_envelope,
        ctx=context(max_total_bytes=128),
    )

    # Time, rollback, and compatibility.
    case(
        "bundle_not_yet_valid",
        "bundle_not_yet_valid",
        envelope=golden_envelope,
        ctx=context(now="2026-09-01T00:00:00Z"),
    )
    case(
        "bundle_expired",
        "bundle_expired",
        envelope=golden_envelope,
        ctx=context(now="2027-06-01T00:00:00Z"),
    )
    case(
        "rollback",
        "rollback_rejected",
        envelope=golden_envelope,
        ctx=context(sequence_floors={"openmed.synthetic.tiny-pii": 8}),
    )
    case(
        "sdk_too_new",
        "sdk_incompatible",
        envelope=golden_envelope,
        ctx=context(target=target(ANDROID, sdk_version="5.0.0")),
    )
    case(
        "sdk_not_declared",
        "sdk_incompatible",
        envelope=golden_envelope,
        ctx=context(target=target(ANDROID, sdk_name="openmedkit-desktop")),
    )
    case(
        "policy_incompatible",
        "policy_incompatible",
        envelope=golden_envelope,
        ctx=context(target=target(ANDROID, policy_version=2)),
    )
    case(
        "wrong_platform",
        "platform_incompatible",
        envelope=golden_envelope,
        ctx=context(
            target=target(
                IOS, platform="watchos", os_version="10.0", runtimes=["coreml"]
            )
        ),
    )
    case(
        "os_too_old",
        "platform_incompatible",
        envelope=golden_envelope,
        ctx=context(target=target(ANDROID, os_version="25")),
    )
    case(
        "memory_too_small",
        "platform_incompatible",
        envelope=golden_envelope,
        ctx=context(target=target(ANDROID, memory_bytes=1024)),
    )

    # File tree.
    case(
        "file_missing",
        "file_missing",
        envelope=golden_envelope,
        tree_remove=("onnx/model.onnx",),
    )
    case(
        "file_modified",
        "file_digest_mismatch",
        envelope=golden_envelope,
        tree_set={"vocab.txt": "[PAD]\n[UNK]\nsyntheti!\n"},
    )
    case(
        "file_oversized",
        "file_size_mismatch",
        envelope=golden_envelope,
        tree_set={"tokenizer.json": GOLDEN_TREE["tokenizer.json"] + " " * 64},
    )
    case(
        "file_truncated",
        "file_size_mismatch",
        envelope=golden_envelope,
        tree_set={"config.json": "{}\n"},
    )
    case(
        "file_added",
        "file_undeclared",
        envelope=golden_envelope,
        tree_set={"onnx/extra.onnx": "unexpected\n"},
    )
    case(
        "file_added_hidden",
        "file_undeclared",
        envelope=golden_envelope,
        tree_set={".DS_Store": "unexpected\n"},
    )
    assert golden_id
    return cases


def trust_policies() -> dict[str, dict]:
    """Return named TEST-ONLY trust policies."""

    a = "openmed-test-root-a"
    b = "openmed-test-root-b"
    golden_id = pb.content_id(canonical_text(golden_manifest()).encode("utf-8"))
    return {
        "default": policy([jwk(a, "2026-01-01T00:00:00Z", "2027-07-01T00:00:00Z")]),
        "key_future": policy([jwk(a, "2027-01-01T00:00:00Z", "2028-01-01T00:00:00Z")]),
        "key_revoked": policy(
            [jwk(a, "2026-01-01T00:00:00Z", "2027-07-01T00:00:00Z", "revoked")]
        ),
        "revoked_bundle": policy(
            [jwk(a, "2026-01-01T00:00:00Z", "2027-07-01T00:00:00Z")],
            revoked=[golden_id],
        ),
        "rotation": policy(
            [
                jwk(a, "2026-01-01T00:00:00Z", "2026-12-01T00:00:00Z"),
                jwk(b, "2026-10-01T00:00:00Z", "2028-01-01T00:00:00Z"),
            ]
        ),
        "threshold_two": policy(
            [
                jwk(a, "2026-01-01T00:00:00Z", "2027-07-01T00:00:00Z"),
                jwk(b, "2026-10-01T00:00:00Z", "2028-01-01T00:00:00Z"),
            ],
            threshold=2,
        ),
    }


CANONICALIZATION_VECTORS = (
    ("member_order_utf16", '{"\\u20ac":1,"\\ud83d\\ude00":2,"\\ufb33":3,"a":4,"A":5}'),
    ("string_escapes", '["\\u0000\\u001f\\b\\t\\n\\f\\r\\"\\\\/\\u007f\\u2028"]'),
    ("non_ascii_literal", '{"text":"\\u00e9\\u4e2d\\ud83d\\ude00"}'),
    ("integers", "[0,-1,9007199254740991,-9007199254740991]"),
    ("literals_and_nesting", '{"b":[true,false,null,{}],"a":{"z":[],"y":"x"}}'),
)
CANONICALIZATION_REJECTS = (
    ("fraction", "1.5"),
    ("beyond_safe_integer", "9007199254740992"),
    ("lone_surrogate", '"\\ud800"'),
)


def splice(base: str, text: str) -> dict:
    """Return the single edit that turns ``base`` into ``text``.

    Readers rebuild a manifest as ``base[:at] + insert + base[at + delete:]``.
    Both strings are ASCII, so offsets are byte offsets in every language.
    """

    prefix = 0
    limit = min(len(base), len(text))
    while prefix < limit and base[prefix] == text[prefix]:
        prefix += 1
    suffix = 0
    while (
        suffix < limit - prefix
        and base[len(base) - 1 - suffix] == text[len(text) - 1 - suffix]
    ):
        suffix += 1
    return {
        "at": prefix,
        "base": "golden",
        "delete": len(base) - prefix - suffix,
        "insert": text[prefix : len(text) - suffix],
    }


def build_vectors() -> dict:
    """Return the complete vector document."""

    policies = {name: value for name, value in sorted(trust_policies().items())}
    trees = {
        "golden": [
            {"path": path, "text": GOLDEN_TREE[path]}
            for path in sorted(GOLDEN_TREE, key=lambda item: item.encode("utf-8"))
        ]
    }
    golden_text = canonical_text(golden_manifest())
    if not golden_text.isascii():
        raise SystemExit("golden manifest must be ASCII so splices are byte offsets")
    manifests: dict[str, dict] = {"golden": {"text": golden_text}}
    manifest_keys = {golden_text: "golden"}
    envelopes: dict[str, str] = {}
    envelope_keys: dict[str, str] = {}
    contexts: dict[str, dict] = {"default": context()}
    context_keys = {canonical_text(context()): "default"}
    cases = []
    for item in build_cases():
        result = evaluate(item, policies, trees)
        intended = item.pop("intended_reason")
        if result["reason"] != intended:
            raise SystemExit(
                f"case {item['id']}: expected {intended}, got {result['reason']}"
            )
        manifest_text = item.pop("manifest")
        if not manifest_text.isascii():
            raise SystemExit(f"case {item['id']}: manifest must be ASCII")
        if manifest_text not in manifest_keys:
            manifest_keys[manifest_text] = item["id"]
            manifests[item["id"]] = splice(golden_text, manifest_text)
        envelope_text = item.pop("envelope")
        if envelope_text not in envelope_keys:
            envelope_keys[envelope_text] = "golden" if not envelopes else item["id"]
            envelopes[envelope_keys[envelope_text]] = envelope_text
        context_text = canonical_text(item["context"])
        if context_text not in context_keys:
            context_keys[context_text] = item["id"]
            contexts[item["id"]] = item["context"]
        item["context"] = context_keys[context_text]
        item["manifest"] = manifest_keys[manifest_text]
        item["envelope"] = envelope_keys[envelope_text]
        item["expected"] = result
        cases.append(item)
    rfc_vectors = []
    for name, public_hex, message_hex, signature_hex in RFC8032_VECTORS:
        public = bytes.fromhex(public_hex)
        if not pb._ed25519_verify(
            public, bytes.fromhex(signature_hex), bytes.fromhex(message_hex)
        ):
            raise SystemExit(f"RFC 8032 {name} does not verify")
        rfc_vectors.append(
            {
                "message_hex": message_hex,
                "name": name,
                "public_hex": public_hex,
                "signature_hex": signature_hex,
            }
        )
    golden = canonical_text(golden_manifest())
    return {
        "canonicalization": [
            {"canonical": canonical_text(json.loads(text)), "id": name, "input": text}
            for name, text in CANONICALIZATION_VECTORS
        ],
        "canonicalization_rejects": [
            {"id": name, "input": text} for name, text in CANONICALIZATION_REJECTS
        ],
        "cases": cases,
        "contexts": contexts,
        "ed25519_rfc8032": rfc_vectors,
        "envelopes": envelopes,
        "golden": {
            "content_id": pb.content_id(golden.encode("utf-8")),
            "manifest": "golden",
            "signing_input_sha256": hashlib.sha256(
                pb.signing_input(golden.encode("utf-8"))
            ).hexdigest(),
        },
        "manifests": manifests,
        "notice": (
            "TEST-ONLY vectors. Keys derive from public labels in "
            "generate_vectors.py; never trust them in production."
        ),
        "payload_type": pb.PAYLOAD_TYPE,
        "reason_codes": list(pb.REASON_CODES),
        "schema": "openmed.portable_bundle.vectors",
        "synthetic_only": True,
        "test_only_keys": True,
        "trees": trees,
        "trust_policies": policies,
        "version": 1,
    }


def evaluate(item: dict, policies: dict, trees: dict) -> dict:
    """Run the Python reference verifier for one case."""

    trust = pb.TrustPolicy.from_mapping(policies[item["trust_policy"]])
    ctx = item["context"]
    verification_target = (
        None if ctx["target"] is None else pb.VerificationTarget(**{**ctx["target"]})
    )
    verification_context = pb.VerificationContext(
        now=ctx["now"],
        target=verification_target,
        sequence_floors=ctx["sequence_floors"],
        understood_extensions=frozenset(ctx["understood_extensions"]),
        max_manifest_bytes=ctx["max_manifest_bytes"],
        max_envelope_bytes=ctx["max_envelope_bytes"],
        max_total_bytes=ctx["max_total_bytes"],
        max_files=ctx["max_files"],
    )
    files = None
    if item["tree"] is not None:
        files = {
            entry["path"]: entry["text"].encode("utf-8")
            for entry in trees[item["tree"]]
        }
        for path in item["tree_remove"]:
            files.pop(path)
        for entry in item["tree_set"]:
            files[entry["path"]] = entry["text"].encode("utf-8")
    result = pb.verify_portable_bundle(
        item["manifest"].encode("utf-8"),
        item["envelope"].encode("utf-8"),
        trust_policy=trust,
        context=verification_context,
        files=files,
    )
    payload = result.to_dict()
    payload.pop("schema")
    payload.pop("schema_version")
    return payload


def render() -> str:
    """Return the serialized vector document."""

    return json.dumps(build_vectors(), indent=2, sort_keys=True) + "\n"


def main() -> int:
    """Write or check ``vectors.json``."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    text = render()
    if args.check:
        if VECTORS_PATH.read_text(encoding="utf-8") != text:
            print("portable bundle vectors are stale")
            return 1
        return 0
    VECTORS_PATH.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
