from __future__ import annotations

import hashlib
import hmac
import json
import logging
import re
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from openmed.skills.bundle_verify import (
    HASH_PREFIX_LENGTH,
    REASON_ENTRY_POINT_MISSING,
    REASON_ENTRY_POINT_NOT_DECLARED,
    REASON_FILE_MISSING,
    REASON_HASH_MISMATCH,
    REASON_MANIFEST_MALFORMED,
    REASON_MANIFEST_VERSION_UNSUPPORTED,
    REASON_SIGNATURE_INVALID,
    REASON_SIGNATURE_PUBLIC_KEY_REQUIRED,
    REASON_SIGNATURE_REQUIRED,
    SUPPORTED_MANIFEST_VERSIONS,
    BundleFileResult,
    BundleVerificationResult,
    BundleVerifier,
    SkillBundleManifest,
    verify_bundle,
)

_HEX64_RE = re.compile(r"[0-9a-f]{64}")
_BUNDLE_ID = "com.example.my-skill"
_SENTINEL_CONTENT = "SYNTHETIC_BUNDLE_SECRET_CONTENT"


def _make_bundle(
    tmp_path,
    files_content,
    manifest_extra=None,
    signature_key=None,
    signature_ed25519_private_key=None,
):
    """Create a bundle dir with manifest.json and files.

    Computes correct SHA-256 hashes. If signature_key is provided, computes
    HMAC-SHA256 over canonical manifest bytes (sorted keys, no whitespace,
    excluding the signature field). If signature_ed25519_private_key is
    provided, computes an Ed25519 signature over the same canonical bytes.
    Returns (bundle_dir, manifest_dict).
    """

    bundle_dir = tmp_path
    files = {}
    for rel_path, content in files_content.items():
        path = bundle_dir / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        data = content.encode("utf-8") if isinstance(content, str) else content
        path.write_bytes(data)
        files[rel_path] = hashlib.sha256(data).hexdigest()

    manifest = {
        "manifest_version": "1.0",
        "bundle_id": _BUNDLE_ID,
        "entry_points": [],
        "files": files,
        "signature_scheme": "none",
        "signature": "",
    }
    if manifest_extra:
        manifest.update(manifest_extra)
    if signature_key is not None:
        manifest["signature_scheme"] = "hmac-sha256"
        canonical_payload = {
            "manifest_version": manifest["manifest_version"],
            "bundle_id": manifest["bundle_id"],
            "entry_points": list(manifest["entry_points"]),
            "files": dict(manifest["files"]),
            "signature_scheme": manifest["signature_scheme"],
        }
        canonical = json.dumps(
            canonical_payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        manifest["signature"] = hmac.new(
            signature_key, canonical, hashlib.sha256
        ).hexdigest()
    if signature_ed25519_private_key is not None:
        manifest["signature_scheme"] = "ed25519"
        canonical_payload = {
            "manifest_version": manifest["manifest_version"],
            "bundle_id": manifest["bundle_id"],
            "entry_points": list(manifest["entry_points"]),
            "files": dict(manifest["files"]),
            "signature_scheme": manifest["signature_scheme"],
        }
        canonical = json.dumps(
            canonical_payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        sig = signature_ed25519_private_key.sign(canonical)
        manifest["signature"] = sig.hex()

    (bundle_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return bundle_dir, manifest


def _write_manifest(bundle_dir, manifest):
    (bundle_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_valid_bundle_no_signature(tmp_path):
    bundle_dir, _ = _make_bundle(
        tmp_path,
        {"main.py": "print('hello')\n"},
        manifest_extra={"entry_points": ["main.py"]},
    )

    result = verify_bundle(bundle_dir)

    assert result.valid is True
    assert result.reason == ""
    assert result.message == ""
    assert result.signature_verified is True
    assert result.bundle_id == _BUNDLE_ID
    assert result.manifest_version == "1.0"
    assert result.entry_points_checked == ("main.py",)
    assert len(result.files) == 1
    assert result.files[0].matched is True


def test_valid_bundle_with_hmac_signature(tmp_path):
    key = b"super-secret-key"
    bundle_dir, _ = _make_bundle(
        tmp_path,
        {"main.py": "print('hello')\n"},
        manifest_extra={"entry_points": ["main.py"]},
        signature_key=key,
    )

    result = verify_bundle(bundle_dir, signature_key=key)

    assert result.valid is True
    assert result.reason == ""
    assert result.signature_verified is True
    assert result.bundle_id == _BUNDLE_ID


def test_manifest_malformed_json(tmp_path):
    (tmp_path / "manifest.json").write_text("{not valid json", encoding="utf-8")

    result = verify_bundle(tmp_path)

    assert result.valid is False
    assert result.reason == REASON_MANIFEST_MALFORMED
    assert result.bundle_id == ""


def test_manifest_validation_failure(tmp_path):
    manifest = {
        "manifest_version": "1.0",
        "bundle_id": "",
        "entry_points": ["main.py"],
        "files": {"main.py": "a" * 64},
        "signature_scheme": "none",
        "signature": "",
    }
    _write_manifest(tmp_path, manifest)
    (tmp_path / "main.py").write_text("print('hi')\n", encoding="utf-8")

    result = verify_bundle(tmp_path)

    assert result.valid is False
    assert result.reason == REASON_MANIFEST_MALFORMED


def test_unsupported_manifest_version(tmp_path):
    bundle_dir, manifest = _make_bundle(tmp_path, {"main.py": "print('hi')\n"})
    manifest["manifest_version"] = "2.0"
    _write_manifest(bundle_dir, manifest)

    result = verify_bundle(bundle_dir)

    assert result.valid is False
    assert result.reason == REASON_MANIFEST_VERSION_UNSUPPORTED
    assert result.bundle_id == _BUNDLE_ID
    assert result.manifest_version == "2.0"


def test_file_missing(tmp_path):
    bundle_dir, manifest = _make_bundle(
        tmp_path,
        {"main.py": "print('hi')\n"},
        manifest_extra={"entry_points": ["main.py"]},
    )
    manifest["files"]["ghost.py"] = "a" * 64
    _write_manifest(bundle_dir, manifest)

    result = verify_bundle(bundle_dir)

    assert result.valid is False
    assert result.reason == REASON_FILE_MISSING


def test_hash_mismatch(tmp_path):
    bundle_dir, _ = _make_bundle(
        tmp_path,
        {"main.py": "print('hi')\n"},
        manifest_extra={"entry_points": ["main.py"]},
    )
    (bundle_dir / "main.py").write_text("print('tampered')\n", encoding="utf-8")

    result = verify_bundle(bundle_dir)

    assert result.valid is False
    assert result.reason == REASON_HASH_MISMATCH
    assert len(result.files) == 1
    assert result.files[0].matched is False
    assert result.files[0].path == "main.py"


def test_entry_point_missing(tmp_path):
    bundle_dir, _ = _make_bundle(
        tmp_path,
        {"main.py": "print('hi')\n"},
        manifest_extra={"entry_points": ["missing.py"]},
    )

    result = verify_bundle(bundle_dir)

    assert result.valid is False
    assert result.reason == REASON_ENTRY_POINT_MISSING
    assert "missing.py" in result.entry_points_checked


def test_entry_point_not_declared_in_files(tmp_path):
    bundle_dir, _ = _make_bundle(
        tmp_path,
        {"main.py": "print('hi')\n"},
        manifest_extra={"entry_points": ["extra.py"]},
    )
    (bundle_dir / "extra.py").write_text("print('extra')\n", encoding="utf-8")

    result = verify_bundle(bundle_dir)

    assert result.valid is False
    assert result.reason == REASON_ENTRY_POINT_NOT_DECLARED
    assert "extra.py" in result.entry_points_checked


def test_signature_required_but_no_key(tmp_path):
    key = b"super-secret-key"
    bundle_dir, _ = _make_bundle(
        tmp_path,
        {"main.py": "print('hi')\n"},
        manifest_extra={"entry_points": ["main.py"]},
        signature_key=key,
    )

    result = verify_bundle(bundle_dir)

    assert result.valid is False
    assert result.reason == REASON_SIGNATURE_REQUIRED
    assert result.signature_verified is False


def test_signature_invalid(tmp_path):
    key = b"super-secret-key"
    bundle_dir, manifest = _make_bundle(
        tmp_path,
        {"main.py": "print('hi')\n"},
        manifest_extra={"entry_points": ["main.py"]},
        signature_key=key,
    )
    manifest["signature"] = "0" * 64
    _write_manifest(bundle_dir, manifest)

    result = verify_bundle(bundle_dir, signature_key=key)

    assert result.valid is False
    assert result.reason == REASON_SIGNATURE_INVALID
    assert result.signature_verified is False


def test_unsupported_signature_scheme_rejected(tmp_path):
    bundle_dir, manifest = _make_bundle(
        tmp_path,
        {"main.py": "print('hi')\n"},
        manifest_extra={"entry_points": ["main.py"]},
    )
    manifest["signature_scheme"] = "rsa-sha256"
    _write_manifest(bundle_dir, manifest)

    result = verify_bundle(bundle_dir)

    assert result.valid is False
    assert result.reason == REASON_MANIFEST_MALFORMED


def test_manifest_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        verify_bundle(tmp_path)


def test_result_to_dict_no_raw_hashes(tmp_path):
    bundle_dir, _ = _make_bundle(
        tmp_path,
        {"main.py": "print('hello')\n"},
        manifest_extra={"entry_points": ["main.py"]},
    )

    result = verify_bundle(bundle_dir)
    serialized = json.dumps(result.to_dict())

    assert _HEX64_RE.search(serialized) is None


def test_file_result_to_dict_uses_prefixes():
    digest = "a" * 64
    other = "b" * 64
    file_result = BundleFileResult(
        path="main.py",
        declared_hash=digest,
        actual_hash=other,
        matched=False,
    )

    data = file_result.to_dict()

    assert "declared_hash" not in data
    assert "actual_hash" not in data
    assert data["declared_hash_prefix"] == digest[:HASH_PREFIX_LENGTH]
    assert data["actual_hash_prefix"] == other[:HASH_PREFIX_LENGTH]
    assert len(data["declared_hash_prefix"]) == HASH_PREFIX_LENGTH
    assert len(data["actual_hash_prefix"]) == HASH_PREFIX_LENGTH
    assert data["matched"] is False


def test_deterministic_results(tmp_path):
    bundle_dir, _ = _make_bundle(
        tmp_path,
        {"main.py": "print('hello')\n"},
        manifest_extra={"entry_points": ["main.py"]},
    )

    first = verify_bundle(bundle_dir)
    second = verify_bundle(bundle_dir)

    assert first == second


def test_supported_versions_override(tmp_path):
    bundle_dir, manifest = _make_bundle(
        tmp_path,
        {"main.py": "print('hi')\n"},
        manifest_extra={"entry_points": ["main.py"]},
    )
    manifest["manifest_version"] = "2.0"
    _write_manifest(bundle_dir, manifest)

    result = verify_bundle(bundle_dir, supported_versions=frozenset({"2.0"}))

    assert result.valid is True
    assert result.manifest_version == "2.0"


def test_verifier_rejects_non_frozenset():
    with pytest.raises(TypeError):
        BundleVerifier(supported_versions=set())


def test_no_raw_values_in_log_messages(tmp_path, caplog):
    caplog.set_level(logging.DEBUG, logger="openmed.skills.bundle_verify")

    valid_dir, _ = _make_bundle(
        tmp_path,
        {"main.py": _SENTINEL_CONTENT + "\n"},
        manifest_extra={"entry_points": ["main.py"]},
    )
    valid_result = verify_bundle(valid_dir)
    assert valid_result.valid is True

    bad_dir, _ = _make_bundle(
        tmp_path / "bad",
        {"main.py": _SENTINEL_CONTENT + "\n"},
        manifest_extra={"entry_points": ["main.py"]},
    )
    (bad_dir / "main.py").write_text("print('tampered')\n", encoding="utf-8")
    bad_result = verify_bundle(bad_dir)
    assert bad_result.valid is False

    for record in caplog.records:
        assert _HEX64_RE.search(record.getMessage()) is None
        assert _SENTINEL_CONTENT not in record.getMessage()


def test_multiple_files_all_match(tmp_path):
    bundle_dir, _ = _make_bundle(
        tmp_path,
        {
            "main.py": "print('main')\n",
            "utils.py": "def helper():\n    pass\n",
            "config.json": '{"key": "value"}\n',
        },
        manifest_extra={"entry_points": ["main.py"]},
    )

    result = verify_bundle(bundle_dir)

    assert result.valid is True
    assert len(result.files) == 3
    assert all(item.matched for item in result.files)


def test_multiple_files_one_mismatch(tmp_path):
    bundle_dir, _ = _make_bundle(
        tmp_path,
        {
            "main.py": "print('main')\n",
            "utils.py": "def helper():\n    pass\n",
            "config.json": '{"key": "value"}\n',
        },
        manifest_extra={"entry_points": ["main.py"]},
    )
    (bundle_dir / "utils.py").write_text(
        "def tampered():\n    pass\n", encoding="utf-8"
    )

    result = verify_bundle(bundle_dir)

    assert result.valid is False
    assert result.reason == REASON_HASH_MISMATCH
    assert len(result.files) == 3
    mismatched = [item for item in result.files if not item.matched]
    matched = [item for item in result.files if item.matched]
    assert len(mismatched) == 1
    assert mismatched[0].path == "utils.py"
    assert len(matched) == 2
    assert {item.path for item in matched} == {"main.py", "config.json"}


def test_empty_entry_points_valid(tmp_path):
    bundle_dir, _ = _make_bundle(
        tmp_path,
        {"main.py": "print('hi')\n"},
    )

    result = verify_bundle(bundle_dir)

    assert result.valid is True
    assert result.entry_points_checked == ()


def test_bundle_id_preserved_in_result(tmp_path):
    valid_dir, _ = _make_bundle(
        tmp_path,
        {"main.py": "print('hi')\n"},
        manifest_extra={"entry_points": ["main.py"]},
    )
    valid_result = verify_bundle(valid_dir)
    assert valid_result.bundle_id == _BUNDLE_ID

    bad_dir, manifest = _make_bundle(
        tmp_path / "bad",
        {"main.py": "print('hi')\n"},
        manifest_extra={"entry_points": ["main.py"]},
    )
    manifest["manifest_version"] = "2.0"
    _write_manifest(bad_dir, manifest)
    bad_result = verify_bundle(bad_dir)
    assert bad_result.bundle_id == _BUNDLE_ID


def test_valid_bundle_with_ed25519_signature(tmp_path):
    private_key = Ed25519PrivateKey.generate()
    public_bytes = private_key.public_key().public_bytes_raw()
    bundle_dir, manifest = _make_bundle(
        tmp_path,
        {"main.py": "print('hi')\n"},
        manifest_extra={"entry_points": ["main.py"]},
        signature_ed25519_private_key=private_key,
    )
    result = verify_bundle(bundle_dir, signature_public_key=public_bytes)
    assert result.valid is True
    assert result.reason == ""
    assert result.signature_verified is True
    assert result.bundle_id == _BUNDLE_ID
    assert result.entry_points_checked == ("main.py",)


def test_ed25519_signature_public_key_required(tmp_path):
    private_key = Ed25519PrivateKey.generate()
    bundle_dir, manifest = _make_bundle(
        tmp_path,
        {"main.py": "print('hi')\n"},
        manifest_extra={"entry_points": ["main.py"]},
        signature_ed25519_private_key=private_key,
    )
    result = verify_bundle(bundle_dir)
    assert result.valid is False
    assert result.reason == REASON_SIGNATURE_PUBLIC_KEY_REQUIRED
    assert result.signature_verified is False


def test_ed25519_signature_invalid(tmp_path):
    private_key = Ed25519PrivateKey.generate()
    other_key = Ed25519PrivateKey.generate()
    public_bytes = other_key.public_key().public_bytes_raw()
    bundle_dir, manifest = _make_bundle(
        tmp_path,
        {"main.py": "print('hi')\n"},
        manifest_extra={"entry_points": ["main.py"]},
        signature_ed25519_private_key=private_key,
    )
    result = verify_bundle(bundle_dir, signature_public_key=public_bytes)
    assert result.valid is False
    assert result.reason == REASON_SIGNATURE_INVALID
    assert result.signature_verified is False


def test_ed25519_signature_tampered_manifest(tmp_path):
    private_key = Ed25519PrivateKey.generate()
    public_bytes = private_key.public_key().public_bytes_raw()
    bundle_dir, manifest = _make_bundle(
        tmp_path,
        {"main.py": "print('hi')\n"},
        manifest_extra={"entry_points": ["main.py"]},
        signature_ed25519_private_key=private_key,
    )
    manifest["bundle_id"] = "com.evil.tampered"
    _write_manifest(bundle_dir, manifest)
    result = verify_bundle(bundle_dir, signature_public_key=public_bytes)
    assert result.valid is False
    assert result.reason == REASON_SIGNATURE_INVALID
    assert result.signature_verified is False


def test_ed25519_with_hmac_key_ignored(tmp_path):
    private_key = Ed25519PrivateKey.generate()
    public_bytes = private_key.public_key().public_bytes_raw()
    bundle_dir, manifest = _make_bundle(
        tmp_path,
        {"main.py": "print('hi')\n"},
        manifest_extra={"entry_points": ["main.py"]},
        signature_ed25519_private_key=private_key,
    )
    result = verify_bundle(
        bundle_dir,
        signature_key=b"wrong-hmac-key",
        signature_public_key=public_bytes,
    )
    assert result.valid is True
    assert result.signature_verified is True


def test_hmac_with_ed25519_key_ignored(tmp_path):
    key = b"super-secret-key"
    bundle_dir, manifest = _make_bundle(
        tmp_path,
        {"main.py": "print('hi')\n"},
        manifest_extra={"entry_points": ["main.py"]},
        signature_key=key,
    )
    private_key = Ed25519PrivateKey.generate()
    public_bytes = private_key.public_key().public_bytes_raw()
    result = verify_bundle(
        bundle_dir,
        signature_key=key,
        signature_public_key=public_bytes,
    )
    assert result.valid is True
    assert result.signature_verified is True


# --- Finding A: Path traversal containment tests ---


def test_path_traversal_in_files_rejected(tmp_path):
    """Manifest declaring a traversal path in files is rejected at construction."""
    manifest = {
        "manifest_version": "1.0",
        "bundle_id": _BUNDLE_ID,
        "entry_points": [],
        "files": {"../../etc/passwd": "a" * 64},
        "signature_scheme": "none",
        "signature": "",
    }
    _write_manifest(tmp_path, manifest)

    result = verify_bundle(tmp_path)

    assert result.valid is False
    assert result.reason == REASON_MANIFEST_MALFORMED


def test_path_traversal_in_entry_points_rejected(tmp_path):
    """Manifest declaring a traversal path in entry_points is rejected at construction."""
    manifest = {
        "manifest_version": "1.0",
        "bundle_id": _BUNDLE_ID,
        "entry_points": ["../../etc/shadow"],
        "files": {"main.py": "a" * 64},
        "signature_scheme": "none",
        "signature": "",
    }
    _write_manifest(tmp_path, manifest)
    (tmp_path / "main.py").write_text("print('hi')\n", encoding="utf-8")

    result = verify_bundle(tmp_path)

    assert result.valid is False
    assert result.reason == REASON_MANIFEST_MALFORMED


def test_absolute_path_in_files_rejected(tmp_path):
    """Manifest declaring an absolute path in files is rejected at construction."""
    manifest = {
        "manifest_version": "1.0",
        "bundle_id": _BUNDLE_ID,
        "entry_points": [],
        "files": {"/etc/passwd": "a" * 64},
        "signature_scheme": "none",
        "signature": "",
    }
    _write_manifest(tmp_path, manifest)

    result = verify_bundle(tmp_path)

    assert result.valid is False
    assert result.reason == REASON_MANIFEST_MALFORMED


def test_absolute_path_in_entry_points_rejected(tmp_path):
    """Manifest declaring an absolute path in entry_points is rejected at construction."""
    manifest = {
        "manifest_version": "1.0",
        "bundle_id": _BUNDLE_ID,
        "entry_points": ["/bin/sh"],
        "files": {"main.py": "a" * 64},
        "signature_scheme": "none",
        "signature": "",
    }
    _write_manifest(tmp_path, manifest)
    (tmp_path / "main.py").write_text("print('hi')\n", encoding="utf-8")

    result = verify_bundle(tmp_path)

    assert result.valid is False
    assert result.reason == REASON_MANIFEST_MALFORMED


def test_nested_subdirectory_path_accepted(tmp_path):
    """Paths into subdirectories within the bundle are valid (no false positives)."""
    bundle_dir, _ = _make_bundle(
        tmp_path,
        {"src/main.py": "print('hi')\n", "src/utils.py": "def f():\n    pass\n"},
        manifest_extra={"entry_points": ["src/main.py"]},
    )

    result = verify_bundle(bundle_dir)

    assert result.valid is True
    assert result.entry_points_checked == ("src/main.py",)


# --- Finding B: Whitespace-padded string rejection tests ---


def test_padded_entry_point_rejected(tmp_path):
    """Whitespace-padded entry point strings are rejected at construction."""
    manifest = {
        "manifest_version": "1.0",
        "bundle_id": _BUNDLE_ID,
        "entry_points": ["  main.py  "],
        "files": {"main.py": "a" * 64},
        "signature_scheme": "none",
        "signature": "",
    }
    _write_manifest(tmp_path, manifest)
    (tmp_path / "main.py").write_text("print('hi')\n", encoding="utf-8")

    result = verify_bundle(tmp_path)

    assert result.valid is False
    assert result.reason == REASON_MANIFEST_MALFORMED


def test_padded_files_key_rejected(tmp_path):
    """Whitespace-padded file path keys are rejected at construction."""
    manifest = {
        "manifest_version": "1.0",
        "bundle_id": _BUNDLE_ID,
        "entry_points": [],
        "files": {"  main.py  ": "a" * 64},
        "signature_scheme": "none",
        "signature": "",
    }
    _write_manifest(tmp_path, manifest)
    (tmp_path / "main.py").write_text("print('hi')\n", encoding="utf-8")

    result = verify_bundle(tmp_path)

    assert result.valid is False
    assert result.reason == REASON_MANIFEST_MALFORMED


def test_unpadded_matching_strings_still_pass(tmp_path):
    """Unpadded entry_points and files keys continue to work (no regression)."""
    bundle_dir, _ = _make_bundle(
        tmp_path,
        {"main.py": "print('hello')\n"},
        manifest_extra={"entry_points": ["main.py"]},
    )

    result = verify_bundle(bundle_dir)

    assert result.valid is True
    assert result.entry_points_checked == ("main.py",)
    assert result.files[0].path == "main.py"
