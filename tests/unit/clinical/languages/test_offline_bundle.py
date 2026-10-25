"""Tests for signed offline clinical language bundles.

Every bundle in this module is built from tiny synthetic files. Nothing here
downloads a model, reads restricted data, or opens a network connection.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import shutil
import socket
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest

from openmed.clinical.languages import offline_bundle
from openmed.clinical.languages.offline_bundle import (
    CHECK_FAILED,
    CHECK_NAMES,
    CHECK_NOT_RUN,
    CHECK_PASSED,
    COMPONENT_EVIDENCE,
    COMPONENT_LEXICON,
    COMPONENT_MODEL,
    COMPONENT_POLICY,
    COMPONENT_SELF_TEST,
    DESCRIPTOR_FILENAME,
    MANIFEST_FILENAME,
    REASON_BUNDLE_ID_MISMATCH,
    REASON_BUNDLE_NOT_VERIFIED,
    REASON_COMPONENT_FILES_MISMATCH,
    REASON_COMPONENT_MISSING,
    REASON_CREDENTIAL_LIKE_FILE,
    REASON_DESCRIPTOR_MALFORMED,
    REASON_ENTRY_POINT_MISMATCH,
    REASON_HOST_FACTS_UNAVAILABLE,
    REASON_INSTALL_COPY_MISMATCH,
    REASON_INSTALL_TARGET_EXISTS,
    REASON_INSUFFICIENT_RAM,
    REASON_INSUFFICIENT_STORAGE,
    REASON_LICENSE_NOT_PERMISSIVE,
    REASON_MANIFEST_FIELDS_UNSUPPORTED,
    REASON_MANIFEST_MISSING,
    REASON_NETWORK_ATTEMPTED,
    REASON_OUTPUT_NOT_EMPTY,
    REASON_RESTRICTED_SOURCE,
    REASON_SELF_TEST_FAILED,
    REASON_SELF_TEST_MALFORMED,
    REASON_SELF_TEST_NOT_SYNTHETIC,
    REASON_SELF_TEST_RUNNER_REQUIRED,
    REASON_SIGNING_KEY_INVALID,
    REASON_SIGNING_KEY_REQUIRED,
    REASON_SOURCE_FILE_MISSING,
    REASON_STORAGE_REQUIREMENT_UNDERSTATED,
    REASON_SYMLINK_PRESENT,
    REASON_UNSIGNED_BUNDLE,
    SELF_TEST_ERROR_INVALID_OUTPUT,
    SELF_TEST_ERROR_RUNNER,
    HostResources,
    OfflineBundleComponent,
    OfflineBundleDescriptor,
    OfflineBundleError,
    OfflineBundleVerification,
    PackagedOfflineBundle,
    ResourceRequirements,
    install_offline_bundle,
    package_offline_bundle,
    run_offline_self_tests,
    verify_offline_bundle,
)
from openmed.skills.bundle_verify import (
    REASON_FILE_UNDECLARED,
    REASON_HASH_MISMATCH,
    REASON_SIGNATURE_DEPENDENCY_MISSING,
    REASON_SIGNATURE_INVALID,
    REASON_SIGNATURE_PUBLIC_KEY_REQUIRED,
    REASON_SIGNATURE_REQUIRED,
    SkillBundleManifest,
    canonical_manifest_bytes,
)

_HMAC_KEY = b"synthetic-offline-bundle-test-key"
_ED25519_SEED = bytes(range(32))
_BUNDLE_ID = "org.openmed.lang.sw.synthetic-deid"
_BUNDLE_VERSION = "1.0.0"
_NAME = "Zawadi Kifaru"
_IDENTIFIER = "SW-0042-SYN"
_CASE_TEXTS = (
    f"Mgonjwa {_NAME} alipimwa leo.",
    f"Kitambulisho {_IDENTIFIER} kimehakikiwa.",
    "Hakuna kitambulisho hapa.",
)
_HOST = HostResources(
    available_ram_bytes=8 * 1024**3,
    available_storage_bytes=64 * 1024**3,
)
_OFFLINE_ENV_VARS = (
    "OPENMED_OFFLINE",
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "HF_DATASETS_OFFLINE",
)


@pytest.fixture(autouse=True)
def _restore_offline_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Restore the offline flags that the socket guard sets for the process."""

    for name in _OFFLINE_ENV_VARS:
        monkeypatch.setenv(name, "0")


@pytest.fixture
def socket_attempts(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Refuse every outbound socket operation and record each attempt."""

    attempts: list[str] = []

    def refuse(*_args: Any, **_kwargs: Any) -> Any:
        attempts.append("network")
        raise OSError("network access is disabled in offline bundle tests")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    return attempts


def _case(case_id: str, text: str, terms: dict[str, str]) -> dict[str, Any]:
    expected = [
        {"start": text.index(term), "end": text.index(term) + len(term), "label": label}
        for term, label in terms.items()
    ]
    return {"case_id": case_id, "synthetic": True, "text": text, "expected": expected}


def _self_test_lines(cases: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(case, sort_keys=True) + "\n" for case in cases)


def _write_sources(root: Path) -> dict[str, Path]:
    """Write tiny synthetic component sources and return them by bundle path."""

    contents: dict[str, str | bytes] = {
        "model/config.json": json.dumps(
            {
                "architecture": "synthetic-lexicon-tagger",
                "labels": ["ID_NUMBER", "NAME"],
            }
        ),
        "model/weights.bin": b"\x00\x01synthetic-weights\x02",
        "lexicons/names.tsv": f"{_NAME}\tNAME\n{_IDENTIFIER}\tID_NUMBER\n",
        "policy/policy.json": json.dumps(
            {"profile": "safe_harbor", "surrogate_locale": "sw_KE"}
        ),
        "selftests/cases.jsonl": _self_test_lines(
            [
                _case("sw-001", _CASE_TEXTS[0], {_NAME: "NAME"}),
                _case("sw-002", _CASE_TEXTS[1], {_IDENTIFIER: "ID_NUMBER"}),
                _case("sw-003", _CASE_TEXTS[2], {}),
            ]
        ),
        "evidence/eval.json": json.dumps(
            {"direct_identifier_recall": 1.0, "synthetic": True}
        ),
    }
    sources: dict[str, Path] = {}
    for rel_path, content in contents.items():
        path = root / "sources" / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content, encoding="utf-8")
        sources[rel_path] = path
    return sources


def _component(
    component_id: str,
    kind: str,
    paths: tuple[str, ...],
    **overrides: Any,
) -> OfflineBundleComponent:
    values: dict[str, Any] = {
        "component_id": component_id,
        "kind": kind,
        "files": paths,
        "license": "Apache-2.0",
        "source": f"synthetic:openmed/{component_id}",
        "data_classification": "public",
    }
    values.update(overrides)
    return OfflineBundleComponent(**values)


def _components(**replacements: OfflineBundleComponent) -> list[OfflineBundleComponent]:
    """Return the five default components, replacing any by role."""

    defaults = {
        "model": _component(
            "model", COMPONENT_MODEL, ("model/config.json", "model/weights.bin")
        ),
        "lexicon": _component(
            "lexicon-names",
            COMPONENT_LEXICON,
            ("lexicons/names.tsv",),
            license="CC0-1.0",
        ),
        "policy": _component("policy", COMPONENT_POLICY, ("policy/policy.json",)),
        "self_test": _component(
            "self-tests",
            COMPONENT_SELF_TEST,
            ("selftests/cases.jsonl",),
            license="CC0-1.0",
            data_classification="synthetic",
        ),
        "evidence": _component(
            "evidence",
            COMPONENT_EVIDENCE,
            ("evidence/eval.json",),
            license="CC-BY-4.0",
            data_classification="synthetic",
        ),
    }
    defaults.update(replacements)
    return list(defaults.values())


def _descriptor(**overrides: Any) -> OfflineBundleDescriptor:
    values: dict[str, Any] = {
        "bundle_id": _BUNDLE_ID,
        "bundle_version": _BUNDLE_VERSION,
        "language": "sw",
        "scripts": ["Latin"],
        "capabilities": ["pii_detection", "deidentification"],
        "resources": ResourceRequirements(
            min_ram_bytes=512 * 1024**2,
            min_storage_bytes=1024**2,
        ),
        "components": _components(),
    }
    values.update(overrides)
    return OfflineBundleDescriptor(**values)


def _package_into(
    output: Path,
    sources: dict[str, Path],
    descriptor: OfflineBundleDescriptor | None = None,
    **keys: Any,
) -> PackagedOfflineBundle:
    descriptor = descriptor if descriptor is not None else _descriptor()
    keys.setdefault("signature_key", _HMAC_KEY)
    selected = {path: sources[path] for path in descriptor.payload_files}
    return package_offline_bundle(descriptor, selected, output, **keys)


def _package(tmp_path: Path, name: str = "bundle", **keys: Any) -> Path:
    return _package_into(tmp_path / name, _write_sources(tmp_path), **keys).path


def _lexicon_runner(bundle_root: Path, text: str) -> list[tuple[int, int, str]]:
    """Synthetic offline 'model': exact matches from the bundled lexicon."""

    lexicon = bundle_root / "lexicons" / "names.tsv"
    spans: list[tuple[int, int, str]] = []
    for line in lexicon.read_text(encoding="utf-8").splitlines():
        term, label = line.split("\t")
        start = text.find(term)
        while start != -1:
            spans.append((start, start + len(term), label))
            start = text.find(term, start + 1)
    return spans


def _resign(
    bundle: Path,
    *,
    key: bytes = _HMAC_KEY,
    manifest_updates: dict[str, Any] | None = None,
) -> None:
    """Recompute digests for every file and re-sign the manifest."""

    manifest = json.loads((bundle / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    manifest["files"] = {
        path.relative_to(bundle).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(bundle.rglob("*"))
        if path.is_file() and path.name != MANIFEST_FILENAME
    }
    manifest.update(manifest_updates or {})
    unsigned = SkillBundleManifest.from_mapping(manifest)
    manifest["signature"] = hmac.new(
        key, canonical_manifest_bytes(unsigned), hashlib.sha256
    ).hexdigest()
    (bundle / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _rewrite_descriptor(bundle: Path, mutate: Callable[[dict[str, Any]], None]) -> None:
    path = bundle / DESCRIPTOR_FILENAME
    descriptor = json.loads(path.read_text(encoding="utf-8"))
    mutate(descriptor)
    path.write_text(json.dumps(descriptor, indent=2, sort_keys=True), encoding="utf-8")
    _resign(bundle)


def _component_named(descriptor: dict[str, Any], component_id: str) -> dict[str, Any]:
    return next(
        item
        for item in descriptor["components"]
        if item["component_id"] == component_id
    )


def _statuses(result: OfflineBundleVerification) -> dict[str, str]:
    return {check.name: check.status for check in result.checks}


def test_packaged_bundle_verifies_without_network(
    tmp_path: Path, socket_attempts: list[str]
) -> None:
    sources = _write_sources(tmp_path)
    packaged = _package_into(tmp_path / "bundle", sources)

    result = verify_offline_bundle(packaged.path, host=_HOST, signature_key=_HMAC_KEY)

    assert socket_attempts == []
    assert result.valid is True
    assert result.reason == ""
    assert result.failures == ()
    assert [check.name for check in result.checks] == list(CHECK_NAMES)
    assert set(_statuses(result).values()) == {CHECK_PASSED}
    assert result.bundle_id == _BUNDLE_ID
    assert result.bundle_version == _BUNDLE_VERSION
    assert result.language == "sw"
    assert result.signature_scheme == "hmac-sha256"
    assert result.manifest_digest == packaged.manifest_digest
    assert result.capabilities == ("deidentification", "pii_detection")
    assert result.resources == ResourceRequirements(512 * 1024**2, 1024**2)
    assert result.self_test_case_count == 3
    assert result.payload_bytes == packaged.payload_bytes
    assert set(result.file_digests) == {
        DESCRIPTOR_FILENAME,
        "evidence/eval.json",
        "lexicons/names.tsv",
        "model/config.json",
        "model/weights.bin",
        "policy/policy.json",
        "selftests/cases.jsonl",
    }
    lexicon_digest = hashlib.sha256(sources["lexicons/names.tsv"].read_bytes())
    assert result.file_digests["lexicons/names.tsv"] == (
        f"sha256:{lexicon_digest.hexdigest()}"
    )
    assert {
        (item.component_id, item.license, item.permissive) for item in result.licenses
    } == {
        ("evidence", "CC-BY-4.0", True),
        ("lexicon-names", "CC0-1.0", True),
        ("model", "Apache-2.0", True),
        ("policy", "Apache-2.0", True),
        ("self-tests", "CC0-1.0", True),
    }
    assert result.descriptor == _descriptor()
    assert result.descriptor.component_files(COMPONENT_MODEL) == (
        "model/config.json",
        "model/weights.bin",
    )


def test_packaging_and_verification_are_deterministic(tmp_path: Path) -> None:
    sources = _write_sources(tmp_path)
    first = _package_into(tmp_path / "first", sources)
    second = _package_into(tmp_path / "second", sources)

    for name in (MANIFEST_FILENAME, DESCRIPTOR_FILENAME):
        assert (first.path / name).read_bytes() == (second.path / name).read_bytes()
    assert first.to_dict() == second.to_dict()
    assert "path" not in first.to_dict()

    reports = [
        verify_offline_bundle(path, host=_HOST, signature_key=_HMAC_KEY).to_dict()
        for path in (first.path, second.path, first.path)
    ]
    assert reports[0] == reports[1] == reports[2]


def test_descriptor_round_trips_and_rejects_unknown_fields(tmp_path: Path) -> None:
    bundle = _package(tmp_path)
    raw = json.loads((bundle / DESCRIPTOR_FILENAME).read_text(encoding="utf-8"))

    descriptor = OfflineBundleDescriptor.from_mapping(raw)

    assert descriptor.to_dict() == raw
    assert descriptor.to_json_bytes() == (bundle / DESCRIPTOR_FILENAME).read_bytes()
    with pytest.raises(OfflineBundleError) as raised:
        OfflineBundleDescriptor.from_mapping({**raw, "activation": "auto"})
    assert raised.value.reason == REASON_DESCRIPTOR_MALFORMED


def test_ed25519_signed_bundle_verifies_with_public_key(tmp_path: Path) -> None:
    ed25519 = pytest.importorskip("cryptography.hazmat.primitives.asymmetric.ed25519")
    serialization = pytest.importorskip("cryptography.hazmat.primitives.serialization")

    def public_bytes(seed: bytes) -> bytes:
        return (
            ed25519.Ed25519PrivateKey.from_private_bytes(seed)
            .public_key()
            .public_bytes(
                serialization.Encoding.Raw,
                serialization.PublicFormat.Raw,
            )
        )

    first = _package(
        tmp_path, "first", signature_key=None, ed25519_private_key=_ED25519_SEED
    )
    second = _package(
        tmp_path, "second", signature_key=None, ed25519_private_key=_ED25519_SEED
    )

    valid = verify_offline_bundle(
        first, host=_HOST, signature_public_key=public_bytes(_ED25519_SEED)
    )
    wrong = verify_offline_bundle(
        first, host=_HOST, signature_public_key=public_bytes(bytes(32))
    )
    missing = verify_offline_bundle(first, host=_HOST)

    assert valid.valid is True
    assert valid.signature_scheme == "ed25519"
    assert (first / MANIFEST_FILENAME).read_bytes() == (
        second / MANIFEST_FILENAME
    ).read_bytes()
    assert wrong.reason == REASON_SIGNATURE_INVALID
    assert missing.reason == REASON_SIGNATURE_PUBLIC_KEY_REQUIRED


def test_ed25519_signing_reports_missing_optional_dependency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(
        sys.modules, "cryptography.hazmat.primitives.asymmetric.ed25519", None
    )

    with pytest.raises(OfflineBundleError) as raised:
        _package(tmp_path, signature_key=None, ed25519_private_key=_ED25519_SEED)

    assert raised.value.reason == REASON_SIGNATURE_DEPENDENCY_MISSING
    assert not (tmp_path / "bundle").exists()


@pytest.mark.parametrize(
    ("keys", "reason"),
    [
        ({"signature_key": None}, REASON_SIGNING_KEY_REQUIRED),
        ({"ed25519_private_key": _ED25519_SEED}, REASON_SIGNING_KEY_REQUIRED),
        ({"signature_key": b""}, REASON_SIGNING_KEY_INVALID),
        (
            {"signature_key": None, "ed25519_private_key": b"short"},
            REASON_SIGNING_KEY_INVALID,
        ),
    ],
)
def test_packaging_requires_exactly_one_valid_signing_key(
    tmp_path: Path, keys: dict[str, Any], reason: str
) -> None:
    if keys.get("ed25519_private_key") == b"short":
        pytest.importorskip("cryptography.hazmat.primitives.asymmetric.ed25519")

    with pytest.raises(OfflineBundleError) as raised:
        _package(tmp_path, **keys)

    assert raised.value.reason == reason
    assert not (tmp_path / "bundle").exists()


def test_tampered_payload_fails_checksum_before_policy_checks(tmp_path: Path) -> None:
    bundle = _package(tmp_path)
    (bundle / "lexicons" / "names.tsv").write_text("tampered\tNAME\n", encoding="utf-8")

    result = verify_offline_bundle(bundle, host=_HOST, signature_key=_HMAC_KEY)

    assert result.valid is False
    assert result.reason == REASON_HASH_MISMATCH
    assert _statuses(result)["integrity"] == CHECK_FAILED
    assert {
        status for name, status in _statuses(result).items() if name != "integrity"
    } == {CHECK_NOT_RUN}
    assert result.descriptor is None
    assert result.capabilities == ()


def test_wrong_or_missing_hmac_key_fails_signature(tmp_path: Path) -> None:
    bundle = _package(tmp_path)

    wrong = verify_offline_bundle(bundle, host=_HOST, signature_key=b"other-key")
    missing = verify_offline_bundle(bundle, host=_HOST)

    assert wrong.reason == REASON_SIGNATURE_INVALID
    assert missing.reason == REASON_SIGNATURE_REQUIRED


def test_unsigned_bundle_is_rejected(tmp_path: Path) -> None:
    bundle = _package(tmp_path)
    manifest = json.loads((bundle / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    manifest.update(signature_scheme="none", signature="")
    (bundle / MANIFEST_FILENAME).write_text(json.dumps(manifest), encoding="utf-8")

    result = verify_offline_bundle(bundle, host=_HOST)

    assert result.reason == REASON_UNSIGNED_BUNDLE


def test_unsigned_manifest_fields_are_rejected(tmp_path: Path) -> None:
    bundle = _package(tmp_path)
    manifest = json.loads((bundle / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    manifest["capabilities"] = ["clinical_decision"]
    (bundle / MANIFEST_FILENAME).write_text(json.dumps(manifest), encoding="utf-8")

    result = verify_offline_bundle(bundle, host=_HOST, signature_key=_HMAC_KEY)

    assert result.reason == REASON_MANIFEST_FIELDS_UNSUPPORTED


def test_missing_manifest_and_entry_point_mismatch(tmp_path: Path) -> None:
    missing = verify_offline_bundle(tmp_path / "absent", host=_HOST)
    bundle = _package(tmp_path)
    _resign(bundle, manifest_updates={"entry_points": []})

    mismatch = verify_offline_bundle(bundle, host=_HOST, signature_key=_HMAC_KEY)

    assert missing.reason == REASON_MANIFEST_MISSING
    assert mismatch.reason == REASON_ENTRY_POINT_MISMATCH


def test_undeclared_file_is_rejected(tmp_path: Path) -> None:
    bundle = _package(tmp_path)
    (bundle / "lexicons" / "extra.tsv").write_text("extra\tNAME\n", encoding="utf-8")

    result = verify_offline_bundle(bundle, host=_HOST, signature_key=_HMAC_KEY)

    assert result.reason == REASON_FILE_UNDECLARED


@pytest.mark.skipif(os.name == "nt", reason="symlinks need privileges on Windows")
def test_symlinks_are_rejected(tmp_path: Path) -> None:
    bundle = _package(tmp_path)
    (bundle / "model" / "alias.bin").symlink_to(bundle / "model" / "weights.bin")
    _resign(bundle)

    result = verify_offline_bundle(bundle, host=_HOST, signature_key=_HMAC_KEY)

    assert result.reason == REASON_SYMLINK_PRESENT


def test_signed_declared_file_outside_components_is_rejected(tmp_path: Path) -> None:
    bundle = _package(tmp_path)
    (bundle / "lexicons" / "extra.tsv").write_text("extra\tNAME\n", encoding="utf-8")
    _resign(bundle)

    result = verify_offline_bundle(bundle, host=_HOST, signature_key=_HMAC_KEY)

    assert result.reason == REASON_COMPONENT_FILES_MISMATCH
    assert _statuses(result)["self_tests"] == CHECK_NOT_RUN


def test_signed_descriptor_with_unknown_field_is_rejected(tmp_path: Path) -> None:
    bundle = _package(tmp_path)
    _rewrite_descriptor(bundle, lambda value: value.update(auto_activate=True))

    result = verify_offline_bundle(bundle, host=_HOST, signature_key=_HMAC_KEY)

    assert result.reason == REASON_DESCRIPTOR_MALFORMED
    assert result.descriptor is None


def test_signed_descriptor_with_duplicate_keys_is_rejected(tmp_path: Path) -> None:
    bundle = _package(tmp_path)
    path = bundle / DESCRIPTOR_FILENAME
    text = path.read_text(encoding="utf-8")
    duplicated = text.replace('"language": "sw"', '"language": "sw", "language": "en"')
    assert duplicated != text
    path.write_text(duplicated, encoding="utf-8")
    _resign(bundle)

    result = verify_offline_bundle(bundle, host=_HOST, signature_key=_HMAC_KEY)

    assert result.reason == REASON_DESCRIPTOR_MALFORMED


def test_descriptor_bundle_id_must_match_manifest(tmp_path: Path) -> None:
    bundle = _package(tmp_path)
    _rewrite_descriptor(bundle, lambda value: value.update(bundle_id="other.bundle"))

    result = verify_offline_bundle(bundle, host=_HOST, signature_key=_HMAC_KEY)

    assert result.reason == REASON_BUNDLE_ID_MISMATCH


def test_policy_failures_are_all_reported_for_a_signed_bundle(tmp_path: Path) -> None:
    bundle = _package(tmp_path)

    def mutate(value: dict[str, Any]) -> None:
        _component_named(value, "model")["license"] = "GPL-3.0-only"
        _component_named(value, "lexicon-names")["source"] = "physionet:mimic-iii"
        value["resources"]["min_storage_bytes"] = 16

    _rewrite_descriptor(bundle, mutate)
    host = HostResources(available_ram_bytes=1024, available_storage_bytes=0)

    result = verify_offline_bundle(bundle, host=host, signature_key=_HMAC_KEY)

    assert result.valid is False
    assert result.failures == (
        REASON_LICENSE_NOT_PERMISSIVE,
        REASON_RESTRICTED_SOURCE,
        REASON_STORAGE_REQUIREMENT_UNDERSTATED,
        REASON_INSUFFICIENT_RAM,
    )
    assert result.reason == REASON_LICENSE_NOT_PERMISSIVE
    model_license = next(
        item for item in result.licenses if item.component_id == "model"
    )
    assert model_license.permissive is False
    assert _statuses(result)["components"] == CHECK_PASSED
    assert _statuses(result)["self_tests"] == CHECK_PASSED


@pytest.mark.parametrize(
    ("ram", "storage", "failures"),
    [
        (1024, 64 * 1024**3, (REASON_INSUFFICIENT_RAM,)),
        (8 * 1024**3, 1024, (REASON_INSUFFICIENT_STORAGE,)),
        (1024, 1024, (REASON_INSUFFICIENT_STORAGE, REASON_INSUFFICIENT_RAM)),
        (512 * 1024**2, 1024**2, ()),
    ],
)
def test_host_ram_and_storage_are_checked_against_injected_facts(
    tmp_path: Path, ram: int, storage: int, failures: tuple[str, ...]
) -> None:
    bundle = _package(tmp_path)
    host = HostResources(available_ram_bytes=ram, available_storage_bytes=storage)

    result = verify_offline_bundle(bundle, host=host, signature_key=_HMAC_KEY)

    assert result.failures == failures
    assert result.valid is (not failures)
    assert result.to_dict()["host"] == host.to_dict()


@pytest.mark.parametrize(
    "license_expression",
    ["GPL-3.0-only", "CC-BY-NC-4.0", "LicenseRef-Proprietary", "MIT AND GPL-2.0"],
)
def test_packaging_rejects_non_permissive_licenses(
    tmp_path: Path, license_expression: str
) -> None:
    sources = _write_sources(tmp_path)
    model = _component(
        "model",
        COMPONENT_MODEL,
        ("model/config.json", "model/weights.bin"),
        license=license_expression,
    )

    with pytest.raises(OfflineBundleError) as raised:
        _package_into(
            tmp_path / "bundle",
            sources,
            _descriptor(components=_components(model=model)),
        )

    assert raised.value.reason == REASON_LICENSE_NOT_PERMISSIVE
    assert license_expression not in str(raised.value)
    assert not (tmp_path / "bundle").exists()


@pytest.mark.parametrize(
    ("overrides", "bundle_path"),
    [
        ({"source": "physionet:mimic-iv/notes"}, "lexicons/names.tsv"),
        ({"source": "nlm:umls-2026aa"}, "lexicons/names.tsv"),
        ({}, "lexicons/snomed_terms.tsv"),
    ],
)
def test_packaging_rejects_restricted_sources(
    tmp_path: Path, overrides: dict[str, Any], bundle_path: str
) -> None:
    sources = _write_sources(tmp_path)
    sources[bundle_path] = sources["lexicons/names.tsv"]
    lexicon = _component(
        "lexicon-names", COMPONENT_LEXICON, (bundle_path,), **overrides
    )

    with pytest.raises(OfflineBundleError) as raised:
        _package_into(
            tmp_path / "bundle",
            sources,
            _descriptor(components=_components(lexicon=lexicon)),
        )

    assert raised.value.reason == REASON_RESTRICTED_SOURCE
    assert "mimic" not in str(raised.value)
    assert "umls" not in str(raised.value)


@pytest.mark.parametrize(
    "extra_path", ["model/server.pem", "model/signing.key", "model/credentials.json"]
)
def test_packaging_rejects_credential_like_files(
    tmp_path: Path, extra_path: str
) -> None:
    sources = _write_sources(tmp_path)
    sources[extra_path] = sources["model/config.json"]
    model = _component(
        "model",
        COMPONENT_MODEL,
        ("model/config.json", "model/weights.bin", extra_path),
    )

    with pytest.raises(OfflineBundleError) as raised:
        _package_into(
            tmp_path / "bundle",
            sources,
            _descriptor(components=_components(model=model)),
        )

    assert raised.value.reason == REASON_CREDENTIAL_LIKE_FILE


@pytest.mark.parametrize(
    "bundle_path",
    [
        "../escape.json",
        "/abs/path.json",
        ".env",
        "model/.hidden",
        "model\\windows.json",
        DESCRIPTOR_FILENAME,
        MANIFEST_FILENAME,
        "Manifest.JSON",
    ],
)
def test_descriptor_rejects_unsafe_bundle_paths(bundle_path: str) -> None:
    with pytest.raises(OfflineBundleError) as raised:
        _component("policy", COMPONENT_POLICY, (bundle_path,))

    assert raised.value.reason == REASON_DESCRIPTOR_MALFORMED


def test_descriptor_rejects_case_colliding_paths() -> None:
    policy = _component("policy", COMPONENT_POLICY, ("model/CONFIG.json",))

    with pytest.raises(OfflineBundleError) as raised:
        _descriptor(components=_components(policy=policy))

    assert raised.value.reason == REASON_DESCRIPTOR_MALFORMED


def test_packaging_requires_every_component_kind(tmp_path: Path) -> None:
    sources = _write_sources(tmp_path)

    with pytest.raises(OfflineBundleError) as raised:
        _package_into(
            tmp_path / "bundle",
            sources,
            _descriptor(components=_components()[:-1]),
        )

    assert raised.value.reason == REASON_COMPONENT_MISSING


def test_self_tests_must_be_synthetic(tmp_path: Path) -> None:
    with pytest.raises(OfflineBundleError) as classification:
        _component(
            "self-tests",
            COMPONENT_SELF_TEST,
            ("selftests/cases.jsonl",),
            data_classification="public",
        )

    sources = _write_sources(tmp_path)
    case = _case("sw-001", _CASE_TEXTS[0], {_NAME: "NAME"})
    case["synthetic"] = False
    sources["selftests/cases.jsonl"].write_text(
        _self_test_lines([case]), encoding="utf-8"
    )
    with pytest.raises(OfflineBundleError) as flagged:
        _package_into(tmp_path / "flagged", sources)

    assert classification.value.reason == REASON_SELF_TEST_NOT_SYNTHETIC
    assert flagged.value.reason == REASON_SELF_TEST_NOT_SYNTHETIC


@pytest.mark.parametrize(
    "lines",
    [
        "not json\n",
        "",
        json.dumps(
            {
                "case_id": "sw-001",
                "synthetic": True,
                "text": "short",
                "expected": [{"start": 2, "end": 99, "label": "NAME"}],
            }
        ),
        json.dumps(
            {"case_id": "sw-001", "synthetic": True, "text": "x", "expected": []}
        )
        + "\n"
        + json.dumps(
            {"case_id": "sw-001", "synthetic": True, "text": "y", "expected": []}
        ),
        json.dumps({"case_id": "sw-001", "synthetic": True, "text": "x"}),
    ],
)
def test_malformed_self_tests_are_rejected(tmp_path: Path, lines: str) -> None:
    sources = _write_sources(tmp_path)
    sources["selftests/cases.jsonl"].write_text(lines, encoding="utf-8")

    with pytest.raises(OfflineBundleError) as raised:
        _package_into(tmp_path / "bundle", sources)

    assert raised.value.reason == REASON_SELF_TEST_MALFORMED
    assert "short" not in str(raised.value)


def test_signed_bundle_with_malformed_self_tests_fails_verification(
    tmp_path: Path,
) -> None:
    bundle = _package(tmp_path)
    (bundle / "selftests" / "cases.jsonl").write_text(
        json.dumps(
            {"case_id": "sw-9", "synthetic": False, "text": "x", "expected": []}
        ),
        encoding="utf-8",
    )
    _resign(bundle)

    result = verify_offline_bundle(bundle, host=_HOST, signature_key=_HMAC_KEY)

    assert result.failures == (REASON_SELF_TEST_NOT_SYNTHETIC,)
    assert result.self_test_case_count == 0


def test_packaging_sources_must_match_declared_files(tmp_path: Path) -> None:
    sources = _write_sources(tmp_path)
    descriptor = _descriptor()
    selected = {path: sources[path] for path in descriptor.payload_files}
    missing_key = {k: v for k, v in selected.items() if k != "evidence/eval.json"}
    missing_file = {**selected, "evidence/eval.json": tmp_path / "absent.json"}

    with pytest.raises(OfflineBundleError) as uncovered:
        package_offline_bundle(
            descriptor, missing_key, tmp_path / "a", signature_key=_HMAC_KEY
        )
    with pytest.raises(OfflineBundleError) as absent:
        package_offline_bundle(
            descriptor, missing_file, tmp_path / "b", signature_key=_HMAC_KEY
        )
    with pytest.raises(TypeError):
        package_offline_bundle(
            descriptor.to_dict(),  # type: ignore[arg-type]
            selected,
            tmp_path / "c",
            signature_key=_HMAC_KEY,
        )

    assert uncovered.value.reason == REASON_COMPONENT_FILES_MISMATCH
    assert absent.value.reason == REASON_SOURCE_FILE_MISSING
    assert not any((tmp_path / name).exists() for name in ("a", "b", "c"))


def test_packaging_storage_requirement_must_cover_payload(tmp_path: Path) -> None:
    sources = _write_sources(tmp_path)

    with pytest.raises(OfflineBundleError) as raised:
        _package_into(
            tmp_path / "bundle",
            sources,
            _descriptor(
                resources=ResourceRequirements(min_ram_bytes=1, min_storage_bytes=8)
            ),
        )

    assert raised.value.reason == REASON_STORAGE_REQUIREMENT_UNDERSTATED
    assert not (tmp_path / "bundle").exists()
    assert not [path for path in tmp_path.iterdir() if path.name.startswith(".")]


def test_packaging_refuses_non_empty_output(tmp_path: Path) -> None:
    sources = _write_sources(tmp_path)
    output = tmp_path / "bundle"
    output.mkdir()
    (output / "keep.txt").write_text("existing", encoding="utf-8")

    with pytest.raises(OfflineBundleError) as raised:
        _package_into(output, sources)

    assert raised.value.reason == REASON_OUTPUT_NOT_EMPTY
    assert (output / "keep.txt").read_text(encoding="utf-8") == "existing"


def test_packaging_into_existing_empty_directory(tmp_path: Path) -> None:
    sources = _write_sources(tmp_path)
    output = tmp_path / "bundle"
    output.mkdir()

    packaged = _package_into(output, sources)

    assert packaged.path == output
    assert verify_offline_bundle(output, host=_HOST, signature_key=_HMAC_KEY).valid


def test_install_runs_verification_and_self_tests_with_sockets_blocked(
    tmp_path: Path, socket_attempts: list[str]
) -> None:
    bundle = _package(tmp_path)
    install_root = tmp_path / "installed"
    roots: list[Path] = []

    def runner(bundle_root: Path, text: str) -> list[tuple[int, int, str]]:
        roots.append(bundle_root)
        return _lexicon_runner(bundle_root, text)

    installation = install_offline_bundle(
        bundle,
        install_root,
        host=_HOST,
        signature_key=_HMAC_KEY,
        self_test_runner=runner,
    )

    assert socket_attempts == []
    assert installation.installed is True
    assert installation.reason == ""
    assert installation.install_path == install_root / _BUNDLE_ID / _BUNDLE_VERSION
    assert installation.self_tests is not None
    assert installation.self_tests.passed is True
    assert installation.self_tests.network_blocked is True
    assert installation.self_tests.case_count == 3
    assert installation.self_tests.passed_count == 3
    assert roots and all(root != installation.install_path for root in roots)
    assert sorted(path.name for path in install_root.iterdir()) == [_BUNDLE_ID]
    reverified = verify_offline_bundle(
        installation.install_path, host=_HOST, signature_key=_HMAC_KEY
    )
    assert reverified.valid is True
    assert reverified.manifest_digest == installation.verification.manifest_digest
    report = installation.to_dict()
    assert report["install_relpath"] == f"{_BUNDLE_ID}/{_BUNDLE_VERSION}"
    assert str(tmp_path) not in json.dumps(report)


def test_install_fails_closed_when_self_tests_reach_for_the_network(
    tmp_path: Path,
) -> None:
    bundle = _package(tmp_path)
    install_root = tmp_path / "installed"
    original_connect = socket.socket.connect

    def network_runner(bundle_root: Path, text: str) -> list[tuple[int, int, str]]:
        socket.create_connection(("127.0.0.1", 9), timeout=0.01)
        return _lexicon_runner(bundle_root, text)

    installation = install_offline_bundle(
        bundle,
        install_root,
        host=_HOST,
        signature_key=_HMAC_KEY,
        self_test_runner=network_runner,
    )

    assert installation.installed is False
    assert installation.reason == REASON_NETWORK_ATTEMPTED
    assert installation.self_tests is not None
    assert installation.self_tests.passed is False
    assert installation.self_tests.case_count == 1
    assert installation.self_tests.cases[0].error == REASON_NETWORK_ATTEMPTED
    assert list(install_root.iterdir()) == []
    assert socket.socket.connect is original_connect


def test_install_detects_network_errors_wrapped_by_runtimes(tmp_path: Path) -> None:
    bundle = _package(tmp_path)

    def wrapping_runner(bundle_root: Path, text: str) -> list[tuple[int, int, str]]:
        try:
            socket.create_connection(("127.0.0.1", 9), timeout=0.01)
        except Exception as exc:
            raise RuntimeError("runtime download failed") from exc
        return []

    installation = install_offline_bundle(
        bundle,
        tmp_path / "installed",
        host=_HOST,
        signature_key=_HMAC_KEY,
        self_test_runner=wrapping_runner,
    )

    assert installation.reason == REASON_NETWORK_ATTEMPTED


def test_install_fails_when_self_tests_miss_expected_spans(tmp_path: Path) -> None:
    bundle = _package(tmp_path)
    install_root = tmp_path / "installed"

    installation = install_offline_bundle(
        bundle,
        install_root,
        host=_HOST,
        signature_key=_HMAC_KEY,
        self_test_runner=lambda _root, _text: [],
    )

    assert installation.installed is False
    assert installation.reason == REASON_SELF_TEST_FAILED
    assert installation.self_tests is not None
    assert [case.passed for case in installation.self_tests.cases] == [
        False,
        False,
        True,
    ]
    assert [case.matched_count for case in installation.self_tests.cases] == [0, 0, 0]
    assert list(install_root.iterdir()) == []


def test_install_requires_a_runner_unless_self_tests_are_deferred(
    tmp_path: Path,
) -> None:
    bundle = _package(tmp_path)

    required = install_offline_bundle(
        bundle, tmp_path / "required", host=_HOST, signature_key=_HMAC_KEY
    )
    deferred = install_offline_bundle(
        bundle,
        tmp_path / "deferred",
        host=_HOST,
        signature_key=_HMAC_KEY,
        require_self_tests=False,
    )

    assert required.installed is False
    assert required.reason == REASON_SELF_TEST_RUNNER_REQUIRED
    assert not (tmp_path / "required").exists()
    assert deferred.installed is True
    assert deferred.self_tests is None
    assert deferred.to_dict()["self_tests"] is None


def test_install_rejects_an_invalid_bundle_without_writing(tmp_path: Path) -> None:
    bundle = _package(tmp_path)
    (bundle / "model" / "weights.bin").write_bytes(b"tampered")
    install_root = tmp_path / "installed"

    installation = install_offline_bundle(
        bundle,
        install_root,
        host=_HOST,
        signature_key=_HMAC_KEY,
        self_test_runner=_lexicon_runner,
    )

    assert installation.installed is False
    assert installation.reason == REASON_HASH_MISMATCH
    assert installation.self_tests is None
    assert not install_root.exists()


def test_install_does_not_replace_an_existing_installation(tmp_path: Path) -> None:
    bundle = _package(tmp_path)
    install_root = tmp_path / "installed"
    first = install_offline_bundle(
        bundle,
        install_root,
        host=_HOST,
        signature_key=_HMAC_KEY,
        self_test_runner=_lexicon_runner,
    )
    assert first.install_path is not None
    marker = (first.install_path / MANIFEST_FILENAME).read_bytes()

    second = install_offline_bundle(
        bundle,
        install_root,
        host=_HOST,
        signature_key=_HMAC_KEY,
        self_test_runner=_lexicon_runner,
    )

    assert second.installed is False
    assert second.reason == REASON_INSTALL_TARGET_EXISTS
    assert (first.install_path / MANIFEST_FILENAME).read_bytes() == marker
    assert sorted(path.name for path in install_root.iterdir()) == [_BUNDLE_ID]


def test_install_rechecks_digests_of_the_staged_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _package(tmp_path)
    install_root = tmp_path / "installed"
    original_copy = shutil.copyfile

    def corrupting_copy(source: Any, target: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_copy(source, target, *args, **kwargs)
        if Path(target).name == "names.tsv":
            Path(target).write_text("corrupted\tNAME\n", encoding="utf-8")
        return result

    monkeypatch.setattr(offline_bundle.shutil, "copyfile", corrupting_copy)

    installation = install_offline_bundle(
        bundle,
        install_root,
        host=_HOST,
        signature_key=_HMAC_KEY,
        self_test_runner=_lexicon_runner,
    )

    assert installation.installed is False
    assert installation.reason == REASON_INSTALL_COPY_MISMATCH
    assert list(install_root.iterdir()) == []


def test_run_offline_self_tests_accepts_common_span_shapes(tmp_path: Path) -> None:
    bundle = _package(tmp_path)
    verification = verify_offline_bundle(bundle, host=_HOST, signature_key=_HMAC_KEY)

    def mixed_runner(bundle_root: Path, text: str) -> list[Any]:
        shaped: list[Any] = [(0, 1, "EXTRA")]
        for start, end, label in _lexicon_runner(bundle_root, text):
            shaped.append({"start": start, "end": end, "label": label})
            shaped.append(SimpleNamespace(start=start, end=end, label=label))
        return shaped

    report = run_offline_self_tests(bundle, mixed_runner, verification=verification)

    assert report.passed is True
    assert [case.predicted_count for case in report.cases] == [2, 2, 1]
    assert [case.matched_count for case in report.cases] == [1, 1, 0]


@pytest.mark.parametrize(
    ("runner", "error"),
    [
        (lambda _root, _text: "Zawadi", SELF_TEST_ERROR_INVALID_OUTPUT),
        (lambda _root, _text: None, SELF_TEST_ERROR_INVALID_OUTPUT),
        (lambda _root, _text: [(0, "1", "NAME")], SELF_TEST_ERROR_INVALID_OUTPUT),
        (lambda _root, _text: 1 / 0, SELF_TEST_ERROR_RUNNER),
    ],
)
def test_runner_failures_are_reported_with_controlled_codes(
    tmp_path: Path, runner: Callable[[Path, str], Any], error: str
) -> None:
    bundle = _package(tmp_path)
    verification = verify_offline_bundle(bundle, host=_HOST, signature_key=_HMAC_KEY)

    report = run_offline_self_tests(bundle, runner, verification=verification)

    assert report.passed is False
    assert report.reason == REASON_SELF_TEST_FAILED
    assert {case.error for case in report.cases} == {error}


def test_self_tests_require_a_valid_verification(tmp_path: Path) -> None:
    bundle = _package(tmp_path)
    invalid = verify_offline_bundle(bundle, host=_HOST, signature_key=b"wrong")

    report = run_offline_self_tests(bundle, _lexicon_runner, verification=invalid)

    assert report.executed is False
    assert report.reason == REASON_BUNDLE_NOT_VERIFIED
    with pytest.raises(TypeError):
        run_offline_self_tests(bundle, "not callable", verification=invalid)  # type: ignore[arg-type]


def test_self_tests_rebind_files_to_verified_digests(tmp_path: Path) -> None:
    bundle = _package(tmp_path)
    verification = verify_offline_bundle(bundle, host=_HOST, signature_key=_HMAC_KEY)
    (bundle / "selftests" / "cases.jsonl").write_text(
        _self_test_lines([_case("sw-x", "Mpya kabisa.", {})]), encoding="utf-8"
    )

    report = run_offline_self_tests(bundle, _lexicon_runner, verification=verification)

    assert report.executed is False
    assert report.reason == REASON_HASH_MISMATCH


def test_reports_logs_and_errors_contain_no_raw_values(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG, logger="openmed")
    bundle = _package(tmp_path)

    installation = install_offline_bundle(
        bundle,
        tmp_path / "installed",
        host=_HOST,
        signature_key=_HMAC_KEY,
        self_test_runner=_lexicon_runner,
    )
    failed = install_offline_bundle(
        bundle,
        tmp_path / "failed",
        host=_HOST,
        signature_key=_HMAC_KEY,
        self_test_runner=lambda _root, _text: [],
    )
    rejected = verify_offline_bundle(bundle, host=_HOST, signature_key=b"wrong")
    with pytest.raises(OfflineBundleError) as raised:
        _descriptor(bundle_id="Bad Bundle Zawadi")

    serialized = "\n".join(
        [
            json.dumps(installation.to_dict(), sort_keys=True),
            json.dumps(failed.to_dict(), sort_keys=True),
            json.dumps(rejected.to_dict(), sort_keys=True),
            str(raised.value),
            caplog.text,
        ]
    )
    for sensitive in (
        _NAME,
        "Zawadi",
        _IDENTIFIER,
        *_CASE_TEXTS,
        _HMAC_KEY.decode("ascii"),
        _HMAC_KEY.hex(),
        str(tmp_path),
    ):
        assert sensitive not in serialized
    assert installation.installed is True
    assert failed.reason == REASON_SELF_TEST_FAILED
    assert "install_path" not in installation.to_dict()


def test_verification_requires_host_facts(tmp_path: Path) -> None:
    bundle = _package(tmp_path)

    with pytest.raises(TypeError):
        verify_offline_bundle(bundle, host=None)  # type: ignore[arg-type]


@pytest.mark.parametrize(("ram", "storage"), [(-1, 0), (0, -5), (True, 0), (1.5, 0)])
def test_host_resources_reject_invalid_facts(ram: Any, storage: Any) -> None:
    with pytest.raises(ValueError):
        HostResources(available_ram_bytes=ram, available_storage_bytes=storage)


@pytest.mark.parametrize(("ram", "storage"), [(0, 1), (1, 0), (True, 1)])
def test_resource_requirements_must_be_positive(ram: Any, storage: Any) -> None:
    with pytest.raises(OfflineBundleError) as raised:
        ResourceRequirements(min_ram_bytes=ram, min_storage_bytes=storage)

    assert raised.value.reason == REASON_DESCRIPTOR_MALFORMED


def test_host_resources_detect_uses_local_probes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    probed: list[Path] = []

    def disk_usage(path: Path) -> SimpleNamespace:
        probed.append(Path(path))
        return SimpleNamespace(total=100, used=40, free=60)

    sysconf_values = {"SC_PHYS_PAGES": 4, "SC_PAGE_SIZE": 4096}
    monkeypatch.setattr(offline_bundle.shutil, "disk_usage", disk_usage)
    # Windows has no os.sysconf; patch it in so the probe runs on every platform.
    monkeypatch.setattr(
        offline_bundle.os, "sysconf", sysconf_values.__getitem__, raising=False
    )

    host = HostResources.detect(tmp_path / "not" / "created" / "yet")

    assert host == HostResources(available_ram_bytes=16384, available_storage_bytes=60)
    assert probed == [tmp_path]


def test_host_resources_detect_fails_closed_without_memory_facts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unavailable(_name: str) -> int:
        raise ValueError("unsupported sysconf name")

    monkeypatch.setattr(offline_bundle.os, "sysconf", unavailable, raising=False)

    with pytest.raises(OfflineBundleError) as raised:
        HostResources.detect(tmp_path)

    assert raised.value.reason == REASON_HOST_FACTS_UNAVAILABLE
