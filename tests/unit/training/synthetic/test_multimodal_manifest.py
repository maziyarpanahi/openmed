"""Synthetic pair linkage, licensing and deterministic artifact verification."""

import hashlib

import pytest

from openmed.training.synthetic.multimodal_manifest import (
    PairAssetProvenance,
    PairTransformation,
    SyntheticPairManifest,
    verify_pair_artifacts,
)


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def asset(payload):
    return PairAssetProvenance(digest(payload), digest(payload), "MIT")


def manifest(passed=True):
    return SyntheticPairManifest(
        asset(b"synthetic image"),
        asset(b"synthetic text"),
        asset(b"synthetic labels"),
        "1" * 64,
        "1.0.0",
        "2" * 64,
        passed,
    )


def test_real_three_artifact_pair_verifies_offline():
    verify_pair_artifacts(
        manifest(),
        {
            "image": b"synthetic image",
            "text": b"synthetic text",
            "labels": b"synthetic labels",
        },
    )


@pytest.mark.parametrize("role", ["image", "text", "labels"])
def test_corrupted_member_cannot_pass_pair_validation(role):
    payloads = {
        "image": b"synthetic image",
        "text": b"synthetic text",
        "labels": b"synthetic labels",
    }
    payloads[role] = b"altered"
    with pytest.raises(ValueError, match="digest mismatch"):
        verify_pair_artifacts(manifest(), payloads)


@pytest.mark.parametrize(
    "license_id", [None, "unknown", "GPL-3.0-only", "LicenseRef-private"]
)
def test_missing_or_nonpermissive_declaration_rejected(license_id):
    with pytest.raises(ValueError, match="SPDX"):
        PairAssetProvenance("0" * 64, "0" * 64, license_id)


def test_disconnected_or_missing_transformation_chain_rejected():
    link = PairTransformation("0" * 64, "1" * 64, "2" * 64, "3" * 64)
    PairAssetProvenance("0" * 64, "1" * 64, "MIT", (link,))
    with pytest.raises(ValueError, match="disconnected"):
        PairAssetProvenance("f" * 64, "1" * 64, "MIT", (link,))
    with pytest.raises(ValueError, match="does not reach"):
        PairAssetProvenance("0" * 64, "1" * 64, "MIT")


def test_failed_validation_refuses_publication_qualification():
    with pytest.raises(ValueError, match="did not pass"):
        manifest(False).require_publishable()


def test_byte_stable_manifest_links_all_assets_without_values():
    first = manifest()
    second = manifest()
    assert first.to_json() == second.to_json()
    assert first.manifest_digest == second.manifest_digest
    assert set(first.to_dict()["assets"]) == {"image", "text", "labels"}
    assert "synthetic image" not in first.to_json()


def test_source_sequence_is_snapshotted():
    links = []
    record = PairAssetProvenance("0" * 64, "0" * 64, "MIT", links)
    links.append(PairTransformation("0" * 64, "1" * 64, "2" * 64, "3" * 64))
    assert record.transformations == ()
