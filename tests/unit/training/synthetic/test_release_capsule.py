"""Canonical capsule reproducibility and offline verification controls."""

import json

import pytest

from openmed.training.synthetic.release_capsule import (
    CapsuleCheck,
    CapsuleGenerator,
    build_release_capsule,
    verify_release_capsule,
)


def generator(external=None):
    return CapsuleGenerator(
        "notes",
        "1.0.0",
        {"count": 2, "locale": "synthetic"},
        "1" * 64,
        "2" * 64,
        external,
    )


def capsule(artifacts=None, external=None):
    artifacts = {"notes": b"synthetic"} if artifacts is None else artifacts
    check = CapsuleCheck("pass", 2, "3" * 64)
    return build_release_capsule(
        [generator(external)],
        artifacts,
        licenses={name: "MIT" for name in artifacts},
        privacy=check,
        utility=check,
    )


def test_complete_capsule_verifies_without_network_or_payload_echo():
    payload = capsule()
    result = verify_release_capsule(payload, {"notes": b"synthetic"})
    assert result["integrity_verified"] is True
    assert result["generation_reproduced"] is False
    assert b'"configuration"' in payload
    assert b'"seed_commitment"' in payload


def test_reordered_artifacts_are_byte_stable():
    assert capsule({"a": b"a", "b": b"b"}) == capsule({"b": b"b", "a": b"a"})


def test_external_model_is_not_claimed_reproducible():
    result = verify_release_capsule(capsule(external="f" * 64), {"notes": b"synthetic"})
    assert result["reproducibility"] == "external_model_required"


@pytest.mark.parametrize(
    "artifacts", [{}, {"notes": b"changed"}, {"notes": b"synthetic", "extra": b"x"}]
)
def test_missing_changed_or_extra_artifact_fails(artifacts):
    with pytest.raises(ValueError):
        verify_release_capsule(capsule(), artifacts)


def test_modified_manifest_fails_integrity():
    wrapper = json.loads(capsule())
    wrapper["manifest"]["generators"][0]["configuration"]["count"] = 3
    with pytest.raises(ValueError):
        verify_release_capsule(json.dumps(wrapper).encode(), {"notes": b"synthetic"})


def test_nonpermissive_or_missing_license_rejected():
    check = CapsuleCheck("pass", 2, "3" * 64)
    for licenses in ({}, {"notes": "GPL-3.0-only"}):
        with pytest.raises(ValueError):
            build_release_capsule(
                [generator()],
                {"notes": b"x"},
                licenses=licenses,
                privacy=check,
                utility=check,
            )


def test_failed_check_is_preserved_not_promoted():
    fail = CapsuleCheck("fail", 2, "3" * 64)
    payload = build_release_capsule(
        [generator()],
        {"notes": b"x"},
        licenses={"notes": "MIT"},
        privacy=fail,
        utility=fail,
    )
    assert verify_release_capsule(payload, {"notes": b"x"})["checks_passed"] is False


def test_configuration_snapshots_and_raw_prompt_rejected():
    options = {"count": 2}
    record = CapsuleGenerator("notes", "1.0.0", options, "1" * 64, "2" * 64)
    options["count"] = 99
    assert record.configuration["count"] == 2
    with pytest.raises(ValueError):
        CapsuleGenerator(
            "notes",
            "1.0.0",
            {"prompt": "arbitrary submitted prose"},
            "1" * 64,
            "2" * 64,
        )


def test_check_status_has_controlled_type_validation():
    with pytest.raises(ValueError):
        CapsuleCheck([], 1, "0" * 64)
