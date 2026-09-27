from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError

import pytest

from openmed.eval.governance import (
    HOLDOUT_MANIFEST_KINDS,
    REASON_COMMITMENT_MISMATCH,
    REASON_INVALID_COMMITMENT,
    REASON_INVALID_PROOF,
    REASON_ITEM_NOT_INCLUDED,
    REASON_VALID,
    HoldoutCommitmentError,
    commit_holdout_manifests,
    create_inclusion_proof,
    verify_holdout_commitment,
    verify_inclusion_proof,
)


def _digest(label: str) -> str:
    return f"sha256:{hashlib.sha256(label.encode('ascii')).hexdigest()}"


def _manifests() -> dict[str, tuple[str, ...]]:
    return {
        kind: tuple(_digest(f"{kind}-{index}") for index in range(5))
        for kind in HOLDOUT_MANIFEST_KINDS
    }


def test_commitment_is_versioned_deterministic_and_content_free() -> None:
    manifests = _manifests()
    reversed_keys = dict(reversed(tuple(manifests.items())))

    first = commit_holdout_manifests("benchmark-2026.09", manifests)
    second = commit_holdout_manifests("benchmark-2026.09", reversed_keys)

    assert first == second
    assert first.to_json() == second.to_json()
    assert first.to_json() == json.dumps(
        first.to_dict(),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    assert first.holdout_version == "benchmark-2026.09"
    assert (
        first.commitment_digest
        == "sha256:de07aeb80c46f17f8e31b7bfcf4e5c29cc09e40a6733317f042c247efb415a0b"
    )
    assert (
        first.manifest_roots["case"]
        == "sha256:c37549e8be90046a6ed645248b9badf2b03a554a380b749c5c07ae265506b953"
    )
    assert tuple(first.manifest_roots) == HOLDOUT_MANIFEST_KINDS
    assert set(first.to_dict()) == {
        "commitment_digest",
        "holdout_version",
        "manifest_item_counts",
        "manifest_roots",
        "schema_version",
    }
    assert not any(
        item_digest in first.to_json()
        for item_digests in manifests.values()
        for item_digest in item_digests
    )
    assert verify_holdout_commitment(first).reason_codes == (REASON_VALID,)


def test_commitment_binds_version_order_and_all_manifest_kinds() -> None:
    manifests = _manifests()
    baseline = commit_holdout_manifests("v1", manifests)

    assert commit_holdout_manifests("v2", manifests) != baseline

    reordered = dict(manifests)
    reordered["case"] = tuple(reversed(reordered["case"]))
    assert (
        commit_holdout_manifests("v1", reordered).manifest_roots["case"]
        != baseline.manifest_roots["case"]
    )

    missing = dict(manifests)
    missing.pop("randomization")
    with pytest.raises(HoldoutCommitmentError, match="manifests: invalid_keys"):
        commit_holdout_manifests("v1", missing)


def test_commitment_rejects_empty_or_non_digest_items_without_echoing_values() -> None:
    manifests = _manifests()
    manifests["label"] = ()
    with pytest.raises(HoldoutCommitmentError, match="manifests.label: empty"):
        commit_holdout_manifests("v1", manifests)

    sensitive_value = "synthetic-private-label-value"
    manifests["label"] = (sensitive_value,)
    with pytest.raises(HoldoutCommitmentError) as raised:
        commit_holdout_manifests("v1", manifests)
    assert sensitive_value not in str(raised.value)


@pytest.mark.parametrize("manifest_kind", HOLDOUT_MANIFEST_KINDS)
@pytest.mark.parametrize("item_index", (0, 3, 4))
def test_selective_proof_verifies_each_manifest_and_odd_tree_position(
    manifest_kind: str,
    item_index: int,
) -> None:
    manifests = _manifests()
    commitment = commit_holdout_manifests("audit-v1", manifests)

    proof = create_inclusion_proof(
        commitment,
        manifest_kind,
        item_index,
        manifests[manifest_kind],
    )
    verification = verify_inclusion_proof(commitment, proof)

    assert verification.valid is True
    assert verification.reason_codes == (REASON_VALID,)
    assert proof.manifest_kind == manifest_kind
    assert proof.item_index == item_index
    assert proof.item_digest == manifests[manifest_kind][item_index]
    assert len(proof.sibling_digests) == 3


def test_proof_generation_requires_the_committed_private_manifest() -> None:
    manifests = _manifests()
    commitment = commit_holdout_manifests("audit-v1", manifests)
    changed_cases = list(manifests["case"])
    changed_cases[2] = _digest("replacement")

    with pytest.raises(HoldoutCommitmentError, match="commitment_mismatch"):
        create_inclusion_proof(commitment, "case", 2, changed_cases)


def test_proof_tampering_and_cross_commitment_reuse_fail_closed() -> None:
    manifests = _manifests()
    commitment = commit_holdout_manifests("audit-v1", manifests)
    proof = create_inclusion_proof(commitment, "label", 2, manifests["label"])

    changed_item = proof.to_dict()
    changed_item["item_digest"] = _digest("changed-item")
    assert verify_inclusion_proof(commitment, changed_item).reason_codes == (
        REASON_ITEM_NOT_INCLUDED,
    )

    changed_sibling = proof.to_dict()
    changed_sibling["sibling_digests"][0] = _digest("changed-sibling")
    assert verify_inclusion_proof(commitment, changed_sibling).reason_codes == (
        REASON_ITEM_NOT_INCLUDED,
    )

    other_commitment = commit_holdout_manifests("audit-v2", manifests)
    assert verify_inclusion_proof(other_commitment, proof).reason_codes == (
        REASON_COMMITMENT_MISMATCH,
    )


def test_public_commitment_tampering_is_detected() -> None:
    commitment = commit_holdout_manifests("audit-v1", _manifests()).to_dict()
    commitment["manifest_item_counts"]["case"] += 1

    result = verify_holdout_commitment(commitment)

    assert result.valid is False
    assert result.reason_codes == (REASON_INVALID_COMMITMENT,)


def test_verifiers_never_raise_or_echo_untrusted_values() -> None:
    sensitive_value = "synthetic-private-holdout-value"
    invalid_commitment = {
        "schema_version": sensitive_value,
        "holdout_version": sensitive_value,
        "manifest_roots": sensitive_value,
        "manifest_item_counts": sensitive_value,
        "commitment_digest": sensitive_value,
    }
    invalid_proof = {
        "schema_version": sensitive_value,
        "commitment_digest": sensitive_value,
        "manifest_kind": sensitive_value,
        "item_index": sensitive_value,
        "item_digest": sensitive_value,
        "sibling_digests": [sensitive_value],
    }

    commitment_result = verify_holdout_commitment(invalid_commitment)
    proof_result = verify_inclusion_proof(invalid_commitment, invalid_proof)
    rendered = json.dumps(
        [commitment_result.to_dict(), proof_result.to_dict()], sort_keys=True
    )

    assert commitment_result.reason_codes == (REASON_INVALID_COMMITMENT,)
    assert proof_result.reason_codes == (REASON_INVALID_COMMITMENT,)
    assert sensitive_value not in rendered


def test_inclusion_proof_structure_fails_closed() -> None:
    manifests = _manifests()
    commitment = commit_holdout_manifests("audit-v1", manifests)
    proof = create_inclusion_proof(commitment, "template", 1, manifests["template"])
    malformed = proof.to_dict()
    malformed["sibling_digests"].pop()

    assert verify_inclusion_proof(commitment, malformed).reason_codes == (
        REASON_INVALID_PROOF,
    )


def test_commitment_objects_are_immutable() -> None:
    commitment = commit_holdout_manifests("audit-v1", _manifests())

    with pytest.raises(TypeError):
        commitment.manifest_roots["case"] = _digest("replacement")  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        commitment.holdout_version = "v2"  # type: ignore[misc]
