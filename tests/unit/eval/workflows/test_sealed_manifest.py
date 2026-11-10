from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError

import pytest

from openmed.eval.workflows import (
    REASON_COMPONENT_DIGEST_MISMATCH,
    REASON_ELIGIBLE,
    REASON_INCOMPLETE_EVALUATION_SNAPSHOT,
    REASON_INCOMPLETE_MANIFEST,
    REASON_INVALID_MANIFEST,
    REASON_MUTABLE_MANIFEST,
    SEALED_MANIFEST_COMPONENTS,
    SealedManifestError,
    seal_workflow_manifest,
    verify_at_evaluation_start,
)


def _digest(label: str) -> str:
    return f"sha256:{hashlib.sha256(label.encode('ascii')).hexdigest()}"


def _component_digests() -> dict[str, str]:
    return {component: _digest(component) for component in SEALED_MANIFEST_COMPONENTS}


def test_sealed_manifest_is_canonical_deterministic_and_immutable() -> None:
    component_digests = _component_digests()
    reverse_order = dict(reversed(tuple(component_digests.items())))

    first = seal_workflow_manifest(component_digests)
    second = seal_workflow_manifest(reverse_order)

    assert first == second
    assert first.to_json() == second.to_json()
    assert (
        first.manifest_digest
        == "sha256:397d70d8eb1775c8b53779586e34412b67f5772f712d442cf8bfdb4362edb9ab"
    )
    assert first.to_json() == json.dumps(
        first.to_dict(),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    assert tuple(first.component_digests) == SEALED_MANIFEST_COMPONENTS
    with pytest.raises(TypeError):
        first.component_digests["model"] = _digest("replacement")  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        first.manifest_digest = _digest("replacement")  # type: ignore[misc]


def test_sealing_requires_exactly_the_governed_sha256_components() -> None:
    incomplete = _component_digests()
    incomplete.pop("tool_inventory")
    with pytest.raises(SealedManifestError, match="invalid_keys"):
        seal_workflow_manifest(incomplete)

    malformed = _component_digests()
    malformed["model"] = "mutable-model-tag"
    with pytest.raises(SealedManifestError, match="component_digests.model") as raised:
        seal_workflow_manifest(malformed)
    assert "mutable-model-tag" not in str(raised.value)


def test_complete_unchanged_submission_is_eligible_for_sealed_results() -> None:
    component_digests = _component_digests()
    manifest = seal_workflow_manifest(component_digests)

    result = verify_at_evaluation_start(
        manifest, dict(reversed(tuple(component_digests.items())))
    )

    assert result.eligible_for_sealed_results is True
    assert result.reason_codes == (REASON_ELIGIBLE,)
    assert result.affected_components == ()


def test_incomplete_manifest_is_ineligible_without_raising() -> None:
    component_digests = _component_digests()
    manifest = seal_workflow_manifest(component_digests).to_dict()
    del manifest["component_digests"]["prompt"]
    del manifest["manifest_digest"]

    result = verify_at_evaluation_start(manifest, component_digests)

    assert result.eligible_for_sealed_results is False
    assert REASON_INCOMPLETE_MANIFEST in result.reason_codes
    assert result.affected_components == ("prompt",)


def test_changed_sealed_payload_is_labeled_mutable() -> None:
    component_digests = _component_digests()
    manifest = seal_workflow_manifest(component_digests).to_dict()
    manifest["component_digests"]["policy"] = _digest("changed-policy")

    result = verify_at_evaluation_start(manifest, manifest["component_digests"])

    assert result.eligible_for_sealed_results is False
    assert REASON_MUTABLE_MANIFEST in result.reason_codes


def test_evaluation_start_detects_component_changes() -> None:
    component_digests = _component_digests()
    manifest = seal_workflow_manifest(component_digests)
    evaluation_digests = dict(component_digests)
    evaluation_digests["container"] = _digest("rebuilt-container")

    result = verify_at_evaluation_start(manifest, evaluation_digests)

    assert result.eligible_for_sealed_results is False
    assert result.reason_codes == (REASON_COMPONENT_DIGEST_MISMATCH,)
    assert result.affected_components == ("container",)


def test_incomplete_evaluation_snapshot_is_ineligible() -> None:
    component_digests = _component_digests()
    manifest = seal_workflow_manifest(component_digests)
    component_digests.pop("tokenizer")

    result = verify_at_evaluation_start(manifest, component_digests)

    assert result.eligible_for_sealed_results is False
    assert REASON_INCOMPLETE_EVALUATION_SNAPSHOT in result.reason_codes
    assert result.affected_components == ("tokenizer",)


def test_verification_report_never_echoes_untrusted_values() -> None:
    sensitive_value = "synthetic-sensitive-submission-value"
    manifest = {
        "schema_version": sensitive_value,
        "component_digests": {"model": sensitive_value},
        "manifest_digest": sensitive_value,
        "unexpected": sensitive_value,
    }

    result = verify_at_evaluation_start(manifest, {"model": sensitive_value})
    rendered = json.dumps(result.to_dict(), sort_keys=True)

    assert result.eligible_for_sealed_results is False
    assert REASON_INVALID_MANIFEST in result.reason_codes
    assert sensitive_value not in rendered
