"""Shared, synthetic Android/Python metadata contracts; no providers or models."""

import json
from pathlib import Path

import pytest

from openmed.multimodal.abstention import AbstentionRecord
from openmed.multimodal.asset_manifest import AssetManifest
from openmed.multimodal.manifest_profiles import AUDIO_V1, DICOM_V1, IMAGE_V1, PDF_V1
from openmed.multimodal.preflight import PreflightFinding
from openmed.multimodal.provider_result import ProviderResultEnvelope

FIXTURE = (
    Path(__file__).resolve().parents[2] / "fixtures/parity/multimodal_contracts_v1.json"
)
VECTORS = json.loads(FIXTURE.read_text(encoding="utf-8"))
PROFILES = {p.modality: p for p in (IMAGE_V1, PDF_V1, DICOM_V1, AUDIO_V1)}


@pytest.mark.parametrize("vector", VECTORS["cases"])
def test_shared_canonical_json(vector):
    """Both runtimes must reproduce the exact committed UTF-8 JSON bytes."""
    payload = vector["canonical_json"]
    kind = vector["kind"]
    if kind == "asset_manifest":
        actual = AssetManifest.from_json(payload).to_json()
    elif kind == "abstention_record":
        actual = AbstentionRecord.from_json(payload).to_json()
    elif kind == "provider_result":
        actual = ProviderResultEnvelope.from_json(payload).to_json()
    elif kind == "preflight_finding":
        finding = PreflightFinding(**json.loads(payload))
        actual = json.dumps(finding.to_dict(), separators=(",", ":"))
    else:
        assert kind == "manifest_profile"
        fields = json.loads(payload)
        profile = PROFILES[fields["modality"]]
        assert fields["version"] == profile.version
        actual = json.dumps(
            {"modality": profile.modality, "version": profile.version},
            separators=(",", ":"),
        )
    assert actual.encode("utf-8") == payload.encode("utf-8")


def test_vectors_are_versioned_synthetic_and_closed():
    """Keep the shared inputs content-free and exercise every mirrored type."""
    assert VECTORS["version"] == 1
    assert VECTORS["synthetic"] is True
    assert {case["kind"] for case in VECTORS["cases"]} == {
        "asset_manifest",
        "manifest_profile",
        "preflight_finding",
        "abstention_record",
        "provider_result",
    }
    allowed = {
        "asset_manifest": {
            "version",
            "asset_id",
            "media_type",
            "sha256",
            "byte_size",
            "pages",
            "width",
            "height",
            "frames",
            "duration_seconds",
        },
        "manifest_profile": {"modality", "version"},
        "preflight_finding": {
            "check",
            "reason_code",
            "field_name",
            "limit",
            "observed",
        },
        "abstention_record": {"schema_version", "stage", "reason"},
        "provider_result": {
            "schema_version",
            "provider_id",
            "model_id",
            "input_digest",
            "output_digest",
            "outcome",
            "abstention_code",
            "duration_ms",
            "count_metadata",
        },
    }
    for case in VECTORS["cases"]:
        assert set(json.loads(case["canonical_json"])) <= allowed[case["kind"]]
