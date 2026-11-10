"""Offline composition of the mirrored Android metadata boundaries."""

import hashlib
import json

import pytest

from openmed.multimodal.abstention import AbstentionRecord
from openmed.multimodal.asset_manifest import AssetManifest
from openmed.multimodal.preflight import preflight_asset
from openmed.multimodal.provider_result import ProviderResultEnvelope


@pytest.mark.integration
def test_synthetic_preflight_abstention_and_provider_envelope_are_content_free():
    """A refusal can cross the contracts without retaining the asset or details."""
    source = b"SYNTH_PRIVATE_CONTENT_SENTINEL"
    manifest = AssetManifest(
        asset_id="synth-001",
        media_type="image/png",
        sha256=hashlib.sha256(source).hexdigest(),
        byte_size=len(source),
        width=2,
        height=3,
    )
    report = preflight_asset(manifest, source)
    assert report.status.value == "abstain"
    record = AbstentionRecord.from_json(report.abstention.to_json())
    assert record.reason.value == "unsupported_media"
    envelope = ProviderResultEnvelope.from_dict(
        {
            "schema_version": "openmed.multimodal.provider_result.v1",
            "provider_id": "synth-adapter",
            "model_id": "synth-model",
            "input_digest": manifest.sha256,
            "outcome": "abstention",
            "abstention_code": record.reason.value,
            "duration_ms": 0,
            "count_metadata": {"input_bytes": len(source)},
        }
    )
    serialized = (
        report.to_json() + record.to_json() + manifest.to_json() + envelope.to_json()
    )
    assert source.decode() not in serialized
    assert "path" not in serialized
    assert "url" not in serialized
    assert json.loads(envelope.to_json())["output_digest"] is None
