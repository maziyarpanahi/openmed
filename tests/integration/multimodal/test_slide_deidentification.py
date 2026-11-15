"""Synthetic offline file-copy and audit integration."""

import json

import pytest

from openmed.multimodal.slide_deidentification import deidentify_tiff_slide
from tests.fixtures.multimodal.slides import SENTINELS, slide


@pytest.mark.integration
def test_copy_roundtrip_keeps_original_and_has_no_identifier_artifacts(tmp_path):
    original, _ = slide(big=True, endian=">")
    source = tmp_path / "synthetic-source.tif"
    destination = tmp_path / "synthetic-clean.tif"
    source.write_bytes(original)
    result = deidentify_tiff_slide(source.read_bytes())
    destination.write_bytes(result.data)
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps(result.to_audit_report()))
    assert source.read_bytes() == original
    assert (
        deidentify_tiff_slide(destination.read_bytes()).outcome
        == "no_associated_images"
    )
    for sentinel in SENTINELS:
        assert sentinel not in destination.read_bytes()
        assert sentinel not in audit.read_bytes()
