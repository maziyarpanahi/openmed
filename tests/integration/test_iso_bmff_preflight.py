"""Shared offline phone-container routing and native parity fixtures."""

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from openmed.multimodal.iso_bmff import BmffPreflightError, preflight_iso_bmff
from openmed.multimodal.media_type import (
    MediaTypeStatus,
    detect_media_type,
    validate_media_type,
)

CASES = json.loads(
    (Path(__file__).parents[1] / "fixtures/multimodal/iso_bmff.json").read_text()
)


@pytest.mark.integration
@pytest.mark.parametrize("case", CASES, ids=lambda case: case["name"])
def test_bounded_phone_media_routing(case):
    source = bytes.fromhex(case["source_hex"])
    detected = detect_media_type(source)
    assert detected == case["media_type"]
    if detected:
        assert validate_media_type(source, detected) is MediaTypeStatus.MATCH
    if "error" in case:
        with pytest.raises(BmffPreflightError) as error:
            preflight_iso_bmff(source, **case.get("limits", {}))
        assert error.value.category == str(error.value) == case["error"]
    else:
        assert asdict(preflight_iso_bmff(source)) == case["expected"]
