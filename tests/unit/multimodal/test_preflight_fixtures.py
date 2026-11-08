"""Complete coverage of the bundled synthetic negative fixture inventory."""

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from openmed.multimodal.preflight_fixtures import (
    MAX_FIXTURE_BYTES,
    PreflightFixtureAsset,
    load_preflight_fixtures,
)

CASE_IDS = frozenset(
    {
        "pdf_truncated",
        "png_truncated",
        "jpeg_truncated",
        "tiff_truncated",
        "dicom_truncated",
        "wav_truncated",
        "type_mismatch",
        "invalid_dimensions",
        "oversized_counts",
        "bad_digest",
        "duplicate_asset",
        "unknown_media",
        "resource_limit",
    }
)


def payload():
    import openmed.multimodal.preflight_fixtures as module

    return json.loads(
        (
            Path(module.__file__).parent / "fixtures/malformed_asset_preflight.json"
        ).read_bytes()
    )


def test_complete_case_id_coverage_and_production_expected_findings():
    cases = load_preflight_fixtures()
    assert {case.case_id for case in cases} == CASE_IDS
    for case in cases:
        assert case.actual_findings() == case.expected_findings
        assert all(0 < len(asset.header) <= 132 for asset in case.assets)


def test_reordered_input_has_identical_immutable_cases():
    data = payload()
    original = load_preflight_fixtures(json.dumps(data))
    data["cases"].reverse()
    assert load_preflight_fixtures(json.dumps(data)) == original
    assert tuple(case.case_id for case in original) == tuple(sorted(CASE_IDS))
    with pytest.raises(FrozenInstanceError):
        original[0].category = "changed"
    with pytest.raises(TypeError):
        original[0].assets[0].manifest["width"] = 0


@pytest.mark.parametrize(
    "mutation",
    [
        lambda x: x.update(schema_version=True),
        lambda x: x.update(synthetic=False),
        lambda x: x.update(private_path="synthetic/private"),
        lambda x: x["cases"].append(x["cases"][0]),
        lambda x: x["cases"][0].update(expected_findings=["invented.reason"]),
        lambda x: x["cases"][0].update(expected_findings=[]),
        lambda x: x["cases"][0]["assets"][0].update(header_hex="00" * 133),
        lambda x: x["cases"][0]["assets"][0].update(header_hex="nothex"),
        lambda x: x["cases"][0]["assets"][0]["manifest"].update(
            patient="synthetic/private"
        ),
        lambda x: x["cases"][0].update(case_id="synthetic/private"),
        lambda x: x["cases"][0]["assets"][0]["manifest"].update(width=True),
    ],
)
def test_malformed_fixtures_reject_without_payload_echo(mutation):
    data = payload()
    mutation(data)
    with pytest.raises(ValueError) as exc:
        load_preflight_fixtures(json.dumps(data))
    assert "synthetic/private" not in str(exc.value)


def test_payload_and_case_count_bounds():
    with pytest.raises(ValueError):
        load_preflight_fixtures(b" " * (MAX_FIXTURE_BYTES + 1))
    data = payload()
    data["cases"] = data["cases"][:1] * 65
    with pytest.raises(ValueError, match="case count"):
        load_preflight_fixtures(json.dumps(data))


def test_duplicate_json_keys_reject_before_fixture_construction():
    with pytest.raises(ValueError, match="JSON"):
        load_preflight_fixtures('{"schema_version":1,"schema_version":1}')


def test_no_real_media_or_sensitive_payload_metadata_in_bundled_cases():
    cases = load_preflight_fixtures()
    forbidden = {
        "patient",
        "name",
        "credential",
        "path",
        "transcript",
        "text",
        "dataset",
        "endpoint",
    }
    for case in cases:
        for asset in case.assets:
            assert not forbidden & set(asset.manifest)
            assert len(asset.header) <= 132
    formats = {asset.manifest["media_type"] for case in cases for asset in case.assets}
    assert {
        "application/pdf",
        "image/png",
        "image/jpeg",
        "image/tiff",
        "application/dicom",
        "audio/wav",
    } <= formats


def test_direct_asset_constructor_rejects_private_metadata():
    with pytest.raises(ValueError):
        PreflightFixtureAsset({"path": "synthetic/private"}, b"\x00")
