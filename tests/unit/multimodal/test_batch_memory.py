import dataclasses
import json
import math

import pytest

from openmed.multimodal.asset_batch import MAX_BATCH_ASSETS
from openmed.multimodal.asset_manifest import MAX_MANIFEST_COUNT, AssetManifest
from openmed.multimodal.batch_memory import (
    BATCH_MEMORY_REASON_CODES,
    MAX_AUDIO_CHANNELS,
    MAX_AUDIO_SAMPLE_RATE_HZ,
    MAX_BYTES_PER_UNIT,
    SATURATION_CEILING,
    BatchMemoryError,
    BatchOutcome,
    MemoryEstimationPolicy,
    plan_batch_memory,
)

FULL = MemoryEstimationPolicy(
    image_bytes_per_pixel=4,
    dicom_bytes_per_pixel=2,
    audio_sample_rate_hz=16_000,
    audio_channels=1,
    audio_bytes_per_sample=4,
)


def manifest(index=0, media_type="image/png", **fields):
    return AssetManifest(
        asset_id=f"asset-{index}",
        media_type=media_type,
        sha256=f"{index:064x}",
        byte_size=1,
        **fields,
    )


def image(index=0, width=10, height=10):
    return manifest(index, "image/png", width=width, height=height)


def plan(assets, policy=FULL, *, budget=10_000, overhead=0):
    return plan_batch_memory(
        assets, policy, budget_bytes=budget, overhead_bytes=overhead
    )


# --- per-modality estimates ---------------------------------------------------


def test_estimates_each_modality_from_manifest_and_policy():
    assets = [
        image(0, width=10, height=20),
        manifest(1, "application/dicom", width=8, height=8, frames=3),
        manifest(2, "audio/wav", duration_seconds=0.5),
    ]
    result = plan(assets, budget=10**6)
    assert result.outcome is BatchOutcome.ACCEPT
    assert [a.modality for a in result.assets] == ["image", "dicom", "audio"]
    # 10*20*4, 8*8*3*2, ceil(0.5*16000)*1*4
    assert [a.estimated_bytes for a in result.assets] == [800, 384, 32_000]


def test_audio_frame_count_is_an_exact_rational_ceiling():
    # 0.1 is not exact in binary; its exact value is slightly above 1/10, so
    # the exact ceiling of 0.1 * 16000 is 1601 frames, not 1600.
    result = plan([manifest(0, "audio/wav", duration_seconds=0.1)], budget=10**6)
    assert result.assets[0].estimated_bytes == 1601 * 4


@pytest.mark.parametrize(
    ("asset", "policy", "field_name"),
    [
        (manifest(0, "image/png", width=10), FULL, "height"),
        (manifest(0, "image/png", height=10), FULL, "width"),
        (image(), MemoryEstimationPolicy(), "image_bytes_per_pixel"),
        (
            manifest(0, "application/dicom", width=4, height=4),
            FULL,
            "frames",
        ),
        (
            manifest(0, "application/dicom", width=4, height=4, frames=1),
            MemoryEstimationPolicy(image_bytes_per_pixel=4),
            "dicom_bytes_per_pixel",
        ),
        (manifest(0, "audio/wav"), FULL, "duration_seconds"),
        (
            manifest(0, "audio/wav", duration_seconds=1.0),
            dataclasses.replace(FULL, audio_channels=None),
            "audio_channels",
        ),
        (manifest(0, "application/dicom+json"), FULL, "media_type"),
    ],
)
def test_missing_geometry_or_factor_is_unevaluable(asset, policy, field_name):
    result = plan([asset], policy)
    assert result.outcome is BatchOutcome.REJECT
    assert result.batches == ()
    (estimate,) = result.assets
    assert estimate.reason_code == "insufficient_metadata"
    assert estimate.field_name == field_name
    assert estimate.estimated_bytes is None


def test_pdf_page_count_never_produces_an_estimate():
    result = plan([manifest(0, "application/pdf", pages=1)], budget=SATURATION_CEILING)
    assert result.outcome is BatchOutcome.REJECT
    (estimate,) = result.assets
    assert (estimate.modality, estimate.reason_code, estimate.estimated_bytes) == (
        "pdf",
        "insufficient_metadata",
        None,
    )


# --- saturation ---------------------------------------------------------------


def test_saturated_estimate_rejects_even_under_the_largest_budget():
    huge = image(0, width=MAX_MANIFEST_COUNT, height=MAX_MANIFEST_COUNT)
    policy = MemoryEstimationPolicy(image_bytes_per_pixel=MAX_BYTES_PER_UNIT)
    result = plan([huge, image(1)], policy, budget=SATURATION_CEILING)
    assert result.outcome is BatchOutcome.REJECT
    assert result.assets[0].reason_code == "estimate_saturated"
    assert result.assets[0].estimated_bytes is None
    assert result.assets[1].reason_code is None


def test_saturation_ceiling_boundary_is_inclusive():
    # 2**63 - 1 == (7 * 7 * 73 * 127 * 337) * 92737 * 649657 exactly.
    at_ceiling = manifest(
        0, "application/dicom", width=153_092_023, height=92_737, frames=649_657
    )
    one_over = manifest(1, "application/dicom", width=1 << 30, height=1 << 30, frames=8)
    policy = MemoryEstimationPolicy(dicom_bytes_per_pixel=1)
    fit = plan([at_ceiling], policy, budget=SATURATION_CEILING)
    assert fit.outcome is BatchOutcome.ACCEPT
    assert fit.assets[0].estimated_bytes == SATURATION_CEILING
    over = plan([one_over], policy, budget=SATURATION_CEILING)
    assert over.assets[0].reason_code == "estimate_saturated"


def test_audio_saturation():
    loud = manifest(0, "audio/wav", duration_seconds=float(MAX_MANIFEST_COUNT))
    policy = MemoryEstimationPolicy(
        audio_sample_rate_hz=MAX_AUDIO_SAMPLE_RATE_HZ,
        audio_channels=MAX_AUDIO_CHANNELS,
        audio_bytes_per_sample=MAX_BYTES_PER_UNIT,
    )
    result = plan([loud], policy, budget=SATURATION_CEILING)
    assert result.assets[0].reason_code == "estimate_saturated"


# --- budgets and boundaries ---------------------------------------------------


def test_zero_budget_rejects():
    result = plan([image()], budget=0)
    assert result.outcome is BatchOutcome.REJECT
    assert result.assets[0].reason_code == "budget_exceeded"


def test_exact_budget_boundary_is_inclusive():
    # 10 * 10 * 4 = 400 bytes, plus 100 bytes of overhead.
    assert plan([image()], budget=500, overhead=100).outcome is BatchOutcome.ACCEPT
    rejected = plan([image()], budget=499, overhead=100)
    assert rejected.outcome is BatchOutcome.REJECT
    assert rejected.assets[0].reason_code == "budget_exceeded"
    # The single-asset estimate is still reported when only the budget fails.
    assert rejected.assets[0].estimated_bytes == 400


def test_overhead_larger_than_budget_rejects():
    result = plan([image()], budget=100, overhead=101)
    assert result.outcome is BatchOutcome.REJECT


def test_single_oversized_asset_rejects_instead_of_splitting():
    result = plan([image(0), image(1, width=100, height=100), image(2)], budget=1_000)
    assert result.outcome is BatchOutcome.REJECT
    assert [a.reason_code for a in result.assets] == [None, "budget_exceeded", None]
    assert result.batches == ()


# --- splits -------------------------------------------------------------------


def test_split_preserves_order_and_charges_overhead_per_batch():
    # Each image is 400 bytes; budget 1000 with 150 overhead fits two per batch.
    assets = [image(i) for i in range(5)]
    result = plan(assets, budget=1_000, overhead=150)
    assert result.outcome is BatchOutcome.SPLIT
    assert [b.positions for b in result.batches] == [(0, 1), (2, 3), (4,)]
    assert [b.estimated_bytes for b in result.batches] == [950, 950, 550]
    assert all(b.estimated_bytes <= 1_000 for b in result.batches)


def test_split_boundary_where_batch_fills_the_budget_exactly():
    assets = [image(i) for i in range(4)]
    result = plan(assets, budget=900, overhead=100)
    assert [b.positions for b in result.batches] == [(0, 1), (2, 3)]
    assert [b.estimated_bytes for b in result.batches] == [900, 900]


def test_split_keeps_input_order_rather_than_asset_id_order():
    assets = [image(3), image(1, width=20, height=20), image(2)]
    result = plan(assets, budget=1_600)
    assert [b.positions for b in result.batches] == [(0,), (1,), (2,)]


def test_split_is_deterministic():
    assets = [image(i, width=10 + i, height=10) for i in range(20)]
    first = plan(assets, budget=2_000, overhead=50)
    second = plan(list(assets), budget=2_000, overhead=50)
    assert first == second
    assert first.to_json() == second.to_json()


def test_mixed_modalities_split():
    assets = [
        image(0),  # 400
        manifest(1, "application/dicom", width=10, height=10, frames=2),  # 400
        manifest(2, "audio/wav", duration_seconds=1 / 64),  # 250 frames * 4 = 1000
        image(3),  # 400
    ]
    result = plan(assets, budget=2_000)
    assert result.outcome is BatchOutcome.SPLIT
    assert [b.positions for b in result.batches] == [(0, 1, 2), (3,)]
    assert [b.estimated_bytes for b in result.batches] == [1_800, 400]


# --- input validation ---------------------------------------------------------


@pytest.mark.parametrize("value", [True, False, -1, 1.0, math.nan, math.inf, "1"])
@pytest.mark.parametrize("name", ["budget_bytes", "overhead_bytes"])
def test_budget_inputs_reject_non_integers(name, value):
    kwargs = {"budget_bytes": 1_000, "overhead_bytes": 0, name: value}
    with pytest.raises(BatchMemoryError):
        plan_batch_memory([image()], FULL, **kwargs)


@pytest.mark.parametrize("name", ["budget_bytes", "overhead_bytes"])
def test_budget_inputs_are_bounded_by_the_ceiling(name):
    kwargs = {"budget_bytes": 1_000, "overhead_bytes": 0, name: SATURATION_CEILING + 1}
    with pytest.raises(BatchMemoryError):
        plan_batch_memory([image()], FULL, **kwargs)


def test_budget_and_overhead_have_no_defaults():
    with pytest.raises(TypeError):
        plan_batch_memory([image()], FULL)  # type: ignore[call-arg]


@pytest.mark.parametrize(
    "field_name",
    [
        "image_bytes_per_pixel",
        "dicom_bytes_per_pixel",
        "audio_sample_rate_hz",
        "audio_channels",
        "audio_bytes_per_sample",
    ],
)
@pytest.mark.parametrize("value", [True, 0, -1, 2.0, math.nan, math.inf])
def test_policy_factors_reject_invalid_values(field_name, value):
    with pytest.raises(BatchMemoryError):
        MemoryEstimationPolicy(**{field_name: value})


@pytest.mark.parametrize(
    ("field_name", "maximum"),
    [
        ("image_bytes_per_pixel", MAX_BYTES_PER_UNIT),
        ("audio_sample_rate_hz", MAX_AUDIO_SAMPLE_RATE_HZ),
        ("audio_channels", MAX_AUDIO_CHANNELS),
    ],
)
def test_policy_factor_bounds_are_inclusive(field_name, maximum):
    MemoryEstimationPolicy(**{field_name: maximum})
    with pytest.raises(BatchMemoryError):
        MemoryEstimationPolicy(**{field_name: maximum + 1})


def test_policy_is_immutable():
    with pytest.raises(dataclasses.FrozenInstanceError):
        FULL.image_bytes_per_pixel = 8  # type: ignore[misc]


def test_rejects_empty_oversized_and_non_manifest_batches():
    with pytest.raises(BatchMemoryError):
        plan([])
    with pytest.raises(BatchMemoryError):
        plan([image()] * (MAX_BATCH_ASSETS + 1))
    with pytest.raises(TypeError):
        plan([image().to_dict()])
    with pytest.raises(TypeError):
        plan(image())
    with pytest.raises(TypeError):
        plan_batch_memory([image()], {}, budget_bytes=1, overhead_bytes=0)


# --- content-free output ------------------------------------------------------


def test_plan_output_is_content_free_and_stable():
    asset = manifest(7, "image/png", width=10, height=10)
    result = plan([asset, manifest(8, "application/pdf", pages=2)])
    payload = result.to_json()
    assert asset.asset_id not in payload
    assert asset.sha256 not in payload
    assert json.loads(payload) == {
        "version": 1,
        "outcome": "reject",
        "budget_bytes": 10_000,
        "overhead_bytes": 0,
        "assets": [
            {
                "position": 0,
                "modality": "image",
                "estimated_bytes": 400,
                "reason_code": None,
                "field_name": None,
            },
            {
                "position": 1,
                "modality": "pdf",
                "estimated_bytes": None,
                "reason_code": "insufficient_metadata",
                "field_name": "width",
            },
        ],
        "batches": [],
    }
    reasons = {a.reason_code for a in result.assets if a.reason_code}
    assert reasons <= BATCH_MEMORY_REASON_CODES
