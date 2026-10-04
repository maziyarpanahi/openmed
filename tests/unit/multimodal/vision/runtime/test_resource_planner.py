"""Tests for the on-device vision memory planner contract."""

from __future__ import annotations

import re
from dataclasses import FrozenInstanceError

import pytest

from openmed.multimodal.vision import runtime as runtime_module
from openmed.multimodal.vision.runtime import (
    CLOUD_FALLBACK_ALLOWED,
    MAX_ACTIVATION_BYTES_PER_PIXEL,
    MAX_CACHE_BYTES_PER_TILE,
    MAX_DEVICE_TIERS,
    MAX_FRAMES,
    MAX_MEMORY_BYTES,
    MAX_MODEL_PARAMETERS,
    MAX_OUTPUT_BYTES,
    MAX_PIXEL_SIDE,
    MAX_TILE_SIZE,
    MAX_WEIGHT_BYTES_PER_PARAMETER,
    RESOURCE_PLAN_REASON_CODES,
    RESOURCE_PLAN_VERSION,
    VLM_TIER_DESKTOP_V1,
    VLM_TIER_MOBILE_V1,
    DeviceTier,
    PreprocessingPlan,
    ResourcePlanOutcome,
    TierCandidate,
    VLMExecutionPlan,
    VLMExecutionPolicy,
    VLMResourceEstimate,
    VLMResourcePlannerError,
    plan_vlm_execution,
)
from openmed.multimodal.vision.runtime import resource_planner as planner

PREPROCESSING = PreprocessingPlan(width=64, height=32, tile_size=16)
GIB = 1 << 30


def _policy(**overrides: object) -> VLMExecutionPolicy:
    values: dict[str, object] = {
        "tiers": (VLM_TIER_MOBILE_V1,),
        "parameter_count": 1000,
        "weight_bytes_per_parameter": 4,
        "activation_bytes_per_pixel": 4,
        "cache_bytes_per_tile": 16,
        "output_bytes": 256,
        "runtime_overhead_bytes": 64,
    }
    values.update(overrides)
    return VLMExecutionPolicy(**values)  # type: ignore[arg-type]


def test_reference_tiers_describe_declared_budgets() -> None:
    assert VLM_TIER_MOBILE_V1.name == "mobile.v1"
    assert VLM_TIER_MOBILE_V1.memory_budget_bytes == 6 * GIB
    assert VLM_TIER_MOBILE_V1.reserved_bytes == GIB
    assert VLM_TIER_MOBILE_V1.available_bytes == 5 * GIB
    assert VLM_TIER_DESKTOP_V1.name == "desktop.v1"
    assert VLM_TIER_DESKTOP_V1.available_bytes == 28 * GIB
    assert VLM_TIER_DESKTOP_V1.to_dict() == {
        "name": "desktop.v1",
        "memory_budget_bytes": 32 * GIB,
        "reserved_bytes": 4 * GIB,
        "available_bytes": 28 * GIB,
    }


def test_tier_already_available_bytes_when_nothing_is_reserved() -> None:
    tier = DeviceTier(name="edge.v1", memory_budget_bytes=4096)
    assert tier.reserved_bytes == 0
    assert tier.available_bytes == 4096


def test_preprocessing_derives_tile_geometry() -> None:
    assert PREPROCESSING.stride == 16
    assert PREPROCESSING.tiles_x == 4
    assert PREPROCESSING.tiles_y == 2
    assert PREPROCESSING.tile_count == 8
    assert PREPROCESSING.to_dict() == {
        "width": 64,
        "height": 32,
        "tile_size": 16,
        "tile_overlap": 0,
        "frames": 1,
        "tiles_x": 4,
        "tiles_y": 2,
        "tile_count": 8,
    }


@pytest.mark.parametrize(
    ("width", "height", "tile_size", "overlap", "expected"),
    [
        (64, 32, 16, 0, (4, 2, 8)),
        (64, 32, 16, 8, (8, 4, 32)),
        (17, 17, 16, 0, (2, 2, 4)),
        (3, 3, 64, 0, (1, 1, 1)),
        (64, 64, 64, 0, (1, 1, 1)),
        (65, 1, 64, 32, (3, 1, 3)),
    ],
)
def test_tile_counts_ceiling_divide_by_stride(
    width: int,
    height: int,
    tile_size: int,
    overlap: int,
    expected: tuple[int, int, int],
) -> None:
    plan = PreprocessingPlan(
        width=width, height=height, tile_size=tile_size, tile_overlap=overlap
    )
    assert (plan.tiles_x, plan.tiles_y, plan.tile_count) == expected


def test_safe_plan_charges_every_documented_component() -> None:
    plan = plan_vlm_execution(_policy(), PREPROCESSING)

    assert plan.outcome is ResourcePlanOutcome.SAFE
    assert plan.tier_name == "mobile.v1"
    assert plan.tier_available_bytes == 5 * GIB
    assert plan.estimate == VLMResourceEstimate(
        model_bytes=4000,
        peak_activation_bytes=8192 + 1024,
        cache_bytes=8 * 16,
        output_bytes=256,
        runtime_overhead_bytes=64,
        total_bytes=4000 + 9216 + 128 + 256 + 64,
        reason_code=None,
        field_name=None,
    )
    assert plan.estimate.total_bytes == 13664
    assert plan.candidates == (
        TierCandidate(name="mobile.v1", available_bytes=5 * GIB, fits=True),
    )


def test_overlapping_tiles_repeat_cache_and_activation_work() -> None:
    overlapping = PreprocessingPlan(width=64, height=32, tile_size=16, tile_overlap=8)
    plan = plan_vlm_execution(_policy(), overlapping)

    assert plan.estimate.cache_bytes == 32 * 16
    assert plan.estimate.peak_activation_bytes == 8192 + 1024
    assert plan.estimate.total_bytes == 4000 + 9216 + 512 + 256 + 64


def test_multiple_frames_scale_the_frame_stack_and_every_tile() -> None:
    frames = PreprocessingPlan(width=64, height=32, tile_size=16, frames=3)
    plan = plan_vlm_execution(_policy(), frames)

    assert plan.estimate.peak_activation_bytes == (64 * 32 * 3 * 4) + (16 * 16 * 3 * 4)
    assert plan.estimate.cache_bytes == 8 * 16


def test_plan_json_is_compact_and_key_ordered() -> None:
    plan = plan_vlm_execution(_policy(), PREPROCESSING)
    document = plan.to_dict()

    assert list(document) == [
        "version",
        "outcome",
        "tier_name",
        "tier_available_bytes",
        "estimate",
        "candidates",
        "preprocessing",
        "cloud_fallback_allowed",
    ]
    assert document["version"] == RESOURCE_PLAN_VERSION
    assert document["outcome"] == "safe"
    assert document["cloud_fallback_allowed"] is False
    assert list(document["estimate"]) == [
        "model_bytes",
        "peak_activation_bytes",
        "cache_bytes",
        "output_bytes",
        "runtime_overhead_bytes",
        "total_bytes",
        "reason_code",
        "field_name",
    ]
    assert list(document["candidates"][0]) == ["name", "available_bytes", "fits"]
    assert list(document["preprocessing"]) == [
        "width",
        "height",
        "tile_size",
        "tile_overlap",
        "frames",
        "tiles_x",
        "tiles_y",
        "tile_count",
    ]
    payload = plan.to_json()
    assert " " not in payload
    assert '"outcome":"safe"' in payload
    assert '"total_bytes":13664' in payload


def test_plans_are_deterministic_for_identical_inputs() -> None:
    first = plan_vlm_execution(_policy(), PREPROCESSING)
    second = plan_vlm_execution(_policy(), PREPROCESSING)

    assert first.to_dict() == second.to_dict()
    assert first.to_json() == second.to_json()


def test_policy_reports_its_declared_factors() -> None:
    assert _policy().to_dict() == {
        "tiers": [
            {
                "name": "mobile.v1",
                "memory_budget_bytes": 6 * GIB,
                "reserved_bytes": GIB,
                "available_bytes": 5 * GIB,
            }
        ],
        "parameter_count": 1000,
        "weight_bytes_per_parameter": 4,
        "activation_bytes_per_pixel": 4,
        "cache_bytes_per_tile": 16,
        "output_bytes": 256,
        "runtime_overhead_bytes": 64,
    }


def test_smallest_fitting_tier_is_selected() -> None:
    tiers = (
        VLM_TIER_DESKTOP_V1,
        DeviceTier(name="panel.v1", memory_budget_bytes=20000),
        DeviceTier(name="edge.v1", memory_budget_bytes=14000),
    )
    plan = plan_vlm_execution(_policy(tiers=tiers), PREPROCESSING)

    assert plan.outcome is ResourcePlanOutcome.SAFE
    assert plan.tier_name == "edge.v1"
    assert plan.tier_available_bytes == 14000


def test_ties_break_on_tier_name_not_declaration_order() -> None:
    tiers = (
        DeviceTier(name="b.v1", memory_budget_bytes=50000),
        DeviceTier(name="a.v1", memory_budget_bytes=50000),
    )
    plan = plan_vlm_execution(_policy(tiers=tiers), PREPROCESSING)

    assert plan.tier_name == "a.v1"
    assert [candidate.name for candidate in plan.candidates] == ["a.v1", "b.v1"]


def test_candidate_report_ignores_declaration_order() -> None:
    tiers = (
        VLM_TIER_DESKTOP_V1,
        DeviceTier(name="edge.v1", memory_budget_bytes=14000),
        VLM_TIER_MOBILE_V1,
    )
    forward = plan_vlm_execution(_policy(tiers=tiers), PREPROCESSING)
    reverse = plan_vlm_execution(_policy(tiers=tuple(reversed(tiers))), PREPROCESSING)

    assert forward.candidates == reverse.candidates
    assert [candidate.name for candidate in forward.candidates] == [
        "edge.v1",
        "mobile.v1",
        "desktop.v1",
    ]
    assert [candidate.available_bytes for candidate in forward.candidates] == [
        14000,
        5 * GIB,
        28 * GIB,
    ]


@pytest.mark.parametrize("budget", [13664, 13665, 20000])
def test_an_exact_or_larger_budget_is_safe(budget: int) -> None:
    tier = DeviceTier(name="exact.v1", memory_budget_bytes=budget)
    plan = plan_vlm_execution(_policy(tiers=(tier,)), PREPROCESSING)

    assert plan.outcome is ResourcePlanOutcome.SAFE
    assert plan.tier_name == "exact.v1"


def test_one_byte_short_abstains_without_a_remoter_fallback() -> None:
    tier = DeviceTier(name="short.v1", memory_budget_bytes=13663)
    plan = plan_vlm_execution(_policy(tiers=(tier,)), PREPROCESSING)

    assert plan.outcome is ResourcePlanOutcome.ABSTAIN
    assert plan.tier_name is None
    assert plan.tier_available_bytes is None
    assert plan.estimate.reason_code == "memory_budget_exceeded"
    assert plan.estimate.field_name is None
    assert plan.estimate.total_bytes == 13664
    assert plan.candidates == (
        TierCandidate(name="short.v1", available_bytes=13663, fits=False),
    )
    document = plan.to_dict()
    assert document["cloud_fallback_allowed"] is False
    assert {
        "provider",
        "endpoint",
        "region",
        "credential",
        "retry",
    }.isdisjoint(document)


@pytest.mark.parametrize(
    ("field", "expected_field"),
    [
        ("parameter_count", "parameter_count"),
        ("weight_bytes_per_parameter", "weight_bytes_per_parameter"),
        ("activation_bytes_per_pixel", "activation_bytes_per_pixel"),
        ("cache_bytes_per_tile", "cache_bytes_per_tile"),
        ("output_bytes", "output_bytes"),
    ],
)
def test_missing_factor_abstains_as_unevaluable(
    field: str, expected_field: str
) -> None:
    plan = plan_vlm_execution(_policy(**{field: None}), PREPROCESSING)

    assert plan.outcome is ResourcePlanOutcome.ABSTAIN
    assert plan.tier_name is None
    assert plan.estimate.reason_code == "insufficient_metadata"
    assert plan.estimate.field_name == expected_field
    assert plan.estimate.total_bytes is None
    assert plan.estimate.model_bytes is None
    assert plan.estimate.peak_activation_bytes is None
    assert plan.estimate.cache_bytes is None
    assert plan.estimate.output_bytes is None
    assert plan.candidates == ()


def test_missing_factor_wins_over_a_tier_that_could_not_fit() -> None:
    tier = DeviceTier(name="tiny.v1", memory_budget_bytes=1)
    plan = plan_vlm_execution(
        _policy(tiers=(tier,), activation_bytes_per_pixel=None), PREPROCESSING
    )

    assert plan.estimate.reason_code == "insufficient_metadata"
    assert plan.estimate.field_name == "activation_bytes_per_pixel"


STACKED_GEOMETRY = PreprocessingPlan(
    width=MAX_PIXEL_SIDE,
    height=MAX_PIXEL_SIDE,
    tile_size=MAX_TILE_SIZE,
    frames=MAX_FRAMES,
)
UNTILED_GEOMETRY = PreprocessingPlan(
    width=MAX_PIXEL_SIDE, height=MAX_PIXEL_SIDE, tile_size=1
)


@pytest.mark.parametrize(
    ("overrides", "geometry", "expected_field"),
    [
        ({"parameter_count": MAX_MODEL_PARAMETERS}, PREPROCESSING, "model"),
        (
            {
                "parameter_count": 0,
                "activation_bytes_per_pixel": MAX_ACTIVATION_BYTES_PER_PIXEL,
                "cache_bytes_per_tile": 0,
                "output_bytes": 0,
                "runtime_overhead_bytes": 0,
            },
            STACKED_GEOMETRY,
            "peak_activation",
        ),
        (
            {
                "parameter_count": 0,
                "activation_bytes_per_pixel": 1,
                "cache_bytes_per_tile": MAX_CACHE_BYTES_PER_TILE,
                "output_bytes": 0,
                "runtime_overhead_bytes": 0,
            },
            UNTILED_GEOMETRY,
            "cache",
        ),
        (
            {
                "parameter_count": MAX_MODEL_PARAMETERS,
                "weight_bytes_per_parameter": 1,
                "output_bytes": MAX_OUTPUT_BYTES,
                "runtime_overhead_bytes": MAX_MEMORY_BYTES - 1,
            },
            PREPROCESSING,
            "total",
        ),
    ],
)
def test_saturating_estimates_abstain(
    overrides: dict[str, object], geometry: PreprocessingPlan, expected_field: str
) -> None:
    plan = plan_vlm_execution(_policy(**overrides), geometry)

    assert plan.outcome is ResourcePlanOutcome.ABSTAIN
    assert plan.tier_name is None
    assert plan.estimate.reason_code == "estimate_saturated"
    assert plan.estimate.field_name == expected_field
    assert plan.estimate.total_bytes is None
    assert plan.candidates == ()


def test_reason_codes_are_a_closed_set() -> None:
    assert RESOURCE_PLAN_REASON_CODES == frozenset(
        {"insufficient_metadata", "estimate_saturated", "memory_budget_exceeded"}
    )
    assert CLOUD_FALLBACK_ALLOWED is False


@pytest.mark.parametrize(
    "name",
    ["", "Mobile", "MOBILE", "1mobile", "-mobile", "mobile/1", "a b", "a@b", "mobile."],
)
def test_tier_names_must_be_bounded_labels(name: str) -> None:
    with pytest.raises(VLMResourcePlannerError) as caught:
        DeviceTier(name=name, memory_budget_bytes=1024)

    assert str(caught.value) == "device tier name must be a bounded label"


@pytest.mark.parametrize("name", ["mobile.v1", "edge-x.v2", "panel_1", "a"])
def test_tier_names_accept_opaque_labels(name: str) -> None:
    assert DeviceTier(name=name, memory_budget_bytes=1024).name == name


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("memory_budget_bytes", 0),
        ("memory_budget_bytes", -1),
        ("memory_budget_bytes", True),
        ("memory_budget_bytes", 1.5),
        ("memory_budget_bytes", MAX_MEMORY_BYTES + 1),
        ("reserved_bytes", -1),
        ("reserved_bytes", True),
        ("reserved_bytes", 1.5),
    ],
)
def test_tier_budgets_are_bounded_integers(field: str, value: object) -> None:
    kwargs: dict[str, object] = {
        "name": "edge.v1",
        "memory_budget_bytes": 1024,
        "reserved_bytes": 0,
    }
    kwargs[field] = value
    if field == "reserved_bytes" and value in (True, 1.5):
        kwargs["memory_budget_bytes"] = MAX_MEMORY_BYTES

    with pytest.raises(VLMResourcePlannerError) as caught:
        DeviceTier(**kwargs)  # type: ignore[arg-type]

    assert str(caught.value) == f"{field} must be a bounded integer"


def test_reserved_bytes_must_leave_a_usable_budget() -> None:
    with pytest.raises(VLMResourcePlannerError) as caught:
        DeviceTier(name="edge.v1", memory_budget_bytes=1024, reserved_bytes=1024)

    assert str(caught.value) == "reserved_bytes must be lower than memory_budget_bytes"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("width", 0),
        ("width", -1),
        ("width", True),
        ("width", 1.5),
        ("width", MAX_PIXEL_SIDE + 1),
        ("height", 0),
        ("tile_size", 0),
        ("tile_size", MAX_TILE_SIZE + 1),
        ("tile_overlap", -1),
        ("tile_overlap", MAX_TILE_SIZE),
        ("frames", 0),
        ("frames", MAX_FRAMES + 1),
    ],
)
def test_preprocessing_geometry_is_bounded(field: str, value: object) -> None:
    kwargs: dict[str, object] = {
        "width": 64,
        "height": 32,
        "tile_size": 16,
        "tile_overlap": 0,
        "frames": 1,
    }
    kwargs[field] = value

    with pytest.raises(VLMResourcePlannerError) as caught:
        PreprocessingPlan(**kwargs)  # type: ignore[arg-type]

    assert str(caught.value) == f"{field} must be a bounded integer"


@pytest.mark.parametrize("overlap", [16, 17, MAX_TILE_SIZE - 1])
def test_tile_overlap_must_be_lower_than_tile_size(overlap: int) -> None:
    with pytest.raises(VLMResourcePlannerError) as caught:
        PreprocessingPlan(width=64, height=64, tile_size=16, tile_overlap=overlap)

    assert str(caught.value) == "tile_overlap must be lower than tile_size"


@pytest.mark.parametrize(
    "tiers",
    [
        [VLM_TIER_MOBILE_V1],
        (),
        "mobile.v1",
        (VLM_TIER_MOBILE_V1,) * (MAX_DEVICE_TIERS + 1),
        (VLM_TIER_MOBILE_V1, VLM_TIER_MOBILE_V1),
        (VLM_TIER_MOBILE_V1, "mobile.v2"),
    ],
)
def test_registered_tiers_are_unique_and_bounded(tiers: object) -> None:
    with pytest.raises(VLMResourcePlannerError) as caught:
        VLMExecutionPolicy(tiers=tiers)  # type: ignore[arg-type]

    assert str(caught.value) in {
        "tiers must be a tuple of DeviceTier",
        "tiers must hold between 1 and MAX_DEVICE_TIERS device tiers",
        "device tier names must be unique",
    }


def test_the_maximum_number_of_tiers_is_accepted() -> None:
    tiers = tuple(
        DeviceTier(name=f"tier{x}.v1", memory_budget_bytes=1 << 40)
        for x in range(MAX_DEVICE_TIERS)
    )
    assert len(VLMExecutionPolicy(tiers=tiers).tiers) == MAX_DEVICE_TIERS


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("parameter_count", -1),
        ("parameter_count", True),
        ("parameter_count", 1.5),
        ("parameter_count", MAX_MODEL_PARAMETERS + 1),
        ("weight_bytes_per_parameter", 0),
        ("weight_bytes_per_parameter", MAX_WEIGHT_BYTES_PER_PARAMETER + 1),
        ("activation_bytes_per_pixel", 0),
        ("activation_bytes_per_pixel", MAX_ACTIVATION_BYTES_PER_PIXEL + 1),
        ("cache_bytes_per_tile", -1),
        ("cache_bytes_per_tile", MAX_CACHE_BYTES_PER_TILE + 1),
        ("output_bytes", -1),
        ("output_bytes", MAX_OUTPUT_BYTES + 1),
        ("runtime_overhead_bytes", -1),
        ("runtime_overhead_bytes", True),
    ],
)
def test_declared_factors_are_bounded(field: str, value: object) -> None:
    with pytest.raises(VLMResourcePlannerError) as caught:
        _policy(**{field: value})

    assert str(caught.value) == f"{field} must be a bounded integer"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("parameter_count", 0),
        ("weight_bytes_per_parameter", 1),
        ("activation_bytes_per_pixel", 1),
        ("cache_bytes_per_tile", 0),
        ("output_bytes", 0),
        ("runtime_overhead_bytes", 0),
    ],
)
def test_declared_factor_boundaries_are_accepted(field: str, value: int) -> None:
    assert getattr(_policy(**{field: value}), field) == value


@pytest.mark.parametrize(
    ("policy", "preprocessing"),
    [
        ("policy", PREPROCESSING),
        (_policy(), "preprocessing"),
        (None, None),
    ],
)
def test_planner_requires_the_declared_types(
    policy: object, preprocessing: object
) -> None:
    with pytest.raises(TypeError):
        plan_vlm_execution(policy, preprocessing)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("instance", "attribute", "value"),
    [
        (VLM_TIER_MOBILE_V1, "name", "other.v1"),
        (PREPROCESSING, "width", 128),
        (_policy(), "parameter_count", 1),
        (plan_vlm_execution(_policy(), PREPROCESSING), "outcome", None),
        (
            plan_vlm_execution(_policy(), PREPROCESSING).estimate,
            "total_bytes",
            1,
        ),
        (TierCandidate(name="a.v1", available_bytes=1, fits=True), "fits", False),
    ],
)
def test_contracts_are_frozen(instance: object, attribute: str, value: object) -> None:
    with pytest.raises(FrozenInstanceError):
        setattr(instance, attribute, value)


def test_rejected_values_are_never_echoed_in_errors() -> None:
    sentinels = ("SITE-A", "Participant@example.com", "123-45-6789", "clinic/host-7")
    for sentinel in sentinels:
        with pytest.raises(VLMResourcePlannerError) as caught:
            DeviceTier(name=sentinel, memory_budget_bytes=1024)

        message = str(caught.value)
        assert message == "device tier name must be a bounded label"
        assert sentinel not in message
        assert re.search(r"[A-Z]", message) is None


def test_plan_diagnostics_hold_only_declared_metadata() -> None:
    plan = plan_vlm_execution(_policy(), PREPROCESSING)
    document = plan.to_dict()

    assert set(document) == {
        "version",
        "outcome",
        "tier_name",
        "tier_available_bytes",
        "estimate",
        "candidates",
        "preprocessing",
        "cloud_fallback_allowed",
    }
    assert set(document["estimate"]) == {
        "model_bytes",
        "peak_activation_bytes",
        "cache_bytes",
        "output_bytes",
        "runtime_overhead_bytes",
        "total_bytes",
        "reason_code",
        "field_name",
    }
    for value in document["estimate"].values():
        assert isinstance(value, (int, str, type(None)))


def test_runtime_package_reexports_the_planner_surface() -> None:
    for name in planner.__all__:
        assert name in runtime_module.__all__
        assert getattr(runtime_module, name) is getattr(planner, name)
    assert planner.__all__ == runtime_module.__all__


def test_planner_module_declares_no_optional_dependency() -> None:
    source = planner.__file__
    assert source is not None
    with open(source, encoding="utf-8") as handle:
        text = handle.read()

    for module in ("requests", "urllib", "socket", "http.client", "boto3"):
        assert f"import {module}" not in text
    assert "cloud_fallback_allowed" in text
