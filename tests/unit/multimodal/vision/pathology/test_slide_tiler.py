"""Multiresolution coverage, conservative filtering and bounded tile work."""

import json
import socket
from dataclasses import replace

import pytest

from openmed.multimodal.vision.pathology import (
    PyramidLevel,
    SlideTileError,
    TileDecision,
    TilePolicy,
    TissueMask,
    mask_from_rgb,
    plan_slide_tiles,
)

SOURCE = "a" * 64


@pytest.fixture
def policy():
    return TilePolicy(
        width=4,
        height=3,
        overlap_x=1,
        overlap_y=1,
        working_bytes_per_pixel=12,
        max_inflight_tiles=2,
        budget_bytes=4096,
        reserved_bytes=128,
        min_tissue_percent=25,
    )


def level(width, height, index=0, numerator=1, denominator=1):
    return PyramidLevel(
        index=index,
        width=width,
        height=height,
        scale_numerator=numerator,
        scale_denominator=denominator,
    )


def mask(width, height, cells, source=SOURCE):
    return TissueMask(
        source_digest=source, width=width, height=height, cells=bytes(cells)
    )


def test_overlap_covers_native_geometry_with_no_redundant_terminal_tiles(policy):
    plan = plan_slide_tiles(SOURCE, 9, 7, (level(9, 7),), policy)
    assert len(plan.tiles) == 9
    assert plan.tiles[0].bounds == (0, 0, 4, 3)
    assert plan.tiles[-1].bounds == (6, 4, 9, 7)
    assert all(tile.bounds == tile.source_bounds for tile in plan.tiles)
    for y in range(7):
        for x in range(9):
            assert any(
                t.bounds[0] <= x < t.bounds[2] and t.bounds[1] <= y < t.bounds[3]
                for t in plan.tiles
            )
    assert all(t.decision is TileDecision.REVIEW for t in plan.tiles)
    assert all(t.sampled_cells == 0 for t in plan.tiles)


def test_multiresolution_order_ids_and_json_are_deterministic_without_network(
    policy, monkeypatch
):
    def deny(*args, **kwargs):
        pytest.fail("unexpected network")

    monkeypatch.setattr(socket, "socket", deny)
    levels = (level(9, 7), level(5, 4, 1, 2))
    first = plan_slide_tiles(SOURCE, 9, 7, levels, policy)
    second = plan_slide_tiles(SOURCE, 9, 7, levels[::-1], policy)
    assert first.to_json() == second.to_json() and first.digest == second.digest
    assert len({t.tile_ref for t in first.tiles}) == len(first.tiles)
    assert first.levels == levels
    assert first.tiles[-1].source_bounds == (6, 4, 9, 7)
    changed = plan_slide_tiles("b" * 64, 9, 7, levels, policy)
    assert changed.tiles[0].tile_ref != first.tiles[0].tile_ref


def test_fractional_native_scale_uses_outward_rounding_and_reports_source_edge_gaps(
    policy,
):
    policy = replace(policy, width=1, height=1, overlap_x=0, overlap_y=0)
    plan = plan_slide_tiles(SOURCE, 11, 6, (level(4, 2, 1, 5, 2),), policy)
    assert plan.tiles[1].bounds == (1, 0, 2, 1)
    assert plan.tiles[1].source_bounds == (2, 0, 5, 3)
    assert plan.tiles[-1].source_bounds == (7, 2, 10, 5)
    assert plan.to_dict()["source_edge_gaps"] == [
        {"level": 1, "right_pixels": 1, "bottom_pixels": 1}
    ]


def test_color_mask_preserves_uncertain_dark_regions_without_retaining_rgb():
    rgb = bytes([250, 250, 250, 160, 50, 120, 20, 20, 20])
    tissue = mask_from_rgb(SOURCE, 3, 1, rgb, white_floor=240, min_chroma=30)
    assert tissue.cells == b"\x00\x01\x02"
    assert "cells=" not in repr(tissue)
    assert rgb.hex() not in repr(tissue)
    assert tissue.digest == mask(3, 1, [0, 1, 2]).digest
    assert tissue.digest != mask(1, 3, [0, 1, 2]).digest


def test_tissue_filter_retains_unknowns_and_records_background_exclusions(policy):
    policy = replace(policy, width=4, height=4, overlap_x=0, overlap_y=0)
    tissue = mask(6, 2, [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 2])
    plan = plan_slide_tiles(SOURCE, 12, 4, (level(12, 4),), policy, tissue_mask=tissue)
    assert [t.decision for t in plan.tiles] == [
        TileDecision.KEEP,
        TileDecision.SKIP,
        TileDecision.REVIEW,
    ]
    assert [(t.sampled_cells, t.tissue_cells, t.unknown_cells) for t in plan.tiles] == [
        (4, 2, 0),
        (4, 0, 0),
        (4, 0, 1),
    ]
    assert plan.selected_tiles == (plan.tiles[0], plan.tiles[2])
    report = plan.to_dict()
    assert len(report["tiles"]) == 3 and report["selected_count"] == 2
    assert report["mask_digest"] == tissue.digest
    assert "cells" not in report and "data" not in report


def test_tissue_percent_boundary_is_inclusive_and_zero_can_disable_background_filter(
    policy,
):
    policy = replace(policy, width=4, height=4, overlap_x=0, overlap_y=0)
    tissue = mask(2, 2, [1, 0, 0, 0])
    assert (
        plan_slide_tiles(SOURCE, 4, 4, (level(4, 4),), policy, tissue_mask=tissue)
        .tiles[0]
        .decision
        is TileDecision.KEEP
    )
    assert (
        plan_slide_tiles(
            SOURCE,
            4,
            4,
            (level(4, 4),),
            replace(policy, min_tissue_percent=26),
            tissue_mask=tissue,
        )
        .tiles[0]
        .decision
        is TileDecision.SKIP
    )
    background = mask(1, 1, [0])
    assert (
        plan_slide_tiles(
            SOURCE,
            4,
            4,
            (level(4, 4),),
            replace(policy, min_tissue_percent=0),
            tissue_mask=background,
        )
        .tiles[0]
        .decision
        is TileDecision.KEEP
    )


def test_mask_cell_centers_use_half_open_bounds_and_unsampled_tiles_are_retained(
    policy,
):
    policy = replace(policy, width=2, height=2, overlap_x=0, overlap_y=0)
    tissue = mask(1, 1, [0])
    plan = plan_slide_tiles(SOURCE, 4, 4, (level(4, 4),), policy, tissue_mask=tissue)
    assert [t.sampled_cells for t in plan.tiles] == [0, 0, 0, 1]
    assert [t.decision for t in plan.tiles] == [TileDecision.REVIEW] * 3 + [
        TileDecision.SKIP
    ]


def test_declared_concurrent_buffers_reject_over_budget_without_partial_plan(policy):
    policy = replace(
        policy, width=4, height=4, overlap_x=0, overlap_y=0, reserved_bytes=100
    )
    required = 100 + 2 * 4 * 4 * 12
    exact = plan_slide_tiles(
        SOURCE, 8, 4, (level(8, 4),), replace(policy, budget_bytes=required)
    )
    assert exact.peak_working_bytes == required
    with pytest.raises(SlideTileError):
        plan_slide_tiles(
            SOURCE, 8, 4, (level(8, 4),), replace(policy, budget_bytes=required - 1)
        )
    one = plan_slide_tiles(
        SOURCE,
        8,
        4,
        (level(8, 4),),
        replace(policy, max_inflight_tiles=1, budget_bytes=100 + 4 * 4 * 12),
    )
    assert [len(batch) for batch in one.iter_batches()] == [1, 1]
    assert tuple(t for batch in one.iter_batches() for t in batch) == one.selected_tiles


def test_all_background_plan_needs_no_tile_buffers_and_emits_no_batches(policy):
    policy = replace(
        policy,
        width=4,
        height=4,
        overlap_x=0,
        overlap_y=0,
        reserved_bytes=8,
        budget_bytes=9,
    )
    plan = plan_slide_tiles(
        SOURCE, 4, 4, (level(4, 4),), policy, tissue_mask=mask(1, 1, [0])
    )
    assert plan.peak_working_bytes == 8
    assert plan.selected_tiles == () and list(plan.iter_batches()) == []
    assert len(plan.tiles) == 1


def test_gigapixel_plan_never_materializes_source_pixels(policy):
    policy = replace(
        policy,
        width=1024,
        height=1024,
        overlap_x=32,
        overlap_y=32,
        working_bytes_per_pixel=8,
        budget_bytes=20 * 1024 * 1024,
    )
    plan = plan_slide_tiles(
        SOURCE, 40000, 40000, (level(40000, 40000), level(10000, 10000, 1, 4)), policy
    )
    assert len(plan.tiles) < 4096
    assert plan.peak_working_bytes <= policy.budget_bytes
    assert plan.tiles[-1].source_bounds[2:] == (40000, 40000)


def test_candidate_limit_precedes_mask_index_or_tile_enumeration(policy, monkeypatch):
    from openmed.multimodal.vision.pathology import slide_tiler

    def fail(*args, **kwargs):
        pytest.fail("mask allocation preceded candidate admission")

    monkeypatch.setattr(slide_tiler, "_MaskIndex", fail)
    with pytest.raises(SlideTileError):
        plan_slide_tiles(
            SOURCE,
            1_000_000,
            1_000_000,
            (level(1_000_000, 1_000_000),),
            policy,
            tissue_mask=mask(1, 1, [1]),
        )


def test_prefix_counts_match_cell_centers_at_fractional_source_boundaries(policy):
    tissue = mask(5, 3, [i % 3 for i in range(15)])
    plan = plan_slide_tiles(
        SOURCE, 11, 7, (level(11, 7), level(7, 5, 1, 3, 2)), policy, tissue_mask=tissue
    )
    for tile in plan.tiles:
        x0, y0, x1, y1 = tile.source_bounds
        cells = [
            tissue.cells[y * 5 + x]
            for y in range(3)
            for x in range(5)
            if 2 * x0 * 5 <= (2 * x + 1) * 11 < 2 * x1 * 5
            and 2 * y0 * 3 <= (2 * y + 1) * 7 < 2 * y1 * 3
        ]
        assert (tile.sampled_cells, tile.tissue_cells, tile.unknown_cells) == (
            len(cells),
            cells.count(1),
            cells.count(2),
        )


@pytest.mark.parametrize(
    "levels",
    [
        (),
        (level(9, 7), level(9, 7)),
        (level(8, 7),),
        (level(5, 4, 1, 2), level(6, 5, 2, 3, 2)),
        (level(8, 6, 1, 11, 10), level(9, 7, 2, 10, 9)),
    ],
)
def test_inconsistent_pyramid_metadata_fails_closed(policy, levels):
    with pytest.raises(SlideTileError, match="^invalid slide tile plan$"):
        plan_slide_tiles(SOURCE, 9, 7, levels, policy)


@pytest.mark.parametrize(
    "change",
    [
        {"width": 0},
        {"height": True},
        {"overlap_x": 4},
        {"working_bytes_per_pixel": 0},
        {"max_inflight_tiles": 0},
        {"budget_bytes": 128},
        {"min_tissue_percent": 101},
    ],
)
def test_invalid_policy_never_echoes_inputs(policy, change):
    with pytest.raises(SlideTileError, match="^invalid slide tile plan$"):
        replace(policy, **change)


def test_malformed_masks_and_cross_source_mask_fail_without_echo(policy):
    for cells in (b"\x03", b"SYNTHETIC-PRIVATE"):
        with pytest.raises(SlideTileError, match="^invalid slide tile plan$"):
            mask(1, 1, cells)
    with pytest.raises(SlideTileError):
        plan_slide_tiles(
            SOURCE, 9, 7, (level(9, 7),), policy, tissue_mask=mask(1, 1, [1], "b" * 64)
        )
    with pytest.raises(SlideTileError):
        mask_from_rgb(
            SOURCE, 1, 1, b"SYNTHETIC-PRIVATE", white_floor=240, min_chroma=30
        )


def test_report_copies_do_not_mutate_plan_and_contain_no_payloads(policy, caplog):
    plan = plan_slide_tiles(
        SOURCE, 9, 7, (level(9, 7),), policy, tissue_mask=mask(1, 1, [2])
    )
    original = plan.to_json()
    report = plan.to_dict()
    report["tiles"][0]["bounds"] = (99, 99, 100, 100)
    assert plan.to_json() == original
    assert json.loads(original)["review_required"] is True
    assert all(
        field not in original
        for field in ('"path":', '"pixels":', '"cells":', '"data":')
    )
    assert caplog.records == []
