"""Synthetic tile completeness, exact geometry and protected feature aggregation."""

import json
from dataclasses import replace
from fractions import Fraction

import pytest

from openmed.multimodal.vision.pathology.slide_tiler import (
    PyramidLevel,
    TilePolicy,
    TissueMask,
    plan_slide_tiles,
)
from openmed.multimodal.vision.pathology.tile_provenance import (
    TileFeatures,
    TileProvenanceError,
    TileTransform,
    TransformKind,
    aggregate_tile_features,
)


def policy(**changes):
    return replace(
        TilePolicy(
            width=4,
            height=3,
            overlap_x=0,
            overlap_y=0,
            working_bytes_per_pixel=12,
            max_inflight_tiles=2,
            budget_bytes=4096,
            reserved_bytes=128,
            min_tissue_percent=25,
        ),
        **changes,
    )


def plan(width=8, height=6, *, levels=None, mask=None, settings=None):
    levels = levels or (
        PyramidLevel(index=0, width=width, height=height, scale_numerator=1),
    )
    return plan_slide_tiles(
        "a" * 64, width, height, levels, settings or policy(), tissue_mask=mask
    )


def observations(source, *, transforms=()):
    return tuple(
        TileFeatures(
            tile_ref=tile.tile_ref,
            plan_digest=source.digest,
            source_digest=source.source_digest,
            level=tile.level,
            native_bounds=tile.bounds,
            source_bounds=tile.source_bounds,
            processed_input_digest=f"{index + 1:064x}",
            provider_digest="b" * 64,
            feature_space_digest="c" * 64,
            transforms=transforms,
            features=(float(index + 1), float(2 * (index + 1))),
        )
        for index, tile in enumerate(source.selected_tiles)
    )


def test_complete_aggregation_computes_features_and_preserves_every_tile():
    source = plan()
    results = observations(source)
    aggregated = aggregate_tile_features(source, results)
    assert aggregated.features == (2.5, 5.0)
    report = json.loads(aggregated.to_json())
    assert report["review_required"] is True
    assert report["uncertain_selection_count"] == 4
    assert report["method"] == "equal_weight_tile_mean"
    assert [item["tile_ref"] for item in report["tiles"]] == [
        tile.tile_ref for tile in source.selected_tiles
    ]
    for tile, item in zip(source.selected_tiles, report["tiles"]):
        assert item["selection"]["bounds"] == list(tile.bounds)
        assert item["evidence"]["region"] == list(tile.source_bounds)
        assert item["evidence"]["source_digest"] == source.source_digest
    assert report["plan"]["source_edge_gaps"] == [
        {"level": 0, "right_pixels": 0, "bottom_pixels": 0}
    ]


def test_arrival_order_does_not_change_reduction_or_lineage():
    source = plan()
    results = observations(source)
    first = aggregate_tile_features(source, results)
    second = aggregate_tile_features(source, tuple(reversed(results)))
    assert first == second


def test_mask_selection_evidence_and_skipped_tiles_survive_aggregation():
    mask = TissueMask(source_digest="a" * 64, width=3, height=1, cells=bytes([1, 0, 2]))
    source = plan(12, 3, mask=mask)
    result = aggregate_tile_features(source, observations(source), tissue_mask=mask)
    report = json.loads(result.to_json())
    assert result.features == (1.5, 3)
    assert report["plan"]["mask_digest"] == mask.digest
    assert [item["decision"] for item in report["plan"]["tiles"]] == [
        "keep_tissue_candidate",
        "skip_background_candidate",
        "keep_uncertain_coverage",
    ]
    assert report["uncertain_selection_count"] == 1
    assert report["tiles"][0]["selection"]["tissue_cells"] == 1
    assert report["tiles"][1]["selection"]["unknown_cells"] == 1
    with pytest.raises(TileProvenanceError, match="selection evidence"):
        aggregate_tile_features(source, observations(source))
    other_mask = replace(mask, cells=bytes([1, 1, 2]))
    with pytest.raises(TileProvenanceError, match="selection evidence"):
        aggregate_tile_features(source, observations(source), tissue_mask=other_mask)


def test_duplicate_missing_and_unknown_tiles_never_produce_partial_aggregation():
    source = plan()
    results = observations(source)
    for malformed in (
        results[:-1],
        results + (results[0],),
        (replace(results[0], tile_ref="d" * 64),) + results[1:],
        (),
    ):
        with pytest.raises(TileProvenanceError):
            aggregate_tile_features(source, malformed)


def test_skipped_tiles_cannot_be_added_back_as_inferred_tissue():
    mask = TissueMask(source_digest="a" * 64, width=2, height=1, cells=bytes([1, 0]))
    source = plan(8, 3, mask=mask)
    accepted = observations(source)
    skipped = source.tiles[1]
    extra = replace(
        accepted[0],
        tile_ref=skipped.tile_ref,
        native_bounds=skipped.bounds,
        source_bounds=skipped.source_bounds,
    )
    with pytest.raises(TileProvenanceError, match="selected tile set"):
        aggregate_tile_features(source, accepted + (extra,), tissue_mask=mask)


@pytest.mark.parametrize(
    "change",
    [
        {"source_digest": "d" * 64},
        {"plan_digest": "d" * 64},
        {"level": 1},
        {"native_bounds": (1, 0, 5, 3)},
        {"source_bounds": (1, 0, 5, 3)},
    ],
)
def test_stale_source_level_or_spatial_declarations_are_rejected(change):
    source = plan()
    results = observations(source)
    with pytest.raises(TileProvenanceError, match="source geometry and plan"):
        aggregate_tile_features(source, (replace(results[0], **change),) + results[1:])


def test_forged_plan_geometry_and_mask_decisions_are_recomputed():
    source = plan()
    altered = replace(
        source,
        tiles=(replace(source.tiles[0], source_bounds=(1, 0, 4, 3)),)
        + source.tiles[1:],
    )
    with pytest.raises(TileProvenanceError, match="selection evidence"):
        aggregate_tile_features(altered, observations(altered))
    for changed in (
        replace(source, peak_working_bytes=1),
        replace(source, policy=replace(source.policy, min_tissue_percent=50)),
    ):
        with pytest.raises(TileProvenanceError):
            aggregate_tile_features(changed, observations(source))


def test_no_selected_tiles_cannot_fabricate_empty_features():
    mask = TissueMask(source_digest="a" * 64, width=1, height=1, cells=b"\x00")
    source = plan(4, 3, mask=mask)
    assert source.selected_tiles == ()
    with pytest.raises(TileProvenanceError):
        aggregate_tile_features(source, (), tissue_mask=mask)


def test_crop_resize_rotation_composition_retains_exact_level_zero_coordinates():
    source = plan(
        8, 6, levels=(PyramidLevel(index=1, width=4, height=3, scale_numerator=2),)
    )
    transforms = (
        TileTransform(
            kind=TransformKind.CROP,
            input_size=(4, 3),
            output_size=(3, 2),
            crop_bounds=(1, 0, 4, 2),
        ),
        TileTransform(kind=TransformKind.RESIZE, input_size=(3, 2), output_size=(6, 4)),
        TileTransform(
            kind=TransformKind.ROTATE,
            input_size=(6, 4),
            output_size=(4, 6),
            quarter_turns=1,
        ),
    )
    report = json.loads(
        aggregate_tile_features(
            source, observations(source, transforms=transforms)
        ).to_json()
    )
    item = report["tiles"][0]
    corners = [
        [Fraction(*coordinate) for coordinate in point]
        for point in item["projected_source_corners"]
    ]
    assert corners == [[2, 4], [2, 0], [8, 0], [8, 4]]
    assert item["output_size"] == [4, 6]
    assert item["evidence"]["region"] == [0, 0, 8, 6]
    assert item["native_bounds"] == [0, 0, 4, 3]
    assert len(item["transforms"]) == 3
    assert not item["source_edge_clipped"]


@pytest.mark.parametrize(
    ("kind", "turns", "size", "expected"),
    [
        (TransformKind.FLIP_HORIZONTAL, None, (4, 3), [[4, 0], [0, 0], [0, 3], [4, 3]]),
        (TransformKind.FLIP_VERTICAL, None, (4, 3), [[0, 3], [4, 3], [4, 0], [0, 0]]),
        (TransformKind.ROTATE, 2, (4, 3), [[4, 3], [0, 3], [0, 0], [4, 0]]),
        (TransformKind.ROTATE, 3, (3, 4), [[4, 0], [4, 3], [0, 3], [0, 0]]),
    ],
)
def test_flip_and_rotation_orientation_is_preserved(kind, turns, size, expected):
    source = plan(4, 3)
    transform = TileTransform(
        kind=kind, input_size=(4, 3), output_size=size, quarter_turns=turns
    )
    item = json.loads(
        aggregate_tile_features(
            source, observations(source, transforms=(transform,))
        ).to_json()
    )["tiles"][0]
    assert [
        [Fraction(*coordinate) for coordinate in point]
        for point in item["projected_source_corners"]
    ] == expected


def test_transform_chains_cannot_skip_or_misdeclare_intermediate_dimensions():
    source = plan(4, 3)
    first = TileTransform(
        kind=TransformKind.RESIZE, input_size=(4, 3), output_size=(8, 6)
    )
    wrong = TileTransform(
        kind=TransformKind.RESIZE, input_size=(4, 3), output_size=(2, 2)
    )
    with pytest.raises(TileProvenanceError, match="inconsistent dimensions"):
        aggregate_tile_features(source, observations(source, transforms=(first, wrong)))
    with pytest.raises(TileProvenanceError):
        TileTransform(
            kind=TransformKind.CROP,
            input_size=(4, 3),
            output_size=(4, 3),
            crop_bounds=(1, 0, 5, 3),
        )
    with pytest.raises(TileProvenanceError):
        TileTransform(
            kind=TransformKind.ROTATE,
            input_size=(4, 3),
            output_size=(4, 3),
            quarter_turns=1,
        )


def test_rational_level_mapping_preserves_clipped_edges_and_unexamined_gaps():
    source = plan(
        9, 7, levels=(PyramidLevel(index=1, width=5, height=4, scale_numerator=2),)
    )
    report = json.loads(aggregate_tile_features(source, observations(source)).to_json())
    last = report["tiles"][-1]
    assert last["evidence"]["region"] == [8, 6, 9, 7]
    assert last["source_edge_clipped"]
    assert last["projected_source_corners"][2] == [[10, 1], [8, 1]]
    short = plan(
        11,
        6,
        levels=(
            PyramidLevel(
                index=1, width=4, height=2, scale_numerator=5, scale_denominator=2
            ),
        ),
    )
    report = json.loads(aggregate_tile_features(short, observations(short)).to_json())
    assert report["plan"]["source_edge_gaps"] == [
        {"level": 1, "right_pixels": 1, "bottom_pixels": 1}
    ]
    assert report["tiles"][0]["output_to_level_zero"][0] == [5, 2]


def test_provider_feature_space_and_dimension_mixing_is_rejected():
    source = plan()
    results = observations(source)
    for change in (
        {"features": (1,)},
        {"provider_digest": "e" * 64},
        {"feature_space_digest": "e" * 64},
    ):
        with pytest.raises(TileProvenanceError):
            aggregate_tile_features(
                source, (replace(results[0], **change),) + results[1:]
            )


def test_tile_identity_and_result_identity_have_distinct_stability():
    source = plan(4, 3)
    original = observations(source)[0]
    transform = TileTransform(
        kind=TransformKind.FLIP_HORIZONTAL, input_size=(4, 3), output_size=(4, 3)
    )
    for changed in (
        replace(original, features=(3, 4)),
        replace(original, processed_input_digest="f" * 64),
        replace(original, transforms=(transform,)),
    ):
        assert changed.tile_ref == original.tile_ref
        assert changed.digest != original.digest
    other = plan_slide_tiles("b" * 64, 4, 3, source.levels, source.policy)
    assert other.tiles[0].tile_ref != original.tile_ref
    multilevel = plan(
        4,
        4,
        levels=(
            PyramidLevel(index=0, width=4, height=4, scale_numerator=1),
            PyramidLevel(index=1, width=2, height=2, scale_numerator=2),
        ),
    )
    assert len({tile.tile_ref for tile in multilevel.tiles}) == len(multilevel.tiles)


def test_feature_payloads_stay_out_of_reports_representations_and_errors():
    source = plan(4, 3)
    observed = replace(observations(source)[0], features=(987654.321, -765432.123))
    result = aggregate_tile_features(source, (observed,))
    assert result.features == observed.features
    for value in observed.features:
        assert (
            str(value)
            not in repr(observed)
            + repr(result)
            + json.dumps(observed.to_report())
            + result.to_json()
        )
    for features in (
        (float("nan"),),
        (float("inf"),),
        (True,),
        (1e101,),
        (10**1000,),
        (),
        (0,) * 4097,
    ):
        with pytest.raises(TileProvenanceError) as raised:
            replace(observed, features=features)
        assert str(raised.value) == "invalid tile aggregation provenance"


def test_feature_and_transform_work_bounds_fail_closed():
    source = plan(36, 24)
    results = tuple(
        replace(item, features=(1,) * 4096) for item in observations(source)
    )
    with pytest.raises(TileProvenanceError):
        aggregate_tile_features(source, results)
    transform = TileTransform(
        kind=TransformKind.FLIP_HORIZONTAL, input_size=(4, 3), output_size=(4, 3)
    )
    with pytest.raises(TileProvenanceError):
        replace(observations(plan(4, 3))[0], transforms=(transform,) * 17)


def test_equal_weight_reduction_is_finite_and_preserves_cancellation():
    source = plan()
    results = observations(source)
    vectors = ((1e100,), (1,), (-1e100,), (3,))
    results = tuple(
        replace(item, features=vector) for item, vector in zip(results, vectors)
    )
    assert aggregate_tile_features(source, results).features == (1,)
