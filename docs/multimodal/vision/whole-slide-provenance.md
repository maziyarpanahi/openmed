# Whole-slide tile provenance through aggregation

`openmed.multimodal.vision.pathology.tile_provenance` verifies a complete set of tile-feature observations against the original [whole-slide tile plan](whole-slide-tiling.md) before computing an aggregate. It preserves stable tile identities, native and level-zero coordinates, tissue-selection evidence, geometric transform history, provider provenance and protected feature digests. Missing, duplicate, skipped, stale or spatially inconsistent tile references block the entire result.

This local Python layer performs no slide decoding, image processing, model loading, file writes or network calls. It verifies declared lineage and computes a deterministic equal-weight mean of supplied features. It does not establish that an upstream adapter actually processed particular pixels, qualify a model or authorize clinical decisions. It does not add a Swift aggregation UI.

## Source identity and selection evidence

`TileFeatures` binds one protected feature vector to the planner's stable `tile_ref`, the complete plan digest, original slide digest, native pyramid level, native bounds and clipped level-zero source bounds. It also records the processed-input digest, provider/weights/configuration digest, feature-space specification digest and ordered geometric transform history. These digests must come from the real upstream inputs and pinned specifications; the module does not manufacture or authenticate that evidence.

The existing planner derives each tile identity from the slide digest, native level geometry and native bounds. The same source tile keeps its identity when features or preprocessing change. `TileFeatures.digest` separately binds the complete result lineage, including transforms, processed input, provider, feature-space definition and feature values. A different slide or pyramid level has a different source tile identity.

`aggregate_tile_features` reconstructs the plan from its original source geometry, levels, policy and optional tissue mask. It compares that fresh plan with the supplied object before trusting tile IDs, coordinates, selection decisions or memory accounting. A manually altered output dataclass is not authoritative. If the plan used a tissue mask, the original mask must be supplied and must match the plan's mask digest and selection evidence.

The result retains the complete plan, including excluded background candidates, sampled/tissue/unknown mask-cell counts, selection reasons and source-edge gaps. Unknown or unsampled coverage remains review-required. A slide with no selected tiles cannot produce fabricated empty features. A selected tile is not evidence that the entire slide was examined.

## Exact geometric transform history

`TileTransform` describes one closed operation with explicit input/output dimensions. Its coordinate system uses pixel edges rather than interpolation centers.

| Operation | Required declaration |
| --- | --- |
| `CROP` | Bounds inside the input and output dimensions equal to the crop extent. |
| `RESIZE` | Explicit original and resulting dimensions. |
| `ROTATE` | One, two or three clockwise quarter turns and corresponding output dimensions. |
| `FLIP_HORIZONTAL` / `FLIP_VERTICAL` | Identical input/output dimensions. |

Every chain starts at the native tile's width and height. Each operation's input dimensions must equal the previous output dimensions. The module computes and composes exact rational output-to-input maps, then includes the native tile origin and pyramid downsample to obtain level-zero coordinates. Reports retain the operations, final dimensions, affine mapping and projected output corners as `[numerator, denominator]` pairs.

Original visual citations use the unchanged [fused evidence](../fused-evidence-graphs.md) source box, slide digest and level-zero geometry. They are not replaced with coordinates from the transformed image. A cropped feature result still cites its original tile; its transform history explains which portion was used. The citation alone must not be interpreted as proof that every original tile pixel contributed to the feature.

Rounded pyramid dimensions can project an edge slightly beyond the declared level-zero extent. The original planner's citation box remains clipped to the source. The report preserves the exact projected corners and explicitly sets `source_edge_clipped`; it does not silently claim evidence outside the slide. Likewise, source-edge gaps from undersized levels remain in the complete plan. Decoder padding, interpolation and pixel processing are upstream responsibilities, not verified by a geometric declaration.

## Complete aggregation gate

The function requires exactly one result for every selected tile and no result for skipped or unknown tiles. It rejects duplicate references, missing results, changed slide/plan/level/bounds, mismatched transform dimensions, different feature-vector lengths and mixed provider or feature-space digests. The feature-space specification should bind the feature layer, schema and normalization conventions. Equal vector length alone is insufficient evidence of compatibility.

Only after every observation passes does the function order results by the original tile plan and compute the mean using stable summation. Arrival order does not change the aggregate or audit report. There is no partial result or partial lineage on failure.

The method is explicitly `equal_weight_tile_mean`. Overlapping tiles and different pyramid levels remain distinct inputs. This is not pixel-area weighting, overlap correction, a spatially unbiased slide estimate or a clinically qualified pooling strategy. Applications needing another aggregation method require its own verified computation and provenance path.

`SlideAggregation.features` contains the protected aggregate vector. `to_json()` returns the separate report without source or aggregate feature values: complete plan, every tile's original citation and selection evidence, transform history, source/provider/feature digests, aggregate digest, input binding and method. `review_required` is always true. Provenance integrity is not clinical approval.

## Synthetic example

This example supplies synthetic slide digests and two short feature vectors. It demonstrates computation and source traceability, not real pathology inference.

```python
import json

from openmed.multimodal.vision.pathology.slide_tiler import (
    PyramidLevel, TilePolicy, plan_slide_tiles,
)
from openmed.multimodal.vision.pathology.tile_provenance import (
    TileFeatures, aggregate_tile_features,
)

policy = TilePolicy(
    width=4, height=3, overlap_x=0, overlap_y=0,
    working_bytes_per_pixel=12, max_inflight_tiles=2,
    budget_bytes=4096, reserved_bytes=128, min_tissue_percent=25,
)
plan = plan_slide_tiles(
    "a" * 64, 8, 3,
    (PyramidLevel(index=0, width=8, height=3, scale_numerator=1),),
    policy,
)
results = tuple(
    TileFeatures(
        tile_ref=tile.tile_ref, plan_digest=plan.digest,
        source_digest=plan.source_digest, level=tile.level,
        native_bounds=tile.bounds, source_bounds=tile.source_bounds,
        processed_input_digest=f"{index + 1:064x}",
        provider_digest="b" * 64, feature_space_digest="c" * 64,
        transforms=(), features=(float(index + 1), float(2 * (index + 1))),
    )
    for index, tile in enumerate(plan.selected_tiles)
)
aggregate = aggregate_tile_features(plan, results)
assert aggregate.features == (1.5, 3.0)
report = json.loads(aggregate.to_json())
assert report["review_required"]
assert report["uncertain_selection_count"] == 2
assert report["tiles"][1]["evidence"]["region"] == [4, 0, 8, 3]
assert "features" not in report["tiles"][0]
assert aggregate == aggregate_tile_features(plan, tuple(reversed(results)))
```

For preprocessing, append explicit `TileTransform` objects in actual execution order and bind the resulting image's real digest. Record all geometric operations. Keeping an empty transform tuple after resizing or cropping would be false upstream evidence; this module cannot detect a dishonest adapter from metadata alone.

## Bounds, privacy and verification

Aggregation accepts at most 4096 tiles, with one to 4096 feature dimensions per tile and at most 262,144 total feature values. Values must be finite real numbers within `[-1e100, 1e100]`; booleans are rejected. These bounds keep summation finite and limit work. Each tile allows at most sixteen transforms, with image sides from one to 8192 pixels. The original planner's source, pyramid, mask and memory bounds also apply.

Feature vectors are excluded from representations and report methods. Fixed `TileProvenanceError` messages contain no feature values, slide paths or patient identifiers. Raw images and tissue-mask bytes do not appear in reports. In-memory feature vectors remain protected clinical data; generic dataclass serialization is not redaction. Digests, geometry and selection metadata remain linkable and require application access and retention controls. Hashes are not de-identification.

```bash
.venv/bin/python -m pytest tests/unit/multimodal/vision/pathology/test_tile_provenance.py -q
```

Synthetic controls cover complete and reordered aggregation, original citations and tissue evidence, stale or forged plans, missing/duplicate/skipped tiles, provider and schema mismatch, composed crop/resize/rotation/flip mappings, rational pyramid edges, privacy and bounded work. They establish software behavior only, not decoder correctness, model accuracy, pixel coverage or clinical qualification.
