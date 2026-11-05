# Whole-slide tile planning

`openmed.multimodal.vision.pathology.plan_slide_tiles()` builds deterministic multiresolution plans for caller-owned slides. It plans native-level tile coordinates, overlap, outward-rounded level-zero coordinates, conservative tissue-mask routing and bounded batches. It never opens a slide, decodes a whole image, loads a model or writes tile pixels.

The plan is an admission contract for an adapter that can read native regions. Source slides remain under caller control. An adapter must verify the slide digest and pyramid metadata, honor the declared buffer factors and batch limit, and release each batch before reading the next. Existing full-frame VLM memory estimates are separate; do not allocate a gigapixel frame merely to satisfy a tile plan. OpenMedKit parity is tracked separately in #3679.

## Plan a synthetic slide

```python
from openmed.multimodal.vision.pathology import (
    PyramidLevel, TilePolicy, TissueMask, plan_slide_tiles,
)

source_digest = "a" * 64  # Synthetic fixture, not a real slide digest.
levels = (
    PyramidLevel(index=0, width=4000, height=2000, scale_numerator=1),
    PyramidLevel(index=1, width=1000, height=500, scale_numerator=4),
)
policy = TilePolicy(
    width=512, height=512, overlap_x=32, overlap_y=32,
    working_bytes_per_pixel=16, max_inflight_tiles=2,
    budget_bytes=16 * 1024 * 1024, reserved_bytes=4 * 1024 * 1024,
    min_tissue_percent=10,
)
mask = TissueMask(
    source_digest=source_digest, width=4, height=2,
    cells=bytes([0, 1, 1, 0, 0, 1, 2, 0]),
)
plan = plan_slide_tiles(
    source_digest, 4000, 2000, levels, policy, tissue_mask=mask,
)
assert plan.peak_working_bytes <= policy.budget_bytes
assert all(len(batch) <= 2 for batch in plan.iter_batches())
assert plan.mask_digest == mask.digest
audit_json = plan.to_json()
```

`iter_batches()` yields metadata only. Read those native regions through the caller's local slide adapter, keeping only one admitted batch resident. It does not open files or invoke a decoder, and it cannot enforce the behavior of arbitrary adapter code. The example's memory factors and mask are synthetic, not measured device requirements or clinically qualified tissue thresholds.

## Native levels and source coordinates

Each requested `PyramidLevel` supplies a native index, width, height and a reduced rational downsample factor. The factor must come from verified pyramid metadata. Do not infer it by dividing rounded dimensions: a 100-pixel source with a 33-pixel level can still have a declared scale of 3, leaving one source-edge pixel outside that level's represented extent. The report records such right/bottom gaps explicitly.

The declared level dimensions must be a floor or ceiling of the source dimensions divided by the scale. Level zero, when requested, has scale 1 and exact source dimensions. Requested levels are sorted by native index and must have strictly increasing scales and nonincreasing dimensions; a subset is allowed. Both axes use the declared scale, so anisotropic pyramids need a different explicitly supported transform instead of a guessed ratio.

Tile bounds use half-open native coordinates `(x0, y0, x1, y1)`. Stride is tile size minus overlap on each axis. The final tile is clipped to the native level edge, and no redundant tile is emitted after an earlier tile already covers that edge. Every native pixel is covered before tissue filtering. Level-zero starts use floor and ends use ceiling, clipped to the true source extent. Fractional scales can therefore introduce a small conservative overlap in source-coordinate rectangles. These rectangles describe provenance geometry, not a claim that every covered source pixel was independently sampled.

## Conservative tissue filtering

A `TissueMask` is a bounded full-slide grid whose cells uniformly partition the level-zero extent. It is bound to the slide digest and uses `0` for background candidate, `1` for tissue candidate and `2` for unknown. Its byte payload is protected in memory and excluded from representations and plan reports. The caller is responsible for the mask's provenance and qualification.

For each tile, the planner counts mask-cell centers inside its half-open level-zero rectangle. Known tissue percentages are compared using integer arithmetic. A tile with any unknown cell, or no sampled center, is retained with `keep_uncertain_coverage`. A sufficiently high known tissue fraction produces `keep_tissue_candidate`; otherwise it produces `skip_background_candidate`. A zero minimum retains every sampled background tile. Missing masks retain every tile for review.

Every candidate, including exclusions, remains in the plan with coordinates, sampled/tissue/unknown counts and a controlled decision. `selected_tiles` and `iter_batches()` omit only known background candidates. A coarse mask can miss small tissue regions even when a tile contains sample centers. These counts are sample coverage, not pixel-level completeness or a clinical tissue classifier. Validate the mask resolution and filtering policy for the intended slide type, and retain all tiles when that evidence is unavailable.

`mask_from_rgb()` can derive the grid from a caller-supplied RGB thumbnail of at most 512 × 512 pixels. Its explicit `white_floor` requires all channels to meet the background threshold; other pixels with at least `min_chroma` channel spread become tissue candidates. Nonwhite low-chroma pixels remain unknown, preserving dark ink and uncertain regions for review. This deterministic color heuristic needs task-specific qualification. It does not extract a thumbnail from a slide, bundle a pathology dataset or diagnose tissue.

## Resource admission

The planner accepts at most 32 native levels and 4,096 total candidate tiles. It checks candidate counts before building mask indexes or enumerating tiles; filtering cannot be used to justify traversing an unbounded grid. Choose coarser requested levels or a separately managed source partition if a full plan exceeds that ceiling. Native tile sides are at most 8,192 pixels. Mask indexing uses two bounded native-integer prefix arrays of at most 513 × 513 entries each; no slide-sized array is created.

Each tile's declared working bytes equal its actual clipped native area times `working_bytes_per_pixel`. Peak admitted working bytes are `reserved_bytes + largest_selected_tile_bytes × min(max_inflight_tiles, selected_count)`. The plan is rejected if this exceeds `budget_bytes`, including by one byte. All-background plans need only the reserve and emit no inference batches. Different-sized tiles are conservatively charged at the largest selected size for concurrent admission.

The caller must account for every simultaneously resident representation in the per-pixel factor and reserve, including model memory, decoder scratch or compressed-block overhead where applicable. The ceiling bounds these declared working buffers. It is not a measurement of total process RSS, and planner metadata, application objects and an adapter that ignores the contract do not become covered by this estimate. Coordinate and mask limits independently bound the planner's own work. No cloud fallback is selected when admission fails.

## Privacy and reproducibility

The stdlib-only planner performs no network calls, logging or file access. Reports contain source/mask digests, native and source coordinates, level metadata, declared budget factors, mask counts and routing decisions. They contain no filenames, slide images, RGB thumbnails, masks, patient metadata or exception details. Saving the returned JSON persists only that metadata; any later pixel caching or export requires a separate explicit application policy.

`tile_ref` hashes source identity, native level geometry and coordinates. It is a reproducible coordinate reference, not a digest of decoded tile pixels. `plan.digest` binds the complete plan, including thresholds, mask digest and exclusions. Digests and spatial metadata remain linkable provenance records and need application access and retention controls. Every plan retains `review_required: true`; it does not authorize autonomous clinical decisions.

```bash
.venv/bin/python -m pytest tests/unit/multimodal/vision/pathology/test_slide_tiler.py -q
```

Synthetic controls cover gigapixel coordinate planning, overlap coverage, multiple levels, fractional scale rounding, source-edge gaps, uncertain masks, threshold boundaries, exact budget rejection, bounded batches, deterministic replay and content-free reporting.
