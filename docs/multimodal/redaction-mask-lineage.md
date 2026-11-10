# Redaction mask lineage

`openmed.multimodal.privacy.propagate_masks` carries protected image regions
through declared affine, crop and raster stages. It checks every stage and
returns either complete mask geometry or a controlled rejection with no partial
masks. It uses exact rational arithmetic, performs no network or file access,
and adds no dependencies.

This is geometry propagation. It does not detect PHI, decode images, paint
redactions or verify that a renderer follows the declared transform. Callers
must use qualified detection, declare the actual rendering/filter behavior and
verify the final redacted pixels. Coordinates and digests remain sensitive
linkage metadata and need the caller's access and retention controls.

## Source evidence and coordinate identities

Pass existing `EvidenceNode` objects of kind `PIXELS` or `OCR`. Their original
source digest, `PageGeometry`, region and optional OCR text offsets stay intact.
One call processes one original asset/page. It never replaces source geometry
with derivative coordinates or loses original citations.

A `CoordinateSpace` binds an opaque 32-character hexadecimal reference to the
original asset SHA-256 and declared page extents/units. The digest consistently
identifies the original asset; it does not claim to hash rendered derivatives.
Every transform must start in exactly the previous space and introduce a new
target reference for the same original asset/page. Matching dimensions alone
do not establish coordinate identity.

Regions use half-open top-left coordinates. Pixel-space extents must be whole
numbers; PDF point-space extents may be fractional. Output mask boxes have
exclusive right/bottom edges. Point-only or coordinate-only final stages return
support polygons with `raster_box=None`; they do not claim a completed raster.

## Affine, crop and raster stages

`Affine(a, b, c, d, e, f)` maps `x' = a*x + b*y + c` and
`y' = d*x + e*y + f`. `later.after(earlier)` composes in execution order.
`inverse()` reverses the coordinate map, not pixel resampling or crop loss.
Use `Fraction` when declaring exact ratios. Floats mean their exact binary
values; there is no floating-point tolerance that can conceal coverage loss.

`TransformStep` binds the affine map to source/target spaces, optional coverage
and explicit `resampling_margin`. The helpers cover common operations:

| Helper | Behavior |
| --- | --- |
| `TransformStep.crop(source, target, region)` | Translation into the declared crop; source/target units and dimensions must agree. Pixel crops require integral edges. Any protected coverage lost by the crop rejects the whole result. |
| `TransformStep.rotate(source, target, rotation)` | Existing `PageRotation` clockwise quarter-turn semantics and swapped extents, with no sampling margin. |
| `TransformStep.scale(source, target, resampling_margin=N)` | Exact declared width/height ratios, including point-to-pixel rendering. The caller supplies the qualified filter footprint in target pixels. |

Pixel-output helpers materialize a raster stage. Direct `TransformStep` calls
also default to `rasterize=True`. Such stages expand protected support to whole
pixel cells before later transforms. A shear followed by its inverse may
therefore require a larger final mask even though the composed coordinate map
is the identity: the intermediate pixels were already affected.

Set `rasterize=False` only when composing coordinates without creating an
intermediate pixel image. This preserves corner polygons through affine stages
without unnecessary bounding-box growth. Point-space stages require this mode.
A coordinate-only stage cannot declare a nonzero sampling margin. End a real
image pipeline with its actual raster stage to obtain raster mask boxes.

Sampling support is explicit. A positive margin expands the protected bounds
before rasterization and remains protected through subsequent stages. Zero is
appropriate only for lossless transforms or a qualified zero-margin sampler.
Convert the renderer's filter footprint to target pixels; do not assume that
one source pixel equals one target pixel. The accepted range is 0–128 pixels.
Coverage lost by padding at an image edge also rejects the result; no clipping
or edge-extrapolation behavior is silently assumed.

## Coverage, failures and bounds

`coverage=None` declares the full target extent. A tuple of `Rectangle` objects
declares their union; an empty tuple declares no output coverage. The algorithm
partitions the union into disjoint cells before checking protected support.
Overlapping rectangles cannot double-count covered area, and an outer bounding
rectangle cannot conceal holes.

Every protected region must remain fully covered after each transform, margin
and materialized raster expansion. Complete or partial crop loss, a singular
affine map, source/space mismatch, arithmetic exhaustion or excessive coverage
work produces a `MaskFailure` with no masks and no usable partial composite.
The failed step is zero-based. Malformed caller contracts raise a fixed-message
`MaskLineageError`, without echoing submitted values.

The bounds are 256 masks, 32 transforms, 64 coverage rectangles per stage,
1,024 disjoint coverage cells and 100,000 polygon/cell intersection tests per
call. Callers may lower `max_clip_tests`. Rational numerators and denominators
are bounded to 512 bits. Small nonsingular transforms remain valid when within
these arithmetic bounds; outward raster rounding prevents subpixel protected
support from collapsing into an empty mask.

Raster support is deliberately conservative and can over-mask or reject a crop
that would fit the mathematical source polygon alone. The immutable source
evidence and exact composed affine map remain separate from expanded mask
support. Successful geometry is not clinical validation or a disclosure grant.

## Runnable synthetic example

```python
from openmed.multimodal.evidence import EvidenceNode, NodeKind, PageGeometry
from openmed.multimodal.privacy import (
    CoordinateSpace, Rectangle, TransformStep, propagate_masks,
)

source = CoordinateSpace(
    space_ref="0" * 32, source_digest="a" * 64,
    geometry=PageGeometry(page=0, width=100, height=80),
)
target = CoordinateSpace(
    space_ref="1" * 32, source_digest=source.source_digest,
    geometry=PageGeometry(page=0, width=80, height=60),
)
protected = EvidenceNode(
    node_id="2" * 32, kind=NodeKind.PIXELS,
    source_digest=source.source_digest, geometry=source.geometry,
    region=(30, 20, 40, 30),
)
crop = TransformStep.crop(source, target, Rectangle(10, 10, 90, 70))
result = propagate_masks(source, (protected,), (crop,))
assert result.accepted
assert result.masks[0].raster_box == (20, 10, 30, 20)
assert protected.region == (30, 20, 40, 30)
```

`result.to_json()` contains opaque references, controlled failure codes,
original source regions, coordinate declarations, rational matrix/support
values and mask bounds. Rational values serialize as `[numerator, denominator]`.
No pixels, OCR text, patient values, paths or free-form diagnostic strings enter
this API. The synthetic tests include inverse composition, raster support,
coverage holes, duplicate coverage, fractional scaling, crop loss and replay:

```sh
.venv/bin/python -m pytest tests/unit/multimodal/privacy/test_mask_lineage.py -q
```
