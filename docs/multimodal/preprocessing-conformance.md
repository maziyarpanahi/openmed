# Executable VLM preprocessing conformance

`openmed.multimodal.preprocessing` and OpenMedKit's
`PreprocessingConformance` verify decoded, caller-supplied tiny RGB/RGBA arrays.
They do not decode DICOM, choose a provider, load weights, or certify diagnostic
image validity. Execution and optional adapter calls stay local. Adapters must
not perform network calls or persist inputs; this seam is trusted local code,
not a sandbox for an untrusted provider.

## Version 1 operator contract

Source and target dimensions are each 1–64 (at most 4096 pixels). All inputs are
explicit row-major tuples of RGB or straight RGBA integer bytes in sRGB sample
space. The execution order is:

1. Compose straight alpha against the declared RGB byte background, without
   intermediate byte rounding or gamma conversion.
2. Apply EXIF orientation 1–8, then a half-open integer crop in oriented space.
3. Resize with half-pixel centers, clamped to the cropped pixel centers. Nearest
   ties round up. Bilinear uses nonzero neighboring weights, with no antialiasing.
4. Order RGB/BGR channels, multiply by scale, subtract mean and divide by std
   in output channel order; flatten HWC/CHW.
5. Round once to float32 or float64. Arithmetic before final rounding is float64.

Unsupported operators and numeric behavior fail with controlled codes. Bicubic,
antialiasing, arbitrary color spaces, premultiplied alpha, quantized tensors,
patch packing and dynamic resolution policies are **not supported v1 profiles**.
In particular, this change does not qualify the existing Compass bicubic/patch
processor or any MLX/Core ML model. A runtime must implement the exact declared
operators; declaring the name of a similar interpolation is insufficient.
No dependencies or weights are added.

## Qualification and downstream binding

```python
from openmed.multimodal.preprocessing import (
    PreprocessingProfile, execute_profile, qualify_profile, verify_preprocessing,
)

profile = PreprocessingProfile(
    source_size=(2, 2), crop=(0, 0, 2, 2), target_size=(3, 3),
    interpolation="bilinear",
)
pixels = ((255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255))
mask = (True, False, False, False)
# Example reference only; replace with an adapter returning actual runtime values.
# In production hash the runtime build AND its configuration instead of this example.
runtime_digest = "a" * 64
qualification = qualify_profile(profile, execute_profile, runtime_digest=runtime_digest)
output = execute_profile(profile, pixels, mask)
evidence = verify_preprocessing(
    profile, pixels, mask, output, qualification, runtime_digest=runtime_digest,
)
evidence.require_current(profile, output, qualification, runtime_digest=runtime_digest)
assert evidence.reviewer_confirmation_required
```

`qualify_profile` compares deterministic RGB/RGBA grids (including alpha 0, 128,
255 and protected masks) to the executable reference. The optional callable
adapter returns actual tensor, shape, dtype, exact geometry and protected mask.
The caller supplies a SHA-256 digest identifying runtime code/build/configuration
and reruns qualification after changing them. A digest is a binding, not a
signed attestation or proof that an adapter honestly ran its declared code.
`verify_preprocessing` also compares every actual output to the reference before
emitting content-free evidence. The NumPy integration fixture verifies the array
conversion seam only and makes no independent provider qualification claim.

Geometry is the exact six-coefficient source-edge to output-edge affine matrix,
stored as reduced rationals `(a, b, c, d, e, f)` with
`x' = a*x + b*y + c`, `y' = d*x + e*y + f`. Tensor comparisons allow only the
profile's absolute tolerance (0–1e-5); geometry, shape and masks compare exactly.
Protected output masks conservatively include every source pixel contributing
with nonzero sampling weight. They are not nearest-neighbor mask resizing and
are not a guarantee of complete redaction: cropped-away or unsampled regions
have no output coverage. A privacy pipeline must separately enforce retention
and complete protected-region coverage, and must not treat these conformance
vectors as identifier-recall evidence.

The profile digest covers version, dimensions, orientation, crop, interpolation,
channel order, background, normalization, layout, dtype and tolerance. Tensor
identity covers actual little-endian rounded bytes plus shape, dtype and profile.
Coordinate identity covers exact geometry, protected coverage and profile.
Qualification binds profile, caller runtime digest and observed vector identities;
downstream evidence binds qualification, tensor and coordinates together.
Changing even one normalization constant or interpolation invalidates prior
qualification and coordinate evidence. Tolerated tensor rounding produces a
different tensor identity, so old evidence cannot silently authorize new bytes.

## Python / OpenMedKit parity and privacy

OpenMedKit exposes the same profile, execution, qualification, verification and
staleness checks. Shared synthetic fixtures in
`tests/fixtures/multimodal/preprocessing-v1.json` pin pixels, protected masks,
all eight orientations and binary identities across both surfaces. Version 1
profile identity uses the UTF-8 prefix `openmed.preprocessing.v1\0`, thirteen
little-endian uint32 values (version, source size, crop, target size, orientation,
background), NUL-terminated interpolation/color/layout/dtype strings, then eight
little-endian float64 values (scale, mean, std, tolerance). Coordinate rationals
and tensor shape/dtype are likewise pinned by these fixtures.

Tensors and masks are transient sensitive payloads, excluded from printable
output descriptions; do not log or serialize them. Evidence includes only
controlled notice codes, review flags and digests. Errors suppress runtime
exception text. There is no telemetry, persistence, cloud fallback or automatic
clinical action. Every qualification and downstream evidence carries
`synthetic_preprocessing_only_non_diagnostic` and requires separate explicit
reviewer confirmation before consequential use. This module records that
requirement; the consuming workflow must enforce it.
