# OpenMedKit multimodal evidence contracts

`openmed.multimodal.apple_contract` and OpenMedKit's `MultimodalPacket` share a versioned metadata projection for local applications. They preserve supported asset manifests, source visual coordinates, registered transcript citations, protected clock bounds and provider outcomes. The contract has no model loader, network call, microphone access or clinical decision path. A valid packet is structural evidence, not authenticated acquisition, source custody, permission to disclose, or clinical approval.

## Source-backed Python projections

The projection functions consume the existing Python contracts, including [fused evidence](fused-evidence-graphs.md), [transcript revisions](streaming/transcript-revisions.md) and [clock alignment](cross-modal-clock-alignment.md). They preserve source holds instead of creating an independent interpretation.

```python
from openmed.multimodal.asset_manifest import AssetManifest
from openmed.multimodal.apple_contract import MultimodalPacket, project_asset

# Synthetic manifest only. The application owns the actual protected asset.
manifest = AssetManifest(
    asset_id="synthetic-image", media_type="image/png", sha256="a" * 64,
    byte_size=128, width=100, height=50,
)
packet = project_asset(manifest)
assert MultimodalPacket(packet.wire).evidence_digest == packet.evidence_digest
assert packet.payload()["width"] == 100
```

| Projection | Python input | Preserved behavior |
| --- | --- | --- |
| `asset` | `AssetManifest` | Supported image/audio/PDF/DICOM metadata, exact source hash, bounded counts and original manifest digest. The original asset identifier becomes an opaque hash. |
| `visual` | Source `EvidenceNode`, optional quarter-turn rotation | Original page, pixel/point units, top-left half-open box and Unicode scalar span. Display geometry is derived and revalidated separately. |
| `transcript` | `RevisionLedger`, exact registered `SpanReference`, declared sample rate | Complete current snapshot digest, original reference, revision, finalization and invalidation. Invalidated coordinates remain absent. |
| `provider` | `ProviderResultEnvelope` | Exact outcome and abstention vocabulary, request/output digests, duration and closed counters. Non-success outcomes have no output digest. |
| `clock` | `AlignmentResult` | Reference, clock and policy provenance, inclusive endpoint uncertainty and guaranteed overlap. Review states contain no translated timestamps. |

`project_visual` accepts source page, pixel and OCR nodes. Derived facts and claims are explicitly unsupported by this source-coordinate projection. Rotation is clockwise and limited to 0, 90, 180 or 270 degrees, using the existing `PageTransform`. Unknown transformations are refused; no arbitrary affine transform is guessed or silently clipped. Both languages recompute the output extent and display rectangle before exposing the packet. A transformation that collapses a positive-area source box through floating-point precision loss is refused.

`project_transcript` validates the exact registered reference and hashes the complete source ledger snapshot. The caller must serialize access to that ledger. Original and current spans use Unicode code points, corresponding to Swift `unicodeScalars`, never UTF-8 byte offsets or grapheme counts. Relative token timing remains in milliseconds. The target sample interval is the outward integer conversion `[floor(start_ms × rate / 1000), ceil(end_ms × rate / 1000)]`. This declared grid conversion is not new acoustic alignment evidence. Unfinalized citations remain explicitly unfinalized; text/token invalidation removes current scalar, time and sample bounds.

Clock projections retain the original result digest and all source bindings. Translated endpoints are inclusive uncertainty envelopes; guaranteed overlap is half-open. Nanoseconds remain signed integers, including values above the exact-integer range of a floating-point number. These protected bounds must not be confused with wall time or an operational log record. Projection does not fit or qualify a new clock model.

## Swift consumption and local adapter seams

```swift
import OpenMedKit

// data is the canonical packet received from a trusted local source boundary.
let packet = try MultimodalPacket.fromJSON(data)
if case .visual(let evidence) = packet.projection {
    // Keep sourceBox for provenance; displayBox is only the validated rotation.
    let sourceBox = evidence.sourceBox
    let displayBox = evidence.displayBox
}
```

`MultimodalProjection` exposes typed asset, visual, transcript, provider and clock values. Construction of the containing packet is validated; no public initializer accepts an unchecked projection. Swift code that produces visual evidence, such as [native document intake](native-document-intake.md), uses `MultimodalPacket.visual(...)`: it derives the display geometry, serializes canonically and admits the result through `fromJSON`, so it produces the same bytes and digest as Python's `project_visual`. `toJSON()` preserves the exact admitted bytes, and `evidenceDigest` is the same lowercase SHA-256 returned by Python. Generic Codable decoding is not an alternative admission path.

Python's `LocalMultimodalAdapter` and Swift's `LocalMultimodalAdapter` provide injected local execution seams. `run_local_adapter` / `LocalMultimodalExecution.run` reject a non-local declaration or an unsupported projection kind before invoking the adapter. A returned provider outcome must bind `input_digest` to the request packet's `evidence_digest`. Missing models, resource limits and validation failures use the existing provider result vocabulary; they never become successful empty outputs. Actual model execution, cancellation, resource enforcement and source custody remain responsibilities of the injected implementation. A declaration is not a network sandbox.

## Wire identity and validation

The exact envelope schema is `openmed.multimodal.apple.v1`, with four keys: `schema_version`, `kind`, `payload`, and `evidence_digest`. Its identity is SHA-256 over canonical JSON containing the first three fields. Canonical JSON is compact, sorted by ASCII field name, and ASCII-escaped. Admission requires byte-for-byte canonical form; whitespace variants, duplicate keys, trailing data and alternate numeric encodings are rejected. Use the projection factories to produce this wire format.

Integer fields use bounded JSON integers, with explicit boolean rejection. Every floating field uses sixteen lowercase hexadecimal digits encoding an IEEE-754 binary64 value in big-endian order. For example, the exact floating representation of `0.1` is `3fb999999999999a`. This avoids depending on Python and Swift JSON float formatting. Both sides reject nonfinite values, negative zero encodings and out-of-range coordinates before exposing typed `Double`/`float` values. Source floating precision is preserved; this is not quantization or a change to existing source contract schemas.

Packets are limited to 64 KiB. Counts, source spans, token references, clock ranges and vocabularies retain explicit limits. Complete key sets and source schema versions are checked. Missing or malformed digests, unknown versions and capabilities, inconsistent transforms, incorrect result bindings and exposed invalidated bounds produce controlled refusals. No filename, source content or arbitrary adapter error is interpolated into diagnostics.

| Refusal | Meaning |
| --- | --- |
| `unsupported_multimodal_schema` | Envelope or source schema is not supported. |
| `unsupported_multimodal_capability` | Projection kind, visual source kind or adapter capability is unsupported. |
| `invalid_multimodal_transform` | Rotation or its derived display geometry is inconsistent. |
| `incomplete_multimodal_provenance` | Required digest/reference is absent, malformed or does not bind the declared result. |
| `invalid_multimodal_contract` | Shape, bounds, canonical encoding, state or provider behavior is invalid. |

A digest detects changes relative to the supplied packet; it does not authenticate who produced it. Applications must check current source custody and review state at use time. Packets contain coordinates, counts and digests instead of raw media or transcript content, but remain sensitive metadata and require access controls. No automatic role assignment, diagnosis, PHI release or model-license decision follows from admission.

## Offline parity validation

The single synthetic fixture is `tests/fixtures/parity/multimodal_contract_v1.json`. Python recreates it from actual source contracts; Swift validates the same bytes, evidence identities, geometry, scalar/sample boundaries and all reason vocabularies. Negative controls cover unknown versions/capabilities, duplicate and unknown keys, numeric coercion and overflow, nonfinite values, incomplete provenance, incorrect transforms, invalidated citations and leaked held timestamps.

```bash
python -m pytest tests/unit/multimodal/test_apple_contract.py -q
cd swift/OpenMedKit
swift test --disable-automatic-resolution --filter MultimodalContractParityTests
```

Package dependencies must already be resolved and cached for an offline build. The tests themselves need no model weights, network access, microphone or real clinical data. The Swift workflow also runs when the shared fixture changes. These checks establish contract parity; they do not qualify any VLM/ASR model or complete application.
