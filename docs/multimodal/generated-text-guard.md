# Generated visual text release guard

Buffered VLM text is a separate PHI channel. The post-generation guard screens
it before release, using an injected **local** detector and identifier commitments
from the same asset's OCR spans, header seeds or region findings. This boundary
adds no model selection, preprocessing, OCR, pixel redaction or arbitration.
It does not qualify a provider or establish clinical accuracy.

## Python

```python
from openmed.multimodal.generated_text_guard import (
    AssetIdentifierDigest, GeneratedTextGuard, GuardPolicy, PHILabel,
    identifier_digest,
)

# Synthetic identifier; real values stay transient in caller-owned memory.
asset_digest = "a" * 64
identifier = AssetIdentifierDigest(
    asset_digest, PHILabel.NAME, identifier_digest("Mira Vale"), 8,
)
guard = GeneratedTextGuard(
    lambda text: (),  # Synthetic negative detector, not production qualification.
    detector_version="synthetic-v1", policy=GuardPolicy(action="redact"),
)
result = guard.release(
    "Label: MIRA   VALE", asset_digest=asset_digest, identifiers=[identifier],
)
assert result.text == "Label: [REDACTED]"
assert result.envelope.guard_metadata["decision"] == "redact"
```

The detector returns `PHISpan(start, end, label, resolved)` using Python Unicode
code-point offsets. Labels are closed: name, MRN, date, identifier. Findings must
be complete and bounded; unresolved critical findings, invalid spans, detector
exceptions, missing detectors and timeouts abstain with `phi_uncertainty`.
Overlap is merged before replacing spans with the fixed `[REDACTED]` marker.
Policies can instead withhold or abstain on any resolved finding. Both produce
an abstention envelope and no released text.

Commitments store only asset digest, label, identifier digest and normalized
length. V1 applies NFKC and case folding per code point and removes whitespace;
all windows of the committed length are checked, independently of detector
recall. This catches spacing/casing echoes, including embedded and repeated
matches. Other transformations (changed punctuation, transliteration, truncated
identifiers, canonical composition across code points) require the local detector;
this is not a universal leakage guarantee. Supply all available same-asset
commitments using this normalization; arbitrary legacy digests are incompatible.
Digests are commitments, not encryption, and still need access controls.

Reports contain controlled decisions/labels, counts and digests only. Detector
version is bound as a SHA-256 digest, policy/normalization is bound by its digest,
and the complete report digest is bound in the provider envelope. Clean outputs
pass unchanged; metadata is deterministic for identical inputs. The output
digest covers only released text, never an abstained generation. Result repr
excludes text. Do not log or persist result text.

`OpenMedMLXVisionLanguageModel.generate`, `generate_with_metadata` and
`generate_vision_text` accept `release_guard`, `asset_digest` and `identifiers`.
Missing guard or asset provenance fails closed. String APIs return an empty
string on abstention; use `generate_with_metadata().release_envelope` to inspect
the decision. The all-zero input digest is an unavailable-provenance sentinel,
used only on this default abstention path. Original generated token IDs are
always suppressed, since they can reconstruct PHI. This intentionally changes
legacy direct-return behavior: callers must inject a local guard to release text.

## OpenMedKit

`GeneratedTextGuard` provides the same buffered policy boundary. Inject its
`Detector` returning `GeneratedPHISpan` in Unicode **scalar** offsets (not UTF-16
or grapheme-cluster indices). `GeneratedIdentifierDigest` commits same-asset
values using `GeneratedTextGuard.identifierDigest`. Foundation case folding is
used per scalar; Python and Swift normalization contracts have separate policy
digests and commitments should be produced on the consuming platform.

Every Compass `generate` overload accepts `releaseGuard`, `assetDigest` and
`identifiers`. `releaseEnvelope` holds only the provider outcome and guard
report, with detector-version, policy and report digests. Missing guard/provenance
returns empty text with `phi_uncertainty`; original token IDs are suppressed.
Everything executes on-device; there is no cloud fallback or autonomous action.

## Bounds and reviewer responsibilities

Each generation is limited to 8192 code points/scalars, 128 commitments of lengths
1–256 and 4096 detector findings; deadlines are positive and at most 60 seconds.
A detector timeout withholds immediately. Arbitrary injected code cannot be
forcibly stopped: it may retain transient input until it returns. Each guard
allows one outstanding detector call and refuses further work until that call
finishes. Detectors must be trusted, local, bounded, and must not log, persist or
transmit text; the guard emits no detector error messages or traceback.

Every report binds the controlled `non_diagnostic_review_required` use-policy
label. `GuardedTextResult` and the Swift envelope expose a fixed non-diagnostic
notice and a requirement for explicit reviewer confirmation.

The guard is a privacy screen, **not diagnostic interpretation**. Consequential
uses must display a non-diagnostic notice and require explicit reviewer
confirmation before any downstream action. The release guard never performs such
an action. Synthetic fake generations test this boundary without bundled weights,
restricted data, network calls or fabricated model-quality evidence. Date shifting,
model quantization qualification and provider recall measurement remain separate
contracts; dates here are conservatively redacted or withheld.
