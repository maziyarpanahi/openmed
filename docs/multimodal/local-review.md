# Local multimodal review in OpenMedKit

`MultimodalReviewView` displays a bounded source region or transcript span, its alternatives, source provenance and uncertainty. The user explicitly accepts a selected alternative, rejects all alternatives or defers. A decision is bound to the admitted evidence, the exact displayed content and its revision. It never finalizes a note, confirms a diagnosis or authorizes disclosure.

## Caller-owned local evidence

Create a `MultimodalReviewCase` from a validated [multimodal packet](openmedkit-contracts.md), an opaque `LocalMultimodalReviewHandle`, a revision digest and one to eight alternative references. Visual conflict and handwriting cases require a visual source box. Speaker cases require a current, non-invalidated transcript span. Each kind requires its corresponding uncertainty reason.

The caller implements `LocalMultimodalReviewSource` on the main actor. Its `currentRevision` and `resolve` methods must be bounded, synchronous and entirely local. Keep the original handle object in the caller's custody map; do not interpret its reference as a path or URL. Return `nil` when access is revoked or the source is absent. Unknown provider errors become a controlled unavailable state without exposing their text.

```swift
import OpenMedKit
import SwiftUI

// The application supplies admitted metadata and its own local source store.
struct ReviewScreen: View {
    let review: MultimodalReviewCase
    let source: any LocalMultimodalReviewSource
    let recordChoice: (MultimodalReviewDecision) -> Void

    var body: some View {
        MultimodalReviewView(review: review, source: source) { decision in
            // Recheck source custody and revision again when applying the choice.
            recordChoice(decision)
        }
    }
}
```

`LocalMultimodalReviewContent` contains protected display material. The revision, source digest and ordered alternative references must match the case. Transcript text must cover exactly the packet's current Unicode-scalar span. Visual content must contain the caller's source crop; any accompanying text is contextual. The component cannot authenticate whether a supplied crop or text actually came from the declared original source. That remains the caller's responsibility.

Content is bounded: text is at most 16,384 UTF-8 bytes, each alternative is at most 4,096 bytes, and there are at most eight alternatives. A crop must be 1–1,024 pixels per dimension, tightly packed 8-bit sRGB with 32-bit premultiplied-last pixels. Pixels are copied into immutable owned memory; unsupported image formats are refused. Convert and crop inside the trusted caller's local source boundary before constructing content.

## Decisions and invalidation

The component checks revision before and after every source resolution. Loading data alone does not enable a decision: actual source pixels or text must enter the active view's viewport. A fresh presentation identity prevents an old visibility callback from approving a newly loaded revision. Replacing the case or source provider resets the view. Hiding the view or making its scene inactive clears protected content and selection; returning to the active scene reloads it.

Accept requires an explicitly selected alternative. A provisional transcript can be rejected or deferred, but cannot be accepted until its source contract is finalized. Reject and defer carry no selected alternative. Every action resolves the source again and compares the complete displayed-content digest, including text, pixels and alternative values. A revision change, content change under an unchanged revision, missing source or revoked local access clears the display and produces no decision. A completed presentation cannot produce a second decision.

Native buttons expose labels, selection state and identifiers to accessibility tools. Keyboard activation uses the same decision gate. On iOS, macOS and visionOS, while a control within this component has keyboard focus, Command-Shift-A accepts, Command-Shift-R rejects and Command-Shift-D defers. Shortcuts do not bypass selection, source presentation, finalization or revision checks. The geometry gate establishes that source content was presented; it does not prove that a person read or understood it. Applications must keep the view visibly available and avoid covering it with unrelated overlays.

`MultimodalReviewDecision` contains the choice, case digest, evidence digest, revision, displayed-content digest and optional opaque alternative reference. `allowsAutomaticFinalization` is always false. This record is not an authenticated review-authority receipt. Validate current source custody and revision at the point of use, then apply the application's separate review and finalization policy. A valid decision must not silently assign a clinical role or finalize a note.

## Privacy and scope

The component performs no network calls, file writes, analytics, model inference, microphone capture or persisted thumbnail generation. It does not log protected content. Content and alternatives have redacted debug descriptions, and the view marks its contents privacy-sensitive. Caller-provided source implementations and callbacks remain trusted application code; a `localOnly` declaration is not a network sandbox. The host application is responsible for source storage, retention, screenshots, accessibility access and any persistence of returned decisions. Clearing references is not a secure-memory-erasure guarantee.

This Swift-only slice owns a reusable on-device view. Python supplies the existing admitted evidence contracts; there is no duplicate Python UI or new inference backend. It uses existing package dependencies and requires the package's existing iOS 17, macOS 14, watchOS 10 or visionOS 1 baseline.

## Synthetic fixtures and validation

`tests/fixtures/multimodal/local_review_v1.json` contains four synthetic cases: ambiguous handwriting, stale visual evidence, missing local content and uncertain speaker attribution. Debug previews in `MultimodalReviewFixtures.swift` use those same cases and generate a tiny synthetic crop in memory. The fixtures contain no real clinical source and are not a model-quality evaluation.

```bash
make format-swift
make lint-swift
cd swift/OpenMedKit
swift test --disable-automatic-resolution --filter MultimodalReviewTests
```

Dependencies must already be cached for offline builds. Focused controls cover unseen and hidden presentations, stale render identities, changed revisions and content, revoked access, source/alternative binding, scalar bounds, controlled errors and replay prevention. The Swift workflow runs when the shared fixture changes. Native fixture review should additionally verify that the visible source enables alternatives, accept returns only the selected reference, changed or missing evidence stays held, and reject/defer never select an alternative.
