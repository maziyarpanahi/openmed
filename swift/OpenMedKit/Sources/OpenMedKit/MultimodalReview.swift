import CoreGraphics
import Foundation
import Observation

public enum MultimodalReviewError: String, Error, Sendable {
    case invalid = "invalid_review_contract"
    case unsupported = "unsupported_review_capability"
    case missing = "review_content_missing"
    case changed = "review_revision_changed"
    case unavailable = "local_review_source_unavailable"
    case unseen = "review_content_not_presented"
    case selection = "review_selection_required"
    case provisional = "review_transcript_not_finalized"
}

public enum MultimodalReviewKind: String, Sendable {
    case visualConflict, handwriting, speakerRole
}

public enum MultimodalReviewUncertainty: String, CaseIterable, Sendable {
    case conflictingValues, ambiguousHandwriting, uncertainSpeaker, provisionalTranscript
}

public enum MultimodalReviewChoice: String, Sendable { case accept, reject, `defer` }

/// Caller-held local capability. Never serialized or interpreted as a file path/URL.
public final class LocalMultimodalReviewHandle: Sendable {
    public let reference = UUID().uuidString.replacingOccurrences(of: "-", with: "").lowercased()
    public init() {}
}

private func reviewHex(_ value: String, count: Int) -> Bool {
    value.utf8.count == count && value.utf8.allSatisfy { (48...57).contains($0) || (97...102).contains($0) }
}

/// Immutable review metadata, bound to an admitted source projection and revision.
public struct MultimodalReviewCase: Sendable {
    public let handle: LocalMultimodalReviewHandle
    public let evidence: MultimodalPacket
    public let revision: String
    public let kind: MultimodalReviewKind
    public let alternatives: [String]
    public let uncertainty: [MultimodalReviewUncertainty]
    public let digest: String

    public init(
        handle: LocalMultimodalReviewHandle, evidence: MultimodalPacket, revision: String,
        kind: MultimodalReviewKind, alternatives: [String], uncertainty: [MultimodalReviewUncertainty]
    ) throws {
        guard reviewHex(revision, count: 64), (1...8).contains(alternatives.count),
            Set(alternatives).count == alternatives.count, alternatives.allSatisfy({ reviewHex($0, count: 32) }),
            !uncertainty.isEmpty, Set(uncertainty).count == uncertainty.count
        else { throw MultimodalReviewError.invalid }
        let requiredReason: MultimodalReviewUncertainty
        switch kind {
        case .visualConflict: requiredReason = .conflictingValues
        case .handwriting: requiredReason = .ambiguousHandwriting
        case .speakerRole: requiredReason = .uncertainSpeaker
        }
        guard uncertainty.contains(requiredReason) else { throw MultimodalReviewError.invalid }
        switch evidence.projection {
        case .visual(let visual):
            guard kind != .speakerRole, visual.sourceBox != nil else { throw MultimodalReviewError.unsupported }
        case .transcript(let transcript):
            guard kind == .speakerRole, transcript.invalidation == nil, transcript.currentScalarSpan != nil else {
                throw MultimodalReviewError.unsupported
            }
        default: throw MultimodalReviewError.unsupported
        }
        self.handle = handle
        self.evidence = evidence
        self.revision = revision
        self.kind = kind
        self.alternatives = alternatives
        self.uncertainty = uncertainty
        digest = ClinicalBrief.hash(
            try ClinicalBrief.canonical([
                "schema": "openmed.multimodal.review.v1", "handle": handle.reference,
                "evidence": evidence.evidenceDigest, "revision": revision, "kind": kind.rawValue,
                "alternatives": alternatives, "uncertainty": uncertainty.map(\.rawValue).sorted(),
            ]))
    }

    var sourceDigest: String {
        switch evidence.projection {
        case .visual(let source): return source.sourceDigest
        case .transcript(let source): return source.snapshotDigest
        default: preconditionFailure("unsupported_review_capability")
        }
    }

    var canAccept: Bool {
        if case .transcript(let source) = evidence.projection { return source.finalized }
        return true
    }
}

/// Protected display text. No values are copied into decisions or diagnostics.
public struct LocalReviewAlternative: CustomStringConvertible, CustomDebugStringConvertible {
    public let reference: String
    public let value: String
    public var description: String { "LocalReviewAlternative(value=<private>)" }
    public var debugDescription: String { description }

    public init(reference: String, value: String) throws {
        guard reviewHex(reference, count: 32), (1...4096).contains(value.utf8.count) else { throw MultimodalReviewError.invalid }
        self.reference = reference
        self.value = value
    }
}

/// Bounded in-memory display material. Images are copied into immutable owned pixels.
/// The caller authenticates source custody and the crop; this component does not.
public struct LocalMultimodalReviewContent: CustomStringConvertible, CustomDebugStringConvertible {
    public let revision: String
    public let sourceDigest: String
    public let text: String?
    public let image: CGImage?
    public let alternatives: [LocalReviewAlternative]
    let digest: String
    public var description: String { "LocalMultimodalReviewContent(content=<private>)" }
    public var debugDescription: String { description }

    public init(
        revision: String, sourceDigest: String, text: String? = nil,
        image: CGImage? = nil, alternatives: [LocalReviewAlternative]
    ) throws {
        guard reviewHex(revision, count: 64), reviewHex(sourceDigest, count: 64),
            (1...8).contains(alternatives.count), Set(alternatives.map(\.reference)).count == alternatives.count,
            text == nil || (1...16_384).contains(text!.utf8.count)
        else { throw MultimodalReviewError.invalid }
        var pixels: [String: Any] = [:]
        if let image {
            guard (1...1024).contains(image.width), (1...1024).contains(image.height),
                image.bitsPerComponent == 8, image.bitsPerPixel == 32, image.bytesPerRow == image.width * 4,
                image.colorSpace?.name == CGColorSpace.sRGB, !image.isMask,
                image.alphaInfo == .premultipliedLast, !image.bitmapInfo.contains(.floatComponents),
                let raw = image.dataProvider?.data, CFDataGetLength(raw) == image.bytesPerRow * image.height
            else { throw MultimodalReviewError.invalid }
            let bytes = Data(bytes: CFDataGetBytePtr(raw)!, count: CFDataGetLength(raw))
            guard let provider = CGDataProvider(data: bytes as CFData),
                let copy = CGImage(
                    width: image.width, height: image.height, bitsPerComponent: 8, bitsPerPixel: 32,
                    bytesPerRow: image.bytesPerRow, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                    bitmapInfo: image.bitmapInfo, provider: provider, decode: nil,
                    shouldInterpolate: false, intent: .defaultIntent)
            else { throw MultimodalReviewError.invalid }
            self.image = copy
            pixels = ["sha256": ClinicalBrief.hash(bytes), "width": image.width, "height": image.height, "format": image.bitmapInfo.rawValue]
        } else {
            self.image = nil
        }
        self.revision = revision
        self.sourceDigest = sourceDigest
        self.text = text
        self.alternatives = alternatives
        digest = ClinicalBrief.hash(
            try ClinicalBrief.canonical([
                "revision": revision, "source": sourceDigest, "text": text as Any? ?? NSNull(),
                "pixels": pixels, "alternatives": alternatives.map { ["reference": $0.reference, "value": $0.value] },
            ]))
    }
}

/// Trusted caller-owned local custody. Calls must be bounded, synchronous and local.
/// Return nil for missing/revoked access. A localOnly declaration is not a sandbox.
@MainActor
public protocol LocalMultimodalReviewSource: AnyObject {
    var localOnly: Bool { get }
    func currentRevision(_ handle: LocalMultimodalReviewHandle) throws -> String?
    func resolve(_ handle: LocalMultimodalReviewHandle) throws -> LocalMultimodalReviewContent?
}

/// A revision-bound human choice, not an authority receipt or a finalized note.
public struct MultimodalReviewDecision: Sendable {
    public let choice: MultimodalReviewChoice
    public let caseDigest: String
    public let evidenceDigest: String
    public let revision: String
    public let displayedContentDigest: String
    public let alternativeReference: String?
    public let allowsAutomaticFinalization = false
}

@MainActor
@Observable
final class MultimodalReviewSession {
    let review: MultimodalReviewCase
    let source: any LocalMultimodalReviewSource
    private(set) var content: LocalMultimodalReviewContent?
    private(set) var refusal: MultimodalReviewError?
    private(set) var presentationID = UUID()
    private(set) var presented = false
    private(set) var completed = false
    private(set) var selected: String?

    init(review: MultimodalReviewCase, source: any LocalMultimodalReviewSource) {
        self.review = review
        self.source = source
    }

    private func resolveCurrent() throws -> LocalMultimodalReviewContent {
        guard source.localOnly else { throw MultimodalReviewError.unavailable }
        guard let before = try source.currentRevision(review.handle) else { throw MultimodalReviewError.missing }
        guard before == review.revision else { throw MultimodalReviewError.changed }
        guard let value = try source.resolve(review.handle) else { throw MultimodalReviewError.missing }
        guard value.revision == review.revision, value.sourceDigest == review.sourceDigest,
            value.alternatives.map(\.reference) == review.alternatives
        else { throw MultimodalReviewError.changed }
        switch review.evidence.projection {
        case .visual:
            guard value.image != nil else { throw MultimodalReviewError.missing }
        case .transcript(let transcript):
            guard let span = transcript.currentScalarSpan, let text = value.text, value.image == nil,
                text.unicodeScalars.count == Int(span.end - span.start)
            else { throw MultimodalReviewError.invalid }
        default: throw MultimodalReviewError.unsupported
        }
        guard source.localOnly else { throw MultimodalReviewError.unavailable }
        guard try source.currentRevision(review.handle) == review.revision else { throw MultimodalReviewError.changed }
        return value
    }

    func load() {
        hide()
        guard !completed else { return }
        do { content = try resolveCurrent() } catch { fail(error) }
    }

    func markPresented(_ identity: UUID) {
        guard identity == presentationID, content != nil, refusal == nil, !completed else { return }
        presented = true
    }

    func hide() {
        content = nil
        selected = nil
        presented = false
        presentationID = UUID()
        refusal = nil
    }

    func select(_ reference: String) {
        guard presented, content != nil, review.alternatives.contains(reference), !completed else { return }
        selected = reference
    }

    private func fail(_ error: Error) {
        hide()
        refusal = (error as? MultimodalReviewError) ?? .unavailable
    }

    var canAct: Bool { presented && content != nil && !completed }
    var canAccept: Bool { canAct && selected != nil && review.canAccept }

    /// Shared gate for pointer, keyboard and accessibility actions; never trusts a UI flag alone.
    func decide(_ choice: MultimodalReviewChoice) -> MultimodalReviewDecision? {
        guard canAct, let shown = content else { return nil }
        if choice == .accept {
            guard review.canAccept else {
                refusal = .provisional
                return nil
            }
            guard selected != nil else { return nil }
        }
        do {
            let current = try resolveCurrent()
            guard current.digest == shown.digest else { throw MultimodalReviewError.changed }
            let result = MultimodalReviewDecision(
                choice: choice, caseDigest: review.digest, evidenceDigest: review.evidence.evidenceDigest,
                revision: review.revision, displayedContentDigest: shown.digest,
                alternativeReference: choice == .accept ? selected : nil)
            completed = true
            hide()
            return result
        } catch {
            fail(error)
            return nil
        }
    }
}
