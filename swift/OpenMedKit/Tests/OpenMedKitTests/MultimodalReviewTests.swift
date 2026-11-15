import CoreGraphics
import Foundation
import XCTest

@testable import OpenMedKit

final class MultimodalReviewTests: XCTestCase {
    @MainActor
    private func loaded(_ id: String = "ambiguous_visual") throws -> (MultimodalReviewSession, MultimodalReviewFixtures.Source) {
        let (review, source) = try MultimodalReviewFixtures.make(id)
        let session = MultimodalReviewSession(review: review, source: source)
        session.load()
        return (session, source)
    }

    @MainActor
    func testAllSharedSyntheticFixturesEnforceTheirExpectedState() throws {
        let rows = try MultimodalReviewFixtures.rows()
        XCTAssertEqual(Set(rows.map(\.id)), Set(["ambiguous_visual", "stale_evidence", "missing_content", "uncertain_speaker"]))
        for row in rows {
            let (session, _) = try loaded(row.id)
            XCTAssertFalse(session.canAct, row.id)
            if row.status == "current" {
                XCTAssertNotNil(session.content, row.id)
                session.markPresented(session.presentationID)
                XCTAssertTrue(session.canAct, row.id)
                session.select(session.review.alternatives[0])
                let decision = try XCTUnwrap(session.decide(.accept), row.id)
                XCTAssertEqual(decision.revision, row.revision)
                XCTAssertEqual(decision.caseDigest, session.review.digest)
                XCTAssertEqual(decision.evidenceDigest, session.review.evidence.evidenceDigest)
                XCTAssertFalse(decision.allowsAutomaticFinalization)
            } else {
                XCTAssertNil(session.content, row.id)
                XCTAssertEqual(session.refusal, row.status == "stale" ? .changed : .missing)
                session.markPresented(session.presentationID)
                XCTAssertNil(session.decide(.accept))
                XCTAssertNil(session.decide(.reject))
                XCTAssertNil(session.decide(.defer))
            }
        }
    }

    @MainActor
    func testUnseenContentCannotSelectOrProduceAnyDecision() throws {
        let (session, _) = try loaded()
        session.select(session.review.alternatives[0])
        XCTAssertNil(session.selected)
        for choice in [MultimodalReviewChoice.accept, .reject, .defer] { XCTAssertNil(session.decide(choice)) }
        session.markPresented(UUID())
        XCTAssertFalse(session.canAct)
        session.markPresented(session.presentationID)
        XCTAssertNil(session.decide(.accept))
        session.select(session.review.alternatives[1])
        XCTAssertEqual(session.decide(.accept)?.alternativeReference, session.review.alternatives[1])
    }

    @MainActor
    func testReloadAndHideInvalidatePresentationAndSelection() throws {
        let (session, _) = try loaded()
        let previous = session.presentationID
        session.markPresented(previous)
        session.select(session.review.alternatives[0])
        session.load()
        session.markPresented(previous)
        XCTAssertFalse(session.canAct)
        XCTAssertNil(session.selected)
        session.markPresented(session.presentationID)
        XCTAssertTrue(session.canAct)
        session.hide()
        XCTAssertNil(session.content)
        XCTAssertFalse(session.canAct)
        XCTAssertNil(session.decide(.reject))
    }

    @MainActor
    func testEveryActionRechecksSourceRevisionAndClearsStaleContent() throws {
        for choice in [MultimodalReviewChoice.accept, .reject, .defer] {
            let (session, source) = try loaded()
            session.markPresented(session.presentationID)
            session.select(session.review.alternatives[0])
            source.revision = String(repeating: "e", count: 64)
            XCTAssertNil(session.decide(choice))
            XCTAssertEqual(session.refusal, .changed)
            XCTAssertNil(session.content)
            XCTAssertFalse(session.canAct)
        }
    }

    @MainActor
    func testChangedDisplayContentCannotReuseAnUnchangedRevision() throws {
        for changeImage in [false, true] {
            let (session, source) = try loaded()
            session.markPresented(session.presentationID)
            session.select(session.review.alternatives[0])
            let original = try XCTUnwrap(source.content)
            source.content = try LocalMultimodalReviewContent(
                revision: original.revision, sourceDigest: original.sourceDigest,
                text: original.text, image: changeImage ? MultimodalReviewFixtures.image("17 ?") : original.image,
                alternatives: changeImage
                    ? original.alternatives
                    : [
                        LocalReviewAlternative(reference: original.alternatives[0].reference, value: "Changed candidate"),
                        original.alternatives[1],
                    ])
            XCTAssertNil(session.decide(.accept))
            XCTAssertEqual(session.refusal, .changed)
            XCTAssertNil(session.content)
        }
    }

    @MainActor
    func testRevokedMissingAndNonlocalSourcesCannotProduceDecisions() throws {
        for failure in 0..<3 {
            let (session, source) = try loaded()
            session.markPresented(session.presentationID)
            session.select(session.review.alternatives[0])
            if failure == 0 { source.revision = nil }
            if failure == 1 { source.content = nil }
            if failure == 2 { source.localOnly = false }
            XCTAssertNil(session.decide(.accept))
            XCTAssertEqual(session.refusal, failure == 2 ? .unavailable : .missing)
            XCTAssertNil(session.content)
        }
    }

    @MainActor
    func testRejectAndDeferDoNotChooseAnAlternativeOrAllowReplay() throws {
        for choice in [MultimodalReviewChoice.reject, .defer] {
            let (session, _) = try loaded("uncertain_speaker")
            session.markPresented(session.presentationID)
            let decision = try XCTUnwrap(session.decide(choice))
            XCTAssertEqual(decision.choice, choice)
            XCTAssertNil(decision.alternativeReference)
            XCTAssertFalse(decision.allowsAutomaticFinalization)
            XCTAssertNil(session.decide(choice))
            session.load()
            XCTAssertNil(session.content)
            XCTAssertTrue(session.completed)
        }
    }

    @MainActor
    func testSourceBindingAndAlternativeOrderMustMatchTheCase() throws {
        for changedSource in [true, false] {
            let (session, source) = try loaded()
            let old = try XCTUnwrap(source.content)
            source.content = try LocalMultimodalReviewContent(
                revision: old.revision, sourceDigest: changedSource ? String(repeating: "e", count: 64) : old.sourceDigest,
                text: old.text, image: old.image,
                alternatives: changedSource ? old.alternatives : old.alternatives.reversed())
            session.load()
            XCTAssertEqual(session.refusal, .changed)
            XCTAssertNil(session.content)
        }
    }

    @MainActor
    func testTextScalarBoundsAndPrivacyDescriptions() throws {
        let (session, source) = try loaded("uncertain_speaker")
        let old = try XCTUnwrap(source.content)
        XCTAssertEqual(String(reflecting: old), "LocalMultimodalReviewContent(content=<private>)")
        XCTAssertEqual(String(reflecting: old.alternatives[0]), "LocalReviewAlternative(value=<private>)")
        source.content = try LocalMultimodalReviewContent(
            revision: old.revision, sourceDigest: old.sourceDigest, text: "different length", alternatives: old.alternatives)
        session.load()
        XCTAssertEqual(session.refusal, .invalid)
        XCTAssertNil(session.content)
    }

    @MainActor
    func testInvalidReviewMetadataAndUnboundedContentAreRefused() throws {
        let (review, source) = try MultimodalReviewFixtures.make("ambiguous_visual")
        XCTAssertThrowsError(
            try MultimodalReviewCase(
                handle: review.handle, evidence: review.evidence, revision: "private input", kind: review.kind,
                alternatives: review.alternatives, uncertainty: review.uncertainty))
        XCTAssertThrowsError(
            try MultimodalReviewCase(
                handle: review.handle, evidence: review.evidence, revision: review.revision, kind: .speakerRole,
                alternatives: review.alternatives, uncertainty: [.uncertainSpeaker]))
        XCTAssertThrowsError(
            try MultimodalReviewCase(
                handle: review.handle, evidence: review.evidence, revision: review.revision, kind: review.kind,
                alternatives: [review.alternatives[0], review.alternatives[0]], uncertainty: review.uncertainty))
        XCTAssertThrowsError(try LocalReviewAlternative(reference: review.alternatives[0], value: String(repeating: "x", count: 4097)))
        XCTAssertThrowsError(
            try LocalMultimodalReviewContent(
                revision: review.revision, sourceDigest: review.sourceDigest, text: String(repeating: "x", count: 16_385),
                alternatives: try XCTUnwrap(source.content).alternatives))
    }

    @MainActor
    func testChangingHandleChangesCaseIdentityWithoutChangingEvidence() throws {
        let (original, _) = try MultimodalReviewFixtures.make("ambiguous_visual")
        let other = try MultimodalReviewCase(
            handle: LocalMultimodalReviewHandle(), evidence: original.evidence, revision: original.revision,
            kind: original.kind, alternatives: original.alternatives, uncertainty: original.uncertainty)
        XCTAssertNotEqual(original.digest, other.digest)
        XCTAssertEqual(original.evidence.evidenceDigest, other.evidence.evidenceDigest)
    }
    @MainActor
    private final class ChangingSource: LocalMultimodalReviewSource {
        var localOnly = true
        var revision: String?
        var content: LocalMultimodalReviewContent?
        var duringResolve: (() throws -> Void)?
        func currentRevision(_ handle: LocalMultimodalReviewHandle) throws -> String? { revision }
        func resolve(_ handle: LocalMultimodalReviewHandle) throws -> LocalMultimodalReviewContent? {
            try duringResolve?()
            return content
        }
    }

    @MainActor
    func testSourceChangesAndPrivateErrorsDuringResolutionAreHeld() throws {
        for mode in 0..<3 {
            let (review, fixture) = try MultimodalReviewFixtures.make("ambiguous_visual")
            let source = ChangingSource()
            source.revision = fixture.revision
            source.content = fixture.content
            let session = MultimodalReviewSession(review: review, source: source)
            session.load()
            session.markPresented(session.presentationID)
            session.select(review.alternatives[0])
            source.duringResolve = { [weak source] in
                if mode == 0 { source?.revision = String(repeating: "e", count: 64) }
                if mode == 1 { source?.localOnly = false }
                if mode == 2 { throw NSError(domain: "synthetic-private-payload", code: 1) }
            }
            XCTAssertNil(session.decide(.accept))
            XCTAssertEqual(session.refusal, mode == 0 ? .changed : .unavailable)
            XCTAssertNil(session.content)
            XCTAssertFalse(session.canAct)
        }
    }

    @MainActor
    func testProvisionalTranscriptMayDeferButCannotBeAccepted() throws {
        let (original, source) = try MultimodalReviewFixtures.make("uncertain_speaker")
        var wire = try XCTUnwrap(JSONSerialization.jsonObject(with: original.evidence.toJSON()) as? [String: Any])
        var payload = try XCTUnwrap(wire["payload"] as? [String: Any])
        payload["finalized"] = false
        wire["payload"] = payload
        wire.removeValue(forKey: "evidence_digest")
        wire["evidence_digest"] = String(ClinicalBrief.hash(try ClinicalBrief.canonical(wire)).dropFirst(7))
        let review = try MultimodalReviewCase(
            handle: original.handle, evidence: MultimodalPacket.fromJSON(ClinicalBrief.canonical(wire)),
            revision: original.revision, kind: original.kind, alternatives: original.alternatives,
            uncertainty: [.uncertainSpeaker, .provisionalTranscript])
        let session = MultimodalReviewSession(review: review, source: source)
        session.load()
        session.markPresented(session.presentationID)
        session.select(review.alternatives[0])
        XCTAssertFalse(session.canAccept)
        XCTAssertNil(session.decide(.accept))
        XCTAssertEqual(session.refusal, .provisional)
        XCTAssertEqual(session.decide(.defer)?.choice, .defer)
    }

    @MainActor
    func testSourcePixelMutationCannotChangeOwnedDisplayedImage() throws {
        let (review, fixture) = try MultimodalReviewFixtures.make("ambiguous_visual")
        let storage = UnsafeMutableRawPointer.allocate(byteCount: 4, alignment: 4)
        storage.storeBytes(of: UInt32.max, as: UInt32.self)
        let provider = try XCTUnwrap(
            CGDataProvider(dataInfo: nil, data: storage, size: 4) { _, pointer, _ in
                pointer.deallocate()
            })
        let image = try XCTUnwrap(
            CGImage(
                width: 1, height: 1, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: 4,
                space: CGColorSpace(name: CGColorSpace.sRGB)!,
                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
                provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent))
        let content = try LocalMultimodalReviewContent(
            revision: review.revision, sourceDigest: review.sourceDigest, image: image,
            alternatives: XCTUnwrap(fixture.content).alternatives)
        let before = try XCTUnwrap(content.image?.dataProvider?.data) as Data
        storage.storeBytes(of: UInt32.zero, as: UInt32.self)
        XCTAssertEqual(try XCTUnwrap(content.image?.dataProvider?.data) as Data, before)
        XCTAssertNotEqual(try XCTUnwrap(image.dataProvider?.data) as Data, before)
    }

}
