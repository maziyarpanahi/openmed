import XCTest

@testable import OpenMedKit

final class AmbientCorrectionsTests: XCTestCase {
    private func opaque(_ number: Int) -> String {
        String(format: "00000000-0000-4000-8000-%012x", number)
    }

    private func digest(_ character: Character) -> String {
        "sha256:" + String(repeating: String(character), count: 64)
    }

    private func fixture() throws -> ([AmbientEvidenceRevision], [AmbientDraftClaim]) {
        let evidence = try (1...4).map { try AmbientEvidenceRevision(evidenceID: opaque($0), revision: 0, digest: digest(Character(String($0)))) }
        let claims = try [
            AmbientDraftClaim(claimID: opaque(11), citations: Array(evidence.prefix(3))),
            AmbientDraftClaim(claimID: opaque(12), citations: Array(evidence.prefix(3))),
            AmbientDraftClaim(claimID: opaque(13), citations: [evidence[3]]),
        ]
        return (evidence, claims)
    }

    private func authority(_ reviewer: String, _ action: String) -> Bool {
        reviewer == opaque(900) && ["correct", "verify", "approve", "export"].contains(action)
    }

    private func ledger() throws -> AmbientCorrectionLedger {
        let (evidence, claims) = try fixture()
        return try AmbientCorrectionLedger(evidence: evidence, claims: claims, authority: authority)
    }

    private func binding(_ book: AmbientCorrectionLedger) throws -> AmbientEventBinding {
        try AmbientEventBinding(
            eventID: opaque(100 + book.events.count), reviewerID: opaque(900),
            baseRevision: book.current.revision, transcriptRevision: book.current.transcriptRevision)
    }

    private func review(_ book: AmbientCorrectionLedger, _ claim: String, _ action: AmbientReviewAction, confirmed: Bool = true) throws {
        let state = try XCTUnwrap(book.current.claims.first { $0.claim.claimID == claim })
        try book.apply(
            .review(
                AmbientClaimReviewEvent(
                    binding: binding(book), claimID: claim, claimRevision: state.revision, action: action,
                    verificationDigest: action == .verify ? digest("a") : nil,
                    reviewerConfirmed: action == .approve && confirmed)))
    }

    private func approveAll(_ book: AmbientCorrectionLedger) throws {
        for state in book.current.claims {
            try review(book, state.claim.claimID, .verify)
            try review(book, state.claim.claimID, .approve)
        }
    }

    private func correction(_ book: AmbientCorrectionLedger, index: Int = 0, kind: AmbientCorrectionKind = .medication) throws -> AmbientLedgerEvent {
        .correction(
            try AmbientCorrectionEvent(
                binding: binding(book), kind: kind,
                changes: [AmbientEvidenceChange(before: book.current.evidence[index], afterDigest: digest("b"))]))
    }

    private func conflict(_ expected: AmbientReviewConflict, _ body: () throws -> Void, file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertThrowsError(try body(), file: file, line: line) { error in
            XCTAssertEqual(error as? AmbientReviewConflict, expected, file: file, line: line)
        }
    }

    func testMedicationNegationSpeakerInvalidateAllDependentApprovals() throws {
        for (index, kind) in [AmbientCorrectionKind.medication, .negation, .speaker].enumerated() {
            let book = try ledger()
            try approveAll(book)
            let old = book.current
            let output = try book.exportReviewed(revision: old.revision, reviewerID: opaque(900), reviewerConfirmed: true)
            XCTAssertTrue(output.nonDiagnostic)
            XCTAssertEqual(output.advisory, AmbientCorrectionLedger.advisory)
            try book.apply(correction(book, index: index, kind: kind))
            for state in book.current.claims.prefix(2) {
                XCTAssertNil(state.verificationEvent)
                XCTAssertNil(state.approvalEvent)
                XCTAssertEqual(state.revision, 1)
                XCTAssertEqual(state.claim.citations[index].revision, 1)
                conflict(.verificationRequired) { try self.review(book, state.claim.claimID, .approve) }
            }
            XCTAssertEqual(book.current.claims[2], old.claims[2])
            XCTAssertEqual(book.current.evidence[index].evidenceID, old.evidence[index].evidenceID)
            XCTAssertTrue(old.claims.allSatisfy { $0.approvalEvent != nil })
            conflict(.staleRevision) { _ = try book.exportReviewed(revision: old.revision, reviewerID: self.opaque(900), reviewerConfirmed: true) }
            conflict(.reviewRequired) { _ = try book.exportReviewed(revision: book.current.revision, reviewerID: self.opaque(900), reviewerConfirmed: true) }
            for state in book.current.claims.prefix(2) {
                try review(book, state.claim.claimID, .verify)
                try review(book, state.claim.claimID, .approve)
            }
            _ = try book.exportReviewed(revision: book.current.revision, reviewerID: opaque(900), reviewerConfirmed: true)
        }
    }

    func testReplayPreservesGraphAndAuthority() throws {
        let book = try ledger()
        try approveAll(book)
        try book.apply(correction(book))
        try approveAll(book)
        let (evidence, claims) = try fixture()
        let replay = try AmbientCorrectionLedger.replay(evidence: evidence, claims: claims, events: book.events, authority: authority)
        XCTAssertEqual(replay.graph, book.graph)
        XCTAssertEqual(replay.events, book.events)
        conflict(.authorityRequired) {
            _ = try AmbientCorrectionLedger.replay(evidence: evidence, claims: claims, events: book.events, authority: { _, _ in false })
        }
    }

    func testStaleDuplicateAndAtomicConflict() throws {
        let book = try ledger()
        let first = try correction(book)
        let original = book.current.evidence[0]
        try book.apply(first)
        let graph = book.graph
        conflict(.duplicateEvent) { try book.apply(first) }
        let staleBinding = try AmbientEventBinding(eventID: opaque(777), reviewerID: opaque(900), baseRevision: 0, transcriptRevision: 0)
        let stale = try AmbientCorrectionEvent(binding: staleBinding, kind: .medication, changes: [AmbientEvidenceChange(before: original, afterDigest: digest("c"))])
        conflict(.staleRevision) { try book.apply(.correction(stale)) }
        let rebound = try AmbientCorrectionEvent(binding: binding(book), kind: .medication, changes: stale.changes)
        conflict(.staleEvidence) { try book.apply(.correction(rebound)) }
        let unknown = try AmbientEvidenceRevision(evidenceID: opaque(999), revision: 0, digest: digest("d"))
        let atomic = try AmbientCorrectionEvent(
            binding: binding(book), kind: .transcript,
            changes: [
                AmbientEvidenceChange(before: book.current.evidence[1], afterDigest: digest("e")),
                AmbientEvidenceChange(before: unknown, afterDigest: digest("f")),
            ])
        conflict(.unknownReference) { try book.apply(.correction(atomic)) }
        XCTAssertEqual(book.graph, graph)
        XCTAssertEqual(book.events, [first])
    }

    func testConfirmationReverificationAndAuthorityFailures() throws {
        let book = try ledger()
        try approveAll(book)
        conflict(.confirmationRequired) { _ = try book.exportReviewed(revision: book.current.revision, reviewerID: self.opaque(900), reviewerConfirmed: false) }
        try review(book, opaque(11), .verify)
        XCTAssertNil(book.current.claims[0].approvalEvent)
        conflict(.confirmationRequired) { try self.review(book, self.opaque(11), .approve, confirmed: false) }
        let (evidence, claims) = try fixture()
        let denied = try AmbientCorrectionLedger(evidence: evidence, claims: claims) { _, _ in
            throw NSError(domain: "Synthetic private payload", code: 1)
        }
        conflict(.authorityRequired) { try denied.apply(self.correction(denied)) }
        XCTAssertEqual(denied.graph.count, 1)
    }

    func testInvalidMetadataAndDuplicateDependencies() throws {
        for value in ["Synthetic Alice", "/private/path", "Bearer synthetic", opaque(1) + "\n"] {
            conflict(.invalidMetadata) { _ = try AmbientEvidenceRevision(evidenceID: value, revision: 0, digest: self.digest("a")) }
        }
        for value in [-1, 9_007_199_254_740_992] {
            conflict(.invalidMetadata) { _ = try AmbientEvidenceRevision(evidenceID: self.opaque(1), revision: value, digest: self.digest("a")) }
        }
        let (evidence, claims) = try fixture()
        conflict(.invalidMetadata) { _ = try AmbientDraftClaim(claimID: self.opaque(1), citations: [evidence[0], evidence[0]]) }
        conflict(.invalidMetadata) { _ = try AmbientEvidenceChange(before: evidence[0], afterDigest: evidence[0].digest) }
        conflict(.invalidMetadata) { _ = try AmbientCorrectionLedger(evidence: evidence + evidence, claims: claims, authority: self.authority) }
    }
}
