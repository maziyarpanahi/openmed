import Foundation

/// Controlled, value-free failures from the local correction boundary.
public enum AmbientReviewConflict: String, Error, LocalizedError {
    case invalidMetadata = "invalid_metadata"
    case staleRevision = "stale_revision"
    case staleEvidence = "stale_evidence"
    case duplicateEvent = "duplicate_event"
    case unknownReference = "unknown_reference"
    case authorityRequired = "authority_required"
    case verificationRequired = "verification_required"
    case reviewRequired = "review_required"
    case confirmationRequired = "confirmation_required"

    public var errorDescription: String? { rawValue }
}

private func requireAmbient(_ condition: Bool, _ code: AmbientReviewConflict = .invalidMetadata) throws {
    guard condition else { throw code }
}

private func ambientOpaque(_ value: String) throws {
    try requireAmbient(value.range(of: "^[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}\\z", options: .regularExpression) != nil)
}

private func ambientDigest(_ value: String) throws {
    try requireAmbient(value.range(of: "^sha256:[0-9a-f]{64}\\z", options: .regularExpression) != nil)
}

private func ambientRevision(_ value: Int) throws {
    try requireAmbient(value >= 0 && value < 9_007_199_254_740_992)
}

/// Stable evidence identity and one immutable revision, without source content.
public struct AmbientEvidenceRevision: Equatable {
    public let evidenceID: String
    public let revision: Int
    public let digest: String

    public init(evidenceID: String, revision: Int, digest: String) throws {
        try ambientOpaque(evidenceID)
        try ambientRevision(revision)
        try ambientDigest(digest)
        self.evidenceID = evidenceID
        self.revision = revision
        self.digest = digest
    }
}

/// A claim with complete direct evidence dependencies supplied by the caller.
public struct AmbientDraftClaim: Equatable {
    public let claimID: String
    public let citations: [AmbientEvidenceRevision]

    public init(claimID: String, citations: [AmbientEvidenceRevision]) throws {
        try ambientOpaque(claimID)
        try requireAmbient(!citations.isEmpty && Set(citations.map(\.evidenceID)).count == citations.count)
        self.claimID = claimID
        self.citations = citations
    }
}

/// Compare-and-set replacement of one evidence unit's keyed digest.
public struct AmbientEvidenceChange: Equatable {
    public let before: AmbientEvidenceRevision
    public let afterDigest: String

    public init(before: AmbientEvidenceRevision, afterDigest: String) throws {
        try ambientDigest(afterDigest)
        try requireAmbient(before.digest != afterDigest)
        self.before = before
        self.afterDigest = afterDigest
    }
}

/// Semantic categories without corrected transcript content or speaker names.
public enum AmbientCorrectionKind: String {
    case medication, negation, speaker, transcript
}

/// Separate trusted verification and explicitly confirmed human approval.
public enum AmbientReviewAction: String {
    case verify, approve
}

/// Versioned event metadata bound to the exact draft and transcript revision.
public struct AmbientEventBinding: Equatable {
    public let eventID: String
    public let reviewerID: String
    public let baseRevision: Int
    public let transcriptRevision: Int
    public let schemaVersion: Int

    public init(eventID: String, reviewerID: String, baseRevision: Int, transcriptRevision: Int, schemaVersion: Int = 1) throws {
        try ambientOpaque(eventID)
        try ambientOpaque(reviewerID)
        try ambientRevision(baseRevision)
        try ambientRevision(transcriptRevision)
        try requireAmbient(schemaVersion == 1)
        self.eventID = eventID
        self.reviewerID = reviewerID
        self.baseRevision = baseRevision
        self.transcriptRevision = transcriptRevision
        self.schemaVersion = schemaVersion
    }
}

/// One atomic human correction; protected content remains outside this ledger.
public struct AmbientCorrectionEvent: Equatable {
    public let binding: AmbientEventBinding
    public let kind: AmbientCorrectionKind
    public let changes: [AmbientEvidenceChange]

    public init(binding: AmbientEventBinding, kind: AmbientCorrectionKind, changes: [AmbientEvidenceChange]) throws {
        try requireAmbient(!changes.isEmpty && Set(changes.map { $0.before.evidenceID }).count == changes.count)
        self.binding = binding
        self.kind = kind
        self.changes = changes
    }
}

/// Revision-bound verification evidence or an explicit human approval.
public struct AmbientClaimReviewEvent: Equatable {
    public let binding: AmbientEventBinding
    public let claimID: String
    public let claimRevision: Int
    public let action: AmbientReviewAction
    public let verificationDigest: String?
    public let reviewerConfirmed: Bool

    public init(
        binding: AmbientEventBinding, claimID: String, claimRevision: Int, action: AmbientReviewAction,
        verificationDigest: String? = nil, reviewerConfirmed: Bool = false
    ) throws {
        try ambientOpaque(claimID)
        try ambientRevision(claimRevision)
        if action == .verify {
            guard let verificationDigest else { throw AmbientReviewConflict.invalidMetadata }
            try ambientDigest(verificationDigest)
            try requireAmbient(!reviewerConfirmed)
        } else {
            try requireAmbient(verificationDigest == nil)
        }
        self.binding = binding
        self.claimID = claimID
        self.claimRevision = claimRevision
        self.action = action
        self.verificationDigest = verificationDigest
        self.reviewerConfirmed = reviewerConfirmed
    }
}

/// Ordered, immutable replay journal entries without source payloads.
public enum AmbientLedgerEvent: Equatable {
    case correction(AmbientCorrectionEvent)
    case review(AmbientClaimReviewEvent)

    fileprivate var binding: AmbientEventBinding {
        switch self {
        case .correction(let event): return event.binding
        case .review(let event): return event.binding
        }
    }
}

/// Claim verification and approval links for exactly one claim revision.
public struct AmbientClaimState: Equatable {
    public let claim: AmbientDraftClaim
    public let revision: Int
    public let verificationEvent: String?
    public let approvalEvent: String?

    fileprivate init(claim: AmbientDraftClaim, revision: Int = 0, verificationEvent: String? = nil, approvalEvent: String? = nil) {
        self.claim = claim
        self.revision = revision
        self.verificationEvent = verificationEvent
        self.approvalEvent = approvalEvent
    }
}

/// Immutable revision graph node; old approvals remain historical only.
public struct AmbientDraftRevision: Equatable {
    public let revision: Int
    public let transcriptRevision: Int
    public let parentRevision: Int?
    public let eventID: String?
    public let evidence: [AmbientEvidenceRevision]
    public let claims: [AmbientClaimState]
}

/// Consequential current reviewed metadata with a mandatory non-diagnostic notice.
public struct AmbientReviewedDraft: Equatable {
    public let schemaVersion: Int
    public let advisory: String
    public let nonDiagnostic: Bool
    public let reviewerConfirmed: Bool
    public let draft: AmbientDraftRevision
}

/// Single-writer, in-memory correction propagation with injected local authority.
///
/// Applications must serialize calls and provide complete direct citation
/// dependencies. Authority actions are `correct`, `verify`, `approve`, `export`.
/// Verification digests refer to actual caller-owned verification evidence;
/// this ledger neither performs inference nor certifies clinical correctness.
/// No network, protected text, ASR, EHR write, or cloud fallback is used.
public final class AmbientCorrectionLedger {
    public static let advisory = "For clinician review; not a diagnosis or treatment recommendation."
    public private(set) var graph: [AmbientDraftRevision]
    public private(set) var events: [AmbientLedgerEvent] = []
    private let authority: (String, String) throws -> Bool

    public var current: AmbientDraftRevision { graph[graph.count - 1] }

    public init(evidence: [AmbientEvidenceRevision], claims: [AmbientDraftClaim], transcriptRevision: Int = 0, authority: @escaping (String, String) throws -> Bool) throws {
        try requireAmbient(!evidence.isEmpty && !claims.isEmpty)
        try ambientRevision(transcriptRevision)
        try requireAmbient(Set(evidence.map(\.evidenceID)).count == evidence.count)
        try requireAmbient(Set(claims.map(\.claimID)).count == claims.count)
        for claim in claims {
            try requireAmbient(claim.citations.allSatisfy { evidence.contains($0) })
        }
        self.authority = authority
        graph = [
            AmbientDraftRevision(
                revision: 0, transcriptRevision: transcriptRevision, parentRevision: nil, eventID: nil,
                evidence: evidence.sorted { $0.evidenceID < $1.evidenceID },
                claims: claims.sorted { $0.claimID < $1.claimID }.map { AmbientClaimState(claim: $0) })
        ]
    }

    private func authorize(_ reviewerID: String, _ action: String) throws {
        try ambientOpaque(reviewerID)
        let granted: Bool
        do { granted = try authority(reviewerID, action) } catch { throw AmbientReviewConflict.authorityRequired }
        try requireAmbient(granted, .authorityRequired)
    }

    /// Atomically apply a correction or review; rejection leaves the graph unchanged.
    @discardableResult
    public func apply(_ event: AmbientLedgerEvent) throws -> AmbientDraftRevision {
        let binding = event.binding
        try requireAmbient(!events.contains { $0.binding.eventID == binding.eventID }, .duplicateEvent)
        let prior = current
        try requireAmbient(binding.baseRevision == prior.revision && binding.transcriptRevision == prior.transcriptRevision, .staleRevision)
        try ambientRevision(prior.revision + 1)
        var evidence = prior.evidence
        var claims = prior.claims
        var transcriptRevision = prior.transcriptRevision
        switch event {
        case .correction(let correction):
            try authorize(binding.reviewerID, "correct")
            var replacements: [String: AmbientEvidenceRevision] = [:]
            for change in correction.changes {
                try requireAmbient(evidence.contains { $0.evidenceID == change.before.evidenceID }, .unknownReference)
                try requireAmbient(evidence.contains(change.before), .staleEvidence)
                replacements[change.before.evidenceID] = try AmbientEvidenceRevision(
                    evidenceID: change.before.evidenceID, revision: change.before.revision + 1, digest: change.afterDigest)
            }
            evidence = evidence.map { replacements[$0.evidenceID] ?? $0 }
            claims = try claims.map { state in
                guard state.claim.citations.contains(where: { replacements[$0.evidenceID] != nil }) else { return state }
                let claim = try AmbientDraftClaim(
                    claimID: state.claim.claimID, citations: state.claim.citations.map { replacements[$0.evidenceID] ?? $0 })
                try ambientRevision(state.revision + 1)
                return AmbientClaimState(claim: claim, revision: state.revision + 1)
            }
            transcriptRevision += 1
            try ambientRevision(transcriptRevision)
        case .review(let review):
            try authorize(binding.reviewerID, review.action.rawValue)
            guard let index = claims.firstIndex(where: { $0.claim.claimID == review.claimID }) else { throw AmbientReviewConflict.unknownReference }
            let state = claims[index]
            try requireAmbient(state.revision == review.claimRevision, .staleRevision)
            if review.action == .verify {
                claims[index] = AmbientClaimState(claim: state.claim, revision: state.revision, verificationEvent: binding.eventID)
            } else {
                try requireAmbient(state.verificationEvent != nil, .verificationRequired)
                try requireAmbient(review.reviewerConfirmed, .confirmationRequired)
                claims[index] = AmbientClaimState(
                    claim: state.claim, revision: state.revision, verificationEvent: state.verificationEvent, approvalEvent: binding.eventID)
            }
        }
        let node = AmbientDraftRevision(
            revision: prior.revision + 1, transcriptRevision: transcriptRevision, parentRevision: prior.revision,
            eventID: binding.eventID, evidence: evidence, claims: claims)
        events.append(event)
        graph.append(node)
        return node
    }

    /// Export current reviewed metadata only after explicit reviewer confirmation.
    public func exportReviewed(revision: Int, reviewerID: String, reviewerConfirmed: Bool) throws -> AmbientReviewedDraft {
        try ambientRevision(revision)
        try requireAmbient(revision == current.revision, .staleRevision)
        try authorize(reviewerID, "export")
        try requireAmbient(reviewerConfirmed, .confirmationRequired)
        try requireAmbient(current.claims.allSatisfy { $0.verificationEvent != nil && $0.approvalEvent != nil }, .reviewRequired)
        return AmbientReviewedDraft(
            schemaVersion: 1, advisory: Self.advisory, nonDiagnostic: true, reviewerConfirmed: true, draft: current)
    }

    /// Replay in order against identical initial metadata, rechecking every grant.
    public static func replay(
        evidence: [AmbientEvidenceRevision], claims: [AmbientDraftClaim], events: [AmbientLedgerEvent], transcriptRevision: Int = 0,
        authority: @escaping (String, String) throws -> Bool
    ) throws -> AmbientCorrectionLedger {
        let ledger = try AmbientCorrectionLedger(evidence: evidence, claims: claims, transcriptRevision: transcriptRevision, authority: authority)
        for event in events { try ledger.apply(event) }
        return ledger
    }
}
