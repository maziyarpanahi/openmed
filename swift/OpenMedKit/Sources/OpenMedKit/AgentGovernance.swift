import Foundation

/// Closed outcome vocabulary shared with Python's WorkflowOutcome.
public enum AgentOutcomeClass: String, CaseIterable, Sendable {
    case success, abstained
    case reviewRequired = "review_required"
    case policyDenied = "policy_denied"
    case failed
}

/// Content-free artifact metadata. Parsing neither fetches nor authenticates it.
public struct AgentArtifactReference: Equatable, Sendable, CustomStringConvertible, CustomDebugStringConvertible {
    public let artifactID: String
    public let kind: Kind
    public let schemaID: String
    public let sha256: String
    public let byteSize: Int64
    private let metadata: AgentJSON

    public enum Kind: String, Sendable { case evidence, preview, fhir, omop, evaluation }

    public static func parse(_ data: Data) throws -> Self { try parseValue(AgentJSON.parse(data)) }
    public func canonicalJSON() -> Data { metadata.canonical() }
    public var description: String { "AgentArtifactReference(<metadata-only>)" }
    public var debugDescription: String { description }

    fileprivate static func parseValue(_ value: AgentJSON) throws -> Self {
        var fields = try value.object(fields: ["artifact_id", "kind", "schema_id", "sha256", "byte_size"], optional: ["version"])
        fields["version"] = fields["version"] ?? .integer(1)
        guard fields["version"] == .integer(1) else { throw AgentGovernanceError.unsupportedVersion }
        let artifactID = try fields["artifact_id"]!.string(pattern: "art_[0-9a-f]{32}")
        guard let kind = Kind(rawValue: try fields["kind"]!.string()) else { throw AgentGovernanceError.invalidMetadata }
        let schemaID = try fields["schema_id"]!.string(pattern: "[a-z][a-z0-9]*(?:[._-][a-z0-9]+)+\\.v[1-9][0-9]*")
        let sha256 = try fields["sha256"]!.string(pattern: "[0-9a-f]{64}")
        let byteSize = try fields["byte_size"]!.integer(minimum: 1)
        return Self(artifactID: artifactID, kind: kind, schemaID: schemaID, sha256: sha256, byteSize: byteSize, metadata: .object(fields))
    }
}

/// A local request for human review; it grants no clinical authority.
public struct AgentReviewerHandoff: Equatable, Sendable, CustomStringConvertible, CustomDebugStringConvertible {
    public static let schemaVersion = "openmed.agent.reviewer_handoff.v1"
    public let runID: String
    public let workflowID: String
    public let reasonCode: String
    public let requestedDecision: RequestedDecision
    public let evidenceReferences: [AgentArtifactReference]
    public let issuedAt: Date
    public let expiresAt: Date
    private let metadata: AgentJSON

    public enum RequestedDecision: String, Sendable {
        case confirmAbstention = "confirm_abstention"
        case reviewEvidence = "review_evidence"
        case resolveEvidenceConflict = "resolve_evidence_conflict"
        case assessSafety = "assess_safety"
        case decideNextStep = "decide_next_step"
    }

    public var requiresHumanReview: Bool { true }
    public var authorizesClinicalAction: Bool { false }
    public var description: String { "AgentReviewerHandoff(<review-request>)" }
    public var debugDescription: String { description }
    public func canonicalJSON() -> Data { metadata.canonical() }

    /// Validate whole-second UTC metadata using the caller's injected local time.
    public static func parse(_ data: Data, now: Date = Date()) throws -> Self {
        guard now.timeIntervalSince1970.isFinite else { throw AgentGovernanceError.invalidMetadata }
        var fields = try AgentJSON.parse(data).object(
            fields: ["run_id", "workflow_id", "reason_code", "requested_decision", "evidence_references", "issued_at", "expires_at"], optional: ["schema_version"])
        fields["schema_version"] = fields["schema_version"] ?? .string(schemaVersion)
        try agentVersion(fields["schema_version"], expected: schemaVersion)
        let runID = try fields["run_id"]!.string(pattern: "run_[0-9a-f]{32}")
        let workflowID = try agentWorkflowID(fields["workflow_id"]!)
        let reason = try fields["reason_code"]!.string()
        guard ["insufficient_evidence", "out_of_scope", "low_confidence", "conflicting_evidence", "safety_review", "human_gate"].contains(reason),
            let decision = RequestedDecision(rawValue: try fields["requested_decision"]!.string())
        else { throw AgentGovernanceError.invalidMetadata }
        let references = try fields["evidence_references"]!.array(maximum: 64).map { try AgentArtifactReference.parseValue($0) }
        guard Set(references.map { $0.artifactID }).count == references.count else { throw AgentGovernanceError.invalidMetadata }
        fields["evidence_references"] = .array(try references.map { try AgentJSON.parse($0.canonicalJSON()) })
        let issued = try timestamp(fields["issued_at"]!)
        let expires = try timestamp(fields["expires_at"]!)
        guard expires > issued else { throw AgentGovernanceError.invalidMetadata }
        guard expires > now else { throw AgentGovernanceError.expired }
        return Self(runID: runID, workflowID: workflowID, reasonCode: reason, requestedDecision: decision, evidenceReferences: references, issuedAt: issued, expiresAt: expires, metadata: .object(fields))
    }

    private static func timestamp(_ value: AgentJSON) throws -> Date {
        let text = try value.string(pattern: "[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z")
        guard !text.hasPrefix("0000") else { throw AgentGovernanceError.invalidMetadata }
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.calendar = Calendar(identifier: .gregorian)
        formatter.timeZone = TimeZone(secondsFromGMT: 0)
        formatter.dateFormat = "yyyy-MM-dd'T'HH:mm:ss'Z'"
        formatter.isLenient = false
        guard let date = formatter.date(from: text), formatter.string(from: date) == text else { throw AgentGovernanceError.invalidMetadata }
        return date
    }
}

/// Exact existing Python ApprovalReceipt metadata, never proof of custody alone.
public struct AgentApprovalReceipt: Equatable, Sendable, CustomStringConvertible, CustomDebugStringConvertible {
    public static let schemaVersion = "openmed.agent.approval_receipt.v2"
    public let actionDigest: String
    public let tokenDigest: String
    private let metadata: AgentJSON

    public static func parse(_ data: Data) throws -> Self {
        let fields = try AgentJSON.parse(data, maximum: 65_536).object(fields: ["schema_version", "action_digest", "token_digest", "code"])
        try agentVersion(fields["schema_version"], expected: schemaVersion)
        let action = try fields["action_digest"]!.string(pattern: agentDigestPattern)
        let token = try fields["token_digest"]!.string(pattern: agentDigestPattern)
        guard fields["code"] == .string("approved") else { throw AgentGovernanceError.invalidMetadata }
        return Self(actionDigest: action, tokenDigest: token, metadata: .object(fields))
    }

    public var authorizesClinicalAction: Bool { false }
    public var receiptDigest: String { metadata.digest() }
    public func canonicalJSON() -> Data { metadata.canonical() }
    public var description: String { "AgentApprovalReceipt(<metadata-only>)" }
    public var debugDescription: String { description }
}

/// A closed observation vocabulary; these states never authorize dispatch.
public enum AgentApprovalEvidenceReason: String, CaseIterable, Sendable {
    case verified, expired, replayed
    case unsupportedAuthority = "unsupported_authority"
    case unrecognizedReceipt = "unrecognized_receipt"
    case authorityUnavailable = "authority_unavailable"
    case futureReceipt = "future_receipt"
    case actionMismatch = "action_mismatch"
    case reviewerRoleMismatch = "reviewer_role_mismatch"
    case nonceStoreUnavailable = "nonce_store_unavailable"
    case clockUnavailable = "clock_unavailable"
}

/// A serialized receipt observation remains untrusted when parsed elsewhere.
public struct AgentApprovalEvidenceResult: Equatable, Sendable, CustomStringConvertible, CustomDebugStringConvertible {
    public static let schemaVersion = "openmed.agent.approval_evidence.v1"
    public let reasonCode: AgentApprovalEvidenceReason
    public let actionDigest: String
    public let receiptDigest: String
    public var authorizesClinicalAction: Bool { false }
    public var description: String { "AgentApprovalEvidenceResult(reason_code=\(reasonCode.rawValue))" }
    public var debugDescription: String { description }

    public static func parse(_ data: Data) throws -> Self {
        let fields = try AgentJSON.parse(data, maximum: 65_536).object(fields: ["schema_version", "status", "reason_code", "action_digest", "receipt_digest"])
        try agentVersion(fields["schema_version"], expected: schemaVersion)
        guard let reason = AgentApprovalEvidenceReason(rawValue: try fields["reason_code"]!.string()),
            fields["status"] == .string(reason == .verified ? "verified" : "refused")
        else { throw AgentGovernanceError.invalidMetadata }
        return Self(reasonCode: reason, actionDigest: try fields["action_digest"]!.string(pattern: agentDigestPattern), receiptDigest: try fields["receipt_digest"]!.string(pattern: agentDigestPattern))
    }

    public func canonicalJSON() -> Data {
        AgentJSON.object([
            "schema_version": .string(Self.schemaVersion), "status": .string(reasonCode == .verified ? "verified" : "refused"),
            "reason_code": .string(reasonCode.rawValue), "action_digest": .string(actionDigest), "receipt_digest": .string(receiptDigest),
        ]).canonical()
    }
}

/// Strict content-free RunSummary projection; no prompts, paths or tool values.
public struct AgentRunEvidence: Equatable, Sendable, CustomStringConvertible, CustomDebugStringConvertible {
    public static let schemaVersion = "openmed.agent.run_summary.v1"
    public let workflowIDs: [String]
    public let outcomeCounts: [AgentOutcomeClass: Int64]
    public let toolCallCount: Int64
    public let durationSeconds: Double
    public let artifactDigests: [String]
    private let metadata: AgentJSON

    public static func parse(_ data: Data) throws -> Self {
        var fields = try AgentJSON.parse(data).object(fields: ["schema_version", "workflow_ids", "outcome_counts", "tool_call_count", "duration_seconds", "artifact_digests"])
        try agentVersion(fields["schema_version"], expected: schemaVersion)
        let ids = try fields["workflow_ids"]!.array(maximum: 1_024).map { try $0.string(pattern: "[A-Za-z0-9](?:[A-Za-z0-9_.-]{0,126}[A-Za-z0-9])?") }
        guard ids == Array(Set(ids)).sorted() else { throw AgentGovernanceError.invalidMetadata }
        let counts = try fields["outcome_counts"]!.object(fields: Set(AgentOutcomeClass.allCases.map { $0.rawValue }))
        var normalized: [AgentOutcomeClass: Int64] = [:]
        for outcome in AgentOutcomeClass.allCases { normalized[outcome] = try counts[outcome.rawValue]!.integer(maximum: 10_000) }
        guard normalized.values.reduce(0, +) <= 10_000 else { throw AgentGovernanceError.invalidMetadata }
        let calls = try fields["tool_call_count"]!.integer(maximum: 10_000_000)
        let duration: Double
        switch fields["duration_seconds"]! {
        case .integer(let value): duration = Double(value)
        case .number(let value): duration = value
        default: throw AgentGovernanceError.invalidMetadata
        }
        guard duration.isFinite, 0 <= duration, duration <= 31_536_000 else { throw AgentGovernanceError.invalidMetadata }
        fields["duration_seconds"] = .number(duration)
        let digests = try fields["artifact_digests"]!.array(maximum: 4_096).map { try $0.string(pattern: agentDigestPattern) }
        guard digests == Array(Set(digests)).sorted() else { throw AgentGovernanceError.invalidMetadata }
        return Self(workflowIDs: ids, outcomeCounts: normalized, toolCallCount: calls, durationSeconds: duration, artifactDigests: digests, metadata: .object(fields))
    }

    public var authorizesClinicalAction: Bool { false }
    public func canonicalJSON() -> Data { metadata.canonical() }
    public var description: String { "AgentRunEvidence(workflows=\(workflowIDs.count), artifacts=\(artifactDigests.count))" }
    public var debugDescription: String { description }
}

/// Supported native OMOP preview: categorical metadata and a checked digest.
/// This value never contains staged row values or a writer/approval credential.
public struct AgentOMOPPreview: Equatable, Sendable, CustomStringConvertible, CustomDebugStringConvertible {
    public static let schemaVersion = "openmed.interop.omop.mutation_batch.v1"
    public let batchDigest: String
    public let previewDigest: String
    public let referenceSnapshotDigest: String
    public let mutationCount: Int
    public let isValid: Bool
    public let mutations: [Mutation]
    public let issues: [Issue]
    private let metadata: AgentJSON

    public enum Operation: String, Sendable { case insert, update, tombstone }
    public enum IssueCode: String, Sendable {
        case duplicateInsert = "duplicate_insert"
        case missingTarget = "missing_target"
        case referencedTombstone = "referenced_tombstone"
        case missingReference = "missing_reference"
    }
    public struct Mutation: Equatable, Sendable, CustomStringConvertible {
        public let ordinal: Int
        public let operation: Operation
        public let table: String
        public let fieldNames: [String]
        public let referenceCount: Int64
        public let rowDigest: String
        public var description: String { "AgentOMOPPreview.Mutation(<metadata-only>)" }
    }
    public struct Issue: Equatable, Sendable, CustomStringConvertible {
        public let ordinal: Int
        public let code: IssueCode
        public let table: String
        public let fieldName: String?
        public var description: String { "AgentOMOPPreview.Issue(code=\(code.rawValue))" }
    }

    public var authorizesClinicalAction: Bool { false }
    public var description: String { "AgentOMOPPreview(mutations=\(mutationCount), valid=\(isValid))" }
    public var debugDescription: String { description }
    public func canonicalJSON() -> Data { metadata.canonical() }

    public static func parse(_ data: Data) throws -> Self {
        let fields = try AgentJSON.parse(data).object(fields: ["schema", "batch_digest", "preview_digest", "reference_snapshot_digest", "mutation_count", "is_valid", "mutations", "issues", "operation_counts"])
        try agentVersion(fields["schema"], expected: schemaVersion)
        let batch = try fields["batch_digest"]!.string(pattern: agentDigestPattern)
        let preview = try fields["preview_digest"]!.string(pattern: agentDigestPattern)
        let snapshot = try fields["reference_snapshot_digest"]!.string(pattern: agentDigestPattern)
        let count = try fields["mutation_count"]!.integer(minimum: 1, maximum: 10_000)
        let valid = try fields["is_valid"]!.bool()
        let mutations = try fields["mutations"]!.array(maximum: 10_000)
        guard mutations.count == count else { throw AgentGovernanceError.invalidMetadata }
        var counts: [String: AgentJSON] = [:]
        var tables: [String] = []
        var typedMutations: [Mutation] = []
        for (index, mutation) in mutations.enumerated() {
            let item = try mutation.object(fields: ["ordinal", "operation", "table", "field_names", "reference_count", "row_digest"])
            guard try item["ordinal"]!.integer(maximum: 9_999) == index else { throw AgentGovernanceError.invalidMetadata }
            let operation = try item["operation"]!.string()
            guard let typedOperation = Operation(rawValue: operation) else { throw AgentGovernanceError.invalidMetadata }
            let previous = try counts[operation]?.integer() ?? 0
            counts[operation] = .integer(previous + 1)
            let table = try item["table"]!.string(pattern: "[a-z][a-z0-9_]{0,62}")
            tables.append(table)
            let names = try item["field_names"]!.array(maximum: 200_000).map { try $0.string(pattern: "[a-z][a-z0-9_]{0,62}") }
            guard names == Array(Set(names)).sorted() else { throw AgentGovernanceError.invalidMetadata }
            let referenceCount = try item["reference_count"]!.integer()
            let rowDigest = try item["row_digest"]!.string(pattern: agentDigestPattern)
            typedMutations.append(Mutation(ordinal: index, operation: typedOperation, table: table, fieldNames: names, referenceCount: referenceCount, rowDigest: rowDigest))
        }
        guard fields["operation_counts"] == .object(counts) else { throw AgentGovernanceError.invalidMetadata }
        let issues = try fields["issues"]!.array(maximum: 200_000)
        guard valid == issues.isEmpty else { throw AgentGovernanceError.invalidMetadata }
        var typedIssues: [Issue] = []
        for issue in issues {
            let item = try issue.object(fields: ["ordinal", "code", "table", "field_name"])
            let ordinal = try item["ordinal"]!.integer(maximum: count - 1)
            guard item["table"] == .string(tables[Int(ordinal)]),
                let code = IssueCode(rawValue: try item["code"]!.string())
            else { throw AgentGovernanceError.invalidMetadata }
            let fieldName = item["field_name"] == .null ? nil : try item["field_name"]!.string(pattern: "[a-z][a-z0-9_]{0,62}")
            typedIssues.append(Issue(ordinal: Int(ordinal), code: code, table: tables[Int(ordinal)], fieldName: fieldName))
        }
        var unsigned = fields
        unsigned.removeValue(forKey: "preview_digest")
        guard AgentJSON.object(unsigned).digest() == preview else { throw AgentGovernanceError.digestMismatch }
        let rows = try mutations.map { try $0.object(fields: ["ordinal", "operation", "table", "field_names", "reference_count", "row_digest"])["row_digest"]! }
        guard AgentJSON.object(["schema": .string(schemaVersion), "row_digests": .array(rows)]).digest() == batch else { throw AgentGovernanceError.digestMismatch }
        return Self(batchDigest: batch, previewDigest: preview, referenceSnapshotDigest: snapshot, mutationCount: Int(count), isValid: valid, mutations: typedMutations, issues: typedIssues, metadata: .object(fields))
    }
}

/// A caller-owned custody lookup response, never supplied by packet contents.
public enum AgentReceiptAuthority: Sendable {
    case recognized(AgentReceiptCustody)
    case unrecognized, unsupported
}

/// A trusted host's local observation of previously consumed approval custody.
/// Role and validity come from its protected store; no wire decoder is provided.
/// This observation grants no authority to execute a clinical action.
public struct AgentReceiptCustody: Sendable, CustomStringConvertible, CustomDebugStringConvertible {
    public let receiptDigest: String
    public let reviewerRole: String
    public let consumedAt: Int64
    public let expiresAt: Int64

    public init(receiptDigest: String, reviewerRole: String, consumedAt: Int64, expiresAt: Int64) throws {
        _ = try AgentJSON.string(receiptDigest).string(pattern: agentDigestPattern)
        _ = try AgentJSON.string(reviewerRole).string(pattern: agentRolePattern)
        guard consumedAt >= 0, consumedAt < expiresAt else { throw AgentGovernanceError.invalidMetadata }
        self.receiptDigest = receiptDigest
        self.reviewerRole = reviewerRole
        self.consumedAt = consumedAt
        self.expiresAt = expiresAt
    }

    public var authorizesClinicalAction: Bool { false }
    public var description: String { "AgentReceiptCustody(<local>)" }
    public var debugDescription: String { description }
}

/// Atomic extra replay protection for local receipt presentation.
/// Application-wide dispatch still requires its original durable nonce custody.
public protocol AgentReceiptReplayStore: Sendable {
    func claim(tokenDigest: String, expiresAt: Int64, now: Int64) throws -> Bool
}

/// Thread-safe process-local presentation store; contains only digests/expiry.
public final class AgentInMemoryReceiptReplayStore: AgentReceiptReplayStore, @unchecked Sendable, CustomStringConvertible {
    private let lock = NSLock()
    private var claims: [String: Int64] = [:]

    public init() {}

    public func claim(tokenDigest: String, expiresAt: Int64, now: Int64) throws -> Bool {
        _ = try AgentJSON.string(tokenDigest).string(pattern: agentDigestPattern)
        guard now >= 0, expiresAt > now else { throw AgentGovernanceError.invalidMetadata }
        lock.lock()
        defer { lock.unlock() }
        claims = claims.filter { $0.value > now }
        guard claims[tokenDigest] == nil else { return false }
        claims[tokenDigest] = expiresAt
        return true
    }

    public var description: String { "AgentInMemoryReceiptReplayStore(<local>)" }
}

/// Observe consumed approval evidence using only application-owned local hooks.
/// No signer, receipt issuer, dispatch callback, EHR or network client is present.
public final class AgentLocalApprovalEvidenceVerifier: Sendable, CustomStringConvertible, CustomDebugStringConvertible {
    private let authority: (@Sendable (String) throws -> AgentReceiptAuthority)?
    private let replayStore: any AgentReceiptReplayStore
    private let clock: @Sendable () throws -> Int64

    public init(
        authority: (@Sendable (String) throws -> AgentReceiptAuthority)? = nil,
        replayStore: any AgentReceiptReplayStore = AgentInMemoryReceiptReplayStore(),
        clock: @escaping @Sendable () throws -> Int64 = { Int64(Date().timeIntervalSince1970) }
    ) {
        self.authority = authority
        self.replayStore = replayStore
        self.clock = clock
    }

    public var description: String { "AgentLocalApprovalEvidenceVerifier(<local>)" }
    public var debugDescription: String { description }

    /// A recognized changed-action/role presentation is burned before refusal.
    /// The returned observation always has authorizesClinicalAction == false.
    public func verify(_ receipt: AgentApprovalReceipt, actionDigest: String, reviewerRole: String) throws -> AgentApprovalEvidenceResult {
        _ = try AgentJSON.string(actionDigest).string(pattern: agentDigestPattern)
        _ = try AgentJSON.string(reviewerRole).string(pattern: agentRolePattern)
        func result(_ reason: AgentApprovalEvidenceReason) -> AgentApprovalEvidenceResult {
            AgentApprovalEvidenceResult(reasonCode: reason, actionDigest: receipt.actionDigest, receiptDigest: receipt.receiptDigest)
        }
        guard let initialTime = try? clock(), initialTime >= 0 else { return result(.clockUnavailable) }
        guard let authority else { return result(.unsupportedAuthority) }
        let decision: AgentReceiptAuthority
        do { decision = try authority(receipt.receiptDigest) } catch { return result(.authorityUnavailable) }
        let custody: AgentReceiptCustody
        switch decision {
        case .unrecognized: return result(.unrecognizedReceipt)
        case .unsupported: return result(.unsupportedAuthority)
        case .recognized(let recognized): custody = recognized
        }
        guard custody.receiptDigest == receipt.receiptDigest else { return result(.unrecognizedReceipt) }
        guard let now = try? clock(), now >= 0 else { return result(.clockUnavailable) }
        if now >= custody.expiresAt { return result(.expired) }
        if now < custody.consumedAt { return result(.futureReceipt) }
        let claimed: Bool
        do { claimed = try replayStore.claim(tokenDigest: receipt.tokenDigest, expiresAt: custody.expiresAt, now: now) } catch { return result(.nonceStoreUnavailable) }
        guard claimed else { return result(.replayed) }
        guard receipt.actionDigest == actionDigest else { return result(.actionMismatch) }
        guard custody.reviewerRole == reviewerRole else { return result(.reviewerRoleMismatch) }
        return result(.verified)
    }
}
