import Foundation
import os

public enum OpenMedClinicianPurpose: String, Codable, Sendable {
    case review, research
    case qualityAssurance = "quality_assurance"
}

public enum OpenMedClinicianModeError: String, Error, Sendable {
    case clinicianModeRequired = "clinician_mode_required"
    case acknowledgementRequired = "acknowledgement_required"
    case invalidAcknowledgement = "invalid_acknowledgement"
}

/// A value-free audit record. Restoring a host-owned acknowledgement revalidates its notice.
public struct OpenMedClinicianModeReceipt: Codable, Equatable, Sendable {
    public let purpose: OpenMedClinicianPurpose
    public let noticeID: String
    public let noticeDigest: String
    private enum CodingKeys: String, CodingKey {
        case purpose
        case noticeID = "notice_id"
        case noticeDigest = "notice_digest"
    }
}

/// Opaque, revocable authority to run review work on the caller's executor.
public struct OpenMedClinicianReviewScope: Sendable {
    fileprivate let enabled = OSAllocatedUnfairLock(initialState: true)
    fileprivate var active: Bool { enabled.withLock { $0 } }
    fileprivate func revoke() { enabled.withLock { $0 = false } }

    fileprivate struct Invocation: Sendable {
        let grant: OpenMedClinicianReviewScope
        let live = OSAllocatedUnfairLock(initialState: true)
        var active: Bool { live.withLock { $0 } && grant.active }
        func finish() { live.withLock { $0 = false } }
    }
    @TaskLocal fileprivate static var invocation: Invocation?

    public func perform<T>(_ operation: () throws -> T) throws -> T {
        guard active else { throw OpenMedClinicianModeError.clinicianModeRequired }
        let call = Invocation(grant: self)
        defer { call.finish() }
        return try Self.$invocation.withValue(call, operation: operation)
    }

    public func performAsync<T>(_ operation: () async throws -> T) async throws -> T {
        guard active else { throw OpenMedClinicianModeError.clinicianModeRequired }
        let call = Invocation(grant: self)
        defer { call.finish() }
        return try await Self.$invocation.withValue(call, operation: operation)
    }
}

/// Host-owned review mode. A new actor starts disabled and performs no persistence.
public actor OpenMedClinicianMode {
    private var grant: OpenMedClinicianReviewScope?
    public init() {}
    deinit { grant?.revoke() }

    @discardableResult
    public func enable(purpose: OpenMedClinicianPurpose, acknowledgement: OpenMedNoticeAcknowledgement?) throws -> OpenMedClinicianModeReceipt {
        guard let ack = acknowledgement else {
            OpenMedUseErrorRecorder.current?.record(.acknowledgementSkipped)
            throw OpenMedClinicianModeError.acknowledgementRequired
        }
        guard let notice = try OpenMedDisclaimer.notice(id: ack.noticeID), notice.tier == .restricted,
            notice.permitsResults(with: ack)
        else {
            OpenMedUseErrorRecorder.current?.record(.acknowledgementSkipped)
            throw OpenMedClinicianModeError.invalidAcknowledgement
        }
        grant?.revoke()
        grant = OpenMedClinicianReviewScope()
        return OpenMedClinicianModeReceipt(purpose: purpose, noticeID: notice.id, noticeDigest: notice.versionDigest)
    }

    public func disable() {
        grant?.revoke()
        grant = nil
    }

    public func reviewScope() throws -> OpenMedClinicianReviewScope {
        guard let grant else { throw OpenMedClinicianModeError.clinicianModeRequired }
        return grant
    }

    public static func require(_ capability: String) throws {
        let tier = try OpenMedIntendedUse.lookup(capability)?.regulatoryTier ?? .unknown
        if tier.obligations.contains("clinician_mode_required"),
            OpenMedClinicianReviewScope.invocation?.active != true
        {
            OpenMedUseErrorRecorder.current?.record(.restrictedCapabilityRefused)
            throw OpenMedClinicianModeError.clinicianModeRequired
        }
    }
}
