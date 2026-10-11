import CryptoKit
import Foundation

/// Closed labels for local generated-text screening.
public enum GeneratedPHILabel: String, Codable, Sendable, CaseIterable {
    case name, mrn, date, identifier
}

/// Value-free finding. Offsets address Unicode scalars, not UTF-16 units.
public struct GeneratedPHISpan: Sendable {
    public let start: Int
    public let end: Int
    public let label: GeneratedPHILabel
    public let resolved: Bool

    public init(start: Int, end: Int, label: GeneratedPHILabel, resolved: Bool = true) {
        self.start = start
        self.end = end
        self.label = label
        self.resolved = resolved
    }
}

/// Commitment supplied by same-asset OCR, header seeds, or region findings.
public struct GeneratedIdentifierDigest: Sendable {
    public let assetDigest: String
    public let label: GeneratedPHILabel
    public let digest: String
    public let normalizedLength: Int

    public init(assetDigest: String, label: GeneratedPHILabel, digest: String, normalizedLength: Int) {
        self.assetDigest = assetDigest
        self.label = label
        self.digest = digest
        self.normalizedLength = normalizedLength
    }
}

/// Deterministic counts, controlled labels and digests only; safe for qualification.
public struct GeneratedTextGuardReport: Codable, Sendable, Equatable {
    public let decision: String
    public let usePolicy: String
    public let detectionCount: Int
    public let labelCounts: [String: Int]
    public let identifierDigests: [String]
    public let detectorVersionDigest: String
    public let policyDigest: String

    public func toJSON() -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        // All fields are closed strings, digests and integers.
        return (try? encoder.encode(self)) ?? Data()
    }
}

/// Metadata-only provider result bound to a release decision and report digest.
public struct GeneratedTextProviderEnvelope: Codable, Sendable, Equatable {
    public let inputDigest: String
    public let outputDigest: String?
    public let outcome: String
    public let abstentionCode: String?
    public let guardReport: GeneratedTextGuardReport
    public let reportDigest: String
    public var requiresReviewerConfirmation: Bool { true }
    public var nonDiagnosticNotice: String {
        "Not diagnostic. Explicit reviewer confirmation is required."
    }
}

/// Released content is separate from the serializable evidence envelope.
public struct GuardedGeneratedText: Sendable, CustomStringConvertible {
    public let text: String?
    public let envelope: GeneratedTextProviderEnvelope
    public var description: String { "GuardedGeneratedText(\(envelope.guardReport.decision))" }
}

private final class GeneratedDetectorBox: @unchecked Sendable {
    private let lock = NSLock()
    private var value: [GeneratedPHISpan]?

    func store(_ value: [GeneratedPHISpan]?) {
        lock.lock()
        defer { lock.unlock() }
        self.value = value
    }

    func read() -> [GeneratedPHISpan]? {
        lock.lock()
        defer { lock.unlock() }
        return value
    }
}

/// Buffered local-only screening; no detector or deadline failure releases text.
///
/// The injected detector must not persist, log or transmit text. Timed-out caller
/// code cannot be terminated; late findings are discarded and this instance rejects
/// further work until its single outstanding detector call finishes.
public final class GeneratedTextGuard: @unchecked Sendable {
    public enum Action: String, Sendable { case redact, withhold, abstain }
    public typealias Detector = @Sendable (String) throws -> [GeneratedPHISpan]

    private let detector: Detector?
    private let versionDigest: String
    private let action: Action
    private let timeout: TimeInterval
    private let policyDigest: String
    private let busy = NSLock()
    private var isBusy = false

    /// Create an offline guard with a bounded deadline. Invalid settings fail closed.
    public init(
        detector: Detector?, detectorVersion: String = "unavailable",
        action: Action = .redact, timeout: TimeInterval = 1
    ) {
        self.detector = detector
        self.versionDigest = Self.hash(detectorVersion)
        self.action = action
        self.timeout = timeout
        self.policyDigest = Self.hash("v1:non_diagnostic_review_required:scalar-nfkc-casefold-no-space:\(action.rawValue):\(timeout):8192")
    }

    /// Hash transient identifier values using scalar NFKC, case folding and no whitespace.
    public static func identifierDigest(_ value: String) -> String {
        hash(normalize(value))
    }

    /// Screen a buffered generation; unresolved findings always yield phi_uncertainty.
    public func release(
        _ text: String, assetDigest: String,
        identifiers: [GeneratedIdentifierDigest] = []
    ) -> GuardedGeneratedText {
        let scalars = Array(text.unicodeScalars)
        var spans: [GeneratedPHISpan] = []
        var decision = "abstain"
        var released: String?
        let valid =
            Self.validDigest(assetDigest) && scalars.count <= 8192
            && identifiers.count <= 128
            && identifiers.allSatisfy {
                $0.assetDigest == assetDigest && Self.validDigest($0.digest)
                    && (1...256).contains($0.normalizedLength)
            }
        if valid, timeout.isFinite, timeout > 0, timeout <= 60,
            let detected = detect(text), detected.count <= 4096,
            detected.allSatisfy({
                $0.resolved && $0.start >= 0 && $0.start < $0.end && $0.end <= scalars.count
            })
        {
            spans = detected
            var normalized = [Unicode.Scalar]()
            var offsets = [Int]()
            for (index, scalar) in scalars.enumerated() {
                let parts = Array(Self.normalize(String(scalar)).unicodeScalars)
                normalized.append(contentsOf: parts)
                offsets.append(contentsOf: Array(repeating: index, count: parts.count))
            }
            let groups = Dictionary(grouping: identifiers, by: { $0.normalizedLength })
            matching: for (length, entries) in groups where length <= normalized.count {
                for start in 0...(normalized.count - length) {
                    let candidate = String(String.UnicodeScalarView(normalized[start..<(start + length)]))
                    let digest = Self.hash(candidate)
                    for entry in entries where entry.digest == digest {
                        spans.append(
                            GeneratedPHISpan(
                                start: offsets[start], end: offsets[start + length - 1] + 1,
                                label: entry.label
                            ))
                        if spans.count > 4096 { break matching }
                    }
                }
            }
            if spans.count > 4096 {
                spans = []
                decision = "abstain"
            } else {
                decision = spans.isEmpty ? "pass" : action.rawValue
                if spans.isEmpty {
                    released = text
                } else if action == .redact {
                    var intervals = [(Int, Int)]()
                    for span in spans.sorted(by: { ($0.start, $0.end) < ($1.start, $1.end) }) {
                        if let last = intervals.last, span.start <= last.1 {
                            intervals[intervals.count - 1] = (last.0, max(last.1, span.end))
                        } else {
                            intervals.append((span.start, span.end))
                        }
                    }
                    var output = scalars
                    for (start, end) in intervals.reversed() {
                        output.replaceSubrange(start..<end, with: "[REDACTED]".unicodeScalars)
                    }
                    released = String(String.UnicodeScalarView(output))
                }
            }
        }
        var counts = [String: Int]()
        for label in GeneratedPHILabel.allCases {
            counts[label.rawValue] = spans.filter { $0.label == label }.count
        }
        let report = GeneratedTextGuardReport(
            decision: decision, usePolicy: "non_diagnostic_review_required", detectionCount: spans.count, labelCounts: counts,
            identifierDigests: valid ? Array(Set(identifiers.map(\.digest))).sorted() : [],
            detectorVersionDigest: versionDigest, policyDigest: policyDigest
        )
        return GuardedGeneratedText(
            text: released,
            envelope: GeneratedTextProviderEnvelope(
                inputDigest: Self.validDigest(assetDigest) ? assetDigest : String(repeating: "0", count: 64),
                outputDigest: released.map(Self.hash),
                outcome: released == nil ? "abstention" : "success",
                abstentionCode: released == nil ? "phi_uncertainty" : nil,
                guardReport: report, reportDigest: Self.hash(report.toJSON())
            ))
    }

    private func detect(_ text: String) -> [GeneratedPHISpan]? {
        guard let detector else { return nil }
        busy.lock()
        if isBusy {
            busy.unlock()
            return nil
        }
        isBusy = true
        busy.unlock()
        let box = GeneratedDetectorBox()
        let done = DispatchSemaphore(value: 0)
        DispatchQueue.global().async { [self] in
            defer {
                busy.lock()
                isBusy = false
                busy.unlock()
                done.signal()
            }
            // Discard arbitrary detector exception messages; never log them.
            box.store(try? detector(text))
        }
        guard done.wait(timeout: .now() + timeout) == .success else { return nil }
        return box.read()
    }

    private static func normalize(_ value: String) -> String {
        value.unicodeScalars.map {
            String($0).precomposedStringWithCompatibilityMapping.folding(
                options: .caseInsensitive, locale: Locale(identifier: "en_US_POSIX")
            ).unicodeScalars.filter { !CharacterSet.whitespacesAndNewlines.contains($0) }
                .map(String.init).joined()
        }.joined()
    }

    private static func validDigest(_ value: String) -> Bool {
        value.utf8.count == 64 && value.utf8.allSatisfy { (48...57).contains($0) || (97...102).contains($0) }
    }

    private static func hash(_ value: String) -> String { hash(Data(value.utf8)) }
    private static func hash(_ value: Data) -> String {
        SHA256.hash(data: value).map { String(format: "%02x", $0) }.joined()
    }
}
