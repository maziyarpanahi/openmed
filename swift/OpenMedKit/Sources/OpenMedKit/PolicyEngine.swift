import CryptoKit
import Foundation

/// One normalized detector span offered to `PolicyEngine.decide`.
///
/// Offsets are half-open Unicode scalar offsets into the source text.
public struct PolicyDetection: Equatable, Sendable {
    public var label: String
    public var start: Int
    public var end: Int
    public var confidencePPM: Int
    /// `model` for model spans, `semantic` for pattern (semantic) units.
    public var source: String
    /// For semantic units, whether the pattern validator passed.
    public var validated: Bool

    public init(
        label: String,
        start: Int,
        end: Int,
        confidencePPM: Int,
        source: String = "model",
        validated: Bool = true
    ) {
        self.label = label
        self.start = start
        self.end = end
        self.confidencePPM = confidencePPM
        self.source = source
        self.validated = validated
    }
}

/// Value-free decision record for one candidate span.
public struct PolicyDecision: Equatable, Sendable {
    public let inputs: [Int]
    /// `model`, `merged`, or `semantic_only`.
    public let kind: String
    public let label: String
    public let canonicalLabel: String
    public let recognized: Bool
    public let start: Int
    public let end: Int
    public let confidencePPM: Int
    public let keepFloorPPM: Int
    public let escalate: Bool
    public let action: String
    public let strictUpgrade: Bool
    /// `applied`, `overlap_loser`, `duplicate`, or `below_threshold`.
    public let outcome: String
    public let span: Int?
}

/// Output of `PolicyEngine.decide`.
public struct PolicyDecisionResult: Equatable, Sendable {
    public let policyName: String
    public let policyDigest: String
    public let result: KitParity.Result
    public let decisions: [PolicyDecision]
}

/// The portable policy decision engine (`openmed.portable_merge` version 1).
///
/// Mirrors `openmed.core.portable_policy.decide`: validation, label
/// normalization, semantic-unit merging, integer parts-per-million threshold
/// decisions, action lookup with caller overrides and the strict no-leak
/// upgrade, exact-duplicate collapse, overlap clustering into union spans,
/// and right-to-left rendering at scalar offsets. Shared vectors in
/// `tests/fixtures/parity/portable_policy_vectors.json` pin the behavior.
public enum PolicyEngine {
    public static let surrogateKeyMinBytes = 16
    public static let surrogateKeyMaxBytes = 64
    static let surrogateDomain = "openmed.portable_surrogate.v1"

    /// Apply a policy to normalized detections.
    ///
    /// - Parameters:
    ///   - policy: Verified policy.
    ///   - text: Source text.
    ///   - detections: Detections with Unicode scalar offsets.
    ///   - overrides: Canonical label to action overrides.
    ///   - surrogateKey: Caller-supplied key (16 to 64 bytes) for keyed
    ///     `replace` surrogates; without it `replace` renders
    ///     `[LABEL_REPLACED]`.
    /// - Throws: `PolicyError` for invalid detections, overrides, or keys.
    public static func decide(
        policy: Policy,
        text: String,
        detections: [PolicyDetection],
        overrides: [String: String] = [:],
        surrogateKey: Data? = nil
    ) throws -> PolicyDecisionResult {
        let scalars = Array(text.unicodeScalars)
        try validate(detections, length: scalars.count)
        let resolvedOverrides = try validate(overrides, taxonomy: policy.taxonomy)
        if let surrogateKey,
            !(surrogateKeyMinBytes...surrogateKeyMaxBytes).contains(surrogateKey.count)
        {
            throw PolicyError("surrogate_key_invalid", "$.surrogate_key")
        }

        var candidates = buildCandidates(policy.taxonomy, detections)
        for index in candidates.indices {
            let threshold = policy.threshold(for: candidates[index].canonicalLabel)
            candidates[index].keepFloorPPM = threshold.keepFloorPPM
            let configured =
                resolvedOverrides[candidates[index].canonicalLabel]
                ?? policy.configuredAction(forCanonical: candidates[index].canonicalLabel)
            if policy.strictNoLeak && configured == .keep {
                candidates[index].action = .mask
                candidates[index].strictUpgrade = true
            } else {
                candidates[index].action = configured
            }
            if candidates[index].confidencePPM >= threshold.keepFloorPPM {
                candidates[index].outcome = "retained"
                candidates[index].escalate =
                    candidates[index].confidencePPM < threshold.escalateBelowPPM
            }
        }

        let retained = candidates.indices.filter { candidates[$0].outcome == "retained" }
        let unique = collapseDuplicates(&candidates, retained, policy: policy)
        var spans: [KitParity.Span] = []
        for cluster in clusters(candidates, unique) {
            let representative = representativeIndex(candidates, cluster, policy: policy)
            let start = cluster.map { candidates[$0].start }.min() ?? 0
            let end = cluster.map { candidates[$0].end }.max() ?? 0
            for member in cluster {
                candidates[member].outcome = member == representative ? "applied" : "overlap_loser"
                candidates[member].span = spans.count
            }
            let winner = candidates[representative]
            let surface = String(String.UnicodeScalarView(scalars[start..<end]))
            spans.append(
                KitParity.Span(
                    label: winner.label,
                    canonicalLabel: winner.canonicalLabel,
                    action: winner.action.rawValue,
                    start: start,
                    end: end,
                    confidencePPM: winner.confidencePPM,
                    replacement: replacement(
                        action: winner.action,
                        canonicalLabel: winner.canonicalLabel,
                        surface: surface,
                        surrogateKey: surrogateKey
                    )
                )
            )
        }
        for index in retained where candidates[index].outcome == "duplicate" {
            let key = duplicateKey(candidates[index])
            if let winner = unique.first(where: { duplicateKey(candidates[$0]) == key }) {
                candidates[index].span = candidates[winner].span
            }
        }

        let output = try KitParity.applySpans(spans, to: text)
        let result = KitParity.Result(deidentifiedText: output, spans: spans)
        try KitParity.validate(result)
        let decisions = candidates.map {
            PolicyDecision(
                inputs: $0.inputs,
                kind: $0.kind,
                label: $0.label,
                canonicalLabel: $0.canonicalLabel,
                recognized: $0.recognized,
                start: $0.start,
                end: $0.end,
                confidencePPM: $0.confidencePPM,
                keepFloorPPM: $0.keepFloorPPM,
                escalate: $0.escalate,
                action: $0.action.rawValue,
                strictUpgrade: $0.strictUpgrade,
                outcome: $0.outcome,
                span: $0.span
            )
        }
        return PolicyDecisionResult(
            policyName: policy.name,
            policyDigest: policy.digest,
            result: result,
            decisions: decisions
        )
    }

    /// Replacement text for one output span; `nil` for `keep`.
    public static func replacement(
        action: PolicyAction,
        canonicalLabel: String,
        surface: String,
        surrogateKey: Data? = nil
    ) -> String? {
        switch action {
        case .keep:
            return nil
        case .mask, .redact:
            return "[\(canonicalLabel)]"
        case .remove:
            return ""
        case .replace:
            guard let surrogateKey else {
                return "[\(canonicalLabel)_REPLACED]"
            }
            let token = surrogateToken(
                key: surrogateKey,
                canonicalLabel: canonicalLabel,
                surface: surface
            )
            return "[\(canonicalLabel)_\(token)]"
        case .hash:
            let digest = SHA256.hash(data: Data(surface.utf8))
            return "\(canonicalLabel)_\(hex(Array(digest.prefix(12))))"
        }
    }

    /// Deterministic keyed surrogate token: HMAC-SHA256 over
    /// `"openmed.portable_surrogate.v1" 0x00 label 0x00 surface`, first 8 bytes
    /// as lowercase hex. The key never leaves the caller.
    public static func surrogateToken(key: Data, canonicalLabel: String, surface: String) -> String {
        var message = Data(surrogateDomain.utf8)
        message.append(0)
        message.append(Data(canonicalLabel.utf8))
        message.append(0)
        message.append(Data(surface.utf8))
        let mac = HMAC<SHA256>.authenticationCode(for: message, using: SymmetricKey(data: key))
        return hex(Array(Data(mac).prefix(8)))
    }

    // MARK: - Internals

    private struct Candidate {
        var inputs: [Int]
        var kind: String
        var label: String
        var canonicalLabel: String
        var recognized: Bool
        var start: Int
        var end: Int
        var confidencePPM: Int
        var keepFloorPPM = 0
        var escalate = false
        var action: PolicyAction = .keep
        var strictUpgrade = false
        var outcome = "below_threshold"
        var span: Int?
    }

    private struct Normalized {
        let index: Int
        let detection: PolicyDetection
        let canonicalLabel: String
        let recognized: Bool
    }

    private static func validate(_ detections: [PolicyDetection], length: Int) throws {
        for (index, detection) in detections.enumerated() {
            let path = "$.detections[\(index)]"
            guard isValidLabel(detection.label) else {
                throw PolicyError("detection_label_invalid", "\(path).label")
            }
            guard detection.start >= 0 else {
                throw PolicyError("detection_offset_invalid", "\(path).start")
            }
            guard detection.end > detection.start else {
                throw PolicyError("detection_offset_invalid", "\(path).end")
            }
            guard detection.end <= length else {
                throw PolicyError("detection_offset_out_of_bounds", "\(path).end")
            }
            guard (0...KitParity.confidenceScale).contains(detection.confidencePPM) else {
                throw PolicyError("detection_confidence_invalid", "\(path).confidence_ppm")
            }
            guard detection.source == "model" || detection.source == "semantic" else {
                throw PolicyError("detection_source_invalid", "\(path).source")
            }
        }
    }

    private static func isValidLabel(_ label: String) -> Bool {
        let scalars = Array(label.unicodeScalars)
        guard (1...80).contains(scalars.count), scalars.allSatisfy(\.isASCII) else {
            return false
        }
        guard ("A"..."Z").contains(scalars[0]) || ("a"..."z").contains(scalars[0]) else {
            return false
        }
        return scalars.dropFirst().allSatisfy {
            ("A"..."Z").contains($0) || ("a"..."z").contains($0) || ("0"..."9").contains($0)
                || $0 == "_" || $0 == ":" || $0 == "-"
        }
    }

    private static func validate(
        _ overrides: [String: String],
        taxonomy: PolicyTaxonomy
    ) throws -> [String: PolicyAction] {
        var resolved: [String: PolicyAction] = [:]
        for label in overrides.keys.sorted() {
            guard taxonomy.policyLabels[label] != nil else {
                throw PolicyError("override_label_unknown", "$.overrides")
            }
            guard let action = overrides[label].flatMap(PolicyAction.init(rawValue:)) else {
                throw PolicyError("override_action_unknown", "$.overrides")
            }
            resolved[label] = action
        }
        return resolved
    }

    private static func detectionPrecedes(_ lhs: Normalized, _ rhs: Normalized) -> Bool {
        let left = lhs.detection
        let right = rhs.detection
        if left.start != right.start { return left.start < right.start }
        if left.end != right.end { return left.end < right.end }
        if lhs.canonicalLabel != rhs.canonicalLabel {
            return asciiLess(lhs.canonicalLabel, rhs.canonicalLabel)
        }
        if left.label != right.label { return asciiLess(left.label, right.label) }
        return lhs.index < rhs.index
    }

    private static func candidatePrecedes(_ lhs: Candidate, _ rhs: Candidate) -> Bool {
        if lhs.start != rhs.start { return lhs.start < rhs.start }
        if lhs.end != rhs.end { return lhs.end < rhs.end }
        if lhs.canonicalLabel != rhs.canonicalLabel {
            return asciiLess(lhs.canonicalLabel, rhs.canonicalLabel)
        }
        if lhs.label != rhs.label { return asciiLess(lhs.label, rhs.label) }
        return lhs.inputs[0] < rhs.inputs[0]
    }

    private static func asciiLess(_ lhs: String, _ rhs: String) -> Bool {
        Array(lhs.utf8).lexicographicallyPrecedes(Array(rhs.utf8))
    }

    private static func buildCandidates(
        _ taxonomy: PolicyTaxonomy,
        _ detections: [PolicyDetection]
    ) -> [Candidate] {
        let normalized = detections.enumerated().map { index, detection in
            let resolved = taxonomy.normalize(detection.label)
            return Normalized(
                index: index,
                detection: detection,
                canonicalLabel: resolved.canonicalLabel,
                recognized: resolved.recognized
            )
        }
        let models = normalized.filter { $0.detection.source == "model" }.sorted(
            by: detectionPrecedes
        )
        let units = normalized.filter { $0.detection.source == "semantic" }.sorted(
            by: detectionPrecedes
        )
        var consumed = Set<Int>()
        var candidates: [Candidate] = []
        for unit in units {
            let overlapping = models.filter {
                $0.detection.start < unit.detection.end && $0.detection.end > unit.detection.start
            }
            guard !overlapping.isEmpty else {
                candidates.append(
                    Candidate(
                        inputs: [unit.index],
                        kind: "semantic_only",
                        label: unit.detection.label,
                        canonicalLabel: unit.canonicalLabel,
                        recognized: unit.recognized,
                        start: unit.detection.start,
                        end: unit.detection.end,
                        confidencePPM: unit.detection.confidencePPM
                    )
                )
                continue
            }
            let dominant = dominantLabel(overlapping)
            let modelLabels = Set(overlapping.map(\.canonicalLabel))
            let unitRank = taxonomy.specificity[unit.canonicalLabel] ?? 1
            let dominantRank = taxonomy.specificity[dominant] ?? 1
            let final =
                modelLabels.contains(unit.canonicalLabel) || dominantRank > unitRank
                ? dominant : unit.canonicalLabel
            var label = unit.detection.label
            var recognized = unit.recognized
            var best: Normalized?
            for entry in overlapping where entry.canonicalLabel == final {
                if best == nil || entry.detection.confidencePPM > best!.detection.confidencePPM {
                    best = entry
                }
            }
            if let best {
                label = best.detection.label
                recognized = best.recognized
            }
            let total = overlapping.reduce(0) { $0 + $1.detection.confidencePPM }
            let average = total / overlapping.count
            let weights = unit.detection.validated ? (6, 4) : (9, 1)
            let confidence =
                (weights.0 * average + weights.1 * unit.detection.confidencePPM + 5) / 10
            for entry in overlapping { consumed.insert(entry.index) }
            candidates.append(
                Candidate(
                    inputs: ([unit.index] + overlapping.map(\.index)).sorted(),
                    kind: "merged",
                    label: label,
                    canonicalLabel: final,
                    recognized: recognized,
                    start: min(unit.detection.start, overlapping.map(\.detection.start).min()!),
                    end: max(unit.detection.end, overlapping.map(\.detection.end).max()!),
                    confidencePPM: confidence
                )
            )
        }
        for entry in models where !consumed.contains(entry.index) {
            candidates.append(
                Candidate(
                    inputs: [entry.index],
                    kind: "model",
                    label: entry.detection.label,
                    canonicalLabel: entry.canonicalLabel,
                    recognized: entry.recognized,
                    start: entry.detection.start,
                    end: entry.detection.end,
                    confidencePPM: entry.detection.confidencePPM
                )
            )
        }
        return candidates.sorted(by: candidatePrecedes)
    }

    private static func dominantLabel(_ entries: [Normalized]) -> String {
        var counts: [String: Int] = [:]
        var sums: [String: Int] = [:]
        for entry in entries {
            counts[entry.canonicalLabel, default: 0] += 1
            sums[entry.canonicalLabel, default: 0] += entry.detection.confidencePPM
        }
        var best = ""
        for label in counts.keys.sorted(by: asciiLess) {
            guard !best.isEmpty else {
                best = label
                continue
            }
            let count = counts[label]!
            let bestCount = counts[best]!
            if count != bestCount {
                if count > bestCount { best = label }
                continue
            }
            if sums[label]! * bestCount > sums[best]! * count {
                best = label
            }
        }
        return best
    }

    /// `true` when `lhs` wins over `rhs`: protective action, specificity,
    /// length, confidence, earlier start, then ASCII label order.
    private static func wins(_ lhs: Candidate, over rhs: Candidate, policy: Policy) -> Bool {
        let lhsProtective = lhs.action != .keep
        let rhsProtective = rhs.action != .keep
        if lhsProtective != rhsProtective { return lhsProtective }
        let lhsRank = policy.taxonomy.specificity[lhs.canonicalLabel] ?? 1
        let rhsRank = policy.taxonomy.specificity[rhs.canonicalLabel] ?? 1
        if lhsRank != rhsRank { return lhsRank > rhsRank }
        let lhsLength = lhs.end - lhs.start
        let rhsLength = rhs.end - rhs.start
        if lhsLength != rhsLength { return lhsLength > rhsLength }
        if lhs.confidencePPM != rhs.confidencePPM { return lhs.confidencePPM > rhs.confidencePPM }
        if lhs.start != rhs.start { return lhs.start < rhs.start }
        if lhs.canonicalLabel != rhs.canonicalLabel {
            return asciiLess(lhs.canonicalLabel, rhs.canonicalLabel)
        }
        if lhs.label != rhs.label { return asciiLess(lhs.label, rhs.label) }
        return lhs.inputs[0] < rhs.inputs[0]
    }

    private static func best(_ members: [Int], in candidates: [Candidate], policy: Policy) -> Int {
        var winner = members[0]
        for member in members.dropFirst()
        where wins(candidates[member], over: candidates[winner], policy: policy) {
            winner = member
        }
        return winner
    }

    private static func duplicateKey(_ candidate: Candidate) -> String {
        "\(candidate.start):\(candidate.end):\(candidate.canonicalLabel)"
    }

    private static func collapseDuplicates(
        _ candidates: inout [Candidate],
        _ retained: [Int],
        policy: Policy
    ) -> [Int] {
        var groups: [String: [Int]] = [:]
        var order: [String] = []
        for index in retained {
            let key = duplicateKey(candidates[index])
            if groups[key] == nil { order.append(key) }
            groups[key, default: []].append(index)
        }
        var unique: [Int] = []
        for key in order {
            let members = groups[key]!
            let winner = best(members, in: candidates, policy: policy)
            for member in members where member != winner {
                candidates[member].outcome = "duplicate"
            }
            unique.append(winner)
        }
        let snapshot = candidates
        return unique.sorted { candidatePrecedes(snapshot[$0], snapshot[$1]) }
    }

    private static func clusters(_ candidates: [Candidate], _ unique: [Int]) -> [[Int]] {
        var result: [[Int]] = []
        var current: [Int] = []
        var currentEnd = -1
        for index in unique {
            if !current.isEmpty && candidates[index].start < currentEnd {
                current.append(index)
                currentEnd = max(currentEnd, candidates[index].end)
                continue
            }
            if !current.isEmpty { result.append(current) }
            current = [index]
            currentEnd = candidates[index].end
        }
        if !current.isEmpty { result.append(current) }
        return result
    }

    private static func representativeIndex(
        _ candidates: [Candidate],
        _ cluster: [Int],
        policy: Policy
    ) -> Int {
        var members = cluster
        if policy.arbitrationMode == "high_recall_union" {
            let strongest = members.map { candidates[$0].action.strength }.max() ?? 0
            members = members.filter { candidates[$0].action.strength == strongest }
        }
        return best(members, in: candidates, policy: policy)
    }

    private static func hex(_ bytes: [UInt8]) -> String {
        bytes.map { String(format: "%02x", $0) }.joined()
    }
}
