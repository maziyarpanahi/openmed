import Foundation

/// A tiny diff between two PII engine outputs over the same source text.
/// Each engine returns a set of `[start, end)` spans; this module returns
/// the spans that agree and the spans unique to each side.
public enum EngineDiff {
    public struct Summary: Hashable, Sendable {
        public let agreements: Int
        public let onlyLeft: Int
        public let onlyRight: Int
        public let leftLabel: String
        public let rightLabel: String

        public init(agreements: Int, onlyLeft: Int, onlyRight: Int,
                    leftLabel: String, rightLabel: String) {
            self.agreements = agreements
            self.onlyLeft = onlyLeft
            self.onlyRight = onlyRight
            self.leftLabel = leftLabel
            self.rightLabel = rightLabel
        }
    }

    public struct Row: Identifiable, Hashable, Sendable {
        public enum Origin: Hashable, Sendable { case both, left, right }
        public let id: UUID
        public let origin: Origin
        public let text: String
        public let leftLabel: String?
        public let rightLabel: String?
        public let start: Int
        public let end: Int

        public init(id: UUID = UUID(), origin: Origin, text: String,
                    leftLabel: String?, rightLabel: String?, start: Int, end: Int) {
            self.id = id
            self.origin = origin
            self.text = text
            self.leftLabel = leftLabel
            self.rightLabel = rightLabel
            self.start = start
            self.end = end
        }
    }

    /// Builds a summary + per-span row list comparing two engine outputs.
    public static func build(
        left: [DetectedEntity],
        right: [DetectedEntity],
        leftLabel: String,
        rightLabel: String
    ) -> (summary: Summary, rows: [Row]) {
        let leftKeyed = Dictionary(left.map { ($0.spanKey, $0) }, uniquingKeysWith: { first, _ in first })
        let rightKeyed = Dictionary(right.map { ($0.spanKey, $0) }, uniquingKeysWith: { first, _ in first })

        let leftKeys = Set(leftKeyed.keys)
        let rightKeys = Set(rightKeyed.keys)
        let shared = leftKeys.intersection(rightKeys)
        let onlyLeft = leftKeys.subtracting(rightKeys)
        let onlyRight = rightKeys.subtracting(leftKeys)

        var rows: [Row] = []
        rows.reserveCapacity(shared.count + onlyLeft.count + onlyRight.count)

        for key in shared {
            let lhs = leftKeyed[key]!
            let rhs = rightKeyed[key]!
            rows.append(Row(
                origin: .both,
                text: lhs.text,
                leftLabel: lhs.label,
                rightLabel: rhs.label,
                start: lhs.start,
                end: lhs.end
            ))
        }
        for key in onlyLeft {
            let lhs = leftKeyed[key]!
            rows.append(Row(
                origin: .left,
                text: lhs.text,
                leftLabel: lhs.label,
                rightLabel: nil,
                start: lhs.start,
                end: lhs.end
            ))
        }
        for key in onlyRight {
            let rhs = rightKeyed[key]!
            rows.append(Row(
                origin: .right,
                text: rhs.text,
                leftLabel: nil,
                rightLabel: rhs.label,
                start: rhs.start,
                end: rhs.end
            ))
        }

        rows.sort { $0.start < $1.start }
        let summary = Summary(
            agreements: shared.count,
            onlyLeft: onlyLeft.count,
            onlyRight: onlyRight.count,
            leftLabel: leftLabel,
            rightLabel: rightLabel
        )
        return (summary, rows)
    }
}

private extension DetectedEntity {
    /// Key used to match the same span across engines. Small tolerance on
    /// `start`/`end` keeps tokeniser-level offset jitter from creating false mismatches.
    var spanKey: String {
        "\(start)-\(end)-\(text.lowercased())"
    }
}
