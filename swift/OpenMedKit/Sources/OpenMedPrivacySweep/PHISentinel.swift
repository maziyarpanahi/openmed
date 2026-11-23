import Foundation

/// Value-free refusal codes raised while configuring a sentinel sweep.
///
/// Codes never carry sentinel values, file names, paths or reader errors.
public enum PHISentinelSweepError: String, Error, Equatable, Sendable {
    case sentinelIdentifierInvalid = "sentinel_identifier_invalid"
    case sentinelLengthOutOfRange = "sentinel_length_out_of_range"
    case sentinelIdentifierDuplicate = "sentinel_identifier_duplicate"
    case sentinelsEmpty = "sentinels_empty"
    case sentinelsTooMany = "sentinels_too_many"
    case boundsInvalid = "sweep_bounds_invalid"
}

/// A synthetic value planted in a workload so app storage and logs can be swept for it.
///
/// Only ``identifier`` is reported. ``plantedValue`` is for feeding the workload
/// under test; it is excluded from descriptions, reflection, errors and reports.
public struct PHISentinel: Hashable, Sendable, CustomStringConvertible,
    CustomDebugStringConvertible, CustomReflectable
{
    /// Shortest accepted sentinel, in UTF-8 bytes, to keep accidental matches rare.
    public static let minimumUTF8Length = 8
    /// Longest accepted sentinel, in UTF-8 bytes.
    public static let maximumUTF8Length = 256
    /// Most sentinels one sweep accepts.
    public static let maximumCount = 64

    /// Controlled identifier matching `[a-z0-9][a-z0-9._-]{0,63}`.
    public let identifier: String
    /// The synthetic value the workload plants. Never reported.
    public let plantedValue: String

    /// Create a sentinel from a controlled identifier and a synthetic value.
    ///
    /// - Throws: ``PHISentinelSweepError/sentinelIdentifierInvalid`` or
    ///   ``PHISentinelSweepError/sentinelLengthOutOfRange``.
    public init(identifier: String, value: String) throws {
        guard Self.isValidIdentifier(identifier) else {
            throw PHISentinelSweepError.sentinelIdentifierInvalid
        }
        let length = value.utf8.count
        guard length >= Self.minimumUTF8Length, length <= Self.maximumUTF8Length else {
            throw PHISentinelSweepError.sentinelLengthOutOfRange
        }
        self.identifier = identifier
        self.plantedValue = value
    }

    /// Create a random synthetic sentinel that cannot collide with other runs.
    public static func synthetic(identifier: String) throws -> PHISentinel {
        var generator = SystemRandomNumberGenerator()
        return try synthetic(identifier: identifier, using: &generator)
    }

    /// Create a synthetic sentinel from an injected random number generator.
    public static func synthetic<Generator: RandomNumberGenerator>(
        identifier: String,
        using generator: inout Generator
    ) throws -> PHISentinel {
        let alphabet = Array("ABCDEFGHJKLMNPQRSTUVWXYZ23456789")
        var value = "OMS"
        for _ in 0..<13 {
            value.append(alphabet[Int(generator.next(upperBound: UInt32(alphabet.count)))])
        }
        return try PHISentinel(identifier: identifier, value: value)
    }

    public var description: String { "PHISentinel(\(identifier))" }

    public var debugDescription: String { description }

    public var customMirror: Mirror {
        Mirror(self, children: ["identifier": identifier], displayStyle: .struct)
    }

    static func isValidIdentifier(_ identifier: String) -> Bool {
        let bytes = Array(identifier.utf8)
        guard (1...64).contains(bytes.count) else { return false }
        for (index, byte) in bytes.enumerated() {
            let alphanumeric = (0x61...0x7A).contains(byte) || (0x30...0x39).contains(byte)
            let separator = byte == 0x2E || byte == 0x5F || byte == 0x2D
            guard alphanumeric || (index > 0 && separator) else { return false }
        }
        return true
    }
}

/// The byte encoding under which a sentinel was found.
public enum SentinelEncoding: String, CaseIterable, Codable, Comparable, Sendable {
    /// The exact UTF-8 bytes of the planted value.
    case utf8
    case utf16le
    case utf16be
    /// Standard base64 at each of the three byte alignments.
    case base64
    /// URL-safe base64 (`-` and `_`) at each byte alignment.
    case base64url
    /// RFC 3986 percent-encoding (upper- or lowercase hex, `%20` or `+` for space).
    case percent

    public static func < (lhs: SentinelEncoding, rhs: SentinelEncoding) -> Bool {
        let order = SentinelEncoding.allCases
        return order.firstIndex(of: lhs)! < order.firstIndex(of: rhs)!
    }
}

/// One byte pattern derived from a sentinel under one encoding.
struct SentinelNeedle {
    let sentinelIndex: Int
    let encoding: SentinelEncoding
    let bytes: Data
}

enum SentinelNeedles {
    /// Derive every distinct byte pattern for a planted value, in encoding order.
    static func make(for value: String, sentinelIndex: Int) -> [SentinelNeedle] {
        let raw = Array(value.utf8)
        var candidates: [(SentinelEncoding, [UInt8])] = [(.utf8, raw)]
        candidates.append((.utf16le, value.utf16.flatMap { [UInt8(truncatingIfNeeded: $0), UInt8($0 >> 8)] }))
        candidates.append((.utf16be, value.utf16.flatMap { [UInt8($0 >> 8), UInt8(truncatingIfNeeded: $0)] }))
        let base64 = alignedBase64(raw)
        candidates += base64.map { (.base64, $0) }
        candidates += base64.map { (.base64url, urlSafe($0)) }
        candidates.append((.percent, percentEncoded(raw, uppercase: true, spaceAsPlus: false)))
        candidates.append((.percent, percentEncoded(raw, uppercase: false, spaceAsPlus: false)))
        candidates.append((.percent, percentEncoded(raw, uppercase: true, spaceAsPlus: true)))

        var seen = Set<[UInt8]>()
        var needles: [SentinelNeedle] = []
        for (encoding, bytes) in candidates where !bytes.isEmpty && seen.insert(bytes).inserted {
            needles.append(SentinelNeedle(sentinelIndex: sentinelIndex, encoding: encoding, bytes: Data(bytes)))
        }
        return needles
    }

    /// Base64 characters fully determined by the value at byte offsets 0, 1 and 2.
    static func alignedBase64(_ raw: [UInt8]) -> [[UInt8]] {
        (0..<3).map { offset in
            let padded = [UInt8](repeating: 0, count: offset) + raw + [0, 0, 0]
            let encoded = Array(Data(padded).base64EncodedString().utf8)
            let first = (8 * offset + 5) / 6
            let end = (8 * (offset + raw.count)) / 6
            return Array(encoded[first..<end])
        }
    }

    static func urlSafe(_ base64: [UInt8]) -> [UInt8] {
        base64.map { byte in
            switch byte {
            case 0x2B: return 0x2D
            case 0x2F: return 0x5F
            default: return byte
            }
        }
    }

    static func percentEncoded(_ raw: [UInt8], uppercase: Bool, spaceAsPlus: Bool) -> [UInt8] {
        let digits = Array((uppercase ? "0123456789ABCDEF" : "0123456789abcdef").utf8)
        var output: [UInt8] = []
        for byte in raw {
            let unreserved =
                (0x41...0x5A).contains(byte) || (0x61...0x7A).contains(byte)
                || (0x30...0x39).contains(byte) || [0x2D, 0x2E, 0x5F, 0x7E].contains(byte)
            if unreserved {
                output.append(byte)
            } else if spaceAsPlus && byte == 0x20 {
                output.append(0x2B)
            } else {
                output += [0x25, digits[Int(byte >> 4)], digits[Int(byte & 0x0F)]]
            }
        }
        return output
    }
}
