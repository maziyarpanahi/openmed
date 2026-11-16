import CryptoKit
import Foundation

/// Verifies a signed OpenMed de-identification audit report on device.
///
/// Python's `AuditReport.sign()` stores a `repro_hash` (`sha256:` plus the
/// SHA-256 of the canonical report without `repro_hash` and `signature`) and an
/// HMAC-SHA256 `signature.value` over the canonical report without `signature`.
/// This verifier recomputes both from the bytes it receives, with a canonical
/// JSON encoder that matches Python's `_canonical_json()`: object members
/// sorted by Unicode scalar value, `,` and `:` separators, every non-ASCII
/// character escaped as lowercase `\uXXXX` (astral characters as surrogate
/// pairs), and floats formatted exactly like Python `repr`.
///
/// Like `AuditReport.verify()`, it tries the spans sorted by `(start, end,
/// canonical_label, action, canonical span JSON)` first and then in the order
/// the report stores them, so reports signed before deterministic span order
/// still verify. The HMAC is compared in constant time, and only `HMAC-SHA256`
/// is accepted.
///
/// Verification is offline and never throws. The result carries a reason code
/// only: it never echoes the report, and the key never appears in a
/// description. The verifier only verifies; it cannot sign, and the caller
/// keeps custody of the key.
///
/// The verifier checks the report exactly as received. A report that Python
/// accepts only after normalizing it, for example one with unknown fields or an
/// integer where Python writes a float, does not verify.
public struct OpenMedReceiptVerifier: Sendable {
    /// The only signature algorithm the verifier accepts.
    public static let signatureAlgorithm = "HMAC-SHA256"

    /// The largest report accepted, in bytes.
    public static let maximumReportBytes = 8 * 1_048_576

    private let key: SymmetricKey

    /// Creates a verifier for reports signed with `key`.
    ///
    /// - Parameter key: The HMAC-SHA256 release key. The caller keeps custody
    ///   of it.
    /// - Returns: `nil` when the key is empty, which Python refuses as well.
    public init?(key: SymmetricKey) {
        guard key.bitCount > 0 else { return nil }
        self.key = key
    }

    /// Verifies a report serialized as JSON.
    ///
    /// - Parameter report: The report's JSON bytes, as `AuditReport.to_json()`
    ///   or `to_dict()` wrote them.
    /// - Returns: The verdict and its reason code.
    public func verify(_ report: Data) -> OpenMedReceiptVerification {
        guard let parsed = ReceiptReport(report) else { return OpenMedReceiptVerification(reason: .malformed) }
        let candidates = parsed.candidates()
        let reproHashMatches = candidates.contains { $0.reproHash == parsed.reproHash }
        guard let signature = parsed.signature else {
            return OpenMedReceiptVerification(reason: .signatureMismatch, reproHashMatches: reproHashMatches)
        }
        guard signature.algorithm == Self.signatureAlgorithm else {
            return OpenMedReceiptVerification(reason: .unsupportedAlgorithm, reproHashMatches: reproHashMatches)
        }
        guard reproHashMatches else { return OpenMedReceiptVerification(reason: .hashMismatch) }
        if let mac = Self.lowercaseHex(signature.value) {
            for candidate in candidates where candidate.reproHash == parsed.reproHash {
                if HMAC<SHA256>.isValidAuthenticationCode(mac, authenticating: candidate.signingPayload, using: key) {
                    return OpenMedReceiptVerification(
                        reason: .valid, matchedSpanOrder: candidate.spanOrder, reproHashMatches: true)
                }
            }
        }
        return OpenMedReceiptVerification(reason: .signatureMismatch, reproHashMatches: true)
    }

    /// Verifies a report serialized as a JSON string.
    ///
    /// - Parameter report: The report's JSON text.
    /// - Returns: The verdict and its reason code.
    public func verify(json report: String) -> OpenMedReceiptVerification {
        verify(Data(report.utf8))
    }

    /// The canonical payloads of a report for each span order, sorted first.
    ///
    /// - Parameter report: The report's JSON bytes.
    /// - Returns: The candidates, or `nil` when the report is malformed.
    static func candidates(_ report: Data) -> [ReceiptCandidate]? {
        ReceiptReport(report)?.candidates()
    }

    /// Decodes exactly 32 bytes written as 64 lowercase hex digits, the form
    /// Python's `hexdigest()` writes; anything else cannot match.
    private static func lowercaseHex(_ text: String) -> Data? {
        let digits = Array(text.utf8)
        guard digits.count == 2 * SHA256.byteCount else { return nil }
        var bytes = Data(capacity: SHA256.byteCount)
        var index = 0
        while index < digits.count {
            guard let high = nibble(digits[index]), let low = nibble(digits[index + 1]) else { return nil }
            bytes.append(high << 4 | low)
            index += 2
        }
        return bytes
    }

    private static func nibble(_ digit: UInt8) -> UInt8? {
        switch digit {
        case 48...57: return digit - 48
        case 97...102: return digit - 87
        default: return nil
        }
    }
}

extension OpenMedReceiptVerifier: CustomStringConvertible, CustomDebugStringConvertible, CustomReflectable {
    public var description: String { "OpenMedReceiptVerifier(algorithm: \(Self.signatureAlgorithm))" }
    public var debugDescription: String { description }
    public var customMirror: Mirror { Mirror(self, children: [], displayStyle: .struct) }
}

/// The verdict on one audit report.
public struct OpenMedReceiptVerification: Sendable, Equatable, CustomStringConvertible {
    /// Why a report did or did not verify. Shared with the Kotlin and web
    /// verifiers.
    public enum Reason: String, Sendable, CaseIterable {
        /// The repro hash and the HMAC match.
        case valid
        /// No span order reproduces the stored `repro_hash`: the report changed.
        case hashMismatch = "hash_mismatch"
        /// The report has no signature, or its HMAC does not match the key.
        case signatureMismatch = "signature_mismatch"
        /// The signature names an algorithm other than `HMAC-SHA256`.
        case unsupportedAlgorithm = "unsupported_algorithm"
        /// The input is not a report the verifier can check: invalid JSON or
        /// UTF-8, duplicate members, non-finite numbers, or a missing or
        /// mistyped `spans`, `repro_hash` or `signature`.
        case malformed
    }

    /// A span order the verifier tries.
    public enum SpanOrder: String, Sendable, CaseIterable {
        /// Spans sorted by `(start, end, canonical_label, action, canonical
        /// span JSON)`, the order current releases sign.
        case sorted
        /// Spans in the order the report stores them, as earlier releases signed.
        case stored
    }

    /// Why the report did or did not verify.
    public let reason: Reason
    /// The span order that verified, when the report is valid.
    public let matchedSpanOrder: SpanOrder?
    /// Whether some span order reproduces the stored `repro_hash`, whatever the
    /// signature, as Python's `verify_repro_hash()` reports.
    public let reproHashMatches: Bool

    init(reason: Reason, matchedSpanOrder: SpanOrder? = nil, reproHashMatches: Bool = false) {
        self.reason = reason
        self.matchedSpanOrder = matchedSpanOrder
        self.reproHashMatches = reproHashMatches
    }

    /// Whether the report verified.
    public var isValid: Bool { reason == .valid }

    public var description: String {
        "OpenMedReceiptVerification(reason: \(reason.rawValue), span_order: \(matchedSpanOrder?.rawValue ?? "none"))"
    }
}

/// The canonical payloads of one span order.
struct ReceiptCandidate: Equatable {
    let spanOrder: OpenMedReceiptVerification.SpanOrder
    /// Canonical bytes of the report without `repro_hash` and `signature`.
    let reproPayload: Data
    /// `sha256:` plus the hex SHA-256 of `reproPayload`.
    let reproHash: String
    /// Canonical bytes of the report without `signature`.
    let signingPayload: Data
}

/// The parts of a report the verifier needs, read without normalizing it.
private struct ReceiptReport {
    struct Signature {
        let algorithm: String?
        let value: String
    }

    let members: [ReceiptJSON.Member]
    let spans: [ReceiptJSON]
    let sortedSpans: [ReceiptJSON]
    let reproHash: String
    let signature: Signature?

    init?(_ data: Data) {
        guard let value = ReceiptJSON.parse(data), case .object(let members) = value,
            case .array(let spans)? = ReceiptJSON.member("spans", in: members),
            case .string(let reproHash)? = ReceiptJSON.member("repro_hash", in: members)
        else { return nil }
        var keyed: [(key: SpanKey, span: ReceiptJSON)] = []
        for span in spans {
            guard let key = SpanKey(span) else { return nil }
            keyed.append((key, span))
        }
        switch ReceiptJSON.member("signature", in: members) {
        case nil, .null?:
            signature = nil
        case .object(let fields)?:
            guard case .string(let value)? = ReceiptJSON.member("value", in: fields) else { return nil }
            if case .string(let algorithm)? = ReceiptJSON.member("algorithm", in: fields) {
                signature = Signature(algorithm: algorithm, value: value)
            } else {
                signature = Signature(algorithm: nil, value: value)
            }
        default:
            return nil
        }
        self.members = members
        self.spans = spans
        self.reproHash = reproHash
        sortedSpans = keyed.sorted { $0.key < $1.key }.map(\.span)
    }

    func candidates() -> [ReceiptCandidate] {
        [(OpenMedReceiptVerification.SpanOrder.sorted, sortedSpans), (.stored, spans)].map { order, spans in
            let repro = ReceiptJSON.encode(payload(spans: spans, dropping: ["repro_hash", "signature"]))
            let signing = ReceiptJSON.encode(payload(spans: spans, dropping: ["signature"]))
            let digest = SHA256.hash(data: repro).map { String(format: "%02x", $0) }.joined()
            return ReceiptCandidate(
                spanOrder: order, reproPayload: repro, reproHash: "sha256:" + digest, signingPayload: signing)
        }
    }

    private func payload(spans: [ReceiptJSON], dropping keys: [String]) -> ReceiptJSON {
        .object(
            members.compactMap { member in
                if keys.contains(where: { ReceiptJSON.sameKey($0, member.key) }) { return nil }
                if ReceiptJSON.sameKey("spans", member.key) { return ReceiptJSON.Member(key: member.key, value: .array(spans)) }
                return member
            })
    }
}

/// Python's span sort key: `(start, end, canonical_label, action, canonical
/// JSON of the span)`, with strings compared by Unicode scalar value.
private struct SpanKey: Comparable {
    let start: Int64
    let end: Int64
    let canonicalLabel: [UInt32]
    let action: [UInt32]
    let canonical: [UInt8]

    init?(_ span: ReceiptJSON) {
        guard case .object(let fields) = span,
            case .integer(let start)? = ReceiptJSON.member("start", in: fields),
            case .integer(let end)? = ReceiptJSON.member("end", in: fields),
            let startValue = Int64(start), let endValue = Int64(end),
            case .string(let label)? = ReceiptJSON.member("canonical_label", in: fields),
            case .string(let action)? = ReceiptJSON.member("action", in: fields)
        else { return nil }
        self.start = startValue
        self.end = endValue
        canonicalLabel = label.unicodeScalars.map(\.value)
        self.action = action.unicodeScalars.map(\.value)
        canonical = Array(ReceiptJSON.encode(span))
    }

    static func < (lhs: SpanKey, rhs: SpanKey) -> Bool {
        if lhs.start != rhs.start { return lhs.start < rhs.start }
        if lhs.end != rhs.end { return lhs.end < rhs.end }
        if lhs.canonicalLabel != rhs.canonicalLabel {
            return lhs.canonicalLabel.lexicographicallyPrecedes(rhs.canonicalLabel)
        }
        if lhs.action != rhs.action { return lhs.action.lexicographicallyPrecedes(rhs.action) }
        return lhs.canonical.lexicographicallyPrecedes(rhs.canonical)
    }
}

/// A JSON value that keeps what Python's canonical form depends on: integer
/// versus float literals, every object member with its exact key scalars, and
/// member order as read.
indirect enum ReceiptJSON {
    struct Member {
        let key: String
        let value: ReceiptJSON
    }

    case object([Member])
    case array([ReceiptJSON])
    case string(String)
    /// An integer literal as Python writes it back: decimal digits, `-` only
    /// for a non-zero value.
    case integer(String)
    case double(Double)
    case boolean(Bool)
    case null

    /// The deepest nesting the parser accepts.
    static let maximumDepth = 64

    /// Parses strict JSON (RFC 8259) without normalizing it.
    ///
    /// - Parameter data: UTF-8 JSON.
    /// - Returns: The value, or `nil` for invalid JSON or UTF-8, a lone
    ///   surrogate, a duplicate member, a non-finite number, excessive nesting
    ///   or more than ``OpenMedReceiptVerifier/maximumReportBytes``.
    static func parse(_ data: Data) -> ReceiptJSON? {
        guard data.count <= OpenMedReceiptVerifier.maximumReportBytes,
            let text = String(data: data, encoding: .utf8)
        else { return nil }
        var parser = ReceiptJSONParser(scalars: Array(text.unicodeScalars))
        guard let value = parser.value(depth: 0) else { return nil }
        parser.skipWhitespace()
        return parser.position == parser.scalars.count ? value : nil
    }

    /// Encodes a value the way Python's `_canonical_json()` does.
    ///
    /// - Parameter value: The value to encode.
    /// - Returns: ASCII bytes.
    static func encode(_ value: ReceiptJSON) -> Data {
        var output = ""
        value.write(into: &output)
        return Data(output.utf8)
    }

    /// The value of the member named `key`, compared by Unicode scalars.
    static func member(_ key: String, in members: [Member]) -> ReceiptJSON? {
        members.first { sameKey(key, $0.key) }?.value
    }

    /// Whether two keys have the same Unicode scalars. Swift's `==` treats
    /// canonically equivalent strings as equal; Python does not.
    static func sameKey(_ lhs: String, _ rhs: String) -> Bool {
        lhs.unicodeScalars.elementsEqual(rhs.unicodeScalars)
    }

    /// Formats a finite double exactly like Python `repr`: the shortest
    /// round-trip digits, positional notation when the decimal point falls
    /// between 4 places before the first digit and 16 places after it, and
    /// otherwise scientific notation with a signed exponent of at least two
    /// digits (`1e-07`, `1e+16`, `5e-324`).
    static func pythonRepr(_ value: Double) -> String {
        let sign = value.sign == .minus ? "-" : ""
        guard value != 0 else { return sign + "0.0" }
        // Swift prints the shortest round-trip digits; only the layout differs.
        let text = value.magnitude.description
        var mantissa = Substring(text)
        var exponent = 0
        if let marker = text.firstIndex(where: { $0 == "e" || $0 == "E" }) {
            mantissa = text[..<marker]
            exponent = Int(text[text.index(after: marker)...]) ?? 0
        }
        var digits = ""
        var integerDigits: Int?
        for character in mantissa {
            if character == "." { integerDigits = digits.count } else { digits.append(character) }
        }
        var point = (integerDigits ?? digits.count) + exponent
        while digits.count > 1, digits.first == "0" {
            digits.removeFirst()
            point -= 1
        }
        while digits.count > 1, digits.last == "0" { digits.removeLast() }
        let count = digits.count
        if point > -4, point <= 16 {
            if point <= 0 { return sign + "0." + String(repeating: "0", count: -point) + digits }
            if point >= count { return sign + digits + String(repeating: "0", count: point - count) + ".0" }
            return sign + digits.prefix(point) + "." + digits.dropFirst(point)
        }
        let power = point - 1
        let fraction = count > 1 ? "." + digits.dropFirst() : ""
        let magnitude = String(abs(power))
        return sign + digits.prefix(1) + fraction + "e" + (power < 0 ? "-" : "+")
            + (magnitude.count < 2 ? "0" : "") + magnitude
    }

    private func write(into output: inout String) {
        switch self {
        case .object(let members):
            output += "{"
            let ordered = members.map { (scalars: $0.key.unicodeScalars.map(\.value), member: $0) }
                .sorted { $0.scalars.lexicographicallyPrecedes($1.scalars) }
            for (index, entry) in ordered.enumerated() {
                if index > 0 { output += "," }
                ReceiptJSON.writeString(entry.member.key, into: &output)
                output += ":"
                entry.member.value.write(into: &output)
            }
            output += "}"
        case .array(let values):
            output += "["
            for (index, value) in values.enumerated() {
                if index > 0 { output += "," }
                value.write(into: &output)
            }
            output += "]"
        case .string(let value): ReceiptJSON.writeString(value, into: &output)
        case .integer(let digits): output += digits
        case .double(let value): output += ReceiptJSON.pythonRepr(value)
        case .boolean(let value): output += value ? "true" : "false"
        case .null: output += "null"
        }
    }

    private static func writeString(_ value: String, into output: inout String) {
        output += "\""
        for scalar in value.unicodeScalars {
            switch scalar.value {
            case 0x22: output += "\\\""
            case 0x5C: output += "\\\\"
            case 0x08: output += "\\b"
            case 0x0C: output += "\\f"
            case 0x0A: output += "\\n"
            case 0x0D: output += "\\r"
            case 0x09: output += "\\t"
            case 0x20...0x7E: output.unicodeScalars.append(scalar)
            case 0...0xFFFF: output += String(format: "\\u%04x", scalar.value)
            default:
                let code = scalar.value - 0x10000
                output += String(format: "\\u%04x\\u%04x", 0xD800 + (code >> 10), 0xDC00 + (code & 0x3FF))
            }
        }
        output += "\""
    }
}

/// A strict recursive-descent JSON parser over Unicode scalars.
private struct ReceiptJSONParser {
    let scalars: [Unicode.Scalar]
    var position = 0

    mutating func skipWhitespace() {
        while position < scalars.count, [0x20, 0x09, 0x0A, 0x0D].contains(scalars[position].value) { position += 1 }
    }

    mutating func value(depth: Int) -> ReceiptJSON? {
        skipWhitespace()
        guard depth <= ReceiptJSON.maximumDepth, position < scalars.count else { return nil }
        switch scalars[position] {
        case "{": return object(depth: depth)
        case "[": return array(depth: depth)
        case "\"": return string().map(ReceiptJSON.string)
        case "t": return literal("true") ? .boolean(true) : nil
        case "f": return literal("false") ? .boolean(false) : nil
        case "n": return literal("null") ? .null : nil
        case "-", "0"..."9": return number()
        default: return nil
        }
    }

    private mutating func object(depth: Int) -> ReceiptJSON? {
        position += 1
        var members: [ReceiptJSON.Member] = []
        var keys: Set<[UInt32]> = []
        skipWhitespace()
        if consume("}") { return .object(members) }
        while true {
            skipWhitespace()
            guard let key = string(), keys.insert(key.unicodeScalars.map(\.value)).inserted else { return nil }
            skipWhitespace()
            guard consume(":"), let value = value(depth: depth + 1) else { return nil }
            members.append(ReceiptJSON.Member(key: key, value: value))
            skipWhitespace()
            if consume("}") { return .object(members) }
            guard consume(",") else { return nil }
        }
    }

    private mutating func array(depth: Int) -> ReceiptJSON? {
        position += 1
        var values: [ReceiptJSON] = []
        skipWhitespace()
        if consume("]") { return .array(values) }
        while true {
            guard let value = value(depth: depth + 1) else { return nil }
            values.append(value)
            skipWhitespace()
            if consume("]") { return .array(values) }
            guard consume(",") else { return nil }
        }
    }

    private mutating func string() -> String? {
        guard consume("\"") else { return nil }
        var result = String.UnicodeScalarView()
        while position < scalars.count {
            let scalar = scalars[position]
            position += 1
            switch scalar {
            case "\"":
                return String(result)
            case "\\":
                guard position < scalars.count else { return nil }
                let escape = scalars[position]
                position += 1
                switch escape {
                case "\"", "\\", "/": result.append(escape)
                case "b": result.append("\u{08}")
                case "f": result.append("\u{0C}")
                case "n": result.append("\n")
                case "r": result.append("\r")
                case "t": result.append("\t")
                case "u":
                    guard let decoded = escapedScalar() else { return nil }
                    result.append(decoded)
                default: return nil
                }
            default:
                guard scalar.value >= 0x20 else { return nil }
                result.append(scalar)
            }
        }
        return nil
    }

    /// Decodes the scalar after `\u`, joining a surrogate pair; a lone
    /// surrogate cannot be represented and fails.
    private mutating func escapedScalar() -> Unicode.Scalar? {
        guard let first = hexQuad() else { return nil }
        if (0xD800...0xDBFF).contains(first) {
            guard consume("\\"), consume("u"), let second = hexQuad(), (0xDC00...0xDFFF).contains(second) else {
                return nil
            }
            return Unicode.Scalar(0x10000 + ((first - 0xD800) << 10) + (second - 0xDC00))
        }
        return Unicode.Scalar(first)
    }

    private mutating func hexQuad() -> UInt32? {
        guard position + 4 <= scalars.count else { return nil }
        var value: UInt32 = 0
        for scalar in scalars[position..<(position + 4)] {
            guard let digit = UInt32(String(scalar), radix: 16) else { return nil }
            value = value << 4 | digit
        }
        position += 4
        return value
    }

    private mutating func number() -> ReceiptJSON? {
        let start = position
        let negative = consume("-")
        guard position < scalars.count else { return nil }
        if !consume("0") {
            guard ("1"..."9").contains(scalars[position]) else { return nil }
            skipDigits()
        }
        var isFloat = false
        if consume(".") {
            isFloat = true
            guard skipDigits() else { return nil }
        }
        if consume("e") || consume("E") {
            isFloat = true
            if !consume("+") { _ = consume("-") }
            guard skipDigits() else { return nil }
        }
        var lexeme = String.UnicodeScalarView()
        lexeme.append(contentsOf: scalars[start..<position])
        let text = String(lexeme)
        if isFloat {
            guard let value = Double(text), value.isFinite else { return nil }
            return .double(value)
        }
        let digits = negative ? String(text.dropFirst()) : text
        return .integer(digits.allSatisfy { $0 == "0" } ? "0" : text)
    }

    @discardableResult
    private mutating func skipDigits() -> Bool {
        let start = position
        while position < scalars.count, ("0"..."9").contains(scalars[position]) { position += 1 }
        return position > start
    }

    private mutating func consume(_ scalar: Unicode.Scalar) -> Bool {
        guard position < scalars.count, scalars[position] == scalar else { return false }
        position += 1
        return true
    }

    private mutating func literal(_ word: String) -> Bool {
        let expected = Array(word.unicodeScalars)
        guard position + expected.count <= scalars.count,
            Array(scalars[position..<(position + expected.count)]) == expected
        else { return false }
        position += expected.count
        return true
    }
}
