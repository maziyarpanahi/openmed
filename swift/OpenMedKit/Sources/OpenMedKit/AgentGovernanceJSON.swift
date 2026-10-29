import CryptoKit
import Foundation

/// Fixed, value-free diagnostics for the local governance metadata boundary.
public enum AgentGovernanceError: String, Error, Sendable {
    case invalidJSON = "invalid_json"
    case tooLarge = "json_too_large"
    case duplicateField = "duplicate_field"
    case invalidFields = "invalid_fields"
    case unsupportedVersion = "unsupported_version"
    case invalidMetadata = "invalid_metadata"
    case digestMismatch = "digest_mismatch"
    case expired = "expired"
}

// Preserve lexical integer/float and boolean distinctions. Foundation's
// permissive NSNumber bridging would accept true as an integer and 1 as a bool.
indirect enum AgentJSON: Equatable, Sendable {
    case object([String: AgentJSON])
    case array([AgentJSON])
    case string(String)
    case integer(Int64)
    case number(Double)
    case boolean(Bool)
    case null

    static func parse(_ data: Data, maximum: Int = 1_048_576) throws -> AgentJSON {
        guard data.count <= maximum else { throw AgentGovernanceError.tooLarge }
        var parser = AgentJSONParser(bytes: Array(data))
        let value = try parser.value(depth: 0)
        parser.whitespace()
        guard parser.position == parser.bytes.count else { throw AgentGovernanceError.invalidJSON }
        return value
    }

    func object(fields: Set<String>, optional: Set<String> = []) throws -> [String: AgentJSON] {
        guard case .object(let values) = self,
            Set(values.keys).isSubset(of: fields.union(optional)),
            fields.isSubset(of: Set(values.keys))
        else { throw AgentGovernanceError.invalidFields }
        return values
    }

    func string(pattern: String? = nil) throws -> String {
        guard case .string(let value) = self else { throw AgentGovernanceError.invalidMetadata }
        if let pattern {
            guard value.range(of: "\\A(?:" + pattern + ")\\z", options: .regularExpression) != nil else {
                throw AgentGovernanceError.invalidMetadata
            }
        }
        return value
    }

    func integer(minimum: Int64 = 0, maximum: Int64 = .max) throws -> Int64 {
        guard case .integer(let value) = self, minimum <= value, value <= maximum else {
            throw AgentGovernanceError.invalidMetadata
        }
        return value
    }

    func bool() throws -> Bool {
        guard case .boolean(let value) = self else { throw AgentGovernanceError.invalidMetadata }
        return value
    }

    func array(maximum: Int) throws -> [AgentJSON] {
        guard case .array(let values) = self, values.count <= maximum else { throw AgentGovernanceError.invalidMetadata }
        return values
    }

    func canonical() -> Data { Data(render().utf8) }

    func digest() -> String {
        "sha256:" + SHA256.hash(data: canonical()).map { String(format: "%02x", $0) }.joined()
    }

    private func render() -> String {
        switch self {
        case .object(let values):
            return "{" + values.keys.sorted().map { AgentJSON.string($0).render() + ":" + values[$0]!.render() }.joined(separator: ",") + "}"
        case .array(let values): return "[" + values.map { $0.render() }.joined(separator: ",") + "]"
        case .integer(let value): return String(value)
        case .number(let value): return String(value)
        case .boolean(let value): return value ? "true" : "false"
        case .null: return "null"
        case .string(let value):
            var result = "\""
            for scalar in value.unicodeScalars {
                switch scalar.value {
                case 34: result += "\\\""
                case 92: result += "\\\\"
                case 8: result += "\\b"
                case 9: result += "\\t"
                case 10: result += "\\n"
                case 12: result += "\\f"
                case 13: result += "\\r"
                case 32...126: result.unicodeScalars.append(scalar)
                case 0...65535: result += String(format: "\\u%04x", scalar.value)
                default:
                    let code = scalar.value - 65536
                    result += String(format: "\\u%04x\\u%04x", 0xD800 + (code >> 10), 0xDC00 + (code & 1023))
                }
            }
            return result + "\""
        }
    }
}

private struct AgentJSONParser {
    let bytes: [UInt8]
    var position = 0
    var nodes = 0

    mutating func whitespace() {
        while position < bytes.count, [9, 10, 13, 32].contains(bytes[position]) { position += 1 }
    }

    mutating func value(depth: Int) throws -> AgentJSON {
        whitespace()
        nodes += 1
        guard depth <= 16, nodes <= 200_000, position < bytes.count else { throw AgentGovernanceError.invalidJSON }
        switch bytes[position] {
        case 123:
            position += 1
            whitespace()
            var result: [String: AgentJSON] = [:]
            if consume(125) { return .object(result) }
            while true {
                whitespace()
                let key = try readString()
                guard result[key] == nil else { throw AgentGovernanceError.duplicateField }
                whitespace()
                guard consume(58) else { throw AgentGovernanceError.invalidJSON }
                result[key] = try value(depth: depth + 1)
                whitespace()
                if consume(125) { return .object(result) }
                guard consume(44) else { throw AgentGovernanceError.invalidJSON }
            }
        case 91:
            position += 1
            whitespace()
            var result: [AgentJSON] = []
            if consume(93) { return .array(result) }
            while true {
                result.append(try value(depth: depth + 1))
                whitespace()
                if consume(93) { return .array(result) }
                guard consume(44) else { throw AgentGovernanceError.invalidJSON }
            }
        case 34: return .string(try readString())
        case 116:
            try literal("true")
            return .boolean(true)
        case 102:
            try literal("false")
            return .boolean(false)
        case 110:
            try literal("null")
            return .null
        case 45, 48...57: return try number()
        default: throw AgentGovernanceError.invalidJSON
        }
    }

    private mutating func consume(_ byte: UInt8) -> Bool {
        guard position < bytes.count, bytes[position] == byte else { return false }
        position += 1
        return true
    }

    private mutating func literal(_ text: String) throws {
        let expected = Array(text.utf8)
        guard position + expected.count <= bytes.count,
            Array(bytes[position..<(position + expected.count)]) == expected
        else { throw AgentGovernanceError.invalidJSON }
        position += expected.count
    }

    private mutating func readString() throws -> String {
        let start = position
        guard consume(34) else { throw AgentGovernanceError.invalidJSON }
        while position < bytes.count {
            let byte = bytes[position]
            position += 1
            if byte == 34 {
                let raw = Data(bytes[start..<position])
                do {
                    guard let decoded = try JSONSerialization.jsonObject(with: raw, options: .fragmentsAllowed) as? String else {
                        throw AgentGovernanceError.invalidJSON
                    }
                    return decoded
                } catch { throw AgentGovernanceError.invalidJSON }
            }
            if byte == 92 {
                guard position < bytes.count else { throw AgentGovernanceError.invalidJSON }
                position += 1
            } else if byte < 32 {
                throw AgentGovernanceError.invalidJSON
            }
        }
        throw AgentGovernanceError.invalidJSON
    }

    private mutating func number() throws -> AgentJSON {
        let start = position
        _ = consume(45)
        guard position < bytes.count else { throw AgentGovernanceError.invalidJSON }
        if !consume(48) {
            guard (49...57).contains(bytes[position]) else { throw AgentGovernanceError.invalidJSON }
            while position < bytes.count, (48...57).contains(bytes[position]) { position += 1 }
        }
        var fractional = false
        if consume(46) {
            fractional = true
            let fractionStart = position
            while position < bytes.count, (48...57).contains(bytes[position]) { position += 1 }
            guard position > fractionStart else { throw AgentGovernanceError.invalidJSON }
        }
        if position < bytes.count, [69, 101].contains(bytes[position]) {
            fractional = true
            position += 1
            if position < bytes.count, [43, 45].contains(bytes[position]) { position += 1 }
            let exponentStart = position
            while position < bytes.count, (48...57).contains(bytes[position]) { position += 1 }
            guard position > exponentStart else { throw AgentGovernanceError.invalidJSON }
        }
        let raw = String(decoding: bytes[start..<position], as: UTF8.self)
        if !fractional {
            guard let value = Int64(raw) else { throw AgentGovernanceError.invalidMetadata }
            return .integer(value)
        }
        guard let value = Double(raw), value.isFinite else { throw AgentGovernanceError.invalidJSON }
        return .number(value)
    }
}

let agentDigestPattern = "sha256:[0-9a-f]{64}"
let agentRolePattern = "role:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?(?:\\.[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?)+/[a-z][a-z0-9-]{0,63}(?:@(?:0|[1-9][0-9]*)\\.(?:0|[1-9][0-9]*)\\.(?:0|[1-9][0-9]*))?"

func agentVersion(_ value: AgentJSON?, expected: String) throws {
    guard value == .string(expected) else { throw AgentGovernanceError.unsupportedVersion }
}

func agentWorkflowID(_ value: AgentJSON) throws -> String {
    let role = try value.string()
    guard role.count <= 512, role.hasPrefix("workflow:") else { throw AgentGovernanceError.invalidMetadata }
    let equivalent = "role:" + role.dropFirst("workflow:".count)
    _ = try AgentJSON.string(equivalent).string(pattern: agentRolePattern)
    guard let slash = role.firstIndex(of: "/"), role[role.index(role.startIndex, offsetBy: 9)..<slash].count <= 253 else {
        throw AgentGovernanceError.invalidMetadata
    }
    return role
}
