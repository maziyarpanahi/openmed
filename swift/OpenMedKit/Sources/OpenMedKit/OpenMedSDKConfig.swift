import Foundation

/// The portable SDK configuration, version 1 of `sdk_config.schema.json`.
///
/// Python and every OpenMedKit port read the same small JSON document with the
/// same keys, types, bounds and defaults, so an app can ship one file in its
/// bundle and share it with its Android or web build:
///
/// | Key | Swift property | Default |
/// | --- | --- | --- |
/// | `confidence_threshold` | ``confidenceThreshold`` | `0.5` |
/// | `local_only` | ``localOnly`` | `true` |
/// | `model_id` | ``modelID`` | `nil` (kit default model) |
/// | `model_revision` | ``modelRevision`` | `nil` |
/// | `onnx_variant` | ``onnxVariant`` | `auto` |
/// | `policy_profile` | ``policyProfile`` | `hipaa_safe_harbor` |
///
/// Decoding is strict: an unknown key, a wrong JSON type (`true` is not a
/// number, `1` is not a boolean), a value out of bounds or a malformed model id
/// fails with ``OpenMedSDKConfigError``, whose issues carry the shared
/// value-free codes in the shared order. `local_only` defaults to `true`, so an
/// empty document stays offline.
///
/// `confidence_threshold` and `policy_profile` are per-call settings, as in
/// Python: pass them to `extractPII(_:confidenceThreshold:)` or
/// `deidentify(_:policy:confidenceThreshold:)`. `onnx_variant` is ignored by
/// the Core ML and MLX backends.
public struct OpenMedSDKConfig: Codable, Sendable, Equatable {
    /// ONNX weight variants. OpenMedKit's Core ML and MLX backends ignore it.
    public enum ONNXVariant: String, Codable, Sendable, CaseIterable {
        case auto
        case int8
        case fp16
        case fp32
    }

    /// The portable policy profiles, as bundled in `Resources/policies`.
    public enum PolicyProfile: String, Codable, Sendable, CaseIterable {
        case australiaPrivacyAct = "australia_privacy_act"
        case canadaPIPEDA = "canada_pipeda"
        case clinicalMinimalRedaction = "clinical_minimal_redaction"
        case gdprArt9Health = "gdpr_art9_health"
        case gdprPseudonymization = "gdpr_pseudonymization"
        case hipaaExpertReviewAssist = "hipaa_expert_review_assist"
        case hipaaSafeHarbor = "hipaa_safe_harbor"
        case researchLimitedDataset = "research_limited_dataset"
        case strictNoLeak = "strict_no_leak"
    }

    /// The six portable keys, in the order issues are reported.
    public enum Field: String, Sendable, CaseIterable, CodingKey {
        case confidenceThreshold = "confidence_threshold"
        case localOnly = "local_only"
        case modelID = "model_id"
        case modelRevision = "model_revision"
        case onnxVariant = "onnx_variant"
        case policyProfile = "policy_profile"
    }

    /// The configuration version this type reads.
    public static let version = 1
    /// The `$id` of `sdk_config.schema.json`.
    public static let schemaID = "https://openmed.life/schemas/sdk_config.schema.json"
    /// The configuration an empty document resolves to.
    public static let defaults = OpenMedSDKConfig(
        validated: 0.5, localOnly: true, modelID: nil, modelRevision: nil, onnxVariant: .auto,
        policyProfile: .hipaaSafeHarbor)

    /// Minimum detection confidence in `[0, 1]`; a per-call setting.
    public let confidenceThreshold: Double
    /// Cache-only, no-egress mode.
    public let localOnly: Bool
    /// Hub repository id of the model, or `nil` for the kit default.
    public let modelID: String?
    /// Pinned revision of ``modelID``.
    public let modelRevision: String?
    /// ONNX weight variant; ignored by Core ML and MLX.
    public let onnxVariant: ONNXVariant
    /// Portable policy profile; a per-call setting.
    public let policyProfile: PolicyProfile

    /// The existing `allowNetworkAccess:` initializer argument this
    /// configuration maps to: `!localOnly`.
    public var allowNetworkAccess: Bool { !localOnly }

    /// Creates a configuration and validates it like a decoded document.
    ///
    /// - Throws: ``OpenMedSDKConfigError`` with `out_of_range`,
    ///   `invalid_format` or `missing_dependency` issues.
    public init(
        confidenceThreshold: Double = 0.5,
        localOnly: Bool = true,
        modelID: String? = nil,
        modelRevision: String? = nil,
        onnxVariant: ONNXVariant = .auto,
        policyProfile: PolicyProfile = .hipaaSafeHarbor
    ) throws {
        let values: [Field: SDKConfigValue] = [
            .confidenceThreshold: .number(confidenceThreshold),
            .localOnly: .boolean(localOnly),
            .modelID: modelID.map(SDKConfigValue.string) ?? .null,
            .modelRevision: modelRevision.map(SDKConfigValue.string) ?? .null,
            .onnxVariant: .string(onnxVariant.rawValue),
            .policyProfile: .string(policyProfile.rawValue),
        ]
        let issues = Self.issues(values, hasUnknownKeys: false)
        guard issues.isEmpty else { throw OpenMedSDKConfigError(issues: issues) }
        self.init(
            validated: confidenceThreshold, localOnly: localOnly, modelID: modelID, modelRevision: modelRevision,
            onnxVariant: onnxVariant, policyProfile: policyProfile)
    }

    private init(
        validated confidenceThreshold: Double, localOnly: Bool, modelID: String?, modelRevision: String?,
        onnxVariant: ONNXVariant, policyProfile: PolicyProfile
    ) {
        self.confidenceThreshold = confidenceThreshold
        self.localOnly = localOnly
        self.modelID = modelID
        self.modelRevision = modelRevision
        self.onnxVariant = onnxVariant
        self.policyProfile = policyProfile
    }

    // MARK: - Loading

    /// Decodes a configuration document.
    ///
    /// The document is read as strict RFC 8259 JSON in UTF-8, the way the
    /// Kotlin and web kits read it, so every runtime resolves the same text to
    /// the same configuration: the last value of a repeated key wins, and a
    /// number too small for a `Double` resolves to `0`. A byte-order mark,
    /// comments, trailing commas, unquoted keys, `NaN`, `Infinity` or text
    /// after the object fail with `config_not_object`, as does nesting deeper
    /// than 512 levels.
    ///
    /// - Parameter data: The JSON document.
    /// - Returns: The configuration, with absent keys set to their defaults.
    /// - Throws: ``OpenMedSDKConfigError``; data that is not a JSON object fails
    ///   with `config_not_object`.
    public static func load(_ data: Data) throws -> OpenMedSDKConfig {
        guard let document = SDKConfigDocument.read(data) else {
            throw OpenMedSDKConfigError(issues: [OpenMedSDKConfigIssue(code: .configNotObject, field: nil)])
        }
        return try OpenMedSDKConfig(resolving: document.values, hasUnknownKeys: document.hasUnknownKeys)
    }

    /// Decodes a configuration file, for example one in the app bundle.
    ///
    /// - Parameter url: A local file URL.
    /// - Returns: The configuration.
    /// - Throws: The file read error, or ``OpenMedSDKConfigError``.
    public static func load(contentsOf url: URL) throws -> OpenMedSDKConfig {
        try load(Data(contentsOf: url))
    }

    /// Returns every issue in a configuration document, in the shared order.
    ///
    /// - Parameter data: The JSON document.
    /// - Returns: The issues; empty when the document is valid.
    public static func validate(_ data: Data) -> [OpenMedSDKConfigIssue] {
        do {
            _ = try load(data)
            return []
        } catch let error as OpenMedSDKConfigError {
            return error.issues
        } catch {
            return [OpenMedSDKConfigIssue(code: .configNotObject, field: nil)]
        }
    }

    // MARK: - Codable

    private struct AnyKey: CodingKey {
        let stringValue: String
        var intValue: Int? { nil }
        init?(stringValue: String) { self.stringValue = stringValue }
        init?(intValue: Int) { nil }
    }

    /// Decodes strictly; see the type documentation for the rules.
    ///
    /// The JSON grammar is the decoder's own. `JSONDecoder` keeps the first
    /// value of a repeated key and accepts a byte-order mark and trailing
    /// commas, which the other runtimes refuse, so read configuration text with
    /// ``load(_:)``.
    ///
    /// - Throws: ``OpenMedSDKConfigError``.
    public init(from decoder: any Decoder) throws {
        let container: KeyedDecodingContainer<AnyKey>
        do {
            container = try decoder.container(keyedBy: AnyKey.self)
        } catch {
            throw OpenMedSDKConfigError(issues: [OpenMedSDKConfigIssue(code: .configNotObject, field: nil)])
        }
        var values: [Field: SDKConfigValue] = [:]
        var hasUnknownKeys = false
        for key in container.allKeys {
            guard let field = Field(rawValue: key.stringValue) else {
                hasUnknownKeys = true
                continue
            }
            values[field] = (try? container.decode(SDKConfigValue.self, forKey: key)) ?? .other
        }
        try self.init(resolving: values, hasUnknownKeys: hasUnknownKeys)
    }

    /// Validates the present keys and fills the absent ones with the defaults.
    private init(resolving values: [Field: SDKConfigValue], hasUnknownKeys: Bool) throws {
        let issues = Self.issues(values, hasUnknownKeys: hasUnknownKeys)
        guard issues.isEmpty else { throw OpenMedSDKConfigError(issues: issues) }
        func string(_ field: Field) -> String? {
            if case .string(let value)? = values[field] { return value }
            return nil
        }
        var threshold = Self.defaults.confidenceThreshold
        if case .number(let value?)? = values[.confidenceThreshold] { threshold = value }
        var localOnly = Self.defaults.localOnly
        if case .boolean(let value)? = values[.localOnly] { localOnly = value }
        self.init(
            validated: threshold, localOnly: localOnly, modelID: string(.modelID), modelRevision: string(.modelRevision),
            onnxVariant: string(.onnxVariant).flatMap(ONNXVariant.init(rawValue:)) ?? Self.defaults.onnxVariant,
            policyProfile: string(.policyProfile).flatMap(PolicyProfile.init(rawValue:)) ?? Self.defaults.policyProfile)
    }

    /// Encodes all six keys, with `null` for an unset model id or revision.
    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: Field.self)
        try container.encode(confidenceThreshold, forKey: .confidenceThreshold)
        try container.encode(localOnly, forKey: .localOnly)
        if let modelID { try container.encode(modelID, forKey: .modelID) } else { try container.encodeNil(forKey: .modelID) }
        if let modelRevision {
            try container.encode(modelRevision, forKey: .modelRevision)
        } else {
            try container.encodeNil(forKey: .modelRevision)
        }
        try container.encode(onnxVariant, forKey: .onnxVariant)
        try container.encode(policyProfile, forKey: .policyProfile)
    }

    // MARK: - Validation

    /// The issues of a document: at most one `unknown_field`, then at most one
    /// issue per present key in ``Field`` order.
    static func issues(_ values: [Field: SDKConfigValue], hasUnknownKeys: Bool) -> [OpenMedSDKConfigIssue] {
        var issues: [OpenMedSDKConfigIssue] = []
        if hasUnknownKeys { issues.append(OpenMedSDKConfigIssue(code: .unknownField, field: nil)) }
        for field in Field.allCases {
            guard let value = values[field] else { continue }
            var code = issue(field, value)
            if code == nil, field == .modelRevision, value != .null, (values[.modelID] ?? .null) == .null {
                code = .missingDependency
            }
            if let code { issues.append(OpenMedSDKConfigIssue(code: code, field: field)) }
        }
        return issues
    }

    private static func issue(_ field: Field, _ value: SDKConfigValue) -> OpenMedSDKConfigIssue.Code? {
        switch field {
        case .confidenceThreshold:
            guard case .number(let number) = value else { return .wrongType }
            guard let number, number.isFinite, (0...1).contains(number) else { return .outOfRange }
            return nil
        case .localOnly:
            guard case .boolean = value else { return .wrongType }
            return nil
        case .modelID, .modelRevision:
            if value == .null { return nil }
            guard case .string(let text) = value else { return .wrongType }
            let valid = field == .modelID ? isModelID(text) : isRevision(text)
            return valid ? nil : .invalidFormat
        case .onnxVariant, .policyProfile:
            guard case .string(let text) = value else { return .wrongType }
            let allowed = field == .onnxVariant ? ONNXVariant(rawValue: text) != nil : PolicyProfile(rawValue: text) != nil
            return allowed ? nil : .notAllowedValue
        }
    }

    /// `name` or `namespace/name`; each part starts with an ASCII letter or
    /// digit, continues with letters, digits, `.`, `_` or `-`, is at most 96
    /// characters, and `..` never appears.
    private static func isModelID(_ text: String) -> Bool {
        let parts = text.split(separator: "/", omittingEmptySubsequences: false)
        guard (1...2).contains(parts.count), !text.contains("..") else { return false }
        return parts.allSatisfy { isSegment($0, maximum: 96) }
    }

    /// One segment of at most 128 characters, with no `/` and no `..`.
    private static func isRevision(_ text: String) -> Bool {
        !text.contains("..") && isSegment(Substring(text), maximum: 128)
    }

    private static func isSegment(_ text: Substring, maximum: Int) -> Bool {
        let scalars = Array(text.unicodeScalars)
        guard let first = scalars.first, scalars.count <= maximum, isAlphanumeric(first) else { return false }
        return scalars.dropFirst().allSatisfy { isAlphanumeric($0) || $0 == "." || $0 == "_" || $0 == "-" }
    }

    private static func isAlphanumeric(_ scalar: Unicode.Scalar) -> Bool {
        ("a"..."z").contains(scalar) || ("A"..."Z").contains(scalar) || ("0"..."9").contains(scalar)
    }
}

/// One value-free configuration problem.
public struct OpenMedSDKConfigIssue: Sendable, Equatable, CustomStringConvertible {
    /// The shared issue codes.
    public enum Code: String, Sendable, CaseIterable {
        /// The document is not a JSON object.
        case configNotObject = "config_not_object"
        /// `model_id` or `model_revision` does not match its pattern.
        case invalidFormat = "invalid_format"
        /// `model_revision` is set while `model_id` is absent or `null`.
        case missingDependency = "missing_dependency"
        /// `onnx_variant` or `policy_profile` is outside its enum.
        case notAllowedValue = "not_allowed_value"
        /// `confidence_threshold` is outside `[0, 1]`.
        case outOfRange = "out_of_range"
        /// At least one key is not a portable key.
        case unknownField = "unknown_field"
        /// The value has the wrong JSON type.
        case wrongType = "wrong_type"
    }

    /// The issue code.
    public let code: Code
    /// The key the issue concerns, or `nil` for document-level issues.
    public let field: OpenMedSDKConfig.Field?

    /// Creates an issue.
    public init(code: Code, field: OpenMedSDKConfig.Field?) {
        self.code = code
        self.field = field
    }

    public var description: String {
        field.map { "\($0.rawValue):\(code.rawValue)" } ?? code.rawValue
    }
}

/// A configuration document failed validation. It carries issue codes and
/// key names only, never an input value or an unknown key's name.
public struct OpenMedSDKConfigError: Error, Sendable, Equatable, LocalizedError, CustomStringConvertible {
    /// The issues, in the shared order.
    public let issues: [OpenMedSDKConfigIssue]

    /// Creates an error from its issues.
    public init(issues: [OpenMedSDKConfigIssue]) {
        self.issues = issues
    }

    public var description: String {
        "invalid SDK configuration: " + issues.map(\.description).joined(separator: ", ")
    }

    public var errorDescription: String? { description }
}

/// A JSON value reduced to what validation needs. `.number(nil)` is a number
/// that is not a finite `Double`; `.other` is an array or an object.
enum SDKConfigValue: Decodable, Equatable, Sendable {
    case null
    case boolean(Bool)
    case number(Double?)
    case string(String)
    case other

    private struct Ignored: Decodable {
        init(from decoder: any Decoder) throws {}
    }

    init(from decoder: any Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self = .null
        } else if let value = try? container.decode(Bool.self) {
            self = .boolean(value)
        } else if let value = try? container.decode(Double.self) {
            self = .number(value)
        } else if let value = try? container.decode(String.self) {
            self = .string(value)
        } else if (try? container.decode([Ignored].self)) != nil || (try? container.decode([String: Ignored].self)) != nil {
            self = .other
        } else {
            // Only a number too large for a Double is left, such as 1e400.
            self = .number(nil)
        }
    }
}

/// The portable keys of a configuration document, read as strict RFC 8259 JSON.
///
/// Foundation's `JSONDecoder` keeps the first value of a repeated key and
/// accepts a leading byte-order mark and trailing commas. Python's `json.loads`,
/// kotlinx.serialization and `JSON.parse` keep the last value, and the Kotlin
/// and web kits refuse everything outside RFC 8259, so this reader does the
/// same. Numbers follow the JSON grammar and round like Python's `float()`: a
/// value too small for a `Double` is `0`, one too large is infinite. The reader
/// never recurses, never retains rejected input, and reports failure as `nil`.
struct SDKConfigDocument {
    /// Deepest accepted nesting, counting the top-level object.
    static let maximumDepth = 512

    /// The last value of each portable key.
    private(set) var values: [OpenMedSDKConfig.Field: SDKConfigValue] = [:]
    /// Whether any other key appears.
    private(set) var hasUnknownKeys = false

    /// Reads a document whose top-level value is an object.
    ///
    /// - Parameter data: The document.
    /// - Returns: The portable keys, or `nil` when `data` is not a JSON object.
    static func read(_ data: Data) -> SDKConfigDocument? {
        var reader = Reader(bytes: [UInt8](data))
        var document = SDKConfigDocument()
        reader.skipWhitespace()
        guard reader.consume(UInt8(ascii: "{")) else { return nil }
        reader.skipWhitespace()
        if !reader.consume(UInt8(ascii: "}")) {
            repeat {
                guard let name = reader.memberName(), let value = reader.memberValue() else { return nil }
                // Compare scalars, not canonically equivalent strings.
                if let field = OpenMedSDKConfig.Field.allCases.first(where: {
                    $0.rawValue.unicodeScalars.elementsEqual(name.unicodeScalars)
                }) {
                    document.values[field] = value
                } else {
                    document.hasUnknownKeys = true
                }
                reader.skipWhitespace()
            } while reader.consume(UInt8(ascii: ","))
            guard reader.consume(UInt8(ascii: "}")) else { return nil }
        }
        reader.skipWhitespace()
        return reader.index == reader.bytes.count ? document : nil
    }

    private struct Reader {
        let bytes: [UInt8]
        var index = 0

        mutating func skipWhitespace() {
            while index < bytes.count, [0x20, 0x09, 0x0A, 0x0D].contains(bytes[index]) { index += 1 }
        }

        mutating func consume(_ byte: UInt8) -> Bool {
            guard index < bytes.count, bytes[index] == byte else { return false }
            index += 1
            return true
        }

        /// An object member's name and the colon after it.
        mutating func memberName() -> String? {
            skipWhitespace()
            guard let name = string() else { return nil }
            skipWhitespace()
            return consume(UInt8(ascii: ":")) ? name : nil
        }

        /// A top-level member's value; arrays and objects are only checked.
        mutating func memberValue() -> SDKConfigValue? {
            skipWhitespace()
            guard index < bytes.count else { return nil }
            if bytes[index] == UInt8(ascii: "[") || bytes[index] == UInt8(ascii: "{") {
                return skipContainer() ? .other : nil
            }
            return scalar()
        }

        /// Checks the array or object at `index` with an explicit stack.
        private mutating func skipContainer() -> Bool {
            var closers: [UInt8] = []
            while true {
                skipWhitespace()
                guard index < bytes.count else { return false }
                let opener = bytes[index]
                if opener == UInt8(ascii: "[") || opener == UInt8(ascii: "{") {
                    // The top-level object is the first level.
                    guard closers.count + 1 < SDKConfigDocument.maximumDepth else { return false }
                    index += 1
                    let closer = opener == UInt8(ascii: "[") ? UInt8(ascii: "]") : UInt8(ascii: "}")
                    skipWhitespace()
                    if !consume(closer) {
                        closers.append(closer)
                        if closer == UInt8(ascii: "}"), memberName() == nil { return false }
                        continue
                    }
                } else if scalar() == nil {
                    return false
                }
                // A value ended: close every finished container, then expect
                // the next element.
                while let closer = closers.last {
                    skipWhitespace()
                    if consume(closer) {
                        closers.removeLast()
                        continue
                    }
                    guard consume(UInt8(ascii: ",")) else { return false }
                    if closer == UInt8(ascii: "}"), memberName() == nil { return false }
                    break
                }
                if closers.isEmpty { return true }
            }
        }

        private mutating func scalar() -> SDKConfigValue? {
            guard index < bytes.count else { return nil }
            switch bytes[index] {
            case UInt8(ascii: "\""): return string().map(SDKConfigValue.string)
            case UInt8(ascii: "t"): return literal("true") ? .boolean(true) : nil
            case UInt8(ascii: "f"): return literal("false") ? .boolean(false) : nil
            case UInt8(ascii: "n"): return literal("null") ? .null : nil
            case UInt8(ascii: "-"), UInt8(ascii: "0")...UInt8(ascii: "9"): return number()
            default: return nil
            }
        }

        private mutating func literal(_ text: String) -> Bool {
            let expected = Array(text.utf8)
            guard bytes[index...].starts(with: expected) else { return false }
            index += expected.count
            return true
        }

        /// `-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?`
        private mutating func number() -> SDKConfigValue? {
            let start = index
            _ = consume(UInt8(ascii: "-"))
            if !consume(UInt8(ascii: "0")) {
                guard index < bytes.count, (UInt8(ascii: "1")...UInt8(ascii: "9")).contains(bytes[index]) else {
                    return nil
                }
                skipDigits()
            }
            if consume(UInt8(ascii: ".")), !skipDigits() { return nil }
            if consume(UInt8(ascii: "e")) || consume(UInt8(ascii: "E")) {
                if !consume(UInt8(ascii: "+")) { _ = consume(UInt8(ascii: "-")) }
                guard skipDigits() else { return nil }
            }
            // Swift rounds to zero or infinity outside the Double range.
            return .number(Double(String(decoding: bytes[start..<index], as: UTF8.self)))
        }

        @discardableResult
        private mutating func skipDigits() -> Bool {
            let start = index
            while index < bytes.count, (UInt8(ascii: "0")...UInt8(ascii: "9")).contains(bytes[index]) { index += 1 }
            return index > start
        }

        /// A string token in UTF-8. A lone surrogate escape reads as U+FFFD,
        /// so the value stays a string as it does in the other runtimes.
        private mutating func string() -> String? {
            guard consume(UInt8(ascii: "\"")) else { return nil }
            var scalars = String.UnicodeScalarView()
            while index < bytes.count {
                let byte = bytes[index]
                switch byte {
                case UInt8(ascii: "\""):
                    index += 1
                    return String(scalars)
                case UInt8(ascii: "\\"):
                    index += 1
                    guard let scalar = escape() else { return nil }
                    scalars.append(scalar)
                case 0..<0x20:
                    return nil
                case 0..<0x80:
                    scalars.append(Unicode.Scalar(byte))
                    index += 1
                default:
                    var iterator = bytes[index...].makeIterator()
                    var decoder = Unicode.UTF8()
                    guard case .scalarValue(let scalar) = decoder.decode(&iterator) else { return nil }
                    scalars.append(scalar)
                    index += UTF8.width(scalar)
                }
            }
            return nil
        }

        private mutating func escape() -> Unicode.Scalar? {
            guard index < bytes.count else { return nil }
            let byte = bytes[index]
            index += 1
            switch byte {
            case UInt8(ascii: "\""), UInt8(ascii: "\\"), UInt8(ascii: "/"): return Unicode.Scalar(byte)
            case UInt8(ascii: "b"): return "\u{08}"
            case UInt8(ascii: "f"): return "\u{0C}"
            case UInt8(ascii: "n"): return "\n"
            case UInt8(ascii: "r"): return "\r"
            case UInt8(ascii: "t"): return "\t"
            case UInt8(ascii: "u"):
                guard var unit = hexQuad() else { return nil }
                if (0xD800...0xDBFF).contains(unit), bytes[index...].starts(with: Array(#"\u"#.utf8)) {
                    let mark = index
                    index += 2
                    guard let low = hexQuad() else { return nil }
                    if (0xDC00...0xDFFF).contains(low) {
                        unit = 0x10000 + ((unit - 0xD800) << 10) + (low - 0xDC00)
                    } else {
                        index = mark
                    }
                }
                return Unicode.Scalar(unit) ?? "\u{FFFD}"
            default:
                return nil
            }
        }

        private mutating func hexQuad() -> UInt32? {
            guard index + 4 <= bytes.count else { return nil }
            var value: UInt32 = 0
            for byte in bytes[index..<index + 4] {
                guard let digit = Character(Unicode.Scalar(byte)).hexDigitValue else { return nil }
                value = value << 4 | UInt32(digit)
            }
            index += 4
            return value
        }
    }
}
