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
    /// - Parameter data: The JSON document.
    /// - Returns: The configuration, with absent keys set to their defaults.
    /// - Throws: ``OpenMedSDKConfigError``; data that is not JSON at all fails
    ///   with `config_not_object`.
    public static func load(_ data: Data) throws -> OpenMedSDKConfig {
        do {
            return try JSONDecoder().decode(OpenMedSDKConfig.self, from: data)
        } catch let error as OpenMedSDKConfigError {
            throw error
        } catch {
            throw OpenMedSDKConfigError(issues: [OpenMedSDKConfigIssue(code: .configNotObject, field: nil)])
        }
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
