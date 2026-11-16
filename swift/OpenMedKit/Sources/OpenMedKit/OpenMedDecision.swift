import Foundation

/// Constants of the OpenMed fixed-option decision contract (request and result
/// schema 1.2.0) shared with Python, REST, Kotlin and the web package.
///
/// Decision results are assistive review artifacts. They never authorize a
/// diagnosis, treatment, enrollment, outreach, ordering or any other
/// autonomous clinical action.
public enum OpenMedDecision {
    public static let schemaVersion = "1.2.0"
    public static let supportedSchemaVersions = ["1.0.0", "1.1.0", "1.2.0"]
    public static let compatibilityPolicy = "same_major"
    public static let policyVersion = "1.0.0"
    public static let defaultCalibrationID = "openmed.synthetic.fixed_option.v1"
    public static let defaultQuestionID = "decision"
    public static let advisory =
        "Decision scores are assistive review evidence, not a diagnosis, treatment recommendation, enrollment decision, or autonomous clinical action."
    public static let maxInputChars = 8192
    public static let maxConfigurableInputChars = 131_072
    public static let maxOptions = 64
    public static let maxQuestions = 32
    public static let minTimeoutMS = 1
    public static let maxTimeoutMS = 30_000
    public static let maxMediaItems = 4
    public static let maxMediaBytes = 8 * 1024 * 1024

    static let baseVersion = "1.0.0"
    static let questionVersion = "1.1.0"
    static let mediaVersion = "1.2.0"
    static let inputBytesPerChar = 4
    static let maxOptionChars = 256
    static let maxOptionBytes = 1024
    static let maxTotalOptionBytes = 16_384
    static let maxInstructionsChars = 2048
    static let maxDescriptionChars = 1024
    static let maxTotalQuestionBytes = 65_536
    static let maxTotalMediaBytes = 16 * 1024 * 1024
    static let mediaTypes: Set<String> = ["image/png", "image/jpeg", "video/mp4", "audio/wav"]
    static let controlledPattern = "[a-z][a-z0-9_.:/-]{0,127}"
    static let questionIDPattern = "[A-Za-z_][A-Za-z0-9_.:-]{0,63}"
    static let versionPattern = "(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)"
    static let digestPattern = "sha256:[0-9a-f]{64}"
}

/// Value-free decision errors. ``reason`` is a fixed code; no request text,
/// option, prompt, identifier or path is ever included.
public enum OpenMedDecisionError: Error, Equatable, Sendable, CustomStringConvertible, LocalizedError {
    /// The request violates the decision contract.
    case invalidRequest(reason: String)
    /// The request declares a contract version or policy this runtime cannot serve.
    case unsupportedContract(reason: String)
    /// A runtime, profile, policy or identity setting is invalid.
    case invalidConfiguration(reason: String)

    /// `invalid_request`, `unsupported_contract` or `invalid_configuration`.
    public var kind: String {
        switch self {
        case .invalidRequest: return "invalid_request"
        case .unsupportedContract: return "unsupported_contract"
        case .invalidConfiguration: return "invalid_configuration"
        }
    }

    /// The fixed reason code.
    public var reason: String {
        switch self {
        case .invalidRequest(let reason), .unsupportedContract(let reason), .invalidConfiguration(let reason):
            return reason
        }
    }

    public var description: String { "\(kind):\(reason)" }
    public var errorDescription: String? { description }
}

/// Supported fixed-output decision shapes.
public enum OpenMedDecisionMode: String, Sendable, CaseIterable {
    case fixedChoice = "fixed_choice"
    case booleanChoice = "boolean_choice"
    case orderedPreference = "ordered_preference"
    case scalarScore = "scalar_score"
    case multiLabel = "multi_label"
    case questionSet = "question_set"
}

/// Lossless public decision states.
public enum OpenMedDecisionState: String, Sendable, CaseIterable {
    case success
    case abstained
    case partial
    case unknown
    case conflict
    case unsupported
    case denied
    case failure
}

/// Versioned thresholds pinned to frozen calibration slices.
///
/// A profile is a plain object today. Reading it from a signed model bundle is
/// a separate step; the runtime only needs the validated values.
public struct OpenMedDecisionCalibrationProfile: Equatable, Sendable {
    /// The profile every kit and the Python runtime use when none is supplied.
    public static let standard = OpenMedDecisionCalibrationProfile(
        uncheckedID: OpenMedDecision.defaultCalibrationID,
        version: "1.0.0",
        method: "deterministic_margin",
        minimumConfidence: 0.70,
        minimumMargin: 0.15,
        multiLabelThreshold: 0.70,
        inDomainDigest: "sha256:a6a2e0596c95b0568fe8871c99774fb1edc5b1db6de692e86172df952642dd6c",
        outOfDomainDigest: "sha256:d20c55e3877710223c6c37f87f6a8aaf9dea1b326087b37c60a81522ac0c9289"
    )

    public let calibrationID: String
    public let version: String
    public let method: String
    public let minimumConfidence: Double
    public let minimumMargin: Double
    public let multiLabelThreshold: Double
    public let inDomainDigest: String
    public let outOfDomainDigest: String

    /// Validates a profile exactly as the Python `DecisionCalibrationProfile`.
    public init(
        calibrationID: String,
        version: String,
        method: String,
        minimumConfidence: Double,
        minimumMargin: Double,
        multiLabelThreshold: Double,
        inDomainDigest: String,
        outOfDomainDigest: String
    ) throws {
        let invalid = OpenMedDecisionError.invalidConfiguration(reason: "calibration_profile_invalid")
        guard DecisionText.matches(calibrationID, OpenMedDecision.controlledPattern),
            DecisionText.matches(version, OpenMedDecision.versionPattern),
            DecisionText.matches(method, OpenMedDecision.controlledPattern),
            [minimumConfidence, minimumMargin, multiLabelThreshold].allSatisfy(DecisionMath.isProbability),
            minimumMargin <= minimumConfidence,
            DecisionText.matches(inDomainDigest, OpenMedDecision.digestPattern),
            DecisionText.matches(outOfDomainDigest, OpenMedDecision.digestPattern)
        else { throw invalid }
        self.init(
            uncheckedID: calibrationID, version: version, method: method,
            minimumConfidence: minimumConfidence, minimumMargin: minimumMargin,
            multiLabelThreshold: multiLabelThreshold, inDomainDigest: inDomainDigest,
            outOfDomainDigest: outOfDomainDigest)
    }

    /// Parses a plain profile object with the schema's snake_case fields.
    public init(json: Data) throws {
        let value: AgentJSON
        do { value = try AgentJSON.parse(json) } catch {
            throw OpenMedDecisionError.invalidConfiguration(reason: "calibration_profile_invalid")
        }
        try self.init(jsonValue: value)
    }

    init(jsonValue value: AgentJSON) throws {
        let invalid = OpenMedDecisionError.invalidConfiguration(reason: "calibration_profile_invalid")
        let fields: Set = [
            "calibration_id", "version", "method", "minimum_confidence", "minimum_margin",
            "multi_label_threshold", "in_domain_digest", "out_of_domain_digest",
        ]
        guard case .object(let payload) = value, Set(payload.keys) == fields,
            case .string(let identifier)? = payload["calibration_id"],
            case .string(let version)? = payload["version"],
            case .string(let method)? = payload["method"],
            let confidence = payload["minimum_confidence"]?.decisionNumber,
            let margin = payload["minimum_margin"]?.decisionNumber,
            let threshold = payload["multi_label_threshold"]?.decisionNumber,
            case .string(let inDomain)? = payload["in_domain_digest"],
            case .string(let outOfDomain)? = payload["out_of_domain_digest"]
        else { throw invalid }
        try self.init(
            calibrationID: identifier, version: version, method: method,
            minimumConfidence: confidence, minimumMargin: margin, multiLabelThreshold: threshold,
            inDomainDigest: inDomain, outOfDomainDigest: outOfDomain)
    }

    private init(
        uncheckedID: String, version: String, method: String, minimumConfidence: Double,
        minimumMargin: Double, multiLabelThreshold: Double, inDomainDigest: String,
        outOfDomainDigest: String
    ) {
        calibrationID = uncheckedID
        self.version = version
        self.method = method
        self.minimumConfidence = minimumConfidence
        self.minimumMargin = minimumMargin
        self.multiLabelThreshold = multiLabelThreshold
        self.inDomainDigest = inDomainDigest
        self.outOfDomainDigest = outOfDomainDigest
    }

    var jsonValue: AgentJSON {
        .object([
            "calibration_id": .string(calibrationID),
            "version": .string(version),
            "method": .string(method),
            "minimum_confidence": .number(minimumConfidence),
            "minimum_margin": .number(minimumMargin),
            "multi_label_threshold": .number(multiLabelThreshold),
            "in_domain_digest": .string(inDomainDigest),
            "out_of_domain_digest": .string(outOfDomainDigest),
        ])
    }
}

/// Namespace, purpose and backend-family allowlist.
public struct OpenMedDecisionAccessPolicy: Equatable, Sendable {
    public static let backendKinds: Set<String> = ["deterministic", "encoder", "cross_encoder", "specialist"]
    /// The Python default: namespace `default`; purposes `analytics`,
    /// `care_review` and `quality`; every backend family.
    public static let standard = OpenMedDecisionAccessPolicy(
        uncheckedNamespaces: ["default"],
        purposes: ["analytics", "care_review", "quality"],
        kinds: backendKinds,
        version: OpenMedDecision.policyVersion
    )

    public let allowedNamespaces: Set<String>
    public let allowedPurposes: Set<String>
    public let allowedBackendKinds: Set<String>
    public let policyVersion: String

    public init(
        allowedNamespaces: Set<String>,
        allowedPurposes: Set<String>,
        allowedBackendKinds: Set<String> = OpenMedDecisionAccessPolicy.backendKinds,
        policyVersion: String = OpenMedDecision.policyVersion
    ) throws {
        guard !allowedNamespaces.isEmpty, !allowedPurposes.isEmpty, !allowedBackendKinds.isEmpty,
            allowedNamespaces.union(allowedPurposes).allSatisfy({
                DecisionText.matches($0, OpenMedDecision.controlledPattern)
            }),
            allowedBackendKinds.isSubset(of: Self.backendKinds),
            DecisionText.matches(policyVersion, OpenMedDecision.versionPattern)
        else { throw OpenMedDecisionError.invalidConfiguration(reason: "access_policy_invalid") }
        self.init(
            uncheckedNamespaces: allowedNamespaces, purposes: allowedPurposes,
            kinds: allowedBackendKinds, version: policyVersion)
    }

    private init(uncheckedNamespaces: Set<String>, purposes: Set<String>, kinds: Set<String>, version: String) {
        allowedNamespaces = uncheckedNamespaces
        allowedPurposes = purposes
        allowedBackendKinds = kinds
        policyVersion = version
    }

    func decision(namespace: String, purpose: String, kind: String) -> OpenMedDecisionAccess {
        let code: String?
        if !allowedNamespaces.contains(namespace) {
            code = "namespace_denied"
        } else if !allowedPurposes.contains(purpose) {
            code = "purpose_denied"
        } else if !allowedBackendKinds.contains(kind) {
            code = "backend_denied"
        } else {
            code = nil
        }
        return OpenMedDecisionAccess(
            allowed: code == nil, code: code, namespace: namespace, purpose: purpose,
            policyVersion: policyVersion)
    }
}

/// The access decision recorded in every result.
public struct OpenMedDecisionAccess: Equatable, Sendable {
    public let allowed: Bool
    public let code: String?
    public let namespace: String
    public let purpose: String
    public let policyVersion: String

    var jsonValue: AgentJSON {
        .object([
            "allowed": .boolean(allowed),
            "code": code.map(AgentJSON.string) ?? .null,
            "namespace": .string(namespace),
            "purpose": .string(purpose),
            "policy_version": .string(policyVersion),
        ])
    }
}

/// Value-free provenance of an on-device encoder decision backend.
///
/// The backend family is always `encoder`, local and non-deterministic. The
/// license is provenance for separately acquired weights, not a redistribution
/// grant.
public struct OpenMedDecisionBackendIdentity: Equatable, Sendable {
    public let backendID: String
    public let runtime: String
    public let modelID: String
    public let revision: String
    public let licenseID: String
    public let licenseSHA256: String?
    public var kind: String { "encoder" }

    public init(
        backendID: String,
        runtime: String,
        modelID: String,
        revision: String,
        licenseID: String,
        licenseSHA256: String? = nil
    ) throws {
        let invalid = OpenMedDecisionError.invalidConfiguration(reason: "backend_identity_invalid")
        guard [backendID, runtime, modelID, revision].allSatisfy({ DecisionText.isBoundedLine($0, maximum: 256) }),
            DecisionText.matches(licenseID, "[A-Za-z0-9][A-Za-z0-9 .()+:/_-]{0,127}"),
            licenseID == licenseID.trimmingCharacters(in: .whitespacesAndNewlines)
        else { throw invalid }
        if let licenseSHA256 {
            guard DecisionText.matches(licenseSHA256, "[0-9a-f]{64}"),
                DecisionText.matches(revision, "[0-9a-f]{40,64}")
            else { throw invalid }
        }
        self.backendID = backendID
        self.runtime = runtime
        self.modelID = modelID
        self.revision = revision
        self.licenseID = licenseID
        self.licenseSHA256 = licenseSHA256
    }

    var jsonValue: AgentJSON {
        var payload: [String: AgentJSON] = [
            "backend_id": .string(backendID),
            "kind": .string(kind),
            "runtime": .string(runtime),
            "model_id": .string(modelID),
            "revision": .string(revision),
            "license_id": .string(licenseID),
            "local": .boolean(true),
            "deterministic": .boolean(false),
        ]
        if let licenseSHA256 { payload["license_sha256"] = .string(licenseSHA256) }
        return .object(payload)
    }
}

/// One named question scored against a request's shared input.
struct OpenMedDecisionQuestion: Sendable {
    let questionID: String
    let mode: OpenMedDecisionMode
    let options: [String]
    let optionDescriptions: [String?]?
    let instructions: String?
    let levels: [String]
}

/// One validated decision request (schema 1.0.0, 1.1.0 or 1.2.0).
///
/// The input text and question wording are transient: they are never part of
/// a result, an error or this value's description.
public struct OpenMedDecisionRequest: Sendable, CustomStringConvertible, CustomDebugStringConvertible, CustomReflectable {
    public let mode: OpenMedDecisionMode
    public let namespace: String
    public let purpose: String
    public let calibrationID: String
    public let timeoutMS: Int
    /// The declared (or resolved) schema version.
    public let schemaVersion: String
    /// The newest supported version not newer than ``schemaVersion``.
    public let contractVersion: String
    public let inputFormat: String
    public let maxInputChars: Int?
    /// Number of media items. On-device runtimes are text-only and refuse media.
    public let mediaCount: Int
    let inputText: String
    let questions: [OpenMedDecisionQuestion]

    /// Parses and validates a request mapping with the schema's field names.
    public init(json: Data) throws {
        let value: AgentJSON
        do { value = try AgentJSON.parse(json, maximum: 4 * 1_048_576) } catch {
            throw OpenMedDecisionError.invalidRequest(reason: "request_json_invalid")
        }
        self = try DecisionRequestParser.parse(value)
    }

    /// Builds a single-question text request through the same validation.
    public init(
        mode: OpenMedDecisionMode,
        inputText: String,
        options: [String],
        namespace: String = "default",
        purpose: String = "care_review",
        calibrationID: String = OpenMedDecision.defaultCalibrationID,
        timeoutMS: Int = 5000,
        schemaVersion: String? = OpenMedDecision.schemaVersion
    ) throws {
        var payload: [String: AgentJSON] = [
            "mode": .string(mode.rawValue),
            "input_text": .string(inputText),
            "options": .array(options.map(AgentJSON.string)),
            "namespace": .string(namespace),
            "purpose": .string(purpose),
            "calibration_id": .string(calibrationID),
            "timeout_ms": .integer(Int64(timeoutMS)),
        ]
        if let schemaVersion { payload["schema_version"] = .string(schemaVersion) }
        self = try DecisionRequestParser.parse(.object(payload))
    }

    init(
        mode: OpenMedDecisionMode, namespace: String, purpose: String, calibrationID: String,
        timeoutMS: Int, schemaVersion: String, contractVersion: String, inputFormat: String,
        maxInputChars: Int?, mediaCount: Int, inputText: String, questions: [OpenMedDecisionQuestion]
    ) {
        self.mode = mode
        self.namespace = namespace
        self.purpose = purpose
        self.calibrationID = calibrationID
        self.timeoutMS = timeoutMS
        self.schemaVersion = schemaVersion
        self.contractVersion = contractVersion
        self.inputFormat = inputFormat
        self.maxInputChars = maxInputChars
        self.mediaCount = mediaCount
        self.inputText = inputText
        self.questions = questions
    }

    /// Identifiers and counts only.
    public var description: String {
        "OpenMedDecisionRequest(mode: \(mode.rawValue), schema: \(contractVersion), questions: \(questions.count), media: \(mediaCount))"
    }
    public var debugDescription: String { description }
    public var customMirror: Mirror {
        Mirror(self, children: ["mode": mode.rawValue, "contractVersion": contractVersion], displayStyle: .struct)
    }
}

/// One calibrated option score, in request order.
public struct OpenMedDecisionOptionScore: Equatable, Sendable {
    public let index: Int
    public let option: String
    public let score: Double
}

/// Calibrated outcome for one named question of a question set.
public struct OpenMedDecisionQuestionResult: Equatable, Sendable {
    public let questionID: String
    public let mode: OpenMedDecisionMode
    public let state: OpenMedDecisionState
    public let code: String?
    public let optionScores: [OpenMedDecisionOptionScore]
    public let choice: String?
    public let choices: [String]
    public let ranking: [String]
    public let scalarScore: Double?
    public let confidence: Double?
    public let margin: Double?
    public let warnings: [String]
    public let reviewReasons: [String]

    public var abstained: Bool { state == .abstained }

    var answerFields: [String: AgentJSON] {
        [
            "mode": .string(mode.rawValue),
            "state": .string(state.rawValue),
            "code": code.map(AgentJSON.string) ?? .null,
            "option_scores": .array(
                optionScores.map {
                    .object(["index": .integer(Int64($0.index)), "option": .string($0.option), "score": .number($0.score)])
                }),
            "choice": choice.map(AgentJSON.string) ?? .null,
            "choices": .array(choices.map(AgentJSON.string)),
            "ranking": .array(ranking.map(AgentJSON.string)),
            "scalar_score": scalarScore.map(AgentJSON.number) ?? .null,
            "confidence": confidence.map(AgentJSON.number) ?? .null,
            "margin": margin.map(AgentJSON.number) ?? .null,
        ]
    }

    var jsonValue: AgentJSON {
        var payload = answerFields
        payload["question_id"] = .string(questionID)
        payload["level_scores"] = .array([])
        payload["expected_level"] = .null
        payload["warnings"] = .array(warnings.map(AgentJSON.string))
        payload["review"] = .object(["required": .boolean(true), "reasons": .array(reviewReasons.map(AgentJSON.string))])
        return .object(payload)
    }
}

/// Canonical decision result; it validates against the 1.2.0 result schema
/// and renders exactly the fields of the request's contract version.
public struct OpenMedDecisionResult: Equatable, Sendable {
    public let mode: OpenMedDecisionMode
    public let state: OpenMedDecisionState
    public let code: String?
    public let optionScores: [OpenMedDecisionOptionScore]
    public let choice: String?
    public let choices: [String]
    public let ranking: [String]
    public let scalarScore: Double?
    public let confidence: Double?
    public let margin: Double?
    /// Per-question results of a `question_set` request; empty otherwise.
    public let questions: [OpenMedDecisionQuestionResult]
    public let calibration: OpenMedDecisionCalibrationProfile
    public let backend: OpenMedDecisionBackendIdentity
    public let access: OpenMedDecisionAccess
    public let warnings: [String]
    public let reviewReasons: [String]
    public let schemaVersion: String

    public var advisory: String { OpenMedDecision.advisory }
    /// Always `false`: results never authorize an autonomous clinical action.
    public var autonomousAction: Bool { false }
    /// Always `true`: every result requires human review.
    public var reviewRequired: Bool { true }
    public var abstained: Bool { state == .abstained }

    /// Canonical JSON (sorted keys) with the schema's field names.
    public func jsonData() -> Data { jsonValue.canonical() }

    var jsonValue: AgentJSON {
        var payload = OpenMedDecisionQuestionResult(
            questionID: OpenMedDecision.defaultQuestionID, mode: mode, state: state, code: code,
            optionScores: optionScores, choice: choice, choices: choices, ranking: ranking,
            scalarScore: scalarScore, confidence: confidence, margin: margin, warnings: [],
            reviewReasons: []
        ).answerFields
        if schemaVersion != OpenMedDecision.baseVersion {
            payload["level_scores"] = .array([])
            payload["expected_level"] = .null
            payload["questions"] = .array(questions.map(\.jsonValue))
        }
        payload["calibration"] = calibration.jsonValue
        payload["backend"] = backend.jsonValue
        payload["access"] = access.jsonValue
        payload["warnings"] = .array(warnings.map(AgentJSON.string))
        payload["review"] = .object(["required": .boolean(true), "reasons": .array(reviewReasons.map(AgentJSON.string))])
        payload["advisory"] = .string(advisory)
        payload["autonomous_action"] = .boolean(false)
        payload["schema_version"] = .string(schemaVersion)
        payload["compatibility_policy"] = .string(OpenMedDecision.compatibilityPolicy)
        payload["extensions"] = .object([:])
        return .object(payload)
    }
}

// MARK: - Request validation

enum DecisionRequestParser {
    private static let baseFields: Set = [
        "mode", "input_text", "options", "namespace", "purpose", "calibration_id", "timeout_ms",
        "schema_version", "compatibility_policy",
    ]
    private static let extensionFields: Set = [
        "question_id", "instructions", "option_descriptions", "levels", "questions", "input_format",
        "max_input_chars", "media",
    ]
    private static let questionFields: Set = [
        "question_id", "mode", "options", "option_descriptions", "instructions", "levels",
    ]

    static func invalid(_ reason: String) -> OpenMedDecisionError { .invalidRequest(reason: reason) }

    static func parse(_ value: AgentJSON) throws -> OpenMedDecisionRequest {
        guard case .object(let payload) = value else { throw invalid("request_not_object") }
        guard Set(payload.keys).isSubset(of: baseFields.union(extensionFields)) else {
            throw invalid("request_field_unsupported")
        }
        guard let modeValue = payload["mode"], let inputValue = payload["input_text"] else {
            throw invalid("request_field_missing")
        }
        guard case .string(let modeRaw) = modeValue, let mode = OpenMedDecisionMode(rawValue: modeRaw) else {
            throw invalid("mode_invalid")
        }
        // Structure first, as the Python mapping parser does.
        let options = try stringList(payload["options"] ?? .array([]), reason: "options_invalid")
        let descriptions = try optionalProseList(payload["option_descriptions"])
        let levels = try stringList(nullable(payload["levels"]) ?? .array([]), reason: "levels_invalid")
        let rawQuestions = try list(nullable(payload["questions"]) ?? .array([]), reason: "questions_invalid")
        let inputFormat: String
        switch nullable(payload["input_format"]) {
        case nil: inputFormat = "text"
        case .string(let format)? where format == "text" || format == "json": inputFormat = format
        default: throw invalid("input_format_invalid")
        }
        let mediaCount = try mediaItems(payload["media"])

        // Values.
        let maxInputChars: Int?
        switch nullable(payload["max_input_chars"]) {
        case nil: maxInputChars = nil
        case .integer(let bound)? where 1 <= bound && bound <= Int64(OpenMedDecision.maxConfigurableInputChars):
            maxInputChars = Int(bound)
        default: throw invalid("max_input_chars_invalid")
        }
        guard case .string(let inputText) = inputValue else { throw invalid("input_invalid") }
        try checkInput(inputText, maximum: maxInputChars ?? OpenMedDecision.maxInputChars, format: inputFormat)

        let instructions = try prose(payload["instructions"], maximum: OpenMedDecision.maxInstructionsChars)
        var questionID: String?
        let questions: [OpenMedDecisionQuestion]
        if mode == .questionSet {
            guard options.isEmpty, descriptions == nil, instructions == nil, levels.isEmpty,
                nullable(payload["question_id"]) == nil
            else { throw invalid("question_set_fields_invalid") }
            questions = try validatedQuestions(rawQuestions)
        } else {
            guard rawQuestions.isEmpty else { throw invalid("questions_require_question_set") }
            if let identifier = nullable(payload["question_id"]) {
                guard case .string(let text) = identifier, DecisionText.matches(text, OpenMedDecision.questionIDPattern)
                else { throw invalid("question_id_invalid") }
                questionID = text
            }
            let question = try validatedQuestion(
                questionID: questionID ?? OpenMedDecision.defaultQuestionID, mode: mode, options: options,
                descriptions: descriptions, instructions: instructions, levels: levels)
            guard questionBytes(question) <= OpenMedDecision.maxTotalQuestionBytes else {
                throw invalid("questions_too_large")
            }
            questions = [question]
        }
        let namespace = try controlled(payload["namespace"] ?? .string("default"), reason: "namespace_invalid")
        let purpose = try controlled(payload["purpose"] ?? .string("care_review"), reason: "purpose_invalid")
        let calibrationID = try controlled(
            payload["calibration_id"] ?? .string(OpenMedDecision.defaultCalibrationID), reason: "calibration_id_invalid")

        let usesQuestionFields =
            mode == .questionSet || questionID != nil || descriptions != nil || instructions != nil
            || !levels.isEmpty || inputFormat != "text" || maxInputChars != nil
        let usesMedia = mediaCount > 0
        let declared: String
        switch nullable(payload["schema_version"]) {
        case nil:
            declared =
                usesMedia
                ? OpenMedDecision.mediaVersion : usesQuestionFields ? OpenMedDecision.questionVersion : OpenMedDecision.baseVersion
        case .string(let version)? where DecisionText.matches(version, OpenMedDecision.versionPattern):
            declared = version
        default: throw invalid("schema_version_invalid")
        }

        // Contract support.
        let unsupported = { (reason: String) in OpenMedDecisionError.unsupportedContract(reason: reason) }
        guard declared.split(separator: ".").first == "1" else { throw unsupported("schema_major_unsupported") }
        let contract = contractVersion(declared)
        if usesQuestionFields && contract == OpenMedDecision.baseVersion {
            throw unsupported("question_fields_require_1_1_0")
        }
        if usesMedia && versionPrecedes(contract, OpenMedDecision.mediaVersion) {
            throw unsupported("media_requires_1_2_0")
        }
        guard
            (payload["compatibility_policy"] ?? .string(OpenMedDecision.compatibilityPolicy))
                == .string(OpenMedDecision.compatibilityPolicy)
        else { throw unsupported("compatibility_policy_unsupported") }

        let timeout: Int
        switch payload["timeout_ms"] ?? .integer(5000) {
        case .integer(let value)
        where Int64(OpenMedDecision.minTimeoutMS) <= value && value <= Int64(OpenMedDecision.maxTimeoutMS):
            timeout = Int(value)
        default: throw invalid("timeout_invalid")
        }
        return OpenMedDecisionRequest(
            mode: mode, namespace: namespace, purpose: purpose, calibrationID: calibrationID, timeoutMS: timeout,
            schemaVersion: declared, contractVersion: contract, inputFormat: inputFormat,
            maxInputChars: maxInputChars, mediaCount: mediaCount, inputText: inputText, questions: questions)
    }

    static func contractVersion(_ declared: String) -> String {
        OpenMedDecision.supportedSchemaVersions.last { !versionPrecedes(declared, $0) } ?? OpenMedDecision.baseVersion
    }

    /// Whether `lhs` is an older semantic version than `rhs`.
    static func versionPrecedes(_ lhs: String, _ rhs: String) -> Bool {
        let key = { (version: String) in version.split(separator: ".").map { Int($0) ?? Int.max } }
        return key(lhs).lexicographicallyPrecedes(key(rhs))
    }

    private static func nullable(_ value: AgentJSON?) -> AgentJSON? {
        if case .null? = value { return nil }
        return value
    }

    private static func list(_ value: AgentJSON, reason: String) throws -> [AgentJSON] {
        guard case .array(let items) = value else { throw invalid(reason) }
        return items
    }

    private static func stringList(_ value: AgentJSON, reason: String) throws -> [String] {
        try list(value, reason: reason).map {
            guard case .string(let text) = $0 else { throw invalid(reason) }
            return text
        }
    }

    private static func optionalProseList(_ value: AgentJSON?) throws -> [String?]? {
        guard let value = nullable(value) else { return nil }
        return try list(value, reason: "option_descriptions_invalid").map {
            try prose($0, maximum: OpenMedDecision.maxDescriptionChars, reason: "option_descriptions_invalid")
        }
    }

    private static func controlled(_ value: AgentJSON, reason: String) throws -> String {
        guard case .string(let text) = value, DecisionText.matches(text, OpenMedDecision.controlledPattern) else {
            throw invalid(reason)
        }
        return text
    }

    private static func mediaItems(_ value: AgentJSON?) throws -> Int {
        guard let value else { return 0 }
        let reason = "media_invalid"
        guard case .array(let items) = value, items.count <= OpenMedDecision.maxMediaItems else { throw invalid(reason) }
        var total = 0
        for item in items {
            guard case .object(let fields) = item,
                Set(fields.keys) == ["data", "byte_size", "media_type", "sha256"],
                case .integer(let size)? = fields["byte_size"],
                0 < size, size <= Int64(OpenMedDecision.maxMediaBytes),
                case .string(let data)? = fields["data"],
                data.unicodeScalars.count == 4 * ((Int(size) + 2) / 3),
                case .string(let mediaType)? = fields["media_type"],
                OpenMedDecision.mediaTypes.contains(mediaType),
                case .string(let digest)? = fields["sha256"],
                DecisionText.matches(digest, "[0-9a-f]{64}")
            else { throw invalid(reason) }
            total += Int(size)
        }
        guard total <= OpenMedDecision.maxTotalMediaBytes else { throw invalid(reason) }
        // Media is refused on device and is never decoded here.
        return items.count
    }

    private static func checkInput(_ text: String, maximum: Int, format: String) throws {
        guard !DecisionText.isBlank(text), !text.unicodeScalars.contains("\u{0}"),
            text.unicodeScalars.count <= maximum,
            text.utf8.count <= maximum * OpenMedDecision.inputBytesPerChar
        else { throw invalid("input_invalid") }
        if format == "json" {
            let parsed = try? JSONSerialization.jsonObject(with: Data(text.utf8))
            guard parsed is [String: Any] || parsed is [Any] else { throw invalid("input_invalid") }
        }
    }

    static func prose(_ value: AgentJSON?, maximum: Int, reason: String = "instructions_invalid") throws -> String? {
        guard let value = nullable(value) else { return nil }
        guard case .string(let text) = value, !DecisionText.isBlank(text), text.unicodeScalars.count <= maximum,
            !text.unicodeScalars.contains(where: {
                ($0.value < 32 && !["\n", "\r", "\t"].contains($0)) || $0.value == 127
            }),
            text.utf8.count <= maximum * OpenMedDecision.inputBytesPerChar
        else { throw invalid(reason) }
        return text
    }

    private static func option(_ value: String, reason: String) throws -> String {
        let stripped = value.trimmingCharacters(in: .whitespacesAndNewlines)
        guard DecisionText.isBoundedLine(stripped, maximum: OpenMedDecision.maxOptionChars),
            stripped.utf8.count <= OpenMedDecision.maxOptionBytes
        else { throw invalid(reason) }
        return stripped
    }

    private static func distinct(_ values: [String], reason: String) throws {
        let normalized = values.map(DecisionText.normalizedOption)
        guard Set(normalized).count == normalized.count,
            values.reduce(0, { $0 + $1.utf8.count }) <= OpenMedDecision.maxTotalOptionBytes
        else { throw invalid(reason) }
    }

    private static func validatedQuestion(
        questionID: String, mode: OpenMedDecisionMode, options rawOptions: [String],
        descriptions: [String?]?, instructions: String?, levels rawLevels: [String]
    ) throws -> OpenMedDecisionQuestion {
        let options = try rawOptions.map { try option($0, reason: "options_invalid") }
        switch mode {
        case .scalarScore:
            guard options.isEmpty else { throw invalid("options_invalid") }
        case .booleanChoice:
            guard options.count == 2 else { throw invalid("options_invalid") }
        default:
            guard (2...OpenMedDecision.maxOptions).contains(options.count) else { throw invalid("options_invalid") }
        }
        if !options.isEmpty { try distinct(options, reason: "options_invalid") }
        if let descriptions {
            guard !options.isEmpty, descriptions.count == options.count else {
                throw invalid("option_descriptions_invalid")
            }
        }
        var levels: [String] = []
        if !rawLevels.isEmpty {
            guard mode == .scalarScore else { throw invalid("levels_invalid") }
            levels = try rawLevels.map { try option($0, reason: "levels_invalid") }
            guard (2...OpenMedDecision.maxOptions).contains(levels.count) else { throw invalid("levels_invalid") }
            try distinct(levels, reason: "levels_invalid")
        }
        return OpenMedDecisionQuestion(
            questionID: questionID, mode: mode, options: options, optionDescriptions: descriptions,
            instructions: instructions, levels: levels)
    }

    private static func validatedQuestions(_ values: [AgentJSON]) throws -> [OpenMedDecisionQuestion] {
        let questions = try values.map { value -> OpenMedDecisionQuestion in
            guard case .object(let payload) = value, Set(payload.keys).isSubset(of: questionFields) else {
                throw invalid("question_invalid")
            }
            guard case .string(let identifier)? = payload["question_id"],
                DecisionText.matches(identifier, OpenMedDecision.questionIDPattern)
            else { throw invalid("question_id_invalid") }
            guard case .string(let modeRaw)? = payload["mode"], let mode = OpenMedDecisionMode(rawValue: modeRaw),
                mode != .questionSet
            else { throw invalid("question_mode_invalid") }
            return try validatedQuestion(
                questionID: identifier, mode: mode,
                options: try stringList(nullable(payload["options"]) ?? .array([]), reason: "options_invalid"),
                descriptions: try optionalProseList(payload["option_descriptions"]),
                instructions: try prose(payload["instructions"], maximum: OpenMedDecision.maxInstructionsChars),
                levels: try stringList(nullable(payload["levels"]) ?? .array([]), reason: "levels_invalid"))
        }
        guard (1...OpenMedDecision.maxQuestions).contains(questions.count),
            Set(questions.map(\.questionID)).count == questions.count,
            questions.reduce(0, { $0 + questionBytes($1) }) <= OpenMedDecision.maxTotalQuestionBytes
        else { throw invalid("questions_invalid") }
        return questions
    }

    private static func questionBytes(_ question: OpenMedDecisionQuestion) -> Int {
        let parts = question.options + (question.optionDescriptions ?? []).compactMap { $0 } + question.levels
        return parts.reduce(0) { $0 + $1.utf8.count } + (question.instructions?.utf8.count ?? 0)
    }
}

// MARK: - Shared helpers

enum DecisionText {
    static func matches(_ value: String, _ pattern: String) -> Bool {
        value.range(of: "\\A(?:" + pattern + ")\\z", options: .regularExpression) != nil
    }

    static func isBlank(_ value: String) -> Bool {
        value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    /// Non-blank, bounded in code points, single-line text without control characters.
    static func isBoundedLine(_ value: String, maximum: Int) -> Bool {
        !isBlank(value) && value.unicodeScalars.count <= maximum
            && !value.unicodeScalars.contains(where: { $0.value < 32 || $0.value == 127 })
    }

    /// NFKC, case folding (approximated by upper- then lower-casing) and
    /// whitespace collapsing, matching the Python option normalization.
    static func normalizedOption(_ value: String) -> String {
        value.precomposedStringWithCompatibilityMapping.uppercased().lowercased()
            .split(whereSeparator: { $0.isWhitespace }).joined(separator: " ")
    }
}

enum DecisionMath {
    static func isProbability(_ value: Double) -> Bool { value.isFinite && 0 <= value && value <= 1 }

    static func unit(_ value: Double) -> Double { min(1, max(0, value)) }

    static func softmax(_ logits: [Double]) -> [Double] {
        let peak = logits.max() ?? 0
        let exps = logits.map { Foundation.exp($0 - peak) }
        var total = 0.0
        for value in exps { total += value }
        return exps.map { unit($0 / total) }
    }

    static func sigmoid(_ value: Double) -> Double {
        if value >= 0 { return unit(1 / (1 + Foundation.exp(-value))) }
        let exp = Foundation.exp(value)
        return unit(exp / (1 + exp))
    }

    static func expectedLevel(_ probabilities: [Double]) -> Double {
        var expected = 0.0
        for (index, value) in probabilities.enumerated() { expected += Double(index) * value }
        return unit(expected / Double(probabilities.count - 1))
    }
}

extension AgentJSON {
    var decisionNumber: Double? {
        switch self {
        case .integer(let value): return Double(value)
        case .number(let value): return value
        default: return nil
        }
    }
}
