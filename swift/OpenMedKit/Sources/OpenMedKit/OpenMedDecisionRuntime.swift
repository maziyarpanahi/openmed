import CryptoKit
import Foundation

#if canImport(Tokenizers)
    import Tokenizers
#endif

/// Tokenizer seam for an encoder decision checkpoint.
///
/// `encode` must return the ids the checkpoint was exported with, including its
/// special tokens, and must never truncate or pad: the runtime refuses inputs
/// over the token budget instead of scoring a truncated text.
public protocol OpenMedDecisionTokenizer {
    func encode(_ text: String) throws -> [Int]
    /// The id of a control token such as `<<LABEL>>`, when the vocabulary has it.
    func tokenID(for token: String) -> Int?
}

extension OpenMedDecisionTokenizer {
    public func tokenID(for token: String) -> Int? { nil }
}

/// A GLiClass-style uni-encoder: `input_ids` and `attention_mask` of shape
/// `[1, tokens]` in, the first row of `logits` (one value per `<<LABEL>>`
/// slot, in label order) out. ONNX and Core ML exports share this contract.
public protocol OpenMedDecisionEncoder {
    func logits(inputIDs: [Int], attentionMask: [Int]) throws -> [Double]
}

/// Value-free encoder adapter failures.
public enum OpenMedDecisionEncoderError: String, Error, Sendable {
    /// The runtime output has no logits tensor.
    case missingOutput = "encoder_output_missing"
    /// The logits tensor has an unexpected rank, shape or type.
    case invalidOutput = "encoder_output_invalid"
    /// The token inputs are empty or misaligned.
    case invalidInput = "encoder_input_invalid"
}

/// One synthetic few-shot example rendered as GLiClass example tokens.
public struct OpenMedGLiClassExample: Equatable, Sendable, CustomStringConvertible, CustomReflectable {
    let text: String
    let labels: [String]

    public init(text: String, labels: [String]) throws {
        let invalid = OpenMedDecisionError.invalidConfiguration(reason: "gliclass_example_invalid")
        guard GLiClassText.isValid(text, maximumChars: 2048, maximumBytes: 8192, multiline: true),
            (1...OpenMedDecision.maxOptions).contains(labels.count)
        else { throw invalid }
        self.text = text
        self.labels = try labels.map { try GLiClassText.label($0, reason: "gliclass_example_invalid") }
    }

    public var description: String { "OpenMedGLiClassExample(labels: \(labels.count))" }
    public var customMirror: Mirror { Mirror(self, children: ["labelCount": labels.count], displayStyle: .struct) }
}

/// Backend settings of a GLiClass v4 uni-encoder decision runtime.
///
/// The instruction prompt, few-shot examples and score levels belong to the
/// runtime, as in the Python `GLiClassDecisionBackend`; requests that carry
/// per-question instructions, option descriptions or levels are reported as
/// `unsupported` rather than silently ignoring their wording.
public struct OpenMedGLiClassConfiguration: Equatable, Sendable, CustomStringConvertible, CustomReflectable {
    public static let defaultScoreLevels = ["very low", "low", "moderate", "high", "very high"]
    public static let defaultMaxLength = 2048
    public static let labelToken = "<<LABEL>>"
    public static let separatorToken = "<<SEP>>"
    public static let exampleToken = "<<EXAMPLE>>"
    /// Prompt-first layout, the default score levels and a 2,048-token budget.
    public static let standard = OpenMedGLiClassConfiguration(
        uncheckedPrompt: nil, examples: [], scoreLevels: defaultScoreLevels, promptFirst: true,
        maxLength: defaultMaxLength, labelTokenID: nil)

    public let prompt: String?
    public let examples: [OpenMedGLiClassExample]
    /// Ordered levels, lowest first, scored as labels for `scalar_score`.
    public let scoreLevels: [String]
    /// `true` for GLiClass v4 (labels and prompt before the text).
    public let promptFirst: Bool
    /// Token budget; longer inputs are refused, never truncated.
    public let maxLength: Int
    /// The `<<LABEL>>` id (the export's `class_token_index`). When `nil`, the
    /// tokenizer's ``OpenMedDecisionTokenizer/tokenID(for:)`` is used.
    public let labelTokenID: Int?

    public init(
        prompt: String? = nil,
        examples: [OpenMedGLiClassExample] = [],
        scoreLevels: [String] = OpenMedGLiClassConfiguration.defaultScoreLevels,
        promptFirst: Bool = true,
        maxLength: Int = OpenMedGLiClassConfiguration.defaultMaxLength,
        labelTokenID: Int? = nil
    ) throws {
        let invalid = OpenMedDecisionError.invalidConfiguration(reason: "gliclass_configuration_invalid")
        if let prompt {
            guard GLiClassText.isValid(prompt, maximumChars: 4096, maximumBytes: 16_384, multiline: true) else {
                throw invalid
            }
        }
        let levels = try scoreLevels.map { try GLiClassText.label($0, reason: "gliclass_configuration_invalid") }
        guard examples.count <= 8, (2...16).contains(levels.count),
            Set(levels.map { $0.uppercased().lowercased() }).count == levels.count,
            maxLength >= 1, (labelTokenID ?? 0) >= 0
        else { throw invalid }
        self.init(
            uncheckedPrompt: prompt, examples: examples, scoreLevels: levels, promptFirst: promptFirst,
            maxLength: maxLength, labelTokenID: labelTokenID)
    }

    private init(
        uncheckedPrompt: String?, examples: [OpenMedGLiClassExample], scoreLevels: [String], promptFirst: Bool,
        maxLength: Int, labelTokenID: Int?
    ) {
        prompt = uncheckedPrompt
        self.examples = examples
        self.scoreLevels = scoreLevels
        self.promptFirst = promptFirst
        self.maxLength = maxLength
        self.labelTokenID = labelTokenID
    }

    /// Counts and the prompt digest only; the prompt and examples are never printed.
    public var description: String {
        "OpenMedGLiClassConfiguration(prompt: \(promptDigest ?? "none"), examples: \(examples.count), levels: \(scoreLevels.count), maxLength: \(maxLength))"
    }

    public var customMirror: Mirror {
        Mirror(
            self,
            children: ["promptDigest": promptDigest as Any, "exampleCount": examples.count, "maxLength": maxLength],
            displayStyle: .struct)
    }

    /// SHA-256 of the instruction prompt, safe to record in audit metadata.
    public var promptDigest: String? {
        prompt.map { "sha256:" + SHA256.hash(data: Data($0.utf8)).map { String(format: "%02x", $0) }.joined() }
    }

    /// Renders the uni-encoder input used by `gliclass` 0.1.21 and the Python
    /// reference: `<<LABEL>>label ... <<SEP>>prompt`, then the text and the
    /// examples (prompt-first), or the text first for older checkpoints.
    ///
    /// The result contains the input text and must never be logged.
    public func formatInput(_ text: String, labels: [String]) -> String {
        var head = labels.map { Self.labelToken + $0 }.joined() + Self.separatorToken
        if let prompt, !prompt.isEmpty { head += prompt }
        var rendered = ""
        if !examples.isEmpty {
            rendered =
                examples.map {
                    Self.exampleToken + $0.text + " \nLabels:\n " + $0.labels.joined(separator: ", ")
                }.joined() + Self.separatorToken
        }
        return promptFirst ? head + text + rendered : text + head + rendered
    }
}

/// On-device fixed-option decisions over a GLiClass-style encoder.
///
/// The runtime validates a decision request (schema 1.0.0, 1.1.0 or 1.2.0),
/// applies the access policy and the supplied calibration profile, scores the
/// options through the injected tokenizer and encoder, and returns a result
/// with the same fields, states, abstention codes and review flags as the
/// Python `openmed.structured.decision.decide`. It is text-only: media
/// requests return `unsupported`. It makes no network call and never logs or
/// returns the input, prompt or examples.
public final class OpenMedEncoderDecisionRuntime {
    public let identity: OpenMedDecisionBackendIdentity
    public let configuration: OpenMedGLiClassConfiguration
    public let policy: OpenMedDecisionAccessPolicy
    private let encoder: any OpenMedDecisionEncoder
    private let tokenizer: any OpenMedDecisionTokenizer
    private let profiles: [String: OpenMedDecisionCalibrationProfile]
    private let clock: () -> TimeInterval

    /// - Parameters:
    ///   - encoder: The encoder, for example ``OpenMedCoreMLDecisionEncoder``.
    ///   - tokenizer: The checkpoint's tokenizer.
    ///   - identity: Provenance recorded in every result.
    ///   - configuration: Prompt, examples, score levels and token budget.
    ///   - policy: Namespace, purpose and backend-family allowlist.
    ///   - calibrationProfiles: Profiles available to requests by
    ///     `calibration_id`; the standard profile when omitted.
    ///   - clock: Monotonic seconds used for the request timeout.
    public init(
        encoder: any OpenMedDecisionEncoder,
        tokenizer: any OpenMedDecisionTokenizer,
        identity: OpenMedDecisionBackendIdentity,
        configuration: OpenMedGLiClassConfiguration = .standard,
        policy: OpenMedDecisionAccessPolicy = .standard,
        calibrationProfiles: [OpenMedDecisionCalibrationProfile] = [.standard],
        clock: @escaping () -> TimeInterval = { ProcessInfo.processInfo.systemUptime }
    ) throws {
        let identifiers = calibrationProfiles.map(\.calibrationID)
        guard !identifiers.isEmpty, Set(identifiers).count == identifiers.count else {
            throw OpenMedDecisionError.invalidConfiguration(reason: "calibration_profiles_invalid")
        }
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.identity = identity
        self.configuration = configuration
        self.policy = policy
        profiles = Dictionary(uniqueKeysWithValues: calibrationProfiles.map { ($0.calibrationID, $0) })
        self.clock = clock
    }

    /// Parses, validates and decides a request mapping.
    ///
    /// - Throws: ``OpenMedDecisionError`` for requests that violate the
    ///   contract. Backend refusals and failures are result states instead.
    public func decide(json: Data) throws -> OpenMedDecisionResult {
        decide(try OpenMedDecisionRequest(json: json))
    }

    /// Decides a validated request. Abstentions, denials, refusals and
    /// failures are typed result states, never fabricated selections.
    public func decide(_ request: OpenMedDecisionRequest) -> OpenMedDecisionResult {
        let access = policy.decision(namespace: request.namespace, purpose: request.purpose, kind: identity.kind)
        guard let profile = profiles[request.calibrationID] else {
            return terminal(request, .unsupported, "calibration_unavailable", .standard, access)
        }
        guard access.allowed else { return terminal(request, .denied, access.code ?? "access_denied", profile, access) }
        guard request.mediaCount == 0 else {
            return terminal(request, .unsupported, "backend_unsupported", profile, access)
        }
        let started = clock()
        let answers = request.questions.map { question -> DecisionAnswer in
            if request.mode == .questionSet && elapsedMS(since: started) > Double(request.timeoutMS) {
                return .terminal(.failure, "backend_timeout")
            }
            return score(question, request: request, profile: profile, started: started)
        }
        return assemble(request, answers: answers, profile: profile, access: access)
    }

    private func elapsedMS(since started: TimeInterval) -> Double { (clock() - started) * 1000 }

    private func score(
        _ question: OpenMedDecisionQuestion, request: OpenMedDecisionRequest,
        profile: OpenMedDecisionCalibrationProfile, started: TimeInterval
    ) -> DecisionAnswer {
        let output: DecisionScores
        do {
            output = try scores(for: question, text: request.inputText)
        } catch let refusal as DecisionRefusal {
            return .terminal(refusal.state, refusal.code)
        } catch {
            return .terminal(.failure, "backend_failure")
        }
        if elapsedMS(since: started) > Double(request.timeoutMS) { return .terminal(.failure, "backend_timeout") }
        return DecisionAnswer.from(output, question: question, profile: profile)
    }

    private func scores(for question: OpenMedDecisionQuestion, text: String) throws -> DecisionScores {
        guard question.instructions == nil, !(question.optionDescriptions ?? []).contains(where: { $0 != nil }),
            question.levels.isEmpty
        else { throw DecisionRefusal.unsupported }
        let labels = question.mode == .scalarScore ? configuration.scoreLevels : question.options
        guard !GLiClassText.hasMarker(text), !labels.contains(where: GLiClassText.hasMarker) else {
            throw DecisionRefusal.unsupported
        }
        let inputIDs = try tokenizer.encode(configuration.formatInput(text, labels: labels))
        guard inputIDs.count <= configuration.maxLength else { throw DecisionRefusal.unsupported }
        if let labelID = configuration.labelTokenID ?? tokenizer.tokenID(for: OpenMedGLiClassConfiguration.labelToken),
            inputIDs.filter({ $0 == labelID }).count != labels.count
        {
            throw DecisionRefusal.conflict
        }
        let raw: [Double]
        do {
            raw = try encoder.logits(inputIDs: inputIDs, attentionMask: Array(repeating: 1, count: inputIDs.count))
        } catch OpenMedDecisionEncoderError.missingOutput, OpenMedDecisionEncoderError.invalidOutput {
            throw DecisionRefusal.conflict
        }
        guard raw.count >= labels.count else { throw DecisionRefusal.conflict }
        let logits = Array(raw.prefix(labels.count))
        guard logits.allSatisfy(\.isFinite) else { throw DecisionRefusal.conflict }
        let probabilities = question.mode == .multiLabel ? logits.map(DecisionMath.sigmoid) : DecisionMath.softmax(logits)
        if question.mode == .scalarScore {
            return .scalar(DecisionMath.expectedLevel(probabilities), confidence: probabilities.max() ?? 0)
        }
        return .options(probabilities)
    }

    private func terminal(
        _ request: OpenMedDecisionRequest, _ state: OpenMedDecisionState, _ code: String,
        _ profile: OpenMedDecisionCalibrationProfile, _ access: OpenMedDecisionAccess
    ) -> OpenMedDecisionResult {
        assemble(
            request, answers: request.questions.map { _ in .terminal(state, code) }, profile: profile, access: access)
    }

    private func assemble(
        _ request: OpenMedDecisionRequest, answers: [DecisionAnswer], profile: OpenMedDecisionCalibrationProfile,
        access: OpenMedDecisionAccess
    ) -> OpenMedDecisionResult {
        var outer = answers.first ?? .terminal(.failure, "backend_failure")
        var questions: [OpenMedDecisionQuestionResult] = []
        if request.mode == .questionSet {
            questions = zip(request.questions, answers).map { question, answer in
                OpenMedDecisionQuestionResult(
                    questionID: question.questionID, mode: question.mode, state: answer.state, code: answer.code,
                    optionScores: answer.optionScores, choice: answer.choice, choices: answer.choices,
                    ranking: answer.ranking, scalarScore: answer.scalarScore, confidence: answer.confidence,
                    margin: answer.margin, warnings: DecisionAnswer.warnings(answer.code),
                    reviewReasons: DecisionAnswer.reviewReasons(answer.code))
            }
            outer = DecisionAnswer.aggregate(answers)
        }
        return OpenMedDecisionResult(
            mode: request.mode, state: outer.state, code: outer.code, optionScores: outer.optionScores,
            choice: outer.choice, choices: outer.choices, ranking: outer.ranking, scalarScore: outer.scalarScore,
            confidence: outer.confidence, margin: outer.margin, questions: questions, calibration: profile,
            backend: identity, access: access, warnings: DecisionAnswer.warnings(outer.code),
            reviewReasons: DecisionAnswer.reviewReasons(outer.code), schemaVersion: request.contractVersion)
    }
}

// MARK: - Calibration policy

enum DecisionScores {
    case options([Double])
    case scalar(Double, confidence: Double)
}

enum DecisionRefusal: Error {
    case unsupported
    case conflict

    var state: OpenMedDecisionState { self == .unsupported ? .unsupported : .conflict }
    var code: String { self == .unsupported ? "backend_unsupported" : "backend_conflict" }
}

struct DecisionAnswer {
    var state: OpenMedDecisionState
    var code: String?
    var optionScores: [OpenMedDecisionOptionScore] = []
    var choice: String?
    var choices: [String] = []
    var ranking: [String] = []
    var scalarScore: Double?
    var confidence: Double?
    var margin: Double?

    static func terminal(_ state: OpenMedDecisionState, _ code: String) -> DecisionAnswer {
        DecisionAnswer(state: state, code: code)
    }

    /// The Python `_decision_from_scores` policy for encoder outputs.
    static func from(
        _ output: DecisionScores, question: OpenMedDecisionQuestion, profile: OpenMedDecisionCalibrationProfile
    ) -> DecisionAnswer {
        switch output {
        case .scalar(let value, let confidence):
            let success = confidence >= profile.minimumConfidence
            return DecisionAnswer(
                state: success ? .success : .abstained, code: success ? nil : "low_confidence",
                scalarScore: success ? value : nil, confidence: confidence)
        case .options(let scores):
            let options = question.options
            let ranked = scores.indices.sorted { scores[$0] != scores[$1] ? scores[$0] > scores[$1] : $0 < $1 }
            let confidence = scores[ranked[0]]
            let margin = confidence - scores[ranked[1]]
            var answer = DecisionAnswer(
                state: .abstained, code: nil,
                optionScores: scores.indices.map { OpenMedDecisionOptionScore(index: $0, option: options[$0], score: scores[$0]) },
                ranking: ranked.map { options[$0] }, confidence: confidence, margin: margin)
            if question.mode == .multiLabel {
                let selected = scores.indices.filter { scores[$0] >= profile.multiLabelThreshold }.map { options[$0] }
                answer.state = selected.isEmpty ? .abstained : .success
                answer.code = selected.isEmpty ? "no_label_above_threshold" : nil
                answer.choice = selected.first
                answer.choices = selected
            } else {
                let clearsConfidence = confidence >= profile.minimumConfidence
                let clearsMargin = margin >= profile.minimumMargin
                answer.state = clearsConfidence && clearsMargin ? .success : .abstained
                answer.code = !clearsConfidence ? "low_confidence" : !clearsMargin ? "ambiguous_scores" : nil
                if answer.state == .success {
                    answer.choice = answer.ranking[0]
                    answer.choices = [answer.ranking[0]]
                }
            }
            return answer
        }
    }

    /// Question-set state: success, the shared state and code, the shared
    /// state with `question_set_<state>`, or `partial`.
    static func aggregate(_ answers: [DecisionAnswer]) -> DecisionAnswer {
        let states = Set(answers.map(\.state))
        if states == [.success] { return DecisionAnswer(state: .success, code: nil) }
        let outcomes = Set(answers.map { "\($0.state.rawValue)|\($0.code ?? "")" })
        if outcomes.count == 1, let first = answers.first { return .terminal(first.state, first.code ?? "") }
        if states.count == 1, let state = states.first { return .terminal(state, "question_set_\(state.rawValue)") }
        return .terminal(.partial, "question_set_partial")
    }

    static func warnings(_ code: String?) -> [String] {
        var values = ["human_review_required"]
        if let code, !values.contains(code) { values.append(code) }
        return values
    }

    static func reviewReasons(_ code: String?) -> [String] {
        ["clinical_use_requires_review"] + (code.map { [$0] } ?? [])
    }
}

enum GLiClassText {
    static let markers = [
        OpenMedGLiClassConfiguration.labelToken, OpenMedGLiClassConfiguration.separatorToken,
        OpenMedGLiClassConfiguration.exampleToken,
    ]

    static func hasMarker(_ value: String) -> Bool { markers.contains { value.contains($0) } }

    static func isValid(_ value: String, maximumChars: Int, maximumBytes: Int, multiline: Bool) -> Bool {
        let allowed: Set<Unicode.Scalar> = multiline ? ["\n", "\r", "\t"] : []
        return !DecisionText.isBlank(value) && value.unicodeScalars.count <= maximumChars
            && !value.unicodeScalars.contains(where: { ($0.value < 32 && !allowed.contains($0)) || $0.value == 127 })
            && value.utf8.count <= maximumBytes && !hasMarker(value)
    }

    static func label(_ value: String, reason: String) throws -> String {
        let stripped = value.trimmingCharacters(in: .whitespacesAndNewlines)
        guard isValid(stripped, maximumChars: 256, maximumBytes: 1024, multiline: false) else {
            throw OpenMedDecisionError.invalidConfiguration(reason: reason)
        }
        return stripped
    }
}

#if canImport(Tokenizers)
    /// Adapts a `swift-transformers` tokenizer loaded from a checkpoint's
    /// `tokenizer.json` (special tokens added, no truncation or padding).
    public struct OpenMedTransformersDecisionTokenizer: OpenMedDecisionTokenizer {
        private let tokenizer: any Tokenizer

        public init(_ tokenizer: any Tokenizer) { self.tokenizer = tokenizer }

        public func encode(_ text: String) throws -> [Int] { tokenizer.encode(text: text, addSpecialTokens: true) }

        public func tokenID(for token: String) -> Int? { tokenizer.convertTokenToId(token) }
    }
#endif
