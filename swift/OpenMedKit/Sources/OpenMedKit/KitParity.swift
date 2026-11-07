import Foundation

/// A value-free kit-parity profile violation.
///
/// Carries a stable reason code and a JSON field path only. Unknown keys are
/// reported at the enclosing object path, so a key name is never echoed.
public struct KitParityError: Error, Equatable, Sendable, CustomStringConvertible,
    LocalizedError
{
    public let code: String
    public let fieldPath: String

    public init(_ code: String, _ fieldPath: String = "$") {
        self.code = code
        self.fieldPath = fieldPath
    }

    public var description: String { "\(code) at \(fieldPath)" }

    public var errorDescription: String? { description }
}

/// OpenMedKit kit-parity result profile (`openmed.kit_parity_result` v1).
///
/// Every OpenMedKit surface projects its native de-identification result into
/// one value-free record and serializes it to the same canonical JSON bytes.
/// The shared golden vectors live in
/// `tests/fixtures/parity/kit_parity_vectors.json`.
///
/// - Offsets are half-open Unicode scalar offsets into the exact source text.
///   UTF-8 and UTF-16 offsets are derived through the conversion helpers.
/// - `canonicalLabel` and `action` come from fixed vocabularies; spans are
///   ordered by `(start, end)`, non-empty, and non-overlapping.
/// - `confidencePPM` is `floor(Double(confidence) * 1_000_000 + 0.5)`, omitted
///   when a surface has no score. `replacement` is omitted for `keep`.
/// - Canonical JSON uses sorted keys, no whitespace, integers only, and RFC
///   8785 string escaping. Compare canonical output as UTF-8 bytes: Swift
///   `String` equality treats canonically equivalent text as equal.
public enum KitParity {
    public static let profile = "openmed.kit_parity_result"
    public static let profileVersion = 1
    public static let offsetUnit = "unicode_scalar"
    public static let confidenceScale = 1_000_000
    /// Offset-contract versions whose scalar and grapheme semantics are accepted.
    public static let offsetContractVersions: Set<Int> = [1]
    /// Span decoder contract versions whose output is accepted.
    public static let decoderContractVersions: Set<Int> = [1]

    public static let actions = ["hash", "keep", "mask", "redact", "remove", "replace"]

    public static let canonicalLabels = [
        "ACCOUNT_NUMBER", "AGE", "AMOUNT", "ANTIBIOTIC", "API_KEY", "BIC",
        "BITCOIN_ADDRESS", "BODY_SITE", "BUILDING_NUMBER", "CONDITION", "CREDIT_CARD",
        "CREDIT_CARD_ISSUER", "CURRENCY", "CVV", "DATE", "DATE_OF_BIRTH", "DIET_TYPE",
        "EMAIL", "ETHEREUM_ADDRESS", "EYE_COLOR", "FEEDING_ROUTE", "FIRST_NAME", "GENDER",
        "GPS_COORDINATES", "HEIGHT", "IBAN", "ID_NUM", "IMEI", "IP_ADDRESS",
        "JOB_DEPARTMENT", "JOB_TITLE", "LAB_TEST", "LAST_NAME", "LITECOIN_ADDRESS",
        "LOCATION", "MAC_ADDRESS", "MASKED_NUMBER", "MEDICATION", "MICROORGANISM",
        "MIDDLE_NAME", "NUTRITIONAL_STATUS", "NUTRITION_TARGET", "OCCUPATION",
        "ORDINAL_DIRECTION", "ORGANIZATION", "OTHER", "PASSWORD", "PERSON", "PHONE", "PIN",
        "PREFIX", "PROCEDURE", "SSN", "STREET_ADDRESS", "SUSCEPTIBILITY", "TIME", "URL",
        "USERNAME", "USER_AGENT", "VEHICLE_REGISTRATION", "VIN", "ZIPCODE",
    ]

    /// One applied span; offsets are Unicode scalar offsets into the source.
    public struct Span: Equatable, Sendable {
        public var label: String
        public var canonicalLabel: String
        public var action: String
        public var start: Int
        public var end: Int
        public var confidencePPM: Int?
        public var replacement: String?

        public init(
            label: String,
            canonicalLabel: String,
            action: String,
            start: Int,
            end: Int,
            confidencePPM: Int? = nil,
            replacement: String? = nil
        ) {
            self.label = label
            self.canonicalLabel = canonicalLabel
            self.action = action
            self.start = start
            self.end = end
            self.confidencePPM = confidencePPM
            self.replacement = replacement
        }
    }

    /// A kit-parity projection of one de-identification result.
    public struct Result: Equatable, Sendable {
        public var deidentifiedText: String
        public var spans: [Span]

        public init(deidentifiedText: String, spans: [Span]) {
            self.deidentifiedText = deidentifiedText
            self.spans = spans
        }
    }

    private static let actionSet = Set(actions)
    private static let labelSet = Set(canonicalLabels)
    private static let topLevelKeys: Set<String> = [
        "deidentified_text", "offset_unit", "profile", "profile_version", "spans",
    ]
    private static let spanKeys: Set<String> = [
        "action", "canonical_label", "confidence_ppm", "end", "label", "replacement", "start",
    ]

    // MARK: - Offset conversions

    /// Convert a Unicode scalar offset into a UTF-16 code-unit offset.
    public static func utf16Offset(in text: String, scalarOffset: Int) throws -> Int {
        let index = try scalarIndex(in: text, scalarOffset: scalarOffset)
        return text.utf16.distance(from: text.utf16.startIndex, to: index)
    }

    /// Convert a Unicode scalar offset into a UTF-8 byte offset.
    public static func utf8Offset(in text: String, scalarOffset: Int) throws -> Int {
        let index = try scalarIndex(in: text, scalarOffset: scalarOffset)
        return text.utf8.distance(from: text.utf8.startIndex, to: index)
    }

    /// Convert a UTF-16 code-unit offset into a Unicode scalar offset.
    ///
    /// An offset between the halves of a surrogate pair throws
    /// `offset_not_scalar_boundary`.
    public static func scalarOffset(in text: String, utf16Offset: Int) throws -> Int {
        guard utf16Offset >= 0, utf16Offset <= text.utf16.count else {
            throw KitParityError("offset_out_of_bounds", "$.offset")
        }
        let index = text.utf16.index(text.utf16.startIndex, offsetBy: utf16Offset)
        return try scalarDistance(in: text, to: index)
    }

    /// Convert a UTF-8 byte offset into a Unicode scalar offset.
    ///
    /// An offset inside a multi-byte sequence throws
    /// `offset_not_scalar_boundary`.
    public static func scalarOffset(in text: String, utf8Offset: Int) throws -> Int {
        guard utf8Offset >= 0, utf8Offset <= text.utf8.count else {
            throw KitParityError("offset_out_of_bounds", "$.offset")
        }
        let index = text.utf8.index(text.utf8.startIndex, offsetBy: utf8Offset)
        return try scalarDistance(in: text, to: index)
    }

    /// Encode a confidence in `[0, 1]` as integer parts per million.
    public static func confidencePPM(_ confidence: Float) throws -> Int {
        let value = Double(confidence)
        guard value.isFinite, value >= 0, value <= 1 else {
            throw KitParityError("confidence_invalid", "$.confidence")
        }
        return Int((value * Double(confidenceScale) + 0.5).rounded(.down))
    }

    // MARK: - Bundle compatibility

    /// Reject bundle contracts this profile cannot interpret.
    ///
    /// Call this before consuming any result produced under a model bundle.
    public static func checkCompatibility(
        offsetContractVersion: Int,
        decoderContractVersion: Int
    ) throws {
        guard offsetContractVersions.contains(offsetContractVersion) else {
            throw KitParityError(
                "offset_contract_version_unsupported",
                "$.offset_contract_version"
            )
        }
        guard decoderContractVersions.contains(decoderContractVersion) else {
            throw KitParityError(
                "decoder_contract_version_unsupported",
                "$.decoder_contract_version"
            )
        }
    }

    /// Gate a bundle's contracts, then parse a result produced under it.
    public static func accept(
        _ data: Data,
        offsetContractVersion: Int,
        decoderContractVersion: Int
    ) throws -> Result {
        try checkCompatibility(
            offsetContractVersion: offsetContractVersion,
            decoderContractVersion: decoderContractVersion
        )
        return try parse(data)
    }

    // MARK: - Validation and serialization

    /// Validate a result against the profile vocabulary and ordering rules.
    public static func validate(_ result: Result) throws {
        var previous: Span?
        for (index, span) in result.spans.enumerated() {
            let path = "$.spans[\(index)]"
            try checkSpan(span, path: path)
            try checkOrder(span, after: previous, path: path)
            previous = span
        }
    }

    /// Return the canonical JSON text of a result.
    ///
    /// Encode it with `.utf8` for the canonical bytes.
    public static func serialize(_ result: Result) throws -> String {
        try validate(result)
        var json = "{\"deidentified_text\":" + quote(result.deidentifiedText)
        json += ",\"offset_unit\":\"\(offsetUnit)\",\"profile\":\"\(profile)\""
        json += ",\"profile_version\":\(profileVersion),\"spans\":["
        for (index, span) in result.spans.enumerated() {
            if index > 0 {
                json += ","
            }
            json += "{\"action\":" + quote(span.action)
            json += ",\"canonical_label\":" + quote(span.canonicalLabel)
            if let confidence = span.confidencePPM {
                json += ",\"confidence_ppm\":\(confidence)"
            }
            json += ",\"end\":\(span.end),\"label\":" + quote(span.label)
            if let replacement = span.replacement {
                json += ",\"replacement\":" + quote(replacement)
            }
            json += ",\"start\":\(span.start)}"
        }
        return json + "]}"
    }

    /// Parse and strictly validate canonical kit-parity result JSON.
    ///
    /// Checks run in the order shared by every surface: JSON syntax, object
    /// shape, profile, profile version, unknown top-level keys, offset unit,
    /// de-identified text, each span, span ordering, then the canonical bytes.
    public static func parse(_ data: Data) throws -> Result {
        let document: JSONValue
        do {
            document = try JSONDecoder().decode(JSONValue.self, from: data)
        } catch {
            throw KitParityError("json_invalid")
        }
        guard case .object(let object) = document else {
            throw KitParityError("result_not_object")
        }
        guard try requiredString(object, "profile", "$") == profile else {
            throw KitParityError("profile_unknown", "$.profile")
        }
        guard try requiredInteger(object, "profile_version", "$") == profileVersion else {
            throw KitParityError("profile_version_unsupported", "$.profile_version")
        }
        guard Set(object.keys).isSubset(of: topLevelKeys) else {
            throw KitParityError("field_unknown", "$")
        }
        guard try requiredString(object, "offset_unit", "$") == offsetUnit else {
            throw KitParityError("offset_unit_unsupported", "$.offset_unit")
        }
        let deidentifiedText = try requiredString(object, "deidentified_text", "$")
        let rawSpans = try requiredArray(object, "spans", "$")

        var spans: [Span] = []
        for (index, raw) in rawSpans.enumerated() {
            let path = "$.spans[\(index)]"
            let span = try parseSpan(raw, path: path)
            try checkOrder(span, after: spans.last, path: path)
            spans.append(span)
        }

        let result = Result(deidentifiedText: deidentifiedText, spans: spans)
        guard Array(try serialize(result).utf8) == Array(data) else {
            throw KitParityError("result_not_canonical")
        }
        return result
    }

    /// Parse canonical kit-parity result JSON text.
    public static func parse(_ json: String) throws -> Result {
        try parse(Data(json.utf8))
    }

    /// Apply span replacements right to left at Unicode scalar offsets.
    public static func applySpans(_ spans: [Span], to source: String) throws -> String {
        // Ordered, non-empty, non-overlapping spans keep every range in bounds.
        try validate(Result(deidentifiedText: "", spans: spans))
        let length = source.unicodeScalars.count
        for (index, span) in spans.enumerated() where span.end > length {
            throw KitParityError("offset_out_of_bounds", "$.spans[\(index)].end")
        }
        var scalars = Array(source.unicodeScalars)
        for span in spans.reversed() {
            guard let replacement = span.replacement else {
                continue
            }
            scalars.replaceSubrange(span.start..<span.end, with: replacement.unicodeScalars)
        }
        var view = String.UnicodeScalarView()
        view.append(contentsOf: scalars)
        return String(view)
    }

    /// Check that a result's offsets and UTF-8 bytes agree with its source.
    public static func verify(_ result: Result, against source: String) throws {
        try validate(result)
        let expected = try applySpans(result.spans, to: source)
        guard Array(expected.utf8) == Array(result.deidentifiedText.utf8) else {
            throw KitParityError("deidentified_text_mismatch", "$.deidentified_text")
        }
    }

    // MARK: - Private helpers

    private enum JSONValue: Decodable {
        case null
        case bool(Bool)
        case integer(Int)
        case number
        case string(String)
        case array([JSONValue])
        case object([String: JSONValue])

        init(from decoder: Decoder) throws {
            let container = try decoder.singleValueContainer()
            if container.decodeNil() {
                self = .null
            } else if let value = try? container.decode(Bool.self) {
                self = .bool(value)
            } else if let value = try? container.decode(Int.self) {
                self = .integer(value)
            } else if (try? container.decode(Double.self)) != nil {
                self = .number
            } else if let value = try? container.decode(String.self) {
                self = .string(value)
            } else if let value = try? container.decode([JSONValue].self) {
                self = .array(value)
            } else {
                self = .object(try container.decode([String: JSONValue].self))
            }
        }
    }

    private static func parseSpan(_ raw: JSONValue, path: String) throws -> Span {
        guard case .object(let object) = raw else {
            throw KitParityError("field_type_invalid", path)
        }
        guard Set(object.keys).isSubset(of: spanKeys) else {
            throw KitParityError("field_unknown", path)
        }
        let label = try requiredString(object, "label", path)
        try checkLabel(label, path: path)
        let canonicalLabel = try requiredString(object, "canonical_label", path)
        try checkCanonicalLabel(canonicalLabel, path: path)
        let action = try requiredString(object, "action", path)
        try checkAction(action, path: path)
        let start = try requiredInteger(object, "start", path)
        try checkStart(start, path: path)
        let end = try requiredInteger(object, "end", path)
        try checkEnd(start: start, end: end, path: path)
        let confidence = try optionalInteger(object, "confidence_ppm", path)
        try checkConfidence(confidence, path: path)
        let replacement = try optionalString(object, "replacement", path)
        try checkReplacement(action: action, replacement: replacement, path: path)
        return Span(
            label: label,
            canonicalLabel: canonicalLabel,
            action: action,
            start: start,
            end: end,
            confidencePPM: confidence,
            replacement: replacement
        )
    }

    private static func checkSpan(_ span: Span, path: String) throws {
        try checkLabel(span.label, path: path)
        try checkCanonicalLabel(span.canonicalLabel, path: path)
        try checkAction(span.action, path: path)
        try checkStart(span.start, path: path)
        try checkEnd(start: span.start, end: span.end, path: path)
        try checkConfidence(span.confidencePPM, path: path)
        try checkReplacement(action: span.action, replacement: span.replacement, path: path)
    }

    private static func checkLabel(_ label: String, path: String) throws {
        let scalars = Array(label.unicodeScalars)
        guard let first = scalars.first, scalars.count <= 80,
            isASCIILetter(first),
            scalars.dropFirst().allSatisfy({
                isASCIILetter($0) || ("0"..."9").contains($0) || "_:-".unicodeScalars.contains($0)
            })
        else {
            throw KitParityError("label_invalid", "\(path).label")
        }
    }

    private static func isASCIILetter(_ scalar: Unicode.Scalar) -> Bool {
        ("A"..."Z").contains(scalar) || ("a"..."z").contains(scalar)
    }

    private static func checkCanonicalLabel(_ label: String, path: String) throws {
        // Compare scalars, not `String` equality, so a canonically equivalent
        // spelling cannot pass as a vocabulary label.
        guard label.unicodeScalars.allSatisfy(\.isASCII), labelSet.contains(label) else {
            throw KitParityError("label_unknown", "\(path).canonical_label")
        }
    }

    private static func checkAction(_ action: String, path: String) throws {
        guard action.unicodeScalars.allSatisfy(\.isASCII), actionSet.contains(action) else {
            throw KitParityError("action_unknown", "\(path).action")
        }
    }

    private static func checkStart(_ start: Int, path: String) throws {
        guard start >= 0 else {
            throw KitParityError("offset_invalid", "\(path).start")
        }
    }

    private static func checkEnd(start: Int, end: Int, path: String) throws {
        guard end > start else {
            throw KitParityError("offset_invalid", "\(path).end")
        }
    }

    private static func checkConfidence(_ confidence: Int?, path: String) throws {
        if let confidence, confidence < 0 || confidence > confidenceScale {
            throw KitParityError("confidence_invalid", "\(path).confidence_ppm")
        }
    }

    private static func checkReplacement(
        action: String,
        replacement: String?,
        path: String
    ) throws {
        if action == "keep" {
            guard replacement == nil else {
                throw KitParityError("replacement_unexpected", "\(path).replacement")
            }
        } else if replacement == nil {
            throw KitParityError("field_missing", "\(path).replacement")
        }
    }

    private static func checkOrder(_ span: Span, after previous: Span?, path: String) throws {
        guard let previous else {
            return
        }
        if span.start < previous.start || (span.start == previous.start && span.end <= previous.end) {
            throw KitParityError("span_order_invalid", path)
        }
        if span.start < previous.end {
            throw KitParityError("span_overlap", path)
        }
    }

    private static func field(
        _ object: [String: JSONValue],
        _ key: String,
        _ path: String
    ) throws -> JSONValue? {
        guard let value = object[key] else {
            return nil
        }
        if case .null = value {
            throw KitParityError("field_null", "\(path).\(key)")
        }
        return value
    }

    private static func requiredString(
        _ object: [String: JSONValue],
        _ key: String,
        _ path: String
    ) throws -> String {
        guard let value = try optionalString(object, key, path) else {
            throw KitParityError("field_missing", "\(path).\(key)")
        }
        return value
    }

    private static func optionalString(
        _ object: [String: JSONValue],
        _ key: String,
        _ path: String
    ) throws -> String? {
        guard let value = try field(object, key, path) else {
            return nil
        }
        guard case .string(let string) = value else {
            throw KitParityError("field_type_invalid", "\(path).\(key)")
        }
        return string
    }

    private static func requiredInteger(
        _ object: [String: JSONValue],
        _ key: String,
        _ path: String
    ) throws -> Int {
        guard let value = try optionalInteger(object, key, path) else {
            throw KitParityError("field_missing", "\(path).\(key)")
        }
        return value
    }

    private static func optionalInteger(
        _ object: [String: JSONValue],
        _ key: String,
        _ path: String
    ) throws -> Int? {
        guard let value = try field(object, key, path) else {
            return nil
        }
        guard case .integer(let integer) = value else {
            throw KitParityError("field_type_invalid", "\(path).\(key)")
        }
        return integer
    }

    private static func requiredArray(
        _ object: [String: JSONValue],
        _ key: String,
        _ path: String
    ) throws -> [JSONValue] {
        guard let value = try field(object, key, path) else {
            throw KitParityError("field_missing", "\(path).\(key)")
        }
        guard case .array(let array) = value else {
            throw KitParityError("field_type_invalid", "\(path).\(key)")
        }
        return array
    }

    private static func quote(_ text: String) -> String {
        var scalars = String.UnicodeScalarView()
        scalars.append("\"")
        for scalar in text.unicodeScalars {
            switch scalar {
            case "\"":
                scalars.append(contentsOf: "\\\"".unicodeScalars)
            case "\\":
                scalars.append(contentsOf: "\\\\".unicodeScalars)
            case "\u{08}":
                scalars.append(contentsOf: "\\b".unicodeScalars)
            case "\u{0C}":
                scalars.append(contentsOf: "\\f".unicodeScalars)
            case "\n":
                scalars.append(contentsOf: "\\n".unicodeScalars)
            case "\r":
                scalars.append(contentsOf: "\\r".unicodeScalars)
            case "\t":
                scalars.append(contentsOf: "\\t".unicodeScalars)
            default:
                if scalar.value < 0x20 {
                    let hex = String(scalar.value, radix: 16)
                    let padded = String(repeating: "0", count: 4 - hex.count) + hex
                    scalars.append(contentsOf: "\\u\(padded)".unicodeScalars)
                } else {
                    scalars.append(scalar)
                }
            }
        }
        scalars.append("\"")
        return String(scalars)
    }

    private static func scalarIndex(in text: String, scalarOffset: Int) throws -> String.Index {
        guard scalarOffset >= 0, scalarOffset <= text.unicodeScalars.count else {
            throw KitParityError("offset_out_of_bounds", "$.offset")
        }
        return text.unicodeScalars.index(text.unicodeScalars.startIndex, offsetBy: scalarOffset)
    }

    private static func scalarDistance(in text: String, to index: String.Index) throws -> Int {
        guard let scalarIndex = index.samePosition(in: text.unicodeScalars) else {
            throw KitParityError("offset_not_scalar_boundary", "$.offset")
        }
        return text.unicodeScalars.distance(from: text.unicodeScalars.startIndex, to: scalarIndex)
    }
}

extension PolicyDeidentificationResult {
    /// Project this policy result into the kit-parity result profile.
    ///
    /// Action records already carry Unicode scalar offsets; they are ordered by
    /// `(start, end)`, the confidence is encoded in parts per million, and
    /// `keep` records omit their replacement.
    public func kitParityResult() throws -> KitParity.Result {
        let ordered = actions.sorted {
            ($0.start, $0.end) < ($1.start, $1.end)
        }
        var spans: [KitParity.Span] = []
        for (index, record) in ordered.enumerated() {
            let confidence: Int
            do {
                confidence = try KitParity.confidencePPM(record.confidence)
            } catch {
                throw KitParityError("confidence_invalid", "$.spans[\(index)].confidence_ppm")
            }
            spans.append(
                KitParity.Span(
                    label: record.label,
                    canonicalLabel: record.canonicalLabel,
                    action: record.action.rawValue,
                    start: record.start,
                    end: record.end,
                    confidencePPM: confidence,
                    replacement: record.replacement
                )
            )
        }
        let result = KitParity.Result(deidentifiedText: redactedText, spans: spans)
        try KitParity.validate(result)
        return result
    }
}
