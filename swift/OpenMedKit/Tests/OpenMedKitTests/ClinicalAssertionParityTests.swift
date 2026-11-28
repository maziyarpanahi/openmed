import CryptoKit
import Foundation
import XCTest

@testable import OpenMedKit

/// Replays `tests/fixtures/parity/clinical_assertion_v1.json`, recorded from
/// Python `assert_context`, through `OpenMedClinicalContext`.
///
/// Every case must match the recorded Python output, including the
/// `reference_gap` cases where that output contradicts the category's intended
/// value (for example `Denies fever and cough.` leaving `cough` affirmed). The
/// gaps are parity with Python, not Swift-only fixes.
final class ClinicalAssertionParityTests: XCTestCase {
    private static let vectorsPath = "tests/fixtures/parity/clinical_assertion_v1.json"
    private static let cueTablePath = "tests/fixtures/parity/context_cues_en_v1.json"
    private static let axes = ["experiencer", "negation", "temporality", "uncertainty"]

    private static func repositoryRoot() -> URL {
        var root = URL(fileURLWithPath: #filePath)
        for _ in 0..<5 { root.deleteLastPathComponent() }
        return root
    }

    private func vectors() throws -> [String: Any] {
        let data = try Data(contentsOf: Self.repositoryRoot().appending(path: Self.vectorsPath))
        let vectors = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(vectors["schema"] as? String, "openmed.tests.clinical_assertion_parity")
        XCTAssertEqual(vectors["version"] as? Int, 1)
        XCTAssertEqual(vectors["synthetic"] as? Bool, true)
        XCTAssertEqual(vectors["offset_unit"] as? String, "unicode_scalar")
        XCTAssertEqual(vectors["language"] as? String, "en")
        return vectors
    }

    private func cases() throws -> [[String: Any]] {
        try XCTUnwrap(vectors()["cases"] as? [[String: Any]])
    }

    private static func span(_ value: Any?) throws -> OpenMedClinicalContext.Span {
        let object = try XCTUnwrap(value as? [String: Any])
        return OpenMedClinicalContext.Span(
            start: try XCTUnwrap(object["start"] as? Int),
            end: try XCTUnwrap(object["end"] as? Int)
        )
    }

    private static func axes(of assertion: OpenMedClinicalAssertion) -> [String: String] {
        [
            "experiencer": assertion.experiencer.rawValue,
            "negation": assertion.negation.rawValue,
            "temporality": assertion.temporality.rawValue,
            "uncertainty": assertion.uncertainty.rawValue,
        ]
    }

    func testEveryCaseReproducesTheRecordedPythonOutput() throws {
        let context = try OpenMedClinicalContext.bundled()
        let cases = try cases()
        XCTAssertGreaterThanOrEqual(cases.count, 40)
        var gapCases = 0
        for item in cases {
            let id = try XCTUnwrap(item["id"] as? String)
            let text = try XCTUnwrap(item["text"] as? String)
            let target = try Self.span(item["target"])
            let targetText = try XCTUnwrap((item["target"] as? [String: Any])?["text"] as? String)
            let scalars = Array(text.unicodeScalars)
            XCTAssertEqual(String(String.UnicodeScalarView(scalars[target.start..<target.end])), targetText, id)
            let sentences = try XCTUnwrap(item["sentences"] as? [Any]).map(Self.span)
            let expected = try XCTUnwrap(item["expected"] as? [String: String])
            if item["reference_gap"] is String { gapCases += 1 }

            let result = try context.assertions(in: text, spans: [target], sentences: sentences)
            let assertion = try XCTUnwrap(result.assertions?.first, id)
            XCTAssertEqual(Self.axes(of: assertion), expected, "case \(id)")
        }
        XCTAssertGreaterThan(gapCases, 0, "the vectors must keep their reference_gap cases")
    }

    func testReferenceGapCasesMatchPythonRatherThanTheIntendedValue() throws {
        let vectors = try vectors()
        let categories = try XCTUnwrap(vectors["categories"] as? [String: [String: Any]])
        let context = try OpenMedClinicalContext.bundled()
        var seen: Set<String> = []
        for item in try cases() {
            guard let gap = item["reference_gap"] as? String else { continue }
            seen.insert(gap)
            let id = try XCTUnwrap(item["id"] as? String)
            let category = try XCTUnwrap(item["category"] as? String)
            let intended = try XCTUnwrap(categories[category]?["intended"] as? [String: String])
            let expected = try XCTUnwrap(item["expected"] as? [String: String])
            XCTAssertNotEqual(expected, intended, "case \(id) is a gap, so Python differs from intended")
            let text = try XCTUnwrap(item["text"] as? String)
            let sentences = try XCTUnwrap(item["sentences"] as? [Any]).map(Self.span)
            let assertion = try XCTUnwrap(
                context.assertions(in: text, spans: [Self.span(item["target"])], sentences: sentences)
                    .assertions?.first
            )
            XCTAssertEqual(Self.axes(of: assertion), expected, "case \(id) must keep Python's recorded gap")
        }
        XCTAssertEqual(seen, Set(try XCTUnwrap(vectors["reference_gap_codes"] as? [String])))
    }

    func testKnownNegationScopeGapIsKeptForParity() throws {
        let context = try OpenMedClinicalContext.bundled()
        let text = "Denies fever and cough."
        let assertions = try XCTUnwrap(
            context.assertions(
                in: text,
                spans: [.init(start: 7, end: 12), .init(start: 17, end: 22)]
            ).assertions
        )
        XCTAssertEqual(assertions.map(\.negation), [.negated, .affirmed])
    }

    func testTheRegexSentenceFallbackReproducesTheVectorsToo() throws {
        let context = try OpenMedClinicalContext.bundled()
        for item in try cases() {
            let id = try XCTUnwrap(item["id"] as? String)
            let text = try XCTUnwrap(item["text"] as? String)
            let expected = try XCTUnwrap(item["expected"] as? [String: String])
            let assertion = try XCTUnwrap(
                context.assertions(in: text, spans: [Self.span(item["target"])]).assertions?.first
            )
            XCTAssertEqual(Self.axes(of: assertion), expected, "case \(id) without caller sentences")
        }
        let windows = OpenMedClinicalContext.regexSentenceWindows(Array("No fever.  Cough!\n\nRash".unicodeScalars))
        XCTAssertEqual(windows, [0..<9, 11..<17, 19..<23])
        XCTAssertEqual(OpenMedClinicalContext.regexSentenceWindows([]), [])
    }

    func testBundledCueTableIsByteIdenticalToTheSharedTable() throws {
        let bundled = try XCTUnwrap(OpenMedClinicalContext.bundledCueTableData())
        let shared = try Data(contentsOf: Self.repositoryRoot().appending(path: Self.cueTablePath))
        let digest = SHA256.hash(data: shared).map { String(format: "%02x", $0) }.joined()
        XCTAssertEqual(
            SHA256.hash(data: bundled).map { String(format: "%02x", $0) }.joined(),
            digest,
            "Resources/clinical/context_cues_en_v1.json is stale; copy \(Self.cueTablePath)"
        )
        XCTAssertEqual(try vectors()["cue_table_sha256"] as? String, digest)
        XCTAssertEqual(try OpenMedClinicalContext.bundled().cueTableSHA256, digest)
    }

    func testCueTableDecodingFailsClosed() throws {
        let shared = try Data(contentsOf: Self.repositoryRoot().appending(path: Self.cueTablePath))
        let edits: [(String, (inout [String: Any]) -> Void)] = [
            ("schema", { $0["schema"] = "openmed.other" }),
            ("version", { $0["version"] = 2 }),
            ("language", { $0["language"] = "es" }),
            ("missing lexicon", { $0["lexicon"] = nil }),
            (
                "token boundaries",
                { table in
                    var lexicon = table["lexicon"] as? [String: Any] ?? [:]
                    lexicon["token_boundaries"] = false
                    table["lexicon"] = lexicon
                }
            ),
            (
                "missing negation",
                { table in
                    var lexicon = table["lexicon"] as? [String: Any] ?? [:]
                    lexicon["negation"] = nil
                    table["lexicon"] = lexicon
                }
            ),
        ]
        for (name, edit) in edits {
            var table = try XCTUnwrap(JSONSerialization.jsonObject(with: shared) as? [String: Any])
            edit(&table)
            XCTAssertThrowsError(try OpenMedClinicalContext(cueTable: JSONSerialization.data(withJSONObject: table)), name) {
                XCTAssertEqual($0 as? OpenMedClinicalContextError, .cueTableInvalid, name)
            }
        }
    }

    func testOnlyEnglishTakesTheEnglishPath() throws {
        let context = try OpenMedClinicalContext.bundled()
        let text = "Paciente niega dolor torácico."
        for language in ["es", "es-MX", "de", "fr_CA", "zh"] {
            let result = try context.assertions(in: text, spans: [.init(start: 15, end: 20)], language: language)
            XCTAssertEqual(result, .unavailable(.unsupportedLanguage), language)
            XCTAssertNil(result.assertions)
            XCTAssertEqual(result.unavailableReason, .unsupportedLanguage)
        }
        for language in [nil, "", "en", "EN", "en-US", "en_GB", " en "] as [String?] {
            XCTAssertTrue(OpenMedClinicalContext.isEnglish(language), language ?? "nil")
            let result = try context.assertions(
                in: "Patient denies chest pain.",
                spans: [.init(start: 15, end: 25)],
                language: language
            )
            XCTAssertEqual(result.assertions?.first?.negation, .negated, language ?? "nil")
        }
    }

    func testOutOfRangeOffsetsFailValueFree() throws {
        let context = try OpenMedClinicalContext.bundled()
        let text = "No fever."
        for span in [OpenMedClinicalContext.Span(start: -1, end: 2), .init(start: 3, end: 2), .init(start: 0, end: 10)] {
            XCTAssertThrowsError(try context.assertions(in: text, spans: [span])) { error in
                XCTAssertEqual(error as? OpenMedClinicalContextError, .invalidSpan)
                XCTAssertFalse(String(reflecting: error).contains("fever"))
            }
        }
        XCTAssertThrowsError(
            try context.assertions(in: text, spans: [.init(start: 3, end: 8)], sentences: [.init(start: 0, end: 12)])
        ) { error in
            XCTAssertEqual(error as? OpenMedClinicalContextError, .invalidSentence)
        }
        XCTAssertEqual(try context.assertions(in: text, spans: []), .asserted([]))
    }

    /// Extra probes whose expected axes were produced by Python
    /// `assert_context(text, [target])` (no caller sentences).
    func testMatchingFollowsPythonCaseTokenAndClauseRules() throws {
        let context = try OpenMedClinicalContext.bundled()
        let probes: [(text: String, target: String, axes: [String])] = [
            ("PATIENT DENIES CHEST PAIN.", "CHEST PAIN", ["patient", "negated", "recent", "certain"]),
            ("Denies\t\n chest pain.", "chest pain", ["patient", "negated", "recent", "certain"]),
            ("No\u{00A0}fever.", "fever", ["patient", "negated", "recent", "certain"]),
            // Token boundaries: "no" inside "Snow" or "nodule" is not a cue.
            ("Snow nodule seen.", "seen", ["patient", "affirmed", "recent", "certain"]),
            // Two negation cues cancel out; a pseudo-negation is masked.
            ("Not without pain.", "pain", ["patient", "affirmed", "recent", "certain"]),
            ("Pneumonia cannot be excluded.", "Pneumonia", ["patient", "affirmed", "recent", "uncertain"]),
            // Python folds the long s onto ASCII s under re.IGNORECASE.
            ("Patient denie\u{17F} rash.", "rash", ["patient", "negated", "recent", "certain"]),
            ("Patient d\u{130}nies rash.", "rash", ["patient", "affirmed", "recent", "certain"]),
            // Commas do not end a clause, so every cue in the sentence applies.
            (
                "History of asthma, no wheezing, mother has eczema.", "eczema",
                ["family", "negated", "historical", "certain"]
            ),
            (
                "Possible pneumonia, ruled out sepsis; if fever recurs, return.", "fever",
                ["patient", "affirmed", "hypothetical", "uncertain"]
            ),
            ("Brother's friend has flu.", "flu", ["other", "affirmed", "recent", "certain"]),
            ("Mother had breast cancer.", "breast cancer", ["family", "affirmed", "recent", "certain"]),
        ]
        for probe in probes {
            let span = try XCTUnwrap(TaskPackTestSupportScalars.span(of: probe.target, in: probe.text))
            let assertion = try XCTUnwrap(context.assertions(in: probe.text, spans: [span]).assertions?.first)
            XCTAssertEqual(
                [
                    assertion.experiencer.rawValue, assertion.negation.rawValue, assertion.temporality.rawValue,
                    assertion.uncertainty.rawValue,
                ],
                probe.axes,
                probe.target
            )
        }
    }
}

/// Scalar-offset helper usable without the MLX-only task pack doubles.
enum TaskPackTestSupportScalars {
    static func span(of needle: String, in text: String) -> OpenMedClinicalContext.Span? {
        guard let range = text.range(of: needle) else { return nil }
        let start = text.unicodeScalars.distance(from: text.unicodeScalars.startIndex, to: range.lowerBound)
        return OpenMedClinicalContext.Span(start: start, end: start + needle.unicodeScalars.count)
    }
}
