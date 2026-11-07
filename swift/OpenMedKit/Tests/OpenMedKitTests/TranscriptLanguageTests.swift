import Foundation
import XCTest

@testable import OpenMedKit

final class TranscriptLanguageTests: XCTestCase {
    func testSharedPythonSwiftSyntheticLanguageDecisions() throws {
        var root = URL(fileURLWithPath: #filePath)
        for _ in 0..<5 { root.deleteLastPathComponent() }
        let data = try Data(contentsOf: root.appending(path: "tests/fixtures/multimodal/transcript_language.json"))
        let cases = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [[String: Any]])
        for item in cases {
            let labels = try XCTUnwrap(item["labels"] as? [String: String])
            let text = try XCTUnwrap(item["text"] as? String)
            let confidence = try XCTUnwrap(item["text_confidence"] as? Double)
            let detector: TranscriptLanguageRouter.Detector = { value in
                try ["Ada", "Lucía"].compactMap { name in
                    guard let range = value.range(of: name) else { return nil }
                    let start = value[..<range.lowerBound].unicodeScalars.count
                    return try TranscriptPHISpan(start: start, end: start + name.unicodeScalars.count)
                }
            }
            let instance = try router(labels, confidence: confidence, detectors: ["en": detector, "es": detector])
            let provider = try (item["provider_tag"] as? String).map { tag in
                try TranscriptLanguageHypothesis(tag: tag, confidence: XCTUnwrap(item["confidence"] as? Double))
            }
            let result = try instance.route(segmentIndex: 5, text: text, providerLanguage: provider)
            let audit = try JSONSerialization.jsonObject(with: JSONEncoder().encode(result.audit)) as? NSDictionary
            XCTAssertEqual(audit, item["expected"] as? NSDictionary)
            if let output = item["output"] as? String {
                XCTAssertEqual(try result.reviewedText(reviewerConfirmed: true), output)
            } else {
                XCTAssertThrowsError(try result.reviewedText(reviewerConfirmed: true))
            }
        }
    }

    private func router(
        _ labels: [String: String], installed: [String] = ["en", "es"],
        confidence: Double = 0.99, detectors: [String: TranscriptLanguageRouter.Detector]? = nil
    ) throws -> TranscriptLanguageRouter {
        try TranscriptLanguageRouter(
            installedPackCodes: installed,
            detectors: detectors ?? Dictionary(uniqueKeysWithValues: installed.map { ($0, { _ in [] }) }),
            candidateLanguages: ["en", "es", "fr"]
        ) { token, _ in
            guard let tag = labels[token] else { return nil }
            return try TranscriptLanguageHypothesis(tag: tag, confidence: confidence)
        }
    }

    private func route(
        _ router: TranscriptLanguageRouter, _ text: String, tag: String = "en", confidence: Double = 0.99
    ) throws -> TranscriptLanguageDecision {
        try router.route(
            segmentIndex: 4, text: text,
            providerLanguage: TranscriptLanguageHypothesis(tag: tag, confidence: confidence))
    }

    func testEnglishAndSpanishUseDeclaredDetectorAndRequireReview() throws {
        for (text, tag) in [("Hello Ada", "en"), ("Hola Lucía", "es")] {
            var calls: [String] = []
            let instance = try router(
                Dictionary(uniqueKeysWithValues: text.split(separator: " ").map { (String($0), tag) }),
                installed: [tag],
                detectors: [
                    tag: { value in
                        calls.append(value)
                        let start = value.split(separator: " ")[0].unicodeScalars.count + 1
                        return [try TranscriptPHISpan(start: start, end: value.unicodeScalars.count)]
                    }
                ])
            let result = try route(instance, text, tag: tag)
            XCTAssertEqual(result.audit.status, "supported")
            XCTAssertEqual(calls, [text])
            XCTAssertThrowsError(try result.reviewedText()) { error in
                XCTAssertEqual(error as? TranscriptLanguageError, .reviewRequired)
            }
            let output = try result.reviewedText(reviewerConfirmed: true)
            XCTAssertEqual(output.unicodeScalars.count, text.unicodeScalars.count)
            XCTAssertFalse(output.contains(tag == "en" ? "Ada" : "Lucía"))
        }
    }

    func testMixedOffsetsNamesAndIdentifiersHaveNoLeakageInAudits() throws {
        let text = "🙂 My name is Ada 123-45-6789. Mi nombre es Lucía 987654321.  "
        let labels = ["My": "en", "name": "en", "is": "en", "Ada": "en", "Mi": "es", "nombre": "es", "es": "es", "Lucía": "es"]
        var calls: [(String, String)] = []
        func detector(_ tag: String, _ names: [String]) -> TranscriptLanguageRouter.Detector {
            { value in
                calls.append((tag, value))
                return try names.map { name in
                    let range = try XCTUnwrap(value.range(of: name))
                    let start = value[..<range.lowerBound].unicodeScalars.count
                    return try TranscriptPHISpan(start: start, end: start + name.unicodeScalars.count)
                }
            }
        }
        let instance = try router(labels, detectors: ["en": detector("en", ["Ada", "123-45-6789"]), "es": detector("es", ["Lucía", "987654321"])])
        let result = try route(instance, text, tag: "EN-us")
        let boundary = text[..<text.range(of: "Mi")!.lowerBound].unicodeScalars.count
        XCTAssertEqual(result.audit.status, "mixed")
        XCTAssertEqual(result.audit.providerTag, "en-US")
        XCTAssertEqual(result.audit.runs.map(\.start), [0, boundary])
        XCTAssertEqual(result.audit.runs.map(\.end), [boundary, text.unicodeScalars.count])
        XCTAssertEqual(calls.map(\.0), ["en", "es"])
        XCTAssertEqual(calls.map(\.1).joined(), text)
        let output = try result.reviewedText(reviewerConfirmed: true)
        let audit = String(decoding: try JSONEncoder().encode(result.audit), as: UTF8.self)
        let report = String(decoding: try JSONSerialization.data(withJSONObject: transcriptLanguageReport([result])), as: UTF8.self)
        for name in ["Ada", "Lucía", "123-45-6789", "987654321"] {
            XCTAssertFalse(output.contains(name))
            XCTAssertFalse(audit.contains(name))
            XCTAssertFalse(report.contains(name))
            XCTAssertFalse(String(describing: result).contains(name))
        }
        XCTAssertEqual(output.unicodeScalars.count, text.unicodeScalars.count)
        XCTAssertEqual(transcriptLanguageReport([result])["tag_counts"], ["en": 1, "es": 1])
        XCTAssertTrue(result.notice.hasPrefix("Non-diagnostic"))
    }

    func testUnsupportedLowConfidenceDisagreementAndUnknownNeverCallDetector() throws {
        let cases: [(String, String, Double, [String: String], String)] = [
            ("Bonjour", "fr", 0.99, ["Bonjour": "fr"], "phi_pack_unavailable"),
            ("Hola", "es", 0.79, ["Hola": "es"], "provider_confidence_low"),
            ("Hola", "en", 0.99, ["Hola": "es"], "language_disagreement"),
            ("Unknown", "en", 0.99, [:], "text_language_uncertain"),
            ("Hello Hola nombre", "en", 0.99, ["Hello": "en", "Hola": "es", "nombre": "es"], "language_disagreement"),
        ]
        for (text, tag, confidence, labels, reason) in cases {
            var calls = 0
            let detector: TranscriptLanguageRouter.Detector = { _ in
                calls += 1
                return []
            }
            let result = try route(router(labels, detectors: ["en": detector, "es": detector]), text, tag: tag, confidence: confidence)
            XCTAssertEqual(result.audit.reasonCode, reason)
            XCTAssertEqual(calls, 0)
            XCTAssertThrowsError(try result.reviewedText(reviewerConfirmed: true)) { error in
                XCTAssertEqual(error as? TranscriptLanguageError, .segmentWithheld)
            }
        }
    }

    func testUninstalledMixedPartWithholdsEntireSegmentBeforeDetection() throws {
        var calls = 0
        let result = try route(
            router(
                ["Hello": "en", "Hola": "es"], installed: ["en"],
                detectors: [
                    "en": { _ in
                        calls += 1
                        return []
                    }
                ]), "Hello Hola")
        XCTAssertEqual(result.audit.status, "unsupported")
        XCTAssertEqual(calls, 0)
    }

    func testMissingProviderPartialEmptyAndTextThreshold() throws {
        let instance = try router(["Hello": "en"])
        XCTAssertEqual(try instance.route(segmentIndex: 0, text: "Hello", providerLanguage: nil).audit.reasonCode, "provider_language_missing")
        XCTAssertEqual(try instance.route(segmentIndex: 0, text: "Hello", providerLanguage: TranscriptLanguageHypothesis(tag: "en", confidence: 0.99), finalized: false).audit.reasonCode, "segment_not_finalized")
        for text in ["", "123"] {
            XCTAssertEqual(try route(instance, text).audit.reasonCode, "text_language_uncertain")
        }
        for confidence in [0.79, 0.8, 0.9] {
            XCTAssertEqual(try route(router(["Hello": "en"], confidence: confidence), "Hello").audit.reasonCode, confidence < 0.8 ? "text_confidence_low" : "language_routed")
        }
    }

    func testDetectorFailureAfterSuccessInvalidSpansAndMergedSpans() throws {
        enum SyntheticFailure: Error { case privatePayload }
        let result = try route(router(["Hello": "en", "Hola": "es"], detectors: ["en": { _ in [] }, "es": { _ in throw SyntheticFailure.privatePayload }]), "Hello Hola")
        XCTAssertEqual(result.audit.reasonCode, "detector_failed")
        XCTAssertTrue(result.phiSpans.isEmpty)
        XCTAssertThrowsError(try result.reviewedText(reviewerConfirmed: true))
        let invalid = try route(router(["Hello": "en"], detectors: ["en": { _ in [try TranscriptPHISpan(start: 0, end: 6)] }, "es": { _ in [] }]), "Hello")
        XCTAssertEqual(invalid.audit.reasonCode, "detector_result_invalid")
        let merged = try route(router(["Hello": "en"], detectors: ["en": { _ in [try TranscriptPHISpan(start: 0, end: 3), try TranscriptPHISpan(start: 2, end: 5)] }, "es": { _ in [] }]), "Hello")
        XCTAssertEqual(merged.phiSpans, [try TranscriptPHISpan(start: 0, end: 5)])
        XCTAssertEqual(try merged.reviewedText(reviewerConfirmed: true), "█████")
    }

    func testIdentifierFailureAndInvalidInputsHaveControlledReasons() throws {
        enum SyntheticFailure: Error { case privatePayload }
        let instance = try TranscriptLanguageRouter(installedPackCodes: [], detectors: [:], candidateLanguages: ["en"]) { _, _ in throw SyntheticFailure.privatePayload }
        XCTAssertEqual(try route(instance, "Hello").audit.reasonCode, "text_identifier_failed")
        for tag in ["en-x-private", "synthetic-payload", "en_US", "en\n", "en-John"] {
            XCTAssertThrowsError(try TranscriptLanguageHypothesis(tag: tag, confidence: 0.99)) { error in
                XCTAssertEqual(error as? TranscriptLanguageError, .invalidLanguageTag)
            }
        }
        XCTAssertThrowsError(try TranscriptLanguageHypothesis(tag: "en", confidence: .nan))
        XCTAssertThrowsError(try TranscriptPHISpan(start: -1, end: 3))
        XCTAssertThrowsError(try TranscriptLanguageRouter(installedPackCodes: ["es"], detectors: [:], candidateLanguages: ["es"]) { _, _ in nil })
    }
}
