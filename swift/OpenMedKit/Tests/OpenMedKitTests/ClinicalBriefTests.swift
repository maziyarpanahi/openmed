import Foundation
import XCTest

@testable import OpenMedKit

final class ClinicalBriefTests: XCTestCase {
    #if canImport(MLX) && canImport(MLXLMCommon) && canImport(MLXLLM) && canImport(MLXNN) && canImport(Tokenizers) && !os(watchOS) && !os(visionOS)
        func testUnloadedRuntimeHasTypedUnavailableRefusal() async {
            let runtime = OpenMedMaple(container: nil)
            await runtime.unload()
            do {
                _ = try await runtime.brief(
                    source: "Synthetic evidence.", originalIdentifiers: [],
                    evaluate: { _, _ in
                        XCTFail("Unavailable runtime must not call evaluation")
                        return Data()
                    },
                    privacyCheck: { _ in
                        XCTFail("Unavailable runtime must not call privacy detector")
                        return true
                    })
                XCTFail("Unavailable runtime must refuse")
            } catch {
                XCTAssertEqual(error as? ClinicalBriefError, .modelUnavailable)
                XCTAssertEqual(error.localizedDescription, "model_unavailable")
            }
        }
    #endif

    func testMultilingualSyntheticComposerPacketsPreserveScalarOffsetsAndRefusals() throws {
        var root = URL(fileURLWithPath: #filePath)
        for _ in 0..<5 { root.deleteLastPathComponent() }
        let data = try Data(contentsOf: root.appending(path: "tests/fixtures/eval/summaries/multilingual_brief_packets.json"))
        let corpus = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(corpus["schema_version"] as? Int, 1)
        XCTAssertEqual(corpus["synthetic_only"] as? Bool, true)
        let rows = try XCTUnwrap(corpus["cases"] as? [[String: Any]])
        XCTAssertEqual(rows.count, 10)
        for row in rows {
            let source = try XCTUnwrap(row["source"] as? String)
            let summary = try XCTUnwrap(row["generator_output"] as? String)
            let wire = Data(try XCTUnwrap(row["evaluation_json"] as? String).utf8)
            if row["scenario"] as? String == "unsupported_provider" {
                // Native validation admits successful reviewed packets only.
                // An unavailable language provider must never become output.
                XCTAssertEqual(summary, "")
                XCTAssertThrowsError(
                    try ClinicalBrief.validate(
                        evaluationJSON: wire, source: source, generatedSummary: summary,
                        originalIdentifiers: [], privacyCheck: { _ in true })
                ) {
                    XCTAssertEqual($0 as? ClinicalBriefError, .invalidPacket)
                }
                continue
            }
            let brief = try ClinicalBrief.validate(
                evaluationJSON: wire, source: source, generatedSummary: summary,
                originalIdentifiers: [], privacyCheck: { _ in true })
            XCTAssertEqual(brief.responseJSON(), wire)
            let audit = String(decoding: brief.auditJSON(), as: UTF8.self)
            XCTAssertFalse(audit.contains(source))
            if row["scenario"] as? String == "preserved" {
                XCTAssertEqual(brief.citations.count, 3)
                let input = Array(source.unicodeScalars)
                let output = Array(summary.unicodeScalars)
                for citation in brief.citations {
                    XCTAssertEqual(
                        Array(input[citation.sourceStart..<citation.sourceEnd]),
                        Array(output[citation.outputStart..<citation.outputEnd]))
                }
                // The source has a non-BMP prefix. Swift Character/UTF-16
                // offsets cannot be substituted for the wire's scalar offsets.
                XCTAssertNotEqual(source.utf16.count, input.count)
                XCTAssertThrowsError(
                    try ClinicalBrief.validate(
                        evaluationJSON: wire, source: source + " changed",
                        generatedSummary: summary, originalIdentifiers: [], privacyCheck: { _ in true }))
                XCTAssertThrowsError(
                    try ClinicalBrief.validate(
                        evaluationJSON: wire, source: source, generatedSummary: summary,
                        originalIdentifiers: [summary], privacyCheck: { _ in true }))
            }
        }
    }

    private func fixtureObject() throws -> [String: Any] {
        var root = URL(fileURLWithPath: #filePath)
        for _ in 0..<5 { root.deleteLastPathComponent() }
        let data = try Data(contentsOf: root.appending(path: "tests/fixtures/clinical/brief_parity/verified.json"))
        return try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    private func fixture() throws -> (String, String, Data) {
        let row = try fixtureObject()
        return (
            try XCTUnwrap(row["source"] as? String),
            try XCTUnwrap(row["generator_output"] as? String),
            Data(try XCTUnwrap(row["evaluation_json"] as? String).utf8)
        )
    }

    func testSharedUnicodeLeakageCasesHavePythonParityAndValueFreeRefusals() throws {
        let rows = try XCTUnwrap(fixtureObject()["leakage_cases"] as? [[String: Any]])
        for row in rows {
            let id = try XCTUnwrap(row["id"] as? String)
            let identifier = try XCTUnwrap(row["surface"] as? String)
            let candidate = try XCTUnwrap(row["candidate"] as? String)
            let leaked = try XCTUnwrap((row["native_leaked"] ?? row["leaked"]) as? Bool)
            let data = Data(try XCTUnwrap(row["evaluation_json"] as? String).utf8)
            XCTAssertEqual(
                try ClinicalBriefLeakageMatcher.contains(
                    identifier,
                    in: ClinicalBriefLeakageMatcher.normalize(candidate)), leaked, id)
            var detectorCalls = 0
            if leaked {
                XCTAssertThrowsError(
                    try ClinicalBrief.validate(
                        evaluationJSON: data,
                        source: candidate, generatedSummary: candidate, originalIdentifiers: [identifier],
                        privacyCheck: { _ in
                            detectorCalls += 1
                            return true
                        }), id
                ) {
                    XCTAssertEqual($0 as? ClinicalBriefError, .privacy, id)
                    XCTAssertEqual($0.localizedDescription, "privacy", id)
                }
                XCTAssertEqual(detectorCalls, 0, id)
            } else {
                let brief = try ClinicalBrief.validate(
                    evaluationJSON: data,
                    source: candidate, generatedSummary: candidate, originalIdentifiers: [identifier],
                    privacyCheck: { _ in
                        detectorCalls += 1
                        return true
                    })
                XCTAssertEqual(brief.summary, candidate, id)
                XCTAssertEqual(detectorCalls, 1, id)
            }
        }
    }

    func testSharedNormalizationIncludesEveryRetainedConfusableMapping() throws {
        let rows = try XCTUnwrap(fixtureObject()["normalization_cases"] as? [[String: String]])
        for row in rows {
            let input = try XCTUnwrap(row["input"])
            let expected = try XCTUnwrap(row["normalized"])
            XCTAssertEqual(Array(ClinicalBriefLeakageMatcher.normalize(input).utf8), Array(expected.utf8))
        }
    }

    private final class CheckpointClock: @unchecked Sendable {
        private let lock = NSLock()
        private var remaining: Int
        init(_ remaining: Int) { self.remaining = remaining }
        func expired() -> Bool {
            lock.lock()
            defer { lock.unlock() }
            remaining -= 1
            return remaining <= 0
        }
    }

    func testDeadlineAtEveryNativeBoundaryRejectsLateOutput() async throws {
        let (source, summary, data) = try fixture()
        // generation, verification, packet validation, rendering and final publication.
        for boundary in 1...6 {
            let clock = CheckpointClock(boundary)
            let cancellation = ClinicalBriefCancellation(deadlineExpired: { clock.expired() })
            do {
                _ = try await ClinicalBrief.compose(
                    source: source, originalIdentifiers: [], cancellation: cancellation,
                    generate: { _ in summary }, evaluate: { _, _ in data },
                    privacyCheck: { _ in true })
                XCTFail("Expected deadline refusal")
            } catch {
                XCTAssertEqual(error as? ClinicalBriefError, .deadlineExceeded)
            }
        }
    }

    func testUncancelledNativeCompositionPreservesPacket() async throws {
        let (source, summary, data) = try fixture()
        let result = try await ClinicalBrief.compose(
            source: source, originalIdentifiers: [], generate: { _ in summary },
            evaluate: { _, _ in data }, privacyCheck: { _ in true })
        XCTAssertEqual(result.responseJSON(), data)
    }

    func testNativeTaskCancellationHasControlledOutcome() async throws {
        let (source, summary, data) = try fixture()
        let task = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            return try await ClinicalBrief.compose(
                source: source, originalIdentifiers: [],
                generate: { _ in summary }, evaluate: { _, _ in data },
                privacyCheck: { _ in true })
        }
        do {
            _ = try await task.value
            XCTFail("Expected cancellation")
        } catch {
            XCTAssertEqual(error as? ClinicalBriefError, .cancelled)
        }
    }

    func testNativeLateGenerationDoesNotReachEvaluator() async throws {
        let (source, summary, _) = try fixture()
        let task = Task {
            try await ClinicalBrief.compose(
                source: source, originalIdentifiers: [],
                generate: { _ in
                    withUnsafeCurrentTask { $0?.cancel() }
                    return summary
                },
                evaluate: { _, _ in
                    XCTFail("Cancelled generation reached evaluation")
                    return Data()
                }, privacyCheck: { _ in true })
        }
        do {
            _ = try await task.value
            XCTFail("Expected cancellation")
        } catch {
            XCTAssertEqual(error as? ClinicalBriefError, .cancelled)
        }
    }

    func testSharedPythonPacketIsByteIdenticalAfterNativeGuards() throws {
        let (source, summary, data) = try fixture()
        let brief = try ClinicalBrief.validate(
            evaluationJSON: data, source: source,
            generatedSummary: summary, originalIdentifiers: [], privacyCheck: { _ in true })
        XCTAssertEqual(brief.responseJSON(), data)
        XCTAssertEqual(brief.summary, summary)
        XCTAssertEqual(brief.citations.count, 3)
        XCTAssertFalse(String(decoding: brief.auditJSON(), as: UTF8.self).contains("dehydration"))
        XCTAssertFalse(brief.description.contains("dehydration"))
    }

    func testChangedSourceAndOutputAreRejected() throws {
        let (source, summary, data) = try fixture()
        for (input, output) in [(source + " changed", summary), (source, summary + " fabricated")] {
            XCTAssertThrowsError(
                try ClinicalBrief.validate(
                    evaluationJSON: data,
                    source: input, generatedSummary: output, originalIdentifiers: [], privacyCheck: { _ in true }))
        }
    }

    func testSourceIdentifierAndPrivacyDetectorFailuresAreRejected() throws {
        let (source, summary, data) = try fixture()
        XCTAssertThrowsError(
            try ClinicalBrief.validate(
                evaluationJSON: data, source: source,
                generatedSummary: summary, originalIdentifiers: ["dehydration"], privacyCheck: { _ in true })
        ) {
            XCTAssertEqual($0 as? ClinicalBriefError, .privacy)
        }
        XCTAssertThrowsError(
            try ClinicalBrief.validate(
                evaluationJSON: data, source: source,
                generatedSummary: summary, originalIdentifiers: [], privacyCheck: { _ in false }))
    }

    func testMalformedAndUnsafePacketsAreRejected() throws {
        let (source, summary, data) = try fixture()
        let packet = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        for key in ["citations", "verdicts", "envelope", "digest"] {
            var mutated = packet
            mutated.removeValue(forKey: key)
            XCTAssertThrowsError(
                try ClinicalBrief.validate(
                    evaluationJSON: JSONSerialization.data(withJSONObject: mutated), source: source,
                    generatedSummary: summary, originalIdentifiers: [], privacyCheck: { _ in true }))
        }
    }

    func testBriefTaskIsStructuredAndNeverStreamsUncheckedText() throws {
        let request = OpenMedMapleRequest(task: .brief, document: "Synthetic evidence.")
        let messages = OpenMedMaplePrompt.messages(for: request)
        XCTAssertTrue(messages.last?.content.contains("verbatim") == true)
        let result = try OpenMedMapleOutputParser.parse(
            "{\"answer\":\"Synthetic evidence.\"}",
            task: .brief, sourceDocument: request.document)
        XCTAssertEqual(result.answer, "Synthetic evidence.")
    }

    private func boundFixture() throws -> (String, ClinicalBriefGeneration, [ClinicalBriefGenerationEvidence], Data) {
        var root = URL(fileURLWithPath: #filePath)
        for _ in 0..<5 { root.deleteLastPathComponent() }
        let data = try Data(contentsOf: root.appending(path: "tests/fixtures/clinical/brief_parity/bound.json"))
        let row = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let decoder = JSONDecoder()
        return (
            try XCTUnwrap(row["source"] as? String),
            try decoder.decode(ClinicalBriefGeneration.self, from: JSONSerialization.data(withJSONObject: XCTUnwrap(row["generation"]))),
            try decoder.decode([ClinicalBriefGenerationEvidence].self, from: JSONSerialization.data(withJSONObject: XCTUnwrap(row["reviewed_evidence"]))),
            Data(try XCTUnwrap(row["evaluation_json"] as? String).utf8)
        )
    }

    func testSharedParaphraseBindingsRetainNativeGuards() throws {
        let (source, generation, evidence, packet) = try boundFixture()
        let summary = generation.claims.map(\.text).joined(separator: " ")
        XCTAssertFalse(String(describing: generation).contains("Dehydration"))
        XCTAssertFalse(String(reflecting: generation).contains("Dehydration"))
        XCTAssertFalse(String(reflecting: generation.claims[0]).contains("Dehydration"))
        let brief = try ClinicalBrief.validate(
            evaluationJSON: packet, source: source, generatedSummary: summary,
            originalIdentifiers: [], boundGeneration: generation, reviewedEvidence: evidence,
            privacyCheck: { _ in true })
        XCTAssertEqual(brief.responseJSON(), packet)
        XCTAssertEqual(brief.citations.count, 3)
        XCTAssertFalse(brief.description.contains("dehydration"))
        XCTAssertFalse(String(decoding: brief.auditJSON(), as: UTF8.self).contains("Dehydration"))
        XCTAssertThrowsError(
            try ClinicalBrief.validate(
                evaluationJSON: packet, source: source, generatedSummary: summary,
                originalIdentifiers: [], privacyCheck: { _ in true }))
        XCTAssertThrowsError(
            try ClinicalBrief.validate(
                evaluationJSON: packet, source: source, generatedSummary: summary,
                originalIdentifiers: ["Dehydration"], boundGeneration: generation, reviewedEvidence: evidence,
                privacyCheck: { _ in true }))
        XCTAssertThrowsError(
            try ClinicalBrief.validate(
                evaluationJSON: packet, source: source, generatedSummary: summary,
                originalIdentifiers: [], boundGeneration: generation, reviewedEvidence: evidence,
                privacyCheck: { _ in false }))
    }

    func testInventedAmbiguousAndChangedNativeBindingsRefuse() throws {
        let (source, generation, evidence, packet) = try boundFixture()
        let summary = generation.claims.map(\.text).joined(separator: " ")
        let changedClaim = ClinicalBriefGeneratedClaim(text: generation.claims[0].text, referenceIDs: ["invented:ref"])
        let multipleClaim = ClinicalBriefGeneratedClaim(text: generation.claims[0].text, referenceIDs: [evidence[0].referenceID, evidence[1].referenceID])
        let emptyClaim = ClinicalBriefGeneratedClaim(text: generation.claims[0].text, referenceIDs: [])
        for candidate in [
            ClinicalBriefGeneration(claims: generation.claims, schemaVersion: 2),
            ClinicalBriefGeneration(claims: []),
            ClinicalBriefGeneration(claims: [changedClaim] + generation.claims.dropFirst()),
            ClinicalBriefGeneration(claims: [multipleClaim] + generation.claims.dropFirst()),
            ClinicalBriefGeneration(claims: [emptyClaim] + generation.claims.dropFirst()),
        ] {
            XCTAssertThrowsError(
                try ClinicalBrief.validate(
                    evaluationJSON: packet, source: source, generatedSummary: summary,
                    originalIdentifiers: [], boundGeneration: candidate, reviewedEvidence: evidence,
                    privacyCheck: { _ in true }))
        }
        for references in [
            [],
            evidence + [evidence[0]],
            [ClinicalBriefGenerationEvidence(referenceID: evidence[0].referenceID, start: 1, end: evidence[0].end)] + evidence.dropFirst(),
            evidence + [ClinicalBriefGenerationEvidence(referenceID: "synthetic:overlap", start: 0, end: 2)],
        ] {
            XCTAssertThrowsError(
                try ClinicalBrief.validate(
                    evaluationJSON: packet, source: source, generatedSummary: summary,
                    originalIdentifiers: [], boundGeneration: generation, reviewedEvidence: references,
                    privacyCheck: { _ in true }))
        }
    }
}
