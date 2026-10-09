import XCTest

@testable import OpenMedKit

final class GeneratedTextGuardTests: XCTestCase {
    private let asset = String(repeating: "a", count: 64)

    func testIdentifierVariantsAndNegativeControls() {
        let guardInstance = GeneratedTextGuard(detector: { _ in [] }, detectorVersion: "synthetic-v1")
        let cases: [(String, String, GeneratedPHILabel, Int)] = [
            ("Mira Vale", "M I R A   V A L E", .name, 8),
            ("ZX1234", "z x 1 2 3 4", .mrn, 6),
            ("2031-04-19", "2031 - 04 - 19", .date, 10),
            ("李明", "李 明", .name, 2),
        ]
        for (value, output, label, length) in cases {
            let evidence = GeneratedIdentifierDigest(
                assetDigest: asset, label: label,
                digest: GeneratedTextGuard.identifierDigest(value), normalizedLength: length
            )
            let result = guardInstance.release("Label: \(output).", assetDigest: asset, identifiers: [evidence])
            XCTAssertEqual(result.text, "Label: [REDACTED].")
            XCTAssertEqual(result.envelope.guardReport.decision, "redact")
            let report = String(decoding: result.envelope.guardReport.toJSON(), as: UTF8.self)
            XCTAssertFalse(report.contains(value))
            XCTAssertFalse(String(describing: result).contains(output))
            let clean = guardInstance.release("Review required.", assetDigest: asset, identifiers: [evidence])
            XCTAssertEqual(clean.text, "Review required.")
            XCTAssertEqual(clean.envelope, guardInstance.release("Review required.", assetDigest: asset, identifiers: [evidence]).envelope)
        }
    }

    func testPoliciesFailureAndUnresolvedFindings() {
        for action in [GeneratedTextGuard.Action.withhold, .abstain] {
            let result = GeneratedTextGuard(detector: { _ in [GeneratedPHISpan(start: 0, end: 9, label: .name)] }, action: action)
                .release("Mira Vale", assetDigest: asset)
            XCTAssertNil(result.text)
            XCTAssertEqual(result.envelope.abstentionCode, "phi_uncertainty")
        }
        struct SyntheticFailure: Error {}
        let broken = GeneratedTextGuard(detector: { _ in throw SyntheticFailure() })
        XCTAssertNil(broken.release("Mira Vale", assetDigest: asset).text)
        XCTAssertNil(GeneratedTextGuard(detector: nil).release("Mira Vale", assetDigest: asset).text)
        for span in [
            GeneratedPHISpan(start: 0, end: 9, label: .name, resolved: false),
            GeneratedPHISpan(start: -1, end: 9, label: .name),
            GeneratedPHISpan(start: 0, end: 99, label: .name),
        ] {
            XCTAssertNil(GeneratedTextGuard(detector: { _ in [span] }).release("Mira Vale", assetDigest: asset).text)
        }
    }

    func testTimeoutWithholdsAndBoundsOutstandingWork() {
        let finish = DispatchSemaphore(value: 0)
        let done = DispatchSemaphore(value: 0)
        let guardInstance = GeneratedTextGuard(
            detector: { _ in
                defer { done.signal() }
                _ = finish.wait(timeout: .now() + 2)
                return []
            }, timeout: 0.01)
        defer {
            finish.signal()
            _ = done.wait(timeout: .now() + 2)
        }
        XCTAssertNil(guardInstance.release("Mira Vale", assetDigest: asset).text)
        XCTAssertNil(guardInstance.release("Mira Vale", assetDigest: asset).text)
    }

    func testSpanUnionAndCrossAssetRejection() {
        let guardInstance = GeneratedTextGuard(detector: { _ in
            [
                GeneratedPHISpan(start: 0, end: 4, label: .name),
                GeneratedPHISpan(start: 3, end: 9, label: .identifier),
            ]
        })
        XCTAssertEqual(guardInstance.release("Mira Vale rest", assetDigest: asset).text, "[REDACTED] rest")
        let wrong = GeneratedIdentifierDigest(
            assetDigest: String(repeating: "b", count: 64), label: .name,
            digest: GeneratedTextGuard.identifierDigest("Mira Vale"), normalizedLength: 8)
        XCTAssertNil(guardInstance.release("Mira Vale rest", assetDigest: asset, identifiers: [wrong]).text)
    }
}
