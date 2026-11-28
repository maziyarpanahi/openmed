import AppIntents
import Foundation
import OpenMedIntents
import XCTest

@testable import OpenMedKit

/// Fails and counts every request, so tests can prove the summary stays offline.
private final class SummaryNoNetworkURLProtocol: URLProtocol {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var count = 0

    static var requestCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return count
    }

    static func reset() {
        lock.lock()
        defer { lock.unlock() }
        count = 0
    }

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() {
        Self.lock.lock()
        Self.count += 1
        Self.lock.unlock()
        client?.urlProtocol(self, didFailWithError: URLError(.notConnectedToInternet))
    }
    override func stopLoading() {}
}

final class PIISummaryIntentTests: XCTestCase {
    private static let firstName = "OMKSUMMARYALPHA7Q4Z"
    private static let secondName = "OMKSUMMARYBRAVO7Q4Z"
    private static let phone = "OMKSUMMARYPHONE7Q4Z"
    private static let sentinels = [firstName, secondName, phone]
    private static let input = "\(firstName) and \(secondName) called from \(phone)."

    private var savedRedactor: OpenMedIntentRedactor!
    private var scratch: URL!

    override func setUpWithError() throws {
        savedRedactor = OpenMedIntentsConfiguration.redactor
        scratch = FileManager.default.temporaryDirectory.appending(
            path: "openmed-summary-\(UUID().uuidString)", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
        SummaryNoNetworkURLProtocol.reset()
        URLProtocol.registerClass(SummaryNoNetworkURLProtocol.self)
    }

    override func tearDownWithError() throws {
        URLProtocol.unregisterClass(SummaryNoNetworkURLProtocol.self)
        OpenMedIntentsConfiguration.redactor = savedRedactor
        try? FileManager.default.removeItem(at: scratch)
    }

    func testSummaryReturnsCountsAndRedactedTextOnly() async throws {
        OpenMedIntentsConfiguration.redactor = Self.fakeRedactor()

        let result = try await PIISummaryIntent(text: Self.input).perform()

        let summary = try Self.summary(of: result)
        XCTAssertEqual(summary.redactedText, "[REDACTED] and [REDACTED] called from [REDACTED].")
        XCTAssertEqual(summary.identifierCount, 3)
        XCTAssertEqual(summary.labelCounts, ["first_name: 2", "phone_number: 1"])

        let renderings = [
            String(describing: result), String(reflecting: result), Self.dump(result),
            String(describing: summary), Self.dump(summary),
            String(describing: summary.displayRepresentation.title),
            String(describing: summary.displayRepresentation.subtitle),
        ]
        for rendering in renderings {
            for sentinel in Self.sentinels {
                XCTAssertFalse(rendering.contains(sentinel), "a detected value leaked")
            }
            for spanField in ["confidence", "replacement", "start:", "end:", "actions"] {
                XCTAssertFalse(rendering.contains(spanField), "\(spanField) leaked")
            }
        }
    }

    func testCleanTextSummarizesToZero() async throws {
        OpenMedIntentsConfiguration.redactor = OpenMedIntentRedactor { text, policy in
            PolicyDeidentificationResult(redactedText: text, policyName: policy.name, actions: [])
        }

        let summary = try Self.summary(of: try await PIISummaryIntent(text: "No identifiers here.").perform())

        XCTAssertEqual(summary.redactedText, "No identifiers here.")
        XCTAssertEqual(summary.identifierCount, 0)
        XCTAssertEqual(summary.labelCounts, [])
    }

    func testNoCachedModelFailsWithoutAnyRequest() async throws {
        OpenMedIntentsConfiguration.redactor = .cachedMLXModel(
            repoID: "OpenMed/Synthetic-Uncached-mlx", cacheDirectory: scratch)

        do {
            _ = try await PIISummaryIntent(text: Self.input).perform()
            XCTFail("an uncached model must fail")
        } catch {
            XCTAssertEqual(error as? OpenMedIntentError, .noCachedModel)
            XCTAssertFalse(error.localizedDescription.isEmpty)
            for sentinel in Self.sentinels {
                XCTAssertFalse(String(reflecting: error).contains(sentinel))
            }
        }
        XCTAssertEqual(SummaryNoNetworkURLProtocol.requestCount, 0)
    }

    func testSummaryIntentNeverOpensTheApp() {
        XCTAssertFalse(PIISummaryIntent.openAppWhenRun)
        XCTAssertEqual(String(localized: PIISummaryIntent.title), "Summarize PII with OpenMed")
    }

    // MARK: - Helpers

    private static func fakeRedactor() -> OpenMedIntentRedactor {
        OpenMedIntentRedactor { text, policy in
            var redacted = text
            var actions: [DeidentifiedSpanAction] = []
            for (value, label) in [(firstName, "first_name"), (secondName, "first_name"), (phone, "phone_number")] {
                let start = text.unicodeScalars.distance(
                    from: text.unicodeScalars.startIndex,
                    to: text.range(of: value)!.lowerBound.samePosition(in: text.unicodeScalars)!)
                actions.append(
                    DeidentifiedSpanAction(
                        label: label, canonicalLabel: label, action: .redact, start: start,
                        end: start + value.unicodeScalars.count, confidence: 0.98, replacement: "[REDACTED]"))
                redacted = redacted.replacingOccurrences(of: value, with: "[REDACTED]")
            }
            return PolicyDeidentificationResult(redactedText: redacted, policyName: policy.name, actions: actions)
        }
    }

    private static func summary(of result: some IntentResult) throws -> OpenMedPIISummary {
        let container = try XCTUnwrap(result as? IntentResultContainer<OpenMedPIISummary, Never, Never, Never>)
        return try XCTUnwrap(container.value)
    }

    private static func dump(_ value: Any) -> String {
        var output = ""
        Swift.dump(value, to: &output)
        return output
    }
}
