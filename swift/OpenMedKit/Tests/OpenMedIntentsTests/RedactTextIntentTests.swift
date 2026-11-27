import AppIntents
import Foundation
import OpenMedIntents
import XCTest

@testable import OpenMedKit

/// Fails and counts every request, so tests can prove intents stay offline.
private final class IntentNoNetworkURLProtocol: URLProtocol {
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

/// Records what a fake pipeline received, behind a lock.
private final class PipelineProbe: @unchecked Sendable {
    private let lock = NSLock()
    private var inputs: [(text: String, policy: String)] = []

    var calls: [(text: String, policy: String)] {
        lock.lock()
        defer { lock.unlock() }
        return inputs
    }

    func record(_ text: String, _ policy: Policy) {
        lock.lock()
        defer { lock.unlock() }
        inputs.append((text, policy.name))
    }
}

final class RedactTextIntentTests: XCTestCase {
    private static let sentinel = "OMKINTENTSENTINEL7Q4Z"
    private static let input = "Patient \(sentinel) visited the clinic."
    private static let redacted = "Patient [REDACTED] visited the clinic."

    private var savedRedactor: OpenMedIntentRedactor!
    private var scratch: URL!

    override func setUpWithError() throws {
        savedRedactor = OpenMedIntentsConfiguration.redactor
        scratch = FileManager.default.temporaryDirectory.appending(
            path: "openmed-intents-\(UUID().uuidString)", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
        IntentNoNetworkURLProtocol.reset()
        URLProtocol.registerClass(IntentNoNetworkURLProtocol.self)
    }

    override func tearDownWithError() throws {
        URLProtocol.unregisterClass(IntentNoNetworkURLProtocol.self)
        OpenMedIntentsConfiguration.redactor = savedRedactor
        try? FileManager.default.removeItem(at: scratch)
    }

    // MARK: - perform()

    func testPerformReturnsRedactedTextFromTheInjectedPipeline() async throws {
        let probe = PipelineProbe()
        OpenMedIntentsConfiguration.redactor = Self.fakeRedactor(probe: probe)

        let result = try await RedactTextIntent(text: Self.input, policyName: "gdpr_pseudonymization").perform()

        XCTAssertEqual(try Self.value(of: result), Self.redacted)
        XCTAssertEqual(probe.calls.map(\.text), [Self.input])
        XCTAssertEqual(probe.calls.map(\.policy), ["gdpr_pseudonymization"])
    }

    func testDefaultPolicyIsTheBundledDefault() async throws {
        let probe = PipelineProbe()
        OpenMedIntentsConfiguration.redactor = Self.fakeRedactor(probe: probe)

        _ = try await RedactTextIntent(text: Self.input).perform()

        XCTAssertEqual(probe.calls.map(\.policy), [Policy.defaultName])
    }

    func testResultCarriesNoSpansOrOriginalValues() async throws {
        OpenMedIntentsConfiguration.redactor = Self.fakeRedactor(probe: PipelineProbe())

        let result = try await RedactTextIntent(text: Self.input).perform()

        let value = try Self.value(of: result)
        XCTAssertFalse(value.contains(Self.sentinel))
        let renderings = [String(describing: result), String(reflecting: result), Self.mirrorDump(result)]
        for rendering in renderings {
            XCTAssertFalse(rendering.contains(Self.sentinel), "original value leaked")
            for spanField in ["first_name", "canonicalLabel", "confidence", "start:", "end:", "actions"] {
                XCTAssertFalse(rendering.contains(spanField), "\(spanField) leaked")
            }
        }
    }

    // MARK: - Cached-model boundary

    func testNoCachedModelFailsWithTheLocalizedMessageAndNoRequests() async throws {
        // Negative control: the stub must see shared-session traffic for the
        // zero-request assertion to mean anything.
        _ = try? await URLSession.shared.data(from: URL(string: "https://intent-stub.invalid/probe")!)
        XCTAssertEqual(IntentNoNetworkURLProtocol.requestCount, 1)
        IntentNoNetworkURLProtocol.reset()
        OpenMedIntentsConfiguration.redactor = .cachedMLXModel(
            repoID: "OpenMed/Synthetic-Uncached-mlx", cacheDirectory: scratch)

        do {
            _ = try await RedactTextIntent(text: Self.input).perform()
            XCTFail("an uncached model must fail")
        } catch {
            XCTAssertEqual(error as? OpenMedIntentError, .noCachedModel)
            XCTAssertEqual(
                error.localizedDescription,
                "No OpenMed model is downloaded on this device. Open the app and download a model first; shortcuts never download models."
            )
            XCTAssertEqual(
                String(localized: (error as? OpenMedIntentError)?.localizedStringResource ?? "missing"),
                error.localizedDescription
            )
        }

        XCTAssertEqual(IntentNoNetworkURLProtocol.requestCount, 0)
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: scratch.path), [])
    }

    func testDefaultRedactorTargetsTheDocumentedModelAndNeverDownloads() async throws {
        XCTAssertEqual(
            OpenMedIntentsConfiguration.defaultModelRepoID, "OpenMed/OpenMed-PII-ClinicalE5-Small-33M-v1-mlx")
        OpenMedIntentsConfiguration.redactor = .cachedMLXModel(
            repoID: OpenMedIntentsConfiguration.defaultModelRepoID, cacheDirectory: scratch)

        do {
            _ = try await RedactTextIntent(text: Self.input).perform()
            XCTFail("an uncached model must fail")
        } catch {
            XCTAssertEqual(error as? OpenMedIntentError, .noCachedModel)
        }
        XCTAssertEqual(IntentNoNetworkURLProtocol.requestCount, 0)
    }

    // MARK: - Validation and error mapping

    func testInvalidInputAndPolicyFailBeforeThePipelineRuns() async throws {
        let probe = PipelineProbe()
        OpenMedIntentsConfiguration.redactor = Self.fakeRedactor(probe: probe)
        let oversized = String(repeating: "a", count: OpenMedIntentsConfiguration.maximumInputUTF8Bytes + 1)
        let cases: [(RedactTextIntent, OpenMedIntentError)] = [
            (RedactTextIntent(text: ""), .emptyInput),
            (RedactTextIntent(text: oversized), .inputTooLarge),
            (RedactTextIntent(text: Self.input, policyName: "not-a-policy-\(Self.sentinel)"), .unknownPolicy),
        ]

        for (intent, expected) in cases {
            do {
                _ = try await intent.perform()
                XCTFail("expected \(expected)")
            } catch {
                XCTAssertEqual(error as? OpenMedIntentError, expected)
            }
        }
        XCTAssertTrue(probe.calls.isEmpty)
    }

    func testPipelineFailuresSurfaceAsValueFreeErrors() async throws {
        let leaking = NSError(
            domain: "Synthetic", code: 7,
            userInfo: [NSLocalizedDescriptionKey: "failed at /private/\(Self.sentinel)/model"])
        OpenMedIntentsConfiguration.redactor = OpenMedIntentRedactor { _, _ in throw leaking }

        do {
            _ = try await RedactTextIntent(text: Self.input).perform()
            XCTFail("a failing pipeline must fail the intent")
        } catch {
            XCTAssertEqual(error as? OpenMedIntentError, .redactionFailed)
            Self.assertValueFree(error)
        }

        OpenMedIntentsConfiguration.redactor = OpenMedIntentRedactor { _, _ in throw CancellationError() }
        do {
            _ = try await RedactTextIntent(text: Self.input).perform()
            XCTFail("cancellation must propagate")
        } catch {
            XCTAssertTrue(error is CancellationError)
        }
    }

    func testEveryErrorHasAValueFreeLocalizedMessage() {
        XCTAssertEqual(Set(OpenMedIntentError.allCases.map(\.reason)).count, OpenMedIntentError.allCases.count)
        for error in OpenMedIntentError.allCases {
            let message = error.localizedDescription
            XCTAssertFalse(message.isEmpty, error.reason)
            XCTAssertFalse(message.hasPrefix("intent.error."), "\(error.reason) has no localized value")
            Self.assertValueFree(error)
        }
    }

    // MARK: - Intent surface

    func testIntentNeverOpensTheAppAndShipsInThePackage() {
        XCTAssertFalse(RedactTextIntent.openAppWhenRun)
        XCTAssertEqual(String(localized: RedactTextIntent.title), "Redact Text with OpenMed")
        let package: any AppIntentsPackage.Type = OpenMedIntentsPackage.self
        XCTAssertTrue(package == OpenMedIntentsPackage.self)
    }

    func testIntentSourcesNeverLogInputOrReachTheNetwork() throws {
        let sources = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appending(path: "Sources/OpenMedIntents", directoryHint: .isDirectory)
        let files = try FileManager.default.contentsOfDirectory(at: sources, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "swift" }
        XCTAssertFalse(files.isEmpty)
        let forbidden = [
            "print(", "debugPrint(", "dump(", "NSLog", "os_log", "Logger(", "OSLog", "URLSession",
            "downloadMLXModel", "startMLXModelDownload", "allowNetworkAccess: true", "import Network",
        ]
        for file in files {
            let source = try String(contentsOf: file, encoding: .utf8)
            for token in forbidden {
                XCTAssertFalse(source.contains(token), "\(file.lastPathComponent) contains \(token)")
            }
        }
    }

    // MARK: - Helpers

    private static func fakeRedactor(probe: PipelineProbe) -> OpenMedIntentRedactor {
        OpenMedIntentRedactor { text, policy in
            probe.record(text, policy)
            let start = text.unicodeScalars.distance(
                from: text.unicodeScalars.startIndex,
                to: text.range(of: sentinel)!.lowerBound.samePosition(in: text.unicodeScalars)!)
            return PolicyDeidentificationResult(
                redactedText: text.replacingOccurrences(of: sentinel, with: "[REDACTED]"),
                policyName: policy.name,
                actions: [
                    DeidentifiedSpanAction(
                        label: "first_name", canonicalLabel: "first_name", action: .redact,
                        start: start, end: start + sentinel.unicodeScalars.count, confidence: 0.99,
                        replacement: "[REDACTED]")
                ]
            )
        }
    }

    private static func value(of result: some IntentResult) throws -> String {
        let container = try XCTUnwrap(result as? IntentResultContainer<String, Never, Never, Never>)
        return try XCTUnwrap(container.value)
    }

    private static func mirrorDump(_ value: Any) -> String {
        var output = ""
        dump(value, to: &output)
        return output
    }

    private static func assertValueFree(_ error: any Error, line: UInt = #line) {
        let nsError = error as NSError
        let renderings = [
            error.localizedDescription, String(describing: error), String(reflecting: error), "\(nsError.userInfo)",
        ]
        for rendering in renderings {
            XCTAssertFalse(rendering.contains(sentinel), "sentinel leaked", line: line)
            XCTAssertFalse(rendering.contains("/private/"), "path leaked", line: line)
        }
    }
}
