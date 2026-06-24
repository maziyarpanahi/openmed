import XCTest

@testable import OpenMedKit

final class LocalFirstGuardTests: XCTestCase {
    func testEntityDescriptionDoesNotExposeRawSpanText() {
        let entity = EntityPrediction(
            label: "full_name",
            text: "John Doe",
            confidence: 0.95,
            start: 8,
            end: 16
        )

        XCTAssertFalse(entity.description.contains("John"))
        XCTAssertFalse(entity.description.contains("Doe"))
        XCTAssertTrue(entity.description.contains("full_name"))
        XCTAssertTrue(entity.description.contains("span=(8:16)"))
        XCTAssertTrue(entity.description.contains("text_hash=sha256:"))
    }

    func testSafeLogSpanSummaryUsesOffsetsLabelsAndHashes() {
        let entity = EntityPrediction(
            label: "ssn",
            text: "123-45-6789",
            confidence: 0.99,
            start: 18,
            end: 29
        )

        let summary = SafeLog.summarize(entity)

        XCTAssertEqual(summary.label, "ssn")
        XCTAssertEqual(summary.start, 18)
        XCTAssertEqual(summary.end, 29)
        XCTAssertEqual(summary.confidence, 0.99, accuracy: 0.0001)
        XCTAssertEqual(summary.textHash, entity.textHash)
        XCTAssertFalse(summary.textHash.contains("123"))
        XCTAssertFalse(summary.textHash.contains("6789"))
    }

    func testSafeLogEventsDoNotEmitRawSpanText() {
        var renderedEvents: [String] = []

        SafeLog.withSink(
            { level, event, metadata in
                renderedEvents.append(
                    ([level.rawValue, event] + metadata.map { "\($0.key)=\($0.value)" })
                        .joined(separator: " ")
                )
            },
            {
                SafeLog.log(.inferenceStarted(operation: .extractPII))
                SafeLog.log(.inferenceCompleted(operation: .extractPII, entityCount: 2))
                SafeLog.log(
                    .inferenceFailed(
                        operation: .analyzeText,
                        errorType: "TokenizerError"
                    ),
                    level: .error
                )
            }
        )

        let combined = renderedEvents.joined(separator: "\n")
        XCTAssertTrue(combined.contains("openmed.inference.started"))
        XCTAssertTrue(combined.contains("operation=extractPII"))
        XCTAssertTrue(combined.contains("entity_count=2"))
        XCTAssertFalse(combined.contains("John Doe"))
        XCTAssertFalse(combined.contains("123-45-6789"))
    }

    func testSourceLoggingRoutesThroughSafeLog() throws {
        let offenders = try sourceFiles().compactMap { url -> String? in
            guard url.lastPathComponent != "SafeLog.swift" else {
                return nil
            }
            let contents = try String(contentsOf: url, encoding: .utf8)
            let hasDirectLog =
                contents.contains("print(")
                || contents.contains("NSLog(")
                || contents.contains("os_log(")
                || contents.contains("Logger(")
            return hasDirectLog ? url.lastPathComponent : nil
        }

        XCTAssertEqual(offenders, [])
    }

    func testSourceNetworkAccessRoutesThroughExplicitDownloadSeam() throws {
        let offenders = try sourceFiles().compactMap { url -> String? in
            guard url.lastPathComponent != "OpenMedNetworkAccess.swift" else {
                return nil
            }
            let contents = try String(contentsOf: url, encoding: .utf8)
            return contents.contains("URLSession.shared") ? url.lastPathComponent : nil
        }

        XCTAssertEqual(offenders, [])
    }

    func testInferenceBodiesDoNotCallNetworkSeam() throws {
        let source = try String(
            contentsOf: sourceRoot().appending(path: "OpenMedKit.swift"),
            encoding: .utf8
        )

        for body in [
            try functionBody(named: "analyzeText", in: source),
            try functionBody(named: "extractPII", in: source),
            try functionBody(named: "extractPIIChunked", in: source),
        ] {
            XCTAssertFalse(body.contains("OpenMedNetworkAccess"))
            XCTAssertFalse(body.contains("URLSession"))
        }
    }

    private func sourceFiles() throws -> [URL] {
        let root = sourceRoot()
        guard
            let enumerator = FileManager.default.enumerator(
                at: root,
                includingPropertiesForKeys: nil
            )
        else {
            return []
        }

        return enumerator.compactMap { item in
            guard let url = item as? URL, url.pathExtension == "swift" else {
                return nil
            }
            return url
        }
    }

    private func sourceRoot() -> URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appending(path: "Sources/OpenMedKit", directoryHint: .isDirectory)
    }

    private func functionBody(named name: String, in source: String) throws -> String {
        guard
            let signatureRange = source.range(of: "public func \(name)("),
            let bodyStart = source[signatureRange.upperBound...].firstIndex(of: "{")
        else {
            throw GuardTestError.missingFunction(name)
        }

        var depth = 0
        var cursor = bodyStart
        while cursor < source.endIndex {
            let character = source[cursor]
            if character == "{" {
                depth += 1
            } else if character == "}" {
                depth -= 1
                if depth == 0 {
                    return String(source[bodyStart...cursor])
                }
            }
            cursor = source.index(after: cursor)
        }

        throw GuardTestError.missingFunction(name)
    }

    private enum GuardTestError: Error {
        case missingFunction(String)
    }
}
