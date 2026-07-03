import CryptoKit
import Foundation

/// A span record used by CoreML/Python parity smoke fixtures.
public struct CoreMLParitySpan: Codable, Equatable, Sendable {
    public let label: String
    public let start: Int
    public let end: Int
    public let text: String

    public init(label: String, start: Int, end: Int, text: String = "") {
        self.label = label
        self.start = start
        self.end = end
        self.text = text
    }

    public init(entity: EntityPrediction) {
        self.init(
            label: entity.label,
            start: entity.start,
            end: entity.end,
            text: entity.text
        )
    }
}

/// One fixture in a Python-generated CoreML parity smoke corpus.
public struct CoreMLParityFixture: Codable, Equatable, Sendable {
    public let fixtureID: String
    public let textSHA256: String
    public let pythonReferenceSpans: [CoreMLParitySpan]

    public init(
        fixtureID: String,
        textSHA256: String,
        pythonReferenceSpans: [CoreMLParitySpan]
    ) {
        self.fixtureID = fixtureID
        self.textSHA256 = textSHA256
        self.pythonReferenceSpans = pythonReferenceSpans
    }

    private enum CodingKeys: String, CodingKey {
        case fixtureID = "fixture_id"
        case textSHA256 = "text_sha256"
        case pythonReferenceSpans = "python_reference_spans"
    }
}

/// A single Swift-vs-Python parity failure.
public struct CoreMLParityFailure: Codable, Equatable, Sendable {
    public let fixtureID: String
    public let reason: String
    public let expected: [CoreMLParitySpan]
    public let observed: [CoreMLParitySpan]

    public init(
        fixtureID: String,
        reason: String,
        expected: [CoreMLParitySpan],
        observed: [CoreMLParitySpan]
    ) {
        self.fixtureID = fixtureID
        self.reason = reason
        self.expected = expected
        self.observed = observed
    }
}

/// Result emitted by the Swift CoreML parity smoke harness.
public struct CoreMLParitySmokeResult: Codable, Equatable, Sendable {
    public let checkedFixtureIDs: [String]
    public let missingFixtureIDs: [String]
    public let failures: [CoreMLParityFailure]

    public var passed: Bool {
        missingFixtureIDs.isEmpty && failures.isEmpty
    }
}

/// Loads Python-generated CoreML parity fixtures and checks Swift spans.
public enum CoreMLParitySmokeHarness {
    public static func loadCorpus(from url: URL) throws -> [CoreMLParityFixture] {
        let data = try Data(contentsOf: url)
        let corpus = try JSONDecoder().decode(CoreMLParityCorpus.self, from: data)
        return corpus.fixtures
    }

    public static func validate(
        fixtures: [CoreMLParityFixture],
        predictions: [String: [EntityPrediction]]
    ) -> CoreMLParitySmokeResult {
        var checked: [String] = []
        var missing: [String] = []
        var failures: [CoreMLParityFailure] = []

        for fixture in fixtures {
            guard let entities = predictions[fixture.fixtureID] else {
                missing.append(fixture.fixtureID)
                continue
            }
            checked.append(fixture.fixtureID)
            let observed = entities.map(CoreMLParitySpan.init(entity:))
            guard observed == fixture.pythonReferenceSpans else {
                failures.append(
                    CoreMLParityFailure(
                        fixtureID: fixture.fixtureID,
                        reason: "Swift spans differ from Python reference",
                        expected: fixture.pythonReferenceSpans,
                        observed: observed
                    )
                )
                continue
            }
        }

        return CoreMLParitySmokeResult(
            checkedFixtureIDs: checked,
            missingFixtureIDs: missing,
            failures: failures
        )
    }

    public static func validateTextHash(
        _ text: String,
        fixture: CoreMLParityFixture
    ) -> Bool {
        sha256Hex(text) == fixture.textSHA256
    }

    public static func sha256Hex(_ text: String) -> String {
        let digest = SHA256.hash(data: Data(text.utf8))
        return digest.map { String(format: "%02x", $0) }.joined()
    }
}

private struct CoreMLParityCorpus: Codable {
    let fixtures: [CoreMLParityFixture]
}
