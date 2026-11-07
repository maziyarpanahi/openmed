#if canImport(MLX) && canImport(Tokenizers) && !os(watchOS) && !os(visionOS)
    import CryptoKit
    import Foundation
    import XCTest

    @testable import OpenMedKit

    /// Runs the repository-root Swift-Kotlin golden in
    /// `fixtures/parity/expected_spans.json` through the public Swift
    /// extraction and policy de-identification path.
    ///
    /// The Kotlin counterpart is `SpanEquivalenceTest` in
    /// `android/openmedkit/src/test/kotlin/com/openmed/openmedkit/parity/`.
    /// Failure messages carry case identifiers, labels, and offsets only.
    final class SpanEquivalenceParityTests: XCTestCase {
        func testFixtureSchemaAndDigestAreCurrent() throws {
            let manifest = try loadManifest()
            let fixture = try loadFixture()

            XCTAssertEqual(manifest.schema, "openmed.native_span_parity_manifest")
            XCTAssertEqual(manifest.version, 1)
            XCTAssertEqual(manifest.fixtureSchema, fixture.schema)
            XCTAssertEqual(manifest.fixtureVersion, fixture.version)
            XCTAssertEqual(manifest.algorithm, "sha256")
            XCTAssertNoThrow(try Self.validate(manifest: manifest, fixture: fixture))
            XCTAssertEqual(
                Set(manifest.consumers.map(\.surface)),
                ["python", "swift", "kotlin"]
            )
            XCTAssertTrue(
                manifest.consumers.contains {
                    $0.test.hasSuffix("SpanEquivalenceParityTests.swift")
                        && $0.gate.contains("SpanEquivalenceParityTests")
                }
            )
            XCTAssertGreaterThanOrEqual(fixture.cases.count, 4)
        }

        func testPublicExtractionMatchesSharedSpans() throws {
            let fixture = try loadValidatedFixture()
            let openmed = OpenMed(recognizer: SharedFixtureRecognizer())

            for fixtureCase in fixture.cases {
                let text = try resolveText(fixtureCase)
                let expected = fixtureCase.spans.map(ScalarSpan.init)

                let extracted = try openmed.extractPII(text, useSmartMerging: false)
                XCTAssertEqual(
                    Self.spanDifferences(
                        caseID: fixtureCase.id,
                        expected: expected,
                        actual: extracted.map(ScalarSpan.init)
                    ),
                    []
                )
                for (index, entity) in extracted.enumerated() {
                    XCTAssertTrue(
                        entity.text
                            == PostProcessing.scalarSubstring(
                                text,
                                start: entity.start,
                                end: entity.end
                            ),
                        "case=\(fixtureCase.id) span=\(index) text does not match offsets"
                    )
                }

                let chunked = try openmed.extractPIIChunked(text, useSmartMerging: false)
                XCTAssertEqual(
                    Self.spanDifferences(
                        caseID: fixtureCase.id,
                        expected: expected,
                        actual: chunked.map(ScalarSpan.init)
                    ),
                    []
                )
            }
        }

        func testUnicodeScalarUTF8AndUTF16ConversionsMatchFixture() throws {
            let fixture = try loadValidatedFixture()
            let recognizer = SharedFixtureRecognizer()
            var sawSurrogatePairShift = false
            var sawMultibyteUTF8Shift = false

            for fixtureCase in fixture.cases {
                let text = try resolveText(fixtureCase)
                let native = try recognizer.nativeMatches(in: text)
                XCTAssertEqual(
                    native.map { "\($0.label)@\($0.range.location)+\($0.range.length)" },
                    fixtureCase.spans.map {
                        "\($0.label)@\($0.utf16Start)+\($0.utf16End - $0.utf16Start)"
                    },
                    "case=\(fixtureCase.id) native NSRange offsets differ"
                )

                for (spanIndex, span) in fixtureCase.spans.enumerated() {
                    let tag = "case=\(fixtureCase.id) span=\(spanIndex)"
                    for (scalar, utf8, utf16) in [
                        (span.start, span.utf8Start, span.utf16Start),
                        (span.end, span.utf8End, span.utf16End),
                    ] {
                        let stringIndex = try XCTUnwrap(
                            PostProcessing.stringIndex(in: text, scalarOffset: scalar),
                            tag
                        )
                        XCTAssertEqual(stringIndex.utf16Offset(in: text), utf16, tag)
                        XCTAssertEqual(
                            text.utf8.distance(from: text.utf8.startIndex, to: stringIndex),
                            utf8,
                            tag
                        )
                        XCTAssertEqual(
                            PostProcessing.scalarOffset(
                                in: text,
                                at: String.Index(utf16Offset: utf16, in: text)
                            ),
                            scalar,
                            tag
                        )
                        XCTAssertEqual(
                            PostProcessing.scalarOffset(
                                in: text,
                                at: text.utf8.index(text.utf8.startIndex, offsetBy: utf8)
                            ),
                            scalar,
                            tag
                        )
                        XCTAssertTrue(PostProcessing.isGraphemeBoundary(scalar, in: text), tag)
                    }
                    sawSurrogatePairShift =
                        sawSurrogatePairShift || span.utf16Start != span.start
                    sawMultibyteUTF8Shift =
                        sawMultibyteUTF8Shift || span.utf8Start != span.utf16Start
                }
            }

            XCTAssertTrue(sawSurrogatePairShift, "fixture lacks a UTF-16 surrogate shift")
            XCTAssertTrue(sawMultibyteUTF8Shift, "fixture lacks a multibyte UTF-8 shift")
        }

        func testPolicyDeidentificationMatchesSharedActionsAndBytes() throws {
            let fixture = try loadValidatedFixture()
            let openmed = OpenMed(recognizer: SharedFixtureRecognizer())

            for fixtureCase in fixture.cases {
                let text = try resolveText(fixtureCase)
                let result = try openmed.deidentify(
                    text,
                    policy: fixtureCase.policy,
                    useSmartMerging: false
                )

                XCTAssertEqual(result.policyName, fixtureCase.policy, fixtureCase.id)
                XCTAssertEqual(
                    result.actions.map(ActionRecord.init),
                    fixtureCase.actions.map(ActionRecord.init),
                    "case=\(fixtureCase.id) policy actions differ"
                )
                XCTAssertNil(
                    Self.utf8Difference(
                        expected: fixtureCase.expectedDeidentifiedText,
                        actual: result.redactedText
                    ),
                    "case=\(fixtureCase.id) de-identified UTF-8 bytes differ"
                )
            }
        }

        func testChangedExpectedSpanFailsParityGate() throws {
            let fixture = try loadValidatedFixture()
            let fixtureCase = try XCTUnwrap(
                fixture.cases.first { $0.id == "emoji_and_combining_marks" }
            )
            let text = try resolveText(fixtureCase)
            let actual = try OpenMed(recognizer: SharedFixtureRecognizer())
                .extractPII(text, useSmartMerging: false)
                .map(ScalarSpan.init)

            let expected = fixtureCase.spans.map(ScalarSpan.init)
            XCTAssertEqual(
                Self.spanDifferences(caseID: fixtureCase.id, expected: expected, actual: actual),
                []
            )

            var shifted = expected
            shifted[2] = ScalarSpan(
                label: shifted[2].label,
                start: shifted[2].start,
                end: shifted[2].end + 1
            )
            XCTAssertEqual(
                Self.spanDifferences(caseID: fixtureCase.id, expected: shifted, actual: actual),
                [
                    "case=\(fixtureCase.id) span=2 end expected=\(expected[2].end + 1) "
                        + "actual=\(expected[2].end) delta=-1"
                ]
            )

            var relabeled = expected
            relabeled[0] = ScalarSpan(
                label: "first_name",
                start: relabeled[0].start,
                end: relabeled[0].end
            )
            XCTAssertEqual(
                Self.spanDifferences(caseID: fixtureCase.id, expected: relabeled, actual: actual),
                [
                    "case=\(fixtureCase.id) span=0 label expected=first_name "
                        + "actual=\(expected[0].label)"
                ]
            )

            XCTAssertEqual(
                Self.spanDifferences(
                    caseID: fixtureCase.id,
                    expected: Array(expected.dropLast()),
                    actual: actual
                ),
                ["case=\(fixtureCase.id) count expected=\(expected.count - 1) actual=\(actual.count)"]
            )

            let changedBytes = try XCTUnwrap(
                Self.utf8Difference(
                    expected: fixtureCase.expectedDeidentifiedText + " ",
                    actual: fixtureCase.expectedDeidentifiedText
                )
            )
            XCTAssertTrue(changedBytes.hasPrefix("first_differing_utf8_byte="))
        }

        func testChangedDigestOrFixtureBytesFailValidation() throws {
            let manifest = try loadManifest()
            let fixture = try loadFixture()

            var staleDigest = manifest
            staleDigest.files[0].sha256 = String(repeating: "0", count: 64)
            XCTAssertThrowsError(try Self.validate(manifest: staleDigest, fixture: fixture)) {
                XCTAssertEqual(
                    $0 as? FixtureError,
                    .digestMismatch(staleDigest.files[0].path)
                )
            }

            var missingFile = manifest
            missingFile.files.removeAll { $0.path == "synthetic_clinical_note.txt" }
            XCTAssertThrowsError(try Self.validate(manifest: missingFile, fixture: fixture)) {
                XCTAssertEqual(
                    $0 as? FixtureError,
                    .undigestedFile("synthetic_clinical_note.txt")
                )
            }

            // A changed expected span in otherwise well-formed fixture bytes must
            // fail the digest gate before any parity assertion runs.
            let span = try XCTUnwrap(fixture.cases.first?.spans.first)
            let original = try Self.fixtureBytes("expected_spans.json")
            let changed = Data(
                String(decoding: original, as: UTF8.self)
                    .replacingOccurrences(
                        of: "\"end\": \(span.end),",
                        with: "\"end\": \(span.end + 1),"
                    )
                    .utf8
            )
            XCTAssertTrue(changed != original, "negative control did not change the fixture")
            let changedFixture = try decoder().decode(Fixture.self, from: changed)
            XCTAssertThrowsError(
                try Self.validate(manifest: manifest, fixture: changedFixture) { name in
                    name == "expected_spans.json" ? changed : try Self.fixtureBytes(name)
                }
            ) {
                XCTAssertEqual($0 as? FixtureError, .digestMismatch("expected_spans.json"))
            }

            var staleVersion = fixture
            staleVersion.version = 1
            XCTAssertThrowsError(try Self.validate(manifest: manifest, fixture: staleVersion)) {
                XCTAssertEqual($0 as? FixtureError, .unsupportedFixture)
            }
        }

        // MARK: - Fixture loading and validation

        private func loadValidatedFixture() throws -> Fixture {
            let fixture = try loadFixture()
            try Self.validate(manifest: try loadManifest(), fixture: fixture)
            return fixture
        }

        private func loadFixture() throws -> Fixture {
            try decoder().decode(Fixture.self, from: Self.fixtureBytes("expected_spans.json"))
        }

        private func loadManifest() throws -> Manifest {
            try decoder().decode(
                Manifest.self,
                from: Self.fixtureBytes("span_parity_manifest.json")
            )
        }

        private func resolveText(_ fixtureCase: Fixture.Case) throws -> String {
            switch (fixtureCase.text, fixtureCase.textFile) {
            case (let text?, nil):
                return text
            case (nil, let name?):
                let data = try Self.fixtureBytes(name)
                return try XCTUnwrap(String(data: data, encoding: .utf8), fixtureCase.id)
            default:
                throw FixtureError.invalidTextSource(fixtureCase.id)
            }
        }

        private static func validate(
            manifest: Manifest,
            fixture: Fixture,
            bytes: (String) throws -> Data = SpanEquivalenceParityTests.fixtureBytes
        ) throws {
            guard
                manifest.schema == "openmed.native_span_parity_manifest",
                manifest.version == 1,
                manifest.algorithm == "sha256",
                fixture.schema == "openmed.native_span_parity",
                fixture.version == 2,
                fixture.offsetUnit == "unicode_scalar",
                manifest.fixtureSchema == fixture.schema,
                manifest.fixtureVersion == fixture.version
            else {
                throw FixtureError.unsupportedFixture
            }

            var required = Set(["expected_spans.json"])
            for fixtureCase in fixture.cases {
                guard (fixtureCase.text == nil) != (fixtureCase.textFile == nil) else {
                    throw FixtureError.invalidTextSource(fixtureCase.id)
                }
                if let name = fixtureCase.textFile {
                    required.insert(name)
                }
            }
            let digested = Set(manifest.files.map(\.path))
            if let missing = required.subtracting(digested).sorted().first {
                throw FixtureError.undigestedFile(missing)
            }

            for file in manifest.files {
                guard sha256Hex(try bytes(file.path)) == file.sha256 else {
                    throw FixtureError.digestMismatch(file.path)
                }
            }
        }

        private static func sha256Hex(_ data: Data) -> String {
            SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        }

        /// Read a file from the repository-root `fixtures/parity` directory.
        private static func fixtureBytes(_ name: String) throws -> Data {
            var url = URL(fileURLWithPath: #filePath)
            for _ in 0..<5 { url.deleteLastPathComponent() }
            return try Data(contentsOf: url.appending(path: "fixtures/parity").appending(path: name))
        }

        private func decoder() -> JSONDecoder {
            let decoder = JSONDecoder()
            decoder.keyDecodingStrategy = .convertFromSnakeCase
            return decoder
        }

        // MARK: - Value-free comparisons

        static func spanDifferences(
            caseID: String,
            expected: [ScalarSpan],
            actual: [ScalarSpan]
        ) -> [String] {
            guard expected.count == actual.count else {
                return ["case=\(caseID) count expected=\(expected.count) actual=\(actual.count)"]
            }
            var differences: [String] = []
            for (index, (lhs, rhs)) in zip(expected, actual).enumerated() {
                let prefix = "case=\(caseID) span=\(index)"
                if lhs.label != rhs.label {
                    differences.append("\(prefix) label expected=\(lhs.label) actual=\(rhs.label)")
                }
                for (field, want, got) in [
                    ("start", lhs.start, rhs.start),
                    ("end", lhs.end, rhs.end),
                ] where want != got {
                    differences.append(
                        "\(prefix) \(field) expected=\(want) actual=\(got) delta=\(got - want)"
                    )
                }
            }
            return differences
        }

        static func utf8Difference(expected: String, actual: String) -> String? {
            let lhs = Array(expected.utf8)
            let rhs = Array(actual.utf8)
            guard lhs != rhs else {
                return nil
            }
            let offset =
                zip(lhs, rhs).enumerated().first { $0.element.0 != $0.element.1 }?.offset
                ?? min(lhs.count, rhs.count)
            return
                "first_differing_utf8_byte=\(offset) expected_bytes=\(lhs.count) actual_bytes=\(rhs.count)"
        }
    }

    // MARK: - Fixture model

    private enum FixtureError: Error, Equatable {
        case unsupportedFixture
        case invalidTextSource(String)
        case undigestedFile(String)
        case digestMismatch(String)
    }

    private struct Manifest: Decodable {
        struct File: Decodable {
            let path: String
            var sha256: String
        }

        struct Consumer: Decodable {
            let surface: String
            let test: String
            let workflow: String
            let gate: String
        }

        let schema: String
        let version: Int
        let fixtureSchema: String
        let fixtureVersion: Int
        let algorithm: String
        var files: [File]
        let updateCommand: String
        let consumers: [Consumer]
    }

    private struct Fixture: Decodable {
        struct Case: Decodable {
            let id: String
            let policy: String
            let text: String?
            let textFile: String?
            let spans: [Span]
            let actions: [Action]
            let expectedDeidentifiedText: String
        }

        struct Span: Decodable {
            let label: String
            let start: Int
            let end: Int
            let utf8Start: Int
            let utf8End: Int
            let utf16Start: Int
            let utf16End: Int
        }

        struct Action: Decodable {
            let label: String
            let canonicalLabel: String
            let action: String
            let start: Int
            let end: Int
            let replacement: String?
        }

        let schema: String
        var version: Int
        let offsetUnit: String
        let cases: [Case]
    }

    struct ScalarSpan: Equatable {
        let label: String
        let start: Int
        let end: Int
    }

    extension ScalarSpan {
        fileprivate init(_ span: Fixture.Span) {
            self.init(label: span.label, start: span.start, end: span.end)
        }

        init(_ entity: EntityPrediction) {
            self.init(label: entity.label, start: entity.start, end: entity.end)
        }
    }

    /// Policy action projection compared across kits: no confidence, no text.
    private struct ActionRecord: Equatable, CustomStringConvertible {
        let label: String
        let canonicalLabel: String
        let action: String
        let start: Int
        let end: Int
        let replacement: String?

        init(_ action: Fixture.Action) {
            label = action.label
            canonicalLabel = action.canonicalLabel
            self.action = action.action
            start = action.start
            end = action.end
            replacement = action.replacement
        }

        init(_ action: DeidentifiedSpanAction) {
            label = action.label
            canonicalLabel = action.canonicalLabel
            self.action = action.action.rawValue
            start = action.start
            end = action.end
            replacement = action.replacement
        }

        var description: String {
            "\(label)/\(canonicalLabel)/\(action)[\(start),\(end))->\(replacement ?? "keep")"
        }
    }

    /// Deterministic recognizer mirroring the Kotlin `LocalPiiRecognizer`.
    ///
    /// Patterns, capture groups, and ordering match
    /// `android/openmedkit/src/main/kotlin/com/openmed/openmedkit/OpenMedKit.kt`.
    /// `NSRegularExpression` reports native UTF-16 ranges, which are converted to
    /// Unicode scalar offsets exactly once, as the shared contract requires.
    private struct SharedFixtureRecognizer: OpenMedSpanRecognizer {
        struct NativeMatch {
            let label: String
            let range: NSRange
        }

        private static let patterns: [(label: String, pattern: String, group: Int)] = [
            ("patient_name", #"(?m)\bPatient:\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)"#, 1),
            ("date_of_birth", #"(?m)\bDOB:\s+(\d{4}-\d{2}-\d{2})"#, 1),
            ("medical_record_number", #"\bMRN-[0-9]+\b"#, 0),
            ("phone_number", #"\b(?:\+?1[-.\s]?)?\d{3}-\d{3}-\d{4}\b"#, 0),
            ("email", #"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"#, 0),
            ("date", #"(?i)\bon\s+(\d{4}-\d{2}-\d{2})"#, 1),
        ]

        func nativeMatches(in text: String) throws -> [NativeMatch] {
            let fullRange = NSRange(text.startIndex..<text.endIndex, in: text)
            var matches: [NativeMatch] = []
            for entry in Self.patterns {
                let regex = try NSRegularExpression(pattern: entry.pattern)
                for match in regex.matches(in: text, range: fullRange) {
                    let range = match.range(at: entry.group)
                    if range.location != NSNotFound {
                        matches.append(NativeMatch(label: entry.label, range: range))
                    }
                }
            }
            return matches.sorted {
                ($0.range.location, NSMaxRange($0.range))
                    < ($1.range.location, NSMaxRange($1.range))
            }
        }

        func recognize(_ text: String) throws -> [EntityPrediction] {
            try nativeMatches(in: text).compactMap { match in
                guard let range = Range(match.range, in: text) else {
                    return nil
                }
                return EntityPrediction(
                    label: match.label,
                    text: String(text[range]),
                    confidence: 1.0,
                    start: PostProcessing.scalarOffset(in: text, at: range.lowerBound),
                    end: PostProcessing.scalarOffset(in: text, at: range.upperBound)
                )
            }
        }
    }
#endif
