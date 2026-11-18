import CryptoKit
import Foundation
import XCTest

@testable import OpenMedKit

/// Runs the repository-root kit-parity golden in
/// `tests/fixtures/parity/kit_parity_vectors.json` (digest-pinned by
/// `fixtures/parity/span_parity_manifest.json`) through the Swift kit-parity
/// profile, the `PolicyDeidentificationResult` projection, and, where MLX is
/// available, the policy de-identification engine.
///
/// Failure messages carry case identifiers, field paths, offsets, labels, and
/// actions only.
final class KitParityVectorTests: XCTestCase {
    func testVectorsDigestAndProfileMatchSwiftConstants() throws {
        let manifest = try Self.loadManifest()
        let bytes = try Self.repositoryBytes(Self.vectorsPath)
        let fixture = try Self.loadVectors(bytes, manifest: manifest)

        XCTAssertEqual(fixture.profile.actions, KitParity.actions)
        XCTAssertEqual(fixture.profile.canonicalLabels, KitParity.canonicalLabels)
        XCTAssertEqual(Set(fixture.profile.offsetContractVersions), KitParity.offsetContractVersions)
        XCTAssertEqual(
            Set(fixture.profile.decoderContractVersions),
            KitParity.decoderContractVersions
        )
        XCTAssertGreaterThanOrEqual(fixture.vectors.count, 13)
        XCTAssertTrue(
            manifest.kitParity.consumers.contains {
                $0.surface == "swift" && $0.test.hasSuffix("KitParityVectorTests.swift")
                    && $0.gate.contains("KitParityVectorTests")
            }
        )

        var tampered = bytes
        tampered[tampered.count - 2] = 0x20
        XCTAssertThrowsError(try Self.loadVectors(tampered, manifest: manifest)) {
            XCTAssertEqual($0 as? FixtureError, .digestMismatch)
        }
        var future = manifest
        future.kitParity.version = 2
        XCTAssertThrowsError(try Self.loadVectors(bytes, manifest: future)) {
            XCTAssertEqual($0 as? FixtureError, .unsupportedVectors)
        }
    }

    func testPolicyVocabularyStaysInsideTheProfile() {
        let labels = Set(KitParity.canonicalLabels)
        for label in KitParity.canonicalLabels {
            XCTAssertTrue(labels.contains(Policy.canonicalLabel(for: label)), label)
        }
        for raw in ["patient_name", "medical_record_number", "phone_number", "unknown_kind"] {
            XCTAssertTrue(labels.contains(Policy.canonicalLabel(for: raw)), raw)
        }
        let actions: [PolicyAction] = [.keep, .redact, .replace, .mask, .remove, .hash]
        XCTAssertEqual(Set(actions.map(\.rawValue)), Set(KitParity.actions))
    }

    func testScalarUTF16AndUTF8ConversionsMatchDerivedOffsets() throws {
        let fixture = try Self.loadValidatedVectors()
        var sawSurrogateShift = false
        var sawMultibyteShift = false
        for vector in fixture.vectors {
            for (index, span) in vector.spans.enumerated() {
                let tag = "case=\(vector.id) span=\(index)"
                for (scalar, utf8, utf16) in [
                    (span.start, span.utf8Start, span.utf16Start),
                    (span.end, span.utf8End, span.utf16End),
                ] {
                    XCTAssertEqual(try KitParity.utf16Offset(in: vector.text, scalarOffset: scalar), utf16, tag)
                    XCTAssertEqual(try KitParity.utf8Offset(in: vector.text, scalarOffset: scalar), utf8, tag)
                    XCTAssertEqual(try KitParity.scalarOffset(in: vector.text, utf16Offset: utf16), scalar, tag)
                    XCTAssertEqual(try KitParity.scalarOffset(in: vector.text, utf8Offset: utf8), scalar, tag)
                    XCTAssertEqual(
                        String.Index(utf16Offset: utf16, in: vector.text),
                        PostProcessing.stringIndex(in: vector.text, scalarOffset: scalar),
                        tag
                    )
                }
                sawSurrogateShift = sawSurrogateShift || span.utf16Start != span.start
                sawMultibyteShift = sawMultibyteShift || span.utf8Start != span.utf16Start
            }
        }
        XCTAssertTrue(sawSurrogateShift, "vectors lack a UTF-16 surrogate shift")
        XCTAssertTrue(sawMultibyteShift, "vectors lack a multibyte UTF-8 shift")

        for record in fixture.offsetRejections {
            XCTAssertThrowsError(try Self.convert(record), record.id) {
                XCTAssertEqual(($0 as? KitParityError)?.code, record.reason, record.id)
            }
        }
    }

    func testPolicyResultsProjectToCanonicalBytes() throws {
        let fixture = try Self.loadValidatedVectors()
        var projected = 0
        for vector in fixture.vectors {
            guard vector.spans.allSatisfy({ $0.confidencePpm != nil }) else {
                // Swift policy actions always carry a score; absent scores are
                // covered by the round-trip test.
                XCTAssertEqual(vector.id, "confidence_encoding")
                continue
            }
            let actions = try vector.spans.reversed().map { span in
                DeidentifiedSpanAction(
                    label: span.label,
                    canonicalLabel: span.canonicalLabel,
                    action: try XCTUnwrap(PolicyAction(rawValue: span.action), vector.id),
                    start: span.start,
                    end: span.end,
                    confidence: Float(try XCTUnwrap(span.confidencePpm)) / 1_000_000,
                    replacement: span.replacement
                )
            }
            let result = PolicyDeidentificationResult(
                redactedText: vector.deidentifiedText,
                policyName: "kit_parity_fixture",
                actions: actions
            )
            let serialized = try KitParity.serialize(result.kitParityResult())
            XCTAssertEqual(
                Self.differences(vector.id, expected: vector.resultJson, actual: serialized),
                []
            )
            XCTAssertEqual(Self.sha256Hex(Data(serialized.utf8)), vector.resultSha256, vector.id)
            projected += 1
        }
        XCTAssertEqual(projected, fixture.vectors.count - 1)
    }

    func testCanonicalResultJSONRoundTripsThroughTheStrictParser() throws {
        let fixture = try Self.loadValidatedVectors()
        for vector in fixture.vectors {
            let data = Data(vector.resultJson.utf8)
            let parsed = try KitParity.parse(data)

            XCTAssertEqual(Array(try KitParity.serialize(parsed).utf8), Array(data), vector.id)
            XCTAssertEqual(
                parsed.spans.map { "\($0.label)/\($0.canonicalLabel)/\($0.action)[\($0.start),\($0.end))" },
                vector.spans.map { "\($0.label)/\($0.canonicalLabel)/\($0.action)[\($0.start),\($0.end))" },
                vector.id
            )
            XCTAssertEqual(parsed.spans.map { $0.confidencePPM != nil }, vector.spans.map { $0.confidencePpm != nil })
            XCTAssertEqual(parsed.spans.map { $0.replacement == nil }, parsed.spans.map { $0.action == "keep" })
            XCTAssertEqual(Array(parsed.deidentifiedText.utf8), Array(vector.deidentifiedText.utf8))
            XCTAssertEqual(
                Self.sha256Hex(Data(vector.deidentifiedText.utf8)),
                vector.deidentifiedSha256,
                vector.id
            )
            XCTAssertNoThrow(try KitParity.verify(parsed, against: vector.text), vector.id)
        }
    }

    func testRejectionsFailWithDocumentedValueFreeReasons() throws {
        let fixture = try Self.loadValidatedVectors()
        let sentinel = fixture.profile.privacySentinel
        for record in fixture.rejections {
            XCTAssertThrowsError(try KitParity.parse(record.input), record.id) { error in
                let failure = error as? KitParityError
                XCTAssertEqual(failure, KitParityError(record.reason, record.fieldPath), record.id)
                XCTAssertEqual(failure?.description, "\(record.reason) at \(record.fieldPath)")
                XCTAssertFalse(String(describing: error).contains(sentinel), record.id)
            }
        }
        for record in fixture.sourceRejections {
            let parsed = try KitParity.parse(record.input)
            XCTAssertThrowsError(try KitParity.verify(parsed, against: record.text), record.id) {
                XCTAssertEqual($0 as? KitParityError, KitParityError(record.reason, record.fieldPath))
            }
        }
    }

    func testCanonicallyEquivalentOutputStillFailsByteComparison() throws {
        let fixture = try Self.loadValidatedVectors()
        let record = try XCTUnwrap(fixture.sourceRejections.first { $0.id == "nfc_normalized_output" })
        let parsed = try KitParity.parse(record.input)
        let expected = try KitParity.applySpans(parsed.spans, to: record.text)

        // Swift `String` equality treats NFC and NFD spellings as equal, so the
        // gate must compare UTF-8 bytes.
        XCTAssertEqual(expected, parsed.deidentifiedText)
        XCTAssertNotEqual(Array(expected.utf8), Array(parsed.deidentifiedText.utf8))
    }

    func testBundleContractsAreGatedBeforeResultsAreParsed() throws {
        let fixture = try Self.loadValidatedVectors()
        let payload = Data(try XCTUnwrap(fixture.vectors.first { $0.id == "emoji_sequences" }).resultJson.utf8)
        for record in fixture.compatibility {
            if record.outcome == "accepted" {
                XCTAssertEqual(
                    try KitParity.accept(
                        payload,
                        offsetContractVersion: record.offsetContractVersion,
                        decoderContractVersion: record.decoderContractVersion
                    ),
                    try KitParity.parse(payload)
                )
                continue
            }
            for candidate in [payload, Data("not json".utf8), Data(fixture.rejections[0].input.utf8)] {
                XCTAssertThrowsError(
                    try KitParity.accept(
                        candidate,
                        offsetContractVersion: record.offsetContractVersion,
                        decoderContractVersion: record.decoderContractVersion
                    ),
                    record.id
                ) {
                    XCTAssertEqual(($0 as? KitParityError)?.code, record.outcome, record.id)
                }
            }
        }
    }

    func testAlteredOffsetLabelActionOrderOrBytesFailTheGate() throws {
        let fixture = try Self.loadValidatedVectors()
        let vector = try XCTUnwrap(fixture.vectors.first { $0.id == "emoji_sequences" })
        let parsed = try KitParity.parse(vector.resultJson)
        let expected = vector.resultJson

        var shifted = parsed
        shifted.spans[0].end += 1
        XCTAssertEqual(
            Self.differences(vector.id, expected: expected, actual: try KitParity.serialize(shifted)),
            ["case=emoji_sequences path=$.spans[0].end expected=\(parsed.spans[0].end) actual=\(parsed.spans[0].end + 1)"]
        )

        var relabeled = parsed
        relabeled.spans[0].canonicalLabel = "OTHER"
        XCTAssertEqual(
            Self.differences(vector.id, expected: expected, actual: try KitParity.serialize(relabeled)),
            ["case=emoji_sequences path=$.spans[0].canonical_label expected=PERSON actual=OTHER"]
        )

        var reaction = parsed
        reaction.spans[1].action = "redact"
        XCTAssertEqual(
            Self.differences(vector.id, expected: expected, actual: try KitParity.serialize(reaction)),
            ["case=emoji_sequences path=$.spans[1].action expected=mask actual=redact"]
        )

        var reordered = parsed
        reordered.spans.swapAt(0, 1)
        XCTAssertThrowsError(try KitParity.serialize(reordered)) {
            XCTAssertEqual($0 as? KitParityError, KitParityError("span_order_invalid", "$.spans[1]"))
        }

        var changedBytes = parsed
        changedBytes.deidentifiedText += " "
        let report = Self.differences(vector.id, expected: expected, actual: try KitParity.serialize(changedBytes))
        XCTAssertTrue(report[0].hasPrefix("case=emoji_sequences path=$.deidentified_text first_differing_utf8_byte="))
        XCTAssertFalse(report.joined().contains("Bo Reyes"))
        XCTAssertThrowsError(try KitParity.verify(changedBytes, against: vector.text)) {
            XCTAssertEqual(($0 as? KitParityError)?.code, "deidentified_text_mismatch")
        }

        // UTF-16 coordinates passed off as scalar offsets must not verify.
        var utf16Coordinates = parsed
        for index in utf16Coordinates.spans.indices {
            utf16Coordinates.spans[index].start = vector.spans[index].utf16Start
            utf16Coordinates.spans[index].end = vector.spans[index].utf16End
        }
        XCTAssertThrowsError(try KitParity.verify(utf16Coordinates, against: vector.text))
    }

    #if canImport(MLX) && canImport(Tokenizers) && !os(watchOS) && !os(visionOS)
        func testPolicyEngineDeidentificationMatchesVectors() throws {
            let fixture = try Self.loadValidatedVectors()
            var ran = 0
            for vector in fixture.vectors {
                guard let engine = vector.engine else {
                    continue
                }
                let entities = engine.detections.map { detection in
                    EntityPrediction(
                        label: detection.label,
                        text: PostProcessing.scalarSubstring(
                            vector.text,
                            start: detection.start,
                            end: detection.end
                        ),
                        confidence: Float(detection.confidencePpm) / 1_000_000,
                        start: detection.start,
                        end: detection.end
                    )
                }
                let result = try OpenMed.deidentify(
                    vector.text,
                    entities: entities,
                    policy: try Policy(named: engine.policy)
                )
                let serialized = try KitParity.serialize(result.kitParityResult())

                XCTAssertEqual(
                    Self.differences(vector.id, expected: vector.resultJson, actual: serialized),
                    []
                )
                XCTAssertEqual(
                    Array(result.redactedText.utf8),
                    Array(vector.deidentifiedText.utf8),
                    "case=\(vector.id) de-identified UTF-8 bytes differ"
                )
                ran += 1
            }
            XCTAssertGreaterThanOrEqual(ran, 10)
        }
    #endif

    // MARK: - Fixture loading

    static let vectorsPath = "tests/fixtures/parity/kit_parity_vectors.json"
    static let manifestPath = "fixtures/parity/span_parity_manifest.json"

    private static func loadValidatedVectors() throws -> Vectors {
        try loadVectors(try repositoryBytes(vectorsPath), manifest: try loadManifest())
    }

    private static func loadManifest() throws -> Manifest {
        try decoder().decode(Manifest.self, from: try repositoryBytes(manifestPath))
    }

    private static func loadVectors(_ bytes: Data, manifest: Manifest) throws -> Vectors {
        let entry = manifest.kitParity
        guard entry.schema == "openmed.kit_parity_vectors", entry.version == 1,
            entry.path == vectorsPath, entry.profile == KitParity.profile,
            entry.profileVersion == KitParity.profileVersion
        else {
            throw FixtureError.unsupportedVectors
        }
        guard sha256Hex(bytes) == entry.sha256 else {
            throw FixtureError.digestMismatch
        }
        let vectors = try decoder().decode(Vectors.self, from: bytes)
        guard vectors.schema == entry.schema, vectors.version == entry.version,
            vectors.profile.name == KitParity.profile,
            vectors.profile.version == KitParity.profileVersion
        else {
            throw FixtureError.unsupportedVectors
        }
        return vectors
    }

    /// Read a file relative to the repository root.
    private static func repositoryBytes(_ path: String) throws -> Data {
        var url = URL(fileURLWithPath: #filePath)
        for _ in 0..<5 { url.deleteLastPathComponent() }
        return try Data(contentsOf: url.appending(path: path))
    }

    private static func decoder() -> JSONDecoder {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return decoder
    }

    private static func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    private static func convert(_ record: Vectors.OffsetRejection) throws -> Int {
        switch record.unit {
        case "scalar":
            return try KitParity.utf16Offset(in: record.text, scalarOffset: record.offset)
        case "utf8":
            return try KitParity.scalarOffset(in: record.text, utf8Offset: record.offset)
        default:
            return try KitParity.scalarOffset(in: record.text, utf16Offset: record.offset)
        }
    }

    // MARK: - Value-free comparisons

    /// Drift report with case identifiers, field paths, offsets, labels, and
    /// actions only; text-bearing fields report a byte offset or a path.
    static func differences(_ caseID: String, expected: String, actual: String) -> [String] {
        guard Array(expected.utf8) != Array(actual.utf8) else {
            return []
        }
        guard let want = try? KitParity.parse(expected), let got = try? KitParity.parse(actual) else {
            return ["case=\(caseID) path=$ bytes differ"]
        }
        var out: [String] = []
        let left = Array(want.deidentifiedText.utf8)
        let right = Array(got.deidentifiedText.utf8)
        if left != right {
            let offset =
                zip(left, right).enumerated().first { $0.element.0 != $0.element.1 }?.offset
                ?? min(left.count, right.count)
            out.append("case=\(caseID) path=$.deidentified_text first_differing_utf8_byte=\(offset)")
        }
        guard want.spans.count == got.spans.count else {
            return out + ["case=\(caseID) path=$.spans count expected=\(want.spans.count) actual=\(got.spans.count)"]
        }
        for (index, (lhs, rhs)) in zip(want.spans, got.spans).enumerated() {
            let prefix = "case=\(caseID) path=$.spans[\(index)]"
            let fields: [(String, String, String)] = [
                ("action", lhs.action, rhs.action),
                ("canonical_label", lhs.canonicalLabel, rhs.canonicalLabel),
                ("confidence_ppm", "\(lhs.confidencePPM.map(String.init) ?? "absent")", "\(rhs.confidencePPM.map(String.init) ?? "absent")"),
                ("end", "\(lhs.end)", "\(rhs.end)"),
                ("label", lhs.label, rhs.label),
                ("start", "\(lhs.start)", "\(rhs.start)"),
            ]
            for (name, wantValue, gotValue) in fields where Array(wantValue.utf8) != Array(gotValue.utf8) {
                out.append("\(prefix).\(name) expected=\(wantValue) actual=\(gotValue)")
            }
            if lhs.replacement.map({ Array($0.utf8) }) != rhs.replacement.map({ Array($0.utf8) }) {
                out.append("\(prefix).replacement")
            }
        }
        return out.isEmpty ? ["case=\(caseID) path=$ bytes differ"] : out
    }
}

// MARK: - Fixture model

private enum FixtureError: Error, Equatable {
    case unsupportedVectors
    case digestMismatch
}

private struct Manifest: Decodable {
    struct Consumer: Decodable {
        let surface: String
        let test: String
        let workflow: String
        let gate: String
    }

    struct KitParityEntry: Decodable {
        let path: String
        let schema: String
        var version: Int
        let profile: String
        let profileVersion: Int
        let sha256: String
        let consumers: [Consumer]
    }

    var kitParity: KitParityEntry
}

private struct Vectors: Decodable {
    struct Profile: Decodable {
        let name: String
        let version: Int
        let actions: [String]
        let canonicalLabels: [String]
        let offsetContractVersions: [Int]
        let decoderContractVersions: [Int]
        let privacySentinel: String
    }

    struct Span: Decodable {
        let label: String
        let canonicalLabel: String
        let action: String
        let start: Int
        let end: Int
        let confidencePpm: Int?
        let replacement: String?
        let utf8Start: Int
        let utf8End: Int
        let utf16Start: Int
        let utf16End: Int
    }

    struct Detection: Decodable {
        let label: String
        let start: Int
        let end: Int
        let confidencePpm: Int
    }

    struct Engine: Decodable {
        let policy: String
        let detections: [Detection]
    }

    struct Vector: Decodable {
        let id: String
        let categories: [String]
        let text: String
        let engine: Engine?
        let spans: [Span]
        let deidentifiedText: String
        let deidentifiedSha256: String
        let resultJson: String
        let resultSha256: String
    }

    struct Rejection: Decodable {
        let id: String
        let input: String
        let reason: String
        let fieldPath: String
    }

    struct SourceRejection: Decodable {
        let id: String
        let text: String
        let input: String
        let reason: String
        let fieldPath: String
    }

    struct OffsetRejection: Decodable {
        let id: String
        let text: String
        let unit: String
        let offset: Int
        let reason: String
    }

    struct Compatibility: Decodable {
        let id: String
        let offsetContractVersion: Int
        let decoderContractVersion: Int
        let outcome: String
    }

    let schema: String
    let version: Int
    let profile: Profile
    let vectors: [Vector]
    let rejections: [Rejection]
    let sourceRejections: [SourceRejection]
    let offsetRejections: [OffsetRejection]
    let compatibility: [Compatibility]
}
