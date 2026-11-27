import Foundation
import OpenMedKit
import XCTest

/// Decodes the shared core-contract examples and the Swift-Kotlin span parity
/// fixture with the generated `OpenMedContract*` types from
/// `scripts/codegen/emit_swift_types.py`.
///
/// Every record must decode and encode back to the same JSON. Failure messages
/// carry schema names, indices and keys only, never payload values.
final class ContractTypesTests: XCTestCase {
    private typealias RoundTrip = (Data) throws -> Data

    private static var roundTrips: [String: RoundTrip] {
        [
            "span": roundTrip(OpenMedContractSpan.self),
            "entity": roundTrip(OpenMedContractEntity.self),
            "action": roundTrip(OpenMedContractAction.self),
            "audit": roundTrip(OpenMedContractAuditEvent.self),
            "attestation": roundTrip(OpenMedContractDataResidencyAttestation.self),
            "relation": roundTrip(OpenMedContractRelation.self),
            "code": roundTrip(OpenMedContractCode.self),
            "provenance": roundTrip(OpenMedContractProvenance.self),
        ]
    }

    func testEverySchemaExampleRoundTripsWithoutLoss() throws {
        let examples = try Self.examples()
        XCTAssertEqual(Set(Self.roundTrips.keys), Set(examples.keys))

        for (schema, records) in examples.sorted(by: { $0.key < $1.key }) {
            let roundTrip = try XCTUnwrap(Self.roundTrips[schema], schema)
            XCTAssertFalse(records.isEmpty, schema)
            for (index, record) in records.enumerated() {
                let original = try JSONSerialization.data(withJSONObject: record)
                XCTAssertEqual(
                    try Self.canonicalJSON(try roundTrip(original)),
                    try Self.canonicalJSON(original),
                    "\(schema)[\(index)] changed in a decode and encode round trip"
                )
            }
        }
    }

    func testParitySpanFixtureDecodesAsContractSpans() throws {
        let fixture = try Self.repositoryJSON("fixtures/parity/expected_spans.json")
        let taxonomy = try Self.repositoryJSON(
            "swift/OpenMedKit/Sources/OpenMedKit/Resources/policies/portable_policy_taxonomy.json"
        )
        let aliases = try XCTUnwrap(taxonomy["aliases"] as? [String: String])
        let labels = try XCTUnwrap(taxonomy["labels"] as? [String: [String: Any]])
        let cases = try XCTUnwrap(fixture["cases"] as? [[String: Any]])
        var decoded = 0

        for fixtureCase in cases {
            let caseID = try XCTUnwrap(fixtureCase["id"] as? String)
            let spans = try XCTUnwrap(fixtureCase["spans"] as? [[String: Any]], caseID)
            let actions = try XCTUnwrap(fixtureCase["actions"] as? [[String: Any]], caseID)
            for (index, span) in spans.enumerated() {
                let context = "case=\(caseID) span=\(index)"
                let label = try XCTUnwrap(span["label"] as? String, context)
                let start = try XCTUnwrap(span["start"] as? Int, context)
                let end = try XCTUnwrap(span["end"] as? Int, context)
                let canonical = aliases[Self.taxonomyKey(label)] ?? label.uppercased()
                let policy = try XCTUnwrap(
                    labels[canonical]?["policy_label"] as? String,
                    context
                )
                let action = actions.first {
                    $0["label"] as? String == label && $0["start"] as? Int == start
                        && $0["end"] as? Int == end
                }
                var replacement: Any = NSNull()
                if let action {
                    XCTAssertEqual(action["canonical_label"] as? String, canonical, context)
                    if let value = action["replacement"] as? String {
                        replacement = value
                    }
                }
                let record: [String: Any] = [
                    "schema_version": 1,
                    "doc_id": caseID,
                    "start": start,
                    "end": end,
                    "text_hash": "hmac-sha256:" + String(repeating: "0", count: 64),
                    "entity_type": label,
                    "canonical_label": canonical,
                    "policy_label": policy,
                    "regulatory_tags": [String](),
                    "score": NSNull(),
                    "detector": "native_span_parity_fixture",
                    "evidence": [
                        "utf8_start": try XCTUnwrap(span["utf8_start"] as? Int, context),
                        "utf8_end": try XCTUnwrap(span["utf8_end"] as? Int, context),
                        "utf16_start": try XCTUnwrap(span["utf16_start"] as? Int, context),
                        "utf16_end": try XCTUnwrap(span["utf16_end"] as? Int, context),
                    ],
                    "action": action?["action"] as? String ?? "keep",
                    "replacement": replacement,
                    "reversible_id": NSNull(),
                    "section": NSNull(),
                    "metadata": ["policy": fixtureCase["policy"] ?? NSNull()],
                ]
                let data = try JSONSerialization.data(withJSONObject: record)
                let contract = try JSONDecoder().decode(OpenMedContractSpan.self, from: data)

                XCTAssertEqual(contract.docId, caseID, context)
                XCTAssertEqual(contract.start, start, context)
                XCTAssertEqual(contract.end, end, context)
                XCTAssertEqual(contract.entityType, label, context)
                XCTAssertEqual(contract.canonicalLabel.rawValue, canonical, context)
                XCTAssertEqual(contract.policyLabel.rawValue, policy, context)
                XCTAssertEqual(contract.evidence["utf16_end"], .integer(span["utf16_end"] as? Int ?? -1), context)
                XCTAssertNil(contract.score, context)
                XCTAssertEqual(
                    try Self.canonicalJSON(try JSONEncoder().encode(contract)),
                    try Self.canonicalJSON(data),
                    context
                )
                decoded += 1
            }
        }
        XCTAssertEqual(decoded, cases.reduce(0) { $0 + (($1["spans"] as? [Any])?.count ?? 0) })
        XCTAssertGreaterThan(decoded, 0)
    }

    func testMissingRequiredKeyFailsEvenWhenItsValueMayBeNull() throws {
        var record = try Self.firstExample("span")
        record.removeValue(forKey: "score")

        XCTAssertThrowsError(try Self.decodeSpan(record)) { error in
            guard case DecodingError.keyNotFound(let key, _) = error else {
                return XCTFail("expected keyNotFound, got \(type(of: error))")
            }
            XCTAssertEqual(key.stringValue, "score")
        }
    }

    func testUndeclaredPropertyFailsWithoutNamingItOrItsValue() throws {
        var record = try Self.firstExample("span")
        record["synthetic_undeclared"] = "SYNTHETIC-SECRET-VALUE"

        XCTAssertThrowsError(try Self.decodeSpan(record)) { error in
            guard case DecodingError.dataCorrupted = error else {
                return XCTFail("expected dataCorrupted, got \(type(of: error))")
            }
            let message = String(describing: error)
            XCTAssertTrue(message.contains("OpenMedContractSpan has an undeclared property"))
            XCTAssertFalse(message.contains("synthetic_undeclared"))
            XCTAssertFalse(message.contains("SYNTHETIC-SECRET-VALUE"))
        }
    }

    func testUnknownEnumValueAndWrongSchemaVersionFailValueFree() throws {
        var unknownLabel = try Self.firstExample("span")
        unknownLabel["canonical_label"] = "SYNTHETIC_UNKNOWN_LABEL"
        var wrongVersion = try Self.firstExample("span")
        wrongVersion["schema_version"] = 2
        var wrongType = try Self.firstExample("span")
        wrongType["start"] = "SYNTHETIC-NOT-AN-OFFSET"

        for (name, record) in [
            ("label", unknownLabel), ("version", wrongVersion), ("type", wrongType),
        ] {
            XCTAssertThrowsError(try Self.decodeSpan(record), name) { error in
                let message = String(describing: error)
                XCTAssertFalse(message.contains("SYNTHETIC_UNKNOWN_LABEL"), name)
                XCTAssertFalse(message.contains("SYNTHETIC-NOT-AN-OFFSET"), name)
            }
        }
    }

    func testInitializerFillsConstantsAndEncodesExplicitNulls() throws {
        let span = OpenMedContractSpan(
            docId: "synthetic-doc",
            start: 0,
            end: 4,
            textHash: "hmac-sha256:" + String(repeating: "a", count: 64),
            entityType: "first_name",
            canonicalLabel: .firstName,
            policyLabel: .sensitiveAttribute,
            regulatoryTags: [],
            score: nil,
            detector: nil,
            evidence: ["nested": .object(["flag": .bool(true), "count": .integer(2)])],
            action: .formatPreserve,
            replacement: nil,
            reversibleId: nil,
            section: nil,
            metadata: [:]
        )
        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: try JSONEncoder().encode(span)) as? [String: Any]
        )

        XCTAssertEqual(span.schemaVersion, 1)
        XCTAssertEqual(object["schema_version"] as? Int, 1)
        XCTAssertEqual(object["action"] as? String, "format_preserve")
        for key in ["score", "detector", "replacement", "reversible_id", "section"] {
            XCTAssertTrue(object[key] is NSNull, key)
        }
        XCTAssertEqual(OpenMedContractCanonicalLabel.allCases.first, .accountNumber)
        XCTAssertEqual(
            OpenMedContractPolicyLabel.allCases.map(\.rawValue),
            ["DIRECT_IDENTIFIER", "QUASI_IDENTIFIER", "SENSITIVE_ATTRIBUTE", "CLINICAL_CONCEPT"]
        )
    }

    func testJSONValueKeepsIntegersNumbersAndBooleansDistinct() throws {
        let data = Data(#"[1, 1.5, true, null, "x", {"k": [false]}]"#.utf8)
        let value = try JSONDecoder().decode(OpenMedContractJSONValue.self, from: data)

        XCTAssertEqual(
            value,
            .array([
                .integer(1), .number(1.5), .bool(true), .null, .string("x"),
                .object(["k": .array([.bool(false)])]),
            ])
        )
        XCTAssertEqual(
            try Self.canonicalJSON(try JSONEncoder().encode(value)),
            try Self.canonicalJSON(data)
        )
    }

    // MARK: - Helpers

    private static func roundTrip<T: Codable & Equatable>(_ type: T.Type) -> RoundTrip {
        { data in
            let decoded = try JSONDecoder().decode(T.self, from: data)
            let encoded = try JSONEncoder().encode(decoded)
            XCTAssertEqual(try JSONDecoder().decode(T.self, from: encoded), decoded)
            return encoded
        }
    }

    private static func decodeSpan(_ record: [String: Any]) throws -> OpenMedContractSpan {
        try JSONDecoder().decode(
            OpenMedContractSpan.self,
            from: try JSONSerialization.data(withJSONObject: record)
        )
    }

    /// Sorted-key JSON that keeps booleans and numbers distinct.
    private static func canonicalJSON(_ data: Data) throws -> String {
        let object = try JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed])
        let sorted = try JSONSerialization.data(
            withJSONObject: object,
            options: [.sortedKeys, .fragmentsAllowed]
        )
        return String(decoding: sorted, as: UTF8.self)
    }

    private static func taxonomyKey(_ label: String) -> String {
        String(label.lowercased().unicodeScalars.filter { $0.isASCII && CharacterSet.alphanumerics.contains($0) })
    }

    private static func examples() throws -> [String: [[String: Any]]] {
        let fixture = try repositoryJSON("tests/fixtures/contracts/core_contract_examples.json")
        XCTAssertEqual(fixture["schema"] as? String, "openmed.core_contract_examples")
        XCTAssertEqual(fixture["version"] as? Int, 1)
        return try XCTUnwrap(fixture["examples"] as? [String: [[String: Any]]])
    }

    private static func firstExample(_ schema: String) throws -> [String: Any] {
        try XCTUnwrap(try examples()[schema]?.first, schema)
    }

    /// Read a JSON object from a repository-relative path.
    private static func repositoryJSON(_ path: String) throws -> [String: Any] {
        var url = URL(fileURLWithPath: #filePath)
        for _ in 0..<5 { url.deleteLastPathComponent() }
        let data = try Data(contentsOf: url.appending(path: path))
        return try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any], path)
    }
}
