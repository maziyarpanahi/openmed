import XCTest

@testable import OpenMedKit

final class PolicyTests: XCTestCase {

    func testLoadsBundledProfilesAndGDPAlias() throws {
        for name in Policy.bundledProfileNames {
            let policy = try Policy(named: name)

            XCTAssertEqual(policy.name, name)
            XCTAssertEqual(policy.schemaVersion, 1)
            XCTAssertFalse(policy.posture.isEmpty)
            XCTAssertFalse(policy.defaultActionBias.isEmpty)
            XCTAssertFalse(policy.actions.isEmpty)
            XCTAssertNotNil(policy.actions["FIRST_NAME"])
        }

        let gdpr = try Policy(named: "gdpr")
        XCTAssertEqual(gdpr.name, "gdpr_pseudonymization")
        XCTAssertEqual(gdpr.action(for: "email"), .replace)

        let gdprHealth = try Policy(named: "gdpr_health")
        XCTAssertEqual(gdprHealth.name, "gdpr_art9_health")
        XCTAssertEqual(gdprHealth.action(for: "condition"), .mask)

        let australia = try Policy(named: "au_privacy")
        XCTAssertEqual(australia.name, "australia_privacy_act")
        XCTAssertEqual(australia.action(for: "medical_record_number"), .mask)
    }

    func testUnknownProfileThrowsClearError() {
        XCTAssertThrowsError(try Policy(named: "unknown_policy")) { error in
            guard case PolicyError.unknownProfile(let name, let allowed) = error else {
                return XCTFail("Unexpected error: \(error)")
            }
            XCTAssertEqual(name, "unknown_policy")
            XCTAssertTrue(allowed.contains("hipaa_safe_harbor"))
            XCTAssertTrue(allowed.contains("gdpr"))
            XCTAssertTrue(allowed.contains("gdpr_health"))
            XCTAssertTrue(allowed.contains("au_privacy"))
        }
    }

    func testStrictNoLeakMasksClinicalAndUnknownLabels() throws {
        let strict = try Policy(named: "strict_no_leak")

        XCTAssertTrue(strict.strictNoLeak)
        XCTAssertEqual(strict.action(for: "antibiotic"), .mask)
        XCTAssertEqual(strict.action(for: "microorganism"), .mask)
        XCTAssertEqual(strict.action(for: "condition"), .mask)
        XCTAssertEqual(strict.action(for: "not_a_known_label"), .mask)

        let clinical = try Policy(named: "clinical_minimal_redaction")
        XCTAssertEqual(clinical.action(for: "antibiotic"), .keep)
        XCTAssertEqual(clinical.action(for: "condition"), .keep)
    }

    #if canImport(MLX) && canImport(Tokenizers) && !os(watchOS) && !os(visionOS)
        func testPolicyProfileTransformPreservesOriginalOffsets() throws {
            let text = "Name Ada DOB 04/01/2026"
            let policy = try Policy(named: "gdpr")
            let entities = [
                entity(label: "first_name", value: "Ada", in: text),
                entity(label: "date_of_birth", value: "04/01/2026", in: text),
            ]

            let result: PolicyDeidentificationResult = OpenMed.deidentify(
                text,
                entities: entities,
                policy: policy
            )

            XCTAssertEqual(
                result.redactedText,
                "Name [FIRST_NAME_REPLACED] DOB [DATE_OF_BIRTH_REPLACED]"
            )
            XCTAssertEqual(result.policyName, "gdpr_pseudonymization")
            XCTAssertEqual(result.actions.map(\.canonicalLabel), ["FIRST_NAME", "DATE_OF_BIRTH"])
            XCTAssertEqual(result.actions.map(\.action), [.replace, .replace])
            XCTAssertEqual(result.actions[0].start, 5)
            XCTAssertEqual(result.actions[0].end, 8)
            XCTAssertEqual(result.actions[1].start, 13)
            XCTAssertEqual(result.actions[1].end, 23)
        }

        func testSyntheticActionToTextTransformMapping() {
            let text =
                "Name Ada Phone 555-0100 SSN 123-45-6789 ID MRN-123 penicillin"
            let policy = Policy(
                name: "synthetic",
                defaultAction: .keep,
                actions: [
                    "FIRST_NAME": .mask,
                    "PHONE": .replace,
                    "SSN": .hash,
                    "ID_NUM": .remove,
                    "ANTIBIOTIC": .keep,
                ]
            )
            let entities = [
                entity(label: "first_name", value: "Ada", in: text),
                entity(label: "phone_number", value: "555-0100", in: text),
                entity(label: "ssn", value: "123-45-6789", in: text),
                entity(label: "medical_record_number", value: "MRN-123", in: text),
                entity(label: "antibiotic", value: "penicillin", in: text),
            ]

            let result = OpenMed.deidentify(text, entities: entities, policy: policy)
            let ssnReplacement = result.actions[2].replacement ?? ""

            XCTAssertTrue(ssnReplacement.hasPrefix("SSN_"))
            XCTAssertFalse(ssnReplacement.contains("123-45-6789"))
            XCTAssertEqual(
                result.redactedText,
                "Name [FIRST_NAME] Phone [PHONE_REPLACED] SSN \(ssnReplacement) ID  penicillin"
            )
            XCTAssertEqual(
                result.actions.map(\.action),
                [.mask, .replace, .hash, .remove, .keep]
            )
            XCTAssertEqual(
                result.actions.map(\.canonicalLabel),
                ["FIRST_NAME", "PHONE", "SSN", "ID_NUM", "ANTIBIOTIC"]
            )
            XCTAssertNil(result.actions[4].replacement)
        }
    #endif

    private func entity(
        label: String,
        value: String,
        in text: String,
        confidence: Float = 0.99
    ) -> EntityPrediction {
        guard let range = text.range(of: value) else {
            XCTFail("Missing fixture value \(value)")
            return EntityPrediction(label: label, text: value, confidence: 0, start: 0, end: 0)
        }

        let start = text.distance(from: text.startIndex, to: range.lowerBound)
        let end = text.distance(from: text.startIndex, to: range.upperBound)
        return EntityPrediction(
            label: label,
            text: value,
            confidence: confidence,
            start: start,
            end: end
        )
    }
}
