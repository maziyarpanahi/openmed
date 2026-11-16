import Foundation
import XCTest

@testable import OpenMedKit

/// Replays `tests/fixtures/parity/identifier_validators_v1.json` through
/// ``OpenMedIdentifierValidators``: every valid value must pass and every
/// invalid value must fail, exactly as the named Python reference decides.
final class IdentifierValidatorParityTests: XCTestCase {
    private typealias Validator = @Sendable (String) -> Bool

    /// The Python reference each Swift function ports.
    private static let validators: [String: Validator] = [
        "openmed.core.anonymizer.providers.clinical_ids.validate_luhn": { OpenMedIdentifierValidators.luhn($0) },
        "openmed.core.anonymizer.providers.clinical_ids.validate_ssn": { OpenMedIdentifierValidators.ssn($0) },
        "openmed.core.anonymizer.providers.clinical_ids.validate_npi": { OpenMedIdentifierValidators.npi($0) },
        "openmed.core.anonymizer.providers.clinical_ids.validate_iban": { OpenMedIdentifierValidators.iban($0) },
        "openmed.core.pii_i18n.validate_aadhaar": { OpenMedIdentifierValidators.aadhaar($0) },
        "openmed.core.anonymizer.providers.clinical_ids.validate_uk_nhs_number": { OpenMedIdentifierValidators.ukNHSNumber($0) },
        "openmed.core.pii_i18n.validate_dutch_bsn": { OpenMedIdentifierValidators.dutchBSN($0) },
        "openmed.core.pii_i18n.validate_spanish_dni": { OpenMedIdentifierValidators.spanishDNI($0) },
        "openmed.core.pii_i18n.validate_french_nir": { OpenMedIdentifierValidators.frenchNIR($0) },
        "openmed.core.pii_i18n.validate_chinese_resident_id": { OpenMedIdentifierValidators.chineseResidentID($0) },
        "openmed.core.anonymizer.providers.clinical_ids.validate_canadian_sin": { OpenMedIdentifierValidators.canadianSIN($0) },
    ]

    func testEveryVectorMatchesThePythonReference() throws {
        let fixture = try loadFixture()
        let cases = try XCTUnwrap(fixture["cases"] as? [[String: Any]])
        XCTAssertEqual(Set(cases.compactMap { $0["reference"] as? String }), Set(Self.validators.keys))
        var checked = 0

        for item in cases {
            let name = try XCTUnwrap(item["validator"] as? String)
            let reference = try XCTUnwrap(item["reference"] as? String, name)
            let validate = try XCTUnwrap(Self.validators[reference], "no Swift port of \(reference)")
            let valid = try XCTUnwrap(item["valid"] as? [String], name)
            let invalid = try XCTUnwrap(item["invalid"] as? [String], name)
            XCTAssertFalse(valid.isEmpty, name)
            XCTAssertFalse(invalid.isEmpty, name)
            // Failure messages carry the validator and position, never the value.
            for (index, value) in valid.enumerated() {
                XCTAssertTrue(validate(value), "\(name) valid[\(index)]")
                checked += 1
            }
            for (index, value) in invalid.enumerated() {
                XCTAssertFalse(validate(value), "\(name) invalid[\(index)]")
                checked += 1
            }
        }
        XCTAssertGreaterThan(checked, 0)
    }

    func testSeparatorsAndUnicodeDigitsFollowThePythonReference() {
        // Verdicts recorded from the Python references for the same inputs.
        let table: [(String, String, Bool)] = [
            ("luhn", " 4992-7378 7789 3288 ", true),
            ("luhn", "\u{664}\u{669}\u{669}\u{662}\u{667}\u{663}\u{667}\u{668}\u{667}\u{667}\u{668}\u{669}\u{663}\u{662}\u{668}\u{668}", false),
            ("ssn", "169 54 6850", true),
            ("ssn", "169546850", true),
            ("ssn", "16954685\u{FF10}", false),
            ("npi", "0716-202-185", true),
            ("npi", "\u{FF10}\u{FF17}\u{FF11}\u{FF16}\u{FF12}\u{FF10}\u{FF12}\u{FF11}\u{FF18}\u{FF15}", false),
            ("iban", "de46 yvok ys9r yapb 2jju kq", true),
            ("iban", "DE46-YVOK-YS9R-YAPB-2JJU-KQ", true),
            ("iban", "DE\u{664}\u{666}YVOKYS9RYAPB2JJUKQ", true),
            ("iban", "DE46YVOKYS9RYAPB2JJUKQ\u{A0}", true),
            ("iban", "ZZ46YVOKYS9RYAPB2JJUKQ", false),
            ("aadhaar", "5882 3384 6871", true),
            ("aadhaar", " 588233846871\u{A}", true),
            ("aadhaar", "\u{A6B}\u{A6E}\u{A6E}\u{A68}\u{A69}\u{A69}\u{A6E}\u{A6A}\u{A6C}\u{A6E}\u{A6D}\u{A67}", true),
            ("aadhaar", "\u{665}\u{668}\u{668}\u{662}\u{663}\u{663}\u{668}\u{664}\u{666}\u{668}\u{667}\u{661}", true),
            ("aadhaar", "5882-3384-6871", false),
            ("aadhaar", "\u{FF15}\u{FF18}\u{FF18}\u{FF12}\u{FF13}\u{FF13}\u{FF18}\u{FF14}\u{FF16}\u{FF18}\u{FF17}\u{FF11}", false),
            ("uk_nhs", "999 000 0018", true),
            ("uk_nhs", "999-000-0018", true),
            ("dutch_bsn", "7123.45.681", true),
            ("dutch_bsn", "12345682", false),
            ("dutch_bsn", "71234568\u{FF11}", false),
            ("spanish_dni", "12345678z", true),
            ("spanish_dni", "1234 5678 Z", true),
            ("spanish_dni", "\u{661}\u{662}\u{663}\u{664}\u{665}\u{666}\u{667}\u{668}Z", true),
            ("spanish_dni", "\u{FF11}\u{FF12}\u{FF13}\u{FF14}\u{FF15}\u{FF16}\u{FF17}\u{FF18}Z", true),
            ("french_nir", "1 85 05 75 001 234 75", true),
            ("french_nir", "1.85.05.75.001.234-75", true),
            ("french_nir", "\u{661}\u{668}\u{665}\u{660}\u{665}\u{667}\u{665}\u{660}\u{660}\u{661}\u{662}\u{663}\u{664}\u{667}\u{665}", false),
            ("french_nir", "1\u{668}\u{665}\u{660}\u{665}\u{667}\u{665}\u{660}\u{660}\u{661}\u{662}\u{663}\u{664}\u{667}5", true),
            ("french_nir", "185057500100909", true),
            ("french_nir", "1850575001009+9", true),
            ("french_nir", "185052A12345633", true),
            ("french_nir", "185052a12345633", true),
            ("french_nir", "185052B12345633", false),
            ("french_nir", "2690520A12345 61", false),
            ("chinese_resident_id", "13244520091012315x", true),
            ("chinese_resident_id", "13244520091012315X ", false),
            ("chinese_resident_id", "\u{663}\u{663}\u{666}\u{666}\u{668}\u{660}\u{661}\u{669}\u{664}\u{661}\u{660}\u{665}\u{660}\u{665}\u{668}\u{664}\u{662}\u{663}", false),
            ("canadian_sin", "975 886 706", true),
            ("canadian_sin", "975-886-706", true),
            ("canadian_sin", "975 886-706", false),
            ("canadian_sin", " 975886706 ", true),
            ("canadian_sin", "\u{669}\u{667}\u{665}\u{668}\u{668}\u{666}\u{667}\u{660}\u{666}", false),
        ]
        let functions: [String: Validator] = [
            "luhn": { OpenMedIdentifierValidators.luhn($0) }, "ssn": { OpenMedIdentifierValidators.ssn($0) },
            "npi": { OpenMedIdentifierValidators.npi($0) }, "iban": { OpenMedIdentifierValidators.iban($0) },
            "aadhaar": { OpenMedIdentifierValidators.aadhaar($0) }, "uk_nhs": { OpenMedIdentifierValidators.ukNHSNumber($0) },
            "dutch_bsn": { OpenMedIdentifierValidators.dutchBSN($0) }, "spanish_dni": { OpenMedIdentifierValidators.spanishDNI($0) },
            "french_nir": { OpenMedIdentifierValidators.frenchNIR($0) },
            "chinese_resident_id": { OpenMedIdentifierValidators.chineseResidentID($0) },
            "canadian_sin": { OpenMedIdentifierValidators.canadianSIN($0) },
        ]
        for (index, (name, value, expected)) in table.enumerated() {
            XCTAssertEqual(functions[name]?(value), expected, "\(name) edge case \(index)")
        }
    }

    func testChineseBirthDatesAreComparedWithAnInjectedToday() throws {
        let id = "13244520091012315X"  // Synthetic vector, born 2009-10-12.
        var components = DateComponents()
        components.calendar = Calendar(identifier: .gregorian)
        components.timeZone = TimeZone.current
        (components.year, components.month, components.day, components.hour) = (2009, 10, 12, 12)
        let birthday = try XCTUnwrap(components.date)
        components.day = 11
        let dayBefore = try XCTUnwrap(components.date)

        XCTAssertTrue(OpenMedIdentifierValidators.chineseResidentID(id, today: birthday))
        XCTAssertFalse(OpenMedIdentifierValidators.chineseResidentID(id, today: dayBefore))
        XCTAssertTrue(OpenMedIdentifierValidators.chineseResidentID(id))
        // A real date is required: February 30 never passes.
        XCTAssertFalse(OpenMedIdentifierValidators.chineseResidentID("132445200902301231"))
    }

    func testValidatorsAreTotalOverHostileInput() {
        let inputs = [
            "", " ", "\u{0}", String(repeating: "9", count: 4096), "😀😀😀😀😀😀😀😀😀", "ß" + String(repeating: "1", count: 14),
            "\u{1C}975886706\u{1F}", "DE" + String(repeating: "Z", count: 40),
        ]
        for function in Self.validators.values {
            for input in inputs { _ = function(input) }
        }
        XCTAssertTrue(OpenMedIdentifierValidators.canadianSIN("\u{1C}975886706\u{1F}"))
    }

    private func loadFixture() throws -> [String: Any] {
        var root = URL(fileURLWithPath: #filePath)
        for _ in 0..<5 { root.deleteLastPathComponent() }
        let data = try Data(contentsOf: root.appending(path: "tests/fixtures/parity/identifier_validators_v1.json"))
        let fixture = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(fixture["synthetic"] as? Bool, true)
        XCTAssertEqual(fixture["fixture"] as? String, "identifier_validators_v1")
        return fixture
    }
}
