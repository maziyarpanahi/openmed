import Foundation
import OpenMedExtensionSupport
import OpenMedPrivacySweep
import XCTest

@testable import OpenMedKit

/// Drives the share/action extension redaction entry points with sentinel inputs, then
/// sweeps this process's storage as far as its ``HostStorageScope`` allows and its
/// unified log.
final class ExtensionSentinelSweepTests: XCTestCase {
    #if os(iOS) || os(macOS)
        func testExtensionRedactionLeavesNoSentinelInContainerOrProcessLog() async throws {
            let start = Date().addingTimeInterval(-1)
            let person = try PHISentinel.synthetic(identifier: "extension.person")
            let birthDate = try PHISentinel(identifier: "extension.birth-date", value: "DOB 0\(Int.random(in: 1...9))/1\(Int.random(in: 0...9))/19\(Int.random(in: 40...99))")
            let texts = [
                "Name \(person.plantedValue) \(birthDate.plantedValue)",
                "Follow-up for \(person.plantedValue)",
            ]
            let handler = ExtensionRedactionHandler { text, policy in
                OpenMed.deidentify(text, entities: Self.entities(in: text, values: [person, birthDate]), policy: policy)
            }

            for text in texts {
                for policyName in ["gdpr", Policy.defaultName, "strict_no_leak"] {
                    _ = try handler.redact(text, policyName: policyName)
                }
            }
            _ = try handler.redact(texts)
            XCTAssertThrowsError(try handler.redact(texts[0], policyName: "unknown_profile"))
            #if os(iOS) && canImport(UIKit)
                let items = ExtensionItemCodec.extensionItems(for: texts)
                let decoded = try await ExtensionItemCodec.plainText(from: items)
                XCTAssertEqual(decoded, texts)
            #endif

            let scope = HostStorageScope.current
            let report = try PHISentinelSweep.currentProcess(
                sentinels: [person, birthDate],
                logSince: start,
                scope: scope
            ).run()
            let attachment = XCTAttachment(data: try report.jsonData(), uniformTypeIdentifier: "public.json")
            attachment.name = "phi-sentinel-sweep-extension.json"
            attachment.lifetime = .keepAlways
            add(attachment)

            XCTAssertTrue(report.clean, report.description)
            XCTAssertTrue(report.passesGate(for: scope), "\(scope.rawValue): \(report)")
        }

        private static func entities(in text: String, values: [PHISentinel]) -> [EntityPrediction] {
            values.compactMap { sentinel in
                guard let range = text.range(of: sentinel.plantedValue) else { return nil }
                let scalars = text.unicodeScalars
                return EntityPrediction(
                    label: sentinel.identifier.hasSuffix("person") ? "first_name" : "date_of_birth",
                    text: sentinel.plantedValue,
                    confidence: 0.99,
                    start: scalars.distance(from: scalars.startIndex, to: range.lowerBound),
                    end: scalars.distance(from: scalars.startIndex, to: range.upperBound)
                )
            }
        }
    #endif
}
