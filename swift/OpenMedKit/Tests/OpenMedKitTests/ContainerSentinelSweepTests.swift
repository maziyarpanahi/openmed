import Foundation
import OpenMedPrivacySweep
import XCTest

@testable import OpenMedKit

/// Runs OpenMedKit entry points on sentinel inputs, then sweeps what the process left
/// behind: the app container when the host has one, the simulated device's shared
/// home for SwiftPM tests on a Simulator, and always this process's unified log. An
/// unsandboxed macOS `swift test` host sweeps the log only, because its home and
/// temporary directories belong to the user.
final class ContainerSentinelSweepTests: XCTestCase {
    func testPolicyDeidentificationLeavesNoSentinelInContainerOrProcessLog() throws {
        let start = Date().addingTimeInterval(-1)
        let person = try PHISentinel.synthetic(identifier: "workload.person")
        let record = try PHISentinel.synthetic(identifier: "workload.record")
        let phone = try PHISentinel(identifier: "workload.phone", value: "+1 (555) 01\(Int.random(in: 10...99))-\(Int.random(in: 1000...9999))")
        let text = "Patient \(person.plantedValue) MRN \(record.plantedValue) phone \(phone.plantedValue)."
        let entities = [
            entity(label: "first_name", value: person.plantedValue, in: text),
            entity(label: "mrn", value: record.plantedValue, in: text),
            entity(label: "phone", value: phone.plantedValue, in: text),
        ]

        try runPolicyWorkload(text: text, entities: entities)

        let scope = HostStorageScope.current
        let report = try PHISentinelSweep.currentProcess(
            sentinels: [person, record, phone],
            logSince: start,
            scope: scope
        ).run()
        attach(report)
        XCTAssertTrue(report.clean, report.description)
        XCTAssertTrue(report.passesGate(for: scope), "\(scope.rawValue): \(report)")
    }

    private func runPolicyWorkload(text: String, entities: [EntityPrediction]) throws {
        for name in Policy.bundledProfileNames {
            let policy = try Policy(named: name)
            #if canImport(MLX) && canImport(Tokenizers) && !os(watchOS) && !os(visionOS)
                let result = OpenMed.deidentify(text, entities: entities, policy: policy)
                XCTAssertEqual(result.policyName, policy.name)
            #else
                XCTAssertEqual(entities.map { policy.action(for: $0.label) }.count, entities.count)
            #endif
        }
    }

    private func attach(_ report: PHISentinelSweepReport) {
        guard let data = try? report.jsonData() else { return }
        let attachment = XCTAttachment(data: data, uniformTypeIdentifier: "public.json")
        attachment.name = "phi-sentinel-sweep.json"
        attachment.lifetime = .keepAlways
        add(attachment)
    }

    private func entity(label: String, value: String, in text: String) -> EntityPrediction {
        let range = text.range(of: value)!
        return EntityPrediction(
            label: label,
            text: value,
            confidence: 0.99,
            start: text.unicodeScalars.distance(from: text.unicodeScalars.startIndex, to: range.lowerBound),
            end: text.unicodeScalars.distance(from: text.unicodeScalars.startIndex, to: range.upperBound)
        )
    }
}
