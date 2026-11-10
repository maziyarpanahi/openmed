import Foundation
import XCTest

@testable import OpenMedKit

final class WatchVisionParityTests: XCTestCase {
    func testWatchAndVisionConfigurationsRequireNanoINT8Budget() {
        for platform in [OpenMedApplePlatform.watchOS, .visionOS] {
            let configuration = PlatformModelConfiguration.configuration(for: platform)

            XCTAssertEqual(configuration.platform, platform)
            XCTAssertEqual(configuration.maximumTier, .nano)
            XCTAssertEqual(configuration.maximumParameterCount, 30_000_000)
            XCTAssertEqual(configuration.maximumResidentMemoryMB, 150)
            XCTAssertEqual(configuration.maximumSequenceLength, 256)
            XCTAssertTrue(configuration.requiresINT8)
        }
    }

    func testWatchAndVisionSelectOnlyCompatibleNanoModel() throws {
        let nano = descriptor(
            identifier: "nano-int8",
            tier: .nano,
            parameterCount: 24_000_000,
            residentMemoryMB: 128,
            isINT8: true
        )
        let tiny = descriptor(
            identifier: "tiny-int8",
            tier: .tiny,
            parameterCount: 44_000_000,
            residentMemoryMB: 220,
            isINT8: true
        )
        let unquantizedNano = descriptor(
            identifier: "nano-fp16",
            tier: .nano,
            parameterCount: 24_000_000,
            residentMemoryMB: 148,
            isINT8: false
        )

        for platform in [OpenMedApplePlatform.watchOS, .visionOS] {
            let configuration = PlatformModelConfiguration.configuration(for: platform)
            let selected = try PlatformModel.selectModel(
                from: [tiny, unquantizedNano, nano],
                configuration: configuration
            )

            XCTAssertEqual(selected, nano)
            XCTAssertFalse(configuration.allows(tiny))
            XCTAssertFalse(configuration.allows(unquantizedNano))
        }
    }

    func testSyntheticNoteRedactsWithIOSReferenceSpansWithinTolerance() throws {
        let note = "Patient Ada Lovelace, MRN TEST-123, called 555-0100."
        let reference = [
            try entity(label: "full_name", value: "Ada Lovelace", in: note),
            try entity(
                label: "medical_record_number",
                value: "TEST-123",
                in: note
            ),
            try entity(label: "phone_number", value: "555-0100", in: note),
        ]

        let result = PlatformModel.redact(note, entities: reference)

        XCTAssertEqual(
            result.deidentifiedText,
            "Patient [FULL_NAME], MRN [MEDICAL_RECORD_NUMBER], called [PHONE_NUMBER]."
        )
        XCTAssertEqual(result.numEntitiesRedacted, reference.count)
        XCTAssertEqual(result.piiEntities.count, reference.count)

        for (observed, expected) in zip(result.piiEntities, reference) {
            XCTAssertEqual(observed.label, expected.label)
            XCTAssertLessThanOrEqual(abs(observed.start - expected.start), 1)
            XCTAssertLessThanOrEqual(abs(observed.end - expected.end), 1)
        }
    }

    #if os(watchOS)
        func testCurrentWatchSimulatorUsesNanoLimits() {
            XCTAssertEqual(PlatformModelConfiguration.current.platform, .watchOS)
            XCTAssertEqual(PlatformModelConfiguration.current.maximumTier, .nano)
        }
    #endif

    #if os(visionOS)
        func testCurrentVisionSimulatorUsesNanoLimits() {
            XCTAssertEqual(PlatformModelConfiguration.current.platform, .visionOS)
            XCTAssertEqual(PlatformModelConfiguration.current.maximumTier, .nano)
        }
    #endif

    private func descriptor(
        identifier: String,
        tier: OpenMedAppleModelTier,
        parameterCount: Int,
        residentMemoryMB: Int,
        isINT8: Bool
    ) -> PlatformModelDescriptor {
        let root = URL(fileURLWithPath: "/synthetic-models")
        return PlatformModelDescriptor(
            identifier: identifier,
            modelURL: root.appending(path: "\(identifier).mlmodelc"),
            id2labelURL: root.appending(path: "\(identifier)-id2label.json"),
            tier: tier,
            parameterCount: parameterCount,
            estimatedResidentMemoryMB: residentMemoryMB,
            isINT8: isINT8
        )
    }

    private func entity(
        label: String,
        value: String,
        in text: String
    ) throws -> EntityPrediction {
        let range = try XCTUnwrap(text.range(of: value))
        return EntityPrediction(
            label: label,
            text: value,
            confidence: 0.99,
            start: text.distance(from: text.startIndex, to: range.lowerBound),
            end: text.distance(from: text.startIndex, to: range.upperBound)
        )
    }
}
