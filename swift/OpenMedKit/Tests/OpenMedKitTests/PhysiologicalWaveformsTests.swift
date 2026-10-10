import Foundation
import XCTest

@testable import OpenMedKit

final class PhysiologicalWaveformsTests: XCTestCase {
    private func fixtures() throws -> [[String: Any]] {
        var root = URL(fileURLWithPath: #filePath)
        for _ in 0..<5 { root.deleteLastPathComponent() }
        let data = try Data(contentsOf: root.appending(path: "tests/fixtures/multimodal/physiological_waveforms.json"))
        return try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [[String: Any]])
    }

    private func channel(_ row: [String: Any], changes: [String: Any] = [:]) throws -> PhysiologicalWaveformChannel {
        let input = row.merging(changes) { _, new in new }
        return try PhysiologicalWaveformChannel(
            kind: try XCTUnwrap(input["kind"] as? String), unit: try XCTUnwrap(input["unit"] as? String),
            sampleRateHz: try XCTUnwrap(input["sample_rate_hz"] as? Double),
            acquisitionMinimum: try XCTUnwrap(input["acquisition_minimum"] as? Double),
            acquisitionMaximum: try XCTUnwrap(input["acquisition_maximum"] as? Double),
            samples: try XCTUnwrap(input["samples"] as? [Any]).map { $0 is NSNull ? nil : $0 as? Double },
            offsetsSeconds: try XCTUnwrap(input["offsets_seconds"] as? [Double]),
            sourceSHA256: try XCTUnwrap(input["source_sha256"] as? String))
    }

    func testSharedPythonFixturesHaveIdenticalReports() throws {
        for row in try fixtures() {
            let item = try channel(row)
            let report = try PhysiologicalRecordingQuality.evaluate([item])
            let data = try report.reportJSON()
            let payload = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
            let channels = try XCTUnwrap(payload["channels"] as? [[String: Any]])
            XCTAssertEqual(channels[0] as NSDictionary, try XCTUnwrap(row["expected"] as? NSDictionary))
            XCTAssertEqual(data, try PhysiologicalRecordingQuality.evaluate([item]).reportJSON())
            XCTAssertFalse(String(decoding: data, as: UTF8.self).contains(String(repeating: "a", count: 64)))
            XCTAssertEqual(Set(payload.keys), ["channel_count", "codes", "channels"])
            XCTAssertThrowsError(try report.requireReviewerConfirmation()) {
                XCTAssertEqual($0 as? PhysiologicalWaveformError, .reviewerConfirmationRequired)
            }
            try report.requireReviewerConfirmation(confirmed: true)
            XCTAssertTrue(PhysiologicalWaveformChannel.nonDiagnosticNotice.contains("not vital-sign"))
            XCTAssertFalse(item.description.contains(String(repeating: "a", count: 64)))
            XCTAssertEqual(item.description, item.debugDescription)
        }
    }

    func testEachKindRejectsUnitsRatesRangesAndSamples() throws {
        for row in try fixtures().filter({ $0["fixture"] as? String == "pass" }) {
            let item = try channel(row)
            let limits = item.kind.constraints
            for rate in [limits.minimumRateHz, limits.maximumRateHz] {
                _ = try channel(row, changes: ["sample_rate_hz": rate, "offsets_seconds": (0..<40).map { Double($0) / rate }])
            }
            let mutations: [([String: Any], PhysiologicalWaveformError)] = [
                (["unit": "synthetic-private-payload"], .invalidUnit),
                (["sample_rate_hz": 0.0], .invalidSamplingRate),
                (["sample_rate_hz": limits.minimumRateHz / 2], .invalidSamplingRate),
                (["sample_rate_hz": limits.maximumRateHz + 1], .invalidSamplingRate),
                (["sample_rate_hz": Double.nan], .invalidSamplingRate),
                (["sample_rate_hz": Double.infinity], .invalidSamplingRate),
                (["acquisition_minimum": limits.minimum - 1], .invalidAcquisitionRange),
                (["acquisition_maximum": limits.maximum + 1], .invalidAcquisitionRange),
                (["acquisition_minimum": limits.maximum], .invalidAcquisitionRange),
                (["samples": [limits.minimum - 1] + item.samples.dropFirst().map { $0 as Any }], .invalidSample),
                (["samples": [limits.maximum + 1] + item.samples.dropFirst().map { $0 as Any }], .invalidSample),
                (["samples": [Double.nan] + item.samples.dropFirst().map { $0 as Any }], .invalidSample),
            ]
            for (changes, code) in mutations {
                XCTAssertThrowsError(try channel(row, changes: changes)) {
                    XCTAssertEqual($0 as? PhysiologicalWaveformError, code)
                    XCTAssertFalse($0.localizedDescription.contains("synthetic-private-payload"))
                }
            }
        }
    }

    func testUnknownKindsTimingProvenanceAndShapeFailClosed() throws {
        let row = try XCTUnwrap(fixtures().first)
        let mutations: [([String: Any], PhysiologicalWaveformError)] = [
            (["kind": "ecg"], .unknownKind),
            (["kind": "synthetic-private-payload"], .unknownKind),
            (["source_sha256": "synthetic-private-payload"], .invalidProvenance),
            (["source_sha256": String(repeating: "A", count: 64)], .invalidProvenance),
            (["samples": [Double]()], .invalidSampleShape),
            (["offsets_seconds": [0.0]], .invalidSampleShape),
            (["offsets_seconds": Array(repeating: 0.0, count: 40)], .invalidTiming),
            (["offsets_seconds": (0..<40).map { Double($0) / 20 }], .invalidTiming),
            (["offsets_seconds": (0..<40).map { Double($0) / 10 + ($0 > 20 ? 1 : 0) }], .invalidTiming),
            (["offsets_seconds": Array(repeating: Double.infinity, count: 40)], .invalidTiming),
            (["offsets_seconds": [-1.0] + (1..<40).map { Double($0) / 10 }], .invalidTiming),
        ]
        for (changes, code) in mutations {
            XCTAssertThrowsError(try channel(row, changes: changes)) {
                XCTAssertEqual($0 as? PhysiologicalWaveformError, code)
            }
        }
    }

    func testMixedRatesRemainIndependentAndResourcesAreBounded() throws {
        let rows = try fixtures().filter { $0["fixture"] as? String == "pass" }
        let items = try rows.map { row in
            let item = try channel(row)
            let rate = item.kind.constraints.minimumRateHz
            return try channel(row, changes: ["sample_rate_hz": rate, "offsets_seconds": (0..<40).map { 2 + Double($0) / rate }])
        }
        let report = try PhysiologicalRecordingQuality.evaluate(items)
        XCTAssertEqual(report.channels.map { $0.kind }, PhysiologicalChannelKind.allCases)
        XCTAssertTrue(report.channels.allSatisfy { $0.state == .pass })
        XCTAssertThrowsError(try PhysiologicalRecordingQuality.evaluate([]))
        XCTAssertThrowsError(try PhysiologicalRecordingQuality.evaluate(Array(repeating: items[0], count: 65)))
        let large = try channel(rows[0], changes: ["samples": Array(repeating: 0.5, count: 20_000), "offsets_seconds": (0..<20_000).map { Double($0) / 10 }])
        XCTAssertThrowsError(try PhysiologicalRecordingQuality.evaluate(Array(repeating: large, count: 51))) {
            XCTAssertEqual($0 as? PhysiologicalWaveformError, .resourceLimit)
        }
    }
}
