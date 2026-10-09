import Foundation
import XCTest

@testable import OpenMedKit

final class EDFReaderTests: XCTestCase {
    private func field(_ value: String, _ size: Int) -> [UInt8] {
        Array(value.utf8) + [UInt8](repeating: 32, count: size - value.utf8.count)
    }

    private func synthetic(kind: String = "EDF", onsets: [String] = ["+0", "+1"], duration: String = "1", declared: String = "2", patient: String = "SYNTHETIC_SECRET", recording: String = "SYNTHETIC_RECORDING", label: String = "ECG II", annotation: String = "SYNTHETIC_ANNOTATION", point: Bool = false, extra: Bool = false, other: Bool = false, annotationOnly: Bool = false) -> Data {
        let ns = (annotationOnly ? 0 : 1) + (kind == "EDF" ? 0 : 1) + (extra ? 1 : 0) + (other ? 1 : 0)
        let fixed = ["0", patient, recording, "09.10.26", "12.34.56", String(256 * (ns + 1)), kind == "EDF" ? "" : kind, declared, duration, String(ns)]
        var bytes: [UInt8] = []
        for (value, size) in zip(fixed, [8, 80, 80, 8, 8, 8, 44, 8, 8, 4]) {
            bytes += field(value, size)
        }
        let signal = [label, "", "mV", "-1", "1", "-2", "2", "", point ? "1" : "4", ""]
        let annotations = ["EDF Annotations", "", "", "-1", "1", "-32768", "32767", "", "128", ""]
        let second = ["EMG", "", "uV", "-10", "10", "-2", "2", "", "2", ""]
        let channels = (annotationOnly ? [] : [signal]) + (other ? [second] : []) + Array(repeating: annotations, count: (kind == "EDF" ? 0 : 1) + (extra ? 1 : 0))
        for (col, size) in [16, 80, 8, 8, 8, 8, 8, 80, 8, 32].enumerated() {
            for channel in channels { bytes += field(channel[col], size) }
        }
        for (index, onset) in onsets.enumerated() {
            let samples: [Int16] = point ? [index == 0 ? -2 : 2] : (index == 0 ? [-2, -1, 0, 2] : [2, 0, -1, -2])
            for sample in (annotationOnly ? [] : samples) {
                let value = UInt16(bitPattern: sample)
                bytes += [UInt8(value & 255), UInt8(value >> 8)]
            }
            if other {
                for sample: Int16 in (index == 0 ? [-2, 2] : [0, 1]) {
                    let value = UInt16(bitPattern: sample)
                    bytes += [UInt8(value & 255), UInt8(value >> 8)]
                }
            }
            if kind != "EDF" {
                let tal = Array((onset + "\u{14}\u{14}\0" + onset + "\u{15}0.5\u{14}" + annotation + "\u{14}other\u{14}\0").utf8)
                bytes += tal + [UInt8](repeating: 0, count: 256 - tal.count)
            }
            if extra {
                let tal = Array((onset + "\u{14}extra\u{14}\0").utf8)
                bytes += tal + [UInt8](repeating: 0, count: 256 - tal.count)
            }
        }
        return Data(bytes)
    }

    private func replaced(_ source: Data, at: Int, size: Int, value: String) -> Data {
        var source = source
        source.replaceSubrange(at..<at + size, with: field(value, size))
        return source
    }

    private func assertCode(_ code: String, _ block: () throws -> EDFRecording, file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertThrowsError(try block(), file: file, line: line) { error in
            XCTAssertEqual((error as? EDFError)?.code, code, file: file, line: line)
            XCTAssertEqual(String(describing: error), code, file: file, line: line)
        }
    }

    func testFormatsScaledSamplesAndReviewGate() throws {
        for kind in ["EDF", "EDF+C", "EDF+D"] {
            let result = try EDFReader.read(synthetic(kind: kind))
            XCTAssertEqual(result.format, kind)
            XCTAssertEqual(result.recordOnsetsSeconds, [0, 1])
            XCTAssertEqual(result.signals[0].label, "ECG II")
            XCTAssertEqual(result.signals[0].samplingRateHz, 4)
            XCTAssertEqual(result.records[0].signals[0].digitalSamples, [-2, -1, 0, 2])
            XCTAssertEqual(result.records[0].signals[0].physicalSamples, [-1, -0.5, 0, 1])
            XCTAssertEqual(result.records[1].signals[0].physicalSamples, [1, 0, -0.5, -1])
            XCTAssertTrue(result.gaps.isEmpty)
            XCTAssertFalse(result.reviewerConfirmed)
            XCTAssertTrue(try result.reviewed(confirmed: true).reviewerConfirmed)
            assertCode("edf_review_required") { try result.reviewed(confirmed: false) }
        }
    }

    func testDiscontinuousGapsAndWindowBoundaries() throws {
        let source = synthetic(kind: "EDF+D", onsets: ["+0.25", "+3.25"])
        let result = try EDFReader.read(source, startSeconds: 0.5, endSeconds: 3.5)
        XCTAssertEqual(result.recordOnsetsSeconds, [0.25, 3.25])
        XCTAssertEqual(result.gaps[0].beforeRecordIndex, 1)
        XCTAssertEqual(result.gaps[0].startSeconds, 1.25)
        XCTAssertEqual(result.gaps[0].endSeconds, 3.25)
        XCTAssertEqual(result.records[0].signals[0].firstSampleIndex, 1)
        XCTAssertEqual(result.records[0].signals[0].digitalSamples, [-1, 0, 2])
        XCTAssertEqual(result.records[1].signals[0].digitalSamples, [2])
        XCTAssertEqual(result.annotations.count, 2)
        XCTAssertEqual(result.annotations[0].onsetSeconds, 0.25)
        XCTAssertEqual(result.annotations[0].durationSeconds, 0.5)
        XCTAssertEqual(result.annotations[0].count, 2)
        XCTAssertTrue(try EDFReader.read(source, startSeconds: 1.25, endSeconds: 3.25).records.isEmpty)
    }

    func testExactDecimalTimingAndUnknownRecordCount() throws {
        let source = synthetic(kind: "EDF+C", onsets: ["+0.1", "+0.2"], duration: "0.1", declared: "-1")
        var limits = EDFLimits()
        limits.maxBytes = source.count
        let result = try EDFReader.read(source, startSeconds: 0.125, endSeconds: 0.2, limits: limits)
        XCTAssertEqual(result.recordOnsetsSeconds, [0.1, 0.2])
        XCTAssertEqual(result.signals[0].samplingRateHz, 40)
        XCTAssertEqual(result.records[0].signals[0].digitalSamples, [-1, 0, 2])
    }

    func testZeroDurationPointsAndNegativeGain() throws {
        let points = try EDFReader.read(synthetic(kind: "EDF+D", onsets: ["+0", "+3"], duration: "0", point: true), endSeconds: 4)
        XCTAssertNil(points.signals[0].samplingRateHz)
        XCTAssertEqual(points.records[1].signals[0].digitalSamples, [2])
        XCTAssertEqual(points.gaps[0].endSeconds, 3)
        var source = synthetic(kind: "EDF+C", extra: true)
        source = replaced(source, at: 256 + 104 * 3, size: 8, value: "1")
        source = replaced(source, at: 256 + 112 * 3, size: 8, value: "-1")
        let result = try EDFReader.read(source)
        XCTAssertEqual(result.records[0].signals[0].physicalSamples, [1, 0.5, 0, -1])
        XCTAssertEqual(result.report()["annotation_count"] as? Int, 6)
    }

    func testPrivacyAndPlaceholderStatus() throws {
        for kind in ["EDF", "EDF+C", "EDF+D"] {
            let result = try EDFReader.read(synthetic(kind: kind, label: "SYNTHETIC_SECRET", annotation: "SYNTHETIC_SECRET 患者 José"))
            let report = String(data: try JSONSerialization.data(withJSONObject: result.report()), encoding: .utf8)!
            let output = String(reflecting: result) + report
            for secret in ["SYNTHETIC_SECRET", "SYNTHETIC_RECORDING", "09.10.26", "12.34.56", "患者", "José"] {
                XCTAssertFalse(output.contains(secret))
            }
            XCTAssertEqual(result.signals[0].label, "withheld")
            XCTAssertTrue(result.patient.present)
            XCTAssertEqual(result.patient.anonymizationStatus, "not_verified")
        }
        let placeholders = try EDFReader.read(synthetic(patient: "X X X X", recording: "Startdate X X X X"))
        XCTAssertEqual(placeholders.patient.anonymizationStatus, "placeholder_only")
        XCTAssertEqual(placeholders.recording.anonymizationStatus, "placeholder_only")
        let empty = try EDFReader.read(synthetic(patient: "", recording: ""))
        XCTAssertFalse(empty.patient.present)
        XCTAssertEqual(empty.recording.anonymizationStatus, "absent")
    }

    func testHeaderFailures() {
        let controls: [(Int, Int, String, String)] = [
            (0, 8, "1", "edf_header_invalid"), (168, 8, "31.02.26", "edf_header_invalid"),
            (176, 8, "24.00.00", "edf_header_invalid"), (184, 8, "257", "edf_header_size_invalid"),
            (236, 8, "0", "edf_record_count_invalid"), (236, 8, "-2", "edf_record_count_invalid"),
            (236, 8, "100001", "edf_record_limit"), (244, 8, "nan", "edf_numeric_invalid"),
            (244, 8, "1E99", "edf_numeric_invalid"), (244, 8, "-1", "edf_duration_limit"),
            (244, 8, "0", "edf_duration_invalid"), (252, 4, "65", "edf_signal_limit"),
            (360, 8, "1", "edf_range_invalid"), (376, 8, "3", "edf_range_invalid"),
            (384, 8, "32768", "edf_range_invalid"), (472, 8, "0", "edf_samples_invalid"),
            (472, 8, "99999999", "edf_record_byte_limit"),
        ]
        for (at, size, value, code) in controls {
            assertCode(code) { try EDFReader.read(replaced(synthetic(), at: at, size: size, value: value)) }
        }
        assertCode("edf_record_count_invalid") { try EDFReader.read(synthetic() + Data([1])) }
        assertCode("edf_truncated") { try EDFReader.read(synthetic().dropLast()) }
        assertCode("edf_record_count_invalid") { try EDFReader.read(synthetic(declared: "1")) }
        assertCode("edf_record_count_invalid") { try EDFReader.read(synthetic(declared: "3")) }
    }

    func testAllResourceBudgets() {
        let setters: [(String, (inout EDFLimits) -> Void)] = [
            ("edf_byte_limit", { $0.maxBytes = 520 }),
            ("edf_signal_limit", { $0.maxSignals = 1 }),
            ("edf_record_limit", { $0.maxRecords = 1 }),
            ("edf_record_byte_limit", { $0.maxRecordBytes = 4 }),
            ("edf_duration_limit", { $0.maxDurationSeconds = 1 }),
            ("edf_window_invalid", { $0.maxWindowSeconds = 1 }),
            ("edf_output_sample_limit", { $0.maxOutputSamples = 7 }),
            ("edf_annotation_limit", { $0.maxAnnotationLists = 3 }),
            ("edf_limits_invalid", { $0.maxBytes = 0 }),
        ]
        for (code, setter) in setters {
            var limits = EDFLimits()
            setter(&limits)
            assertCode(code) { try EDFReader.read(synthetic(kind: "EDF+C"), limits: limits) }
        }
    }

    func testTimingAndWindowNegativeControls() {
        assertCode("edf_record_timing_invalid") { try EDFReader.read(synthetic(kind: "EDF+C", onsets: ["+0", "+2"])) }
        assertCode("edf_record_timing_invalid") { try EDFReader.read(synthetic(kind: "EDF+D", onsets: ["+0", "+0.5"])) }
        assertCode("edf_timekeeping_invalid") { try EDFReader.read(synthetic(kind: "EDF+D", onsets: ["+1", "+2"])) }
        for (start, end) in [(-1.0, 1.0), (0, 0), (1, 0), (Double.nan, 1), (0, Double.infinity), (0, 3601)] {
            assertCode("edf_window_invalid") { try EDFReader.read(synthetic(), startSeconds: start, endSeconds: end) }
        }
    }

    func testMalformedAnnotations() {
        let tals: [[UInt8]] = [
            Array("+0\n\u{14}\u{14}\0".utf8), Array("+0 \u{14}\u{14}\0".utf8),
            Array("+0\u{14}secret\u{14}\0".utf8), Array("+0\u{15}0.1\u{14}\u{14}\0".utf8),
            Array("0\u{14}\u{14}\0".utf8), Array("+0\u{14}\u{14}\0\0x".utf8),
            Array("+0\u{14}\u{14}\0+0\u{14}".utf8) + [255, 20, 0],
        ]
        for tal in tals {
            var source = synthetic(kind: "EDF+C")
            source.replaceSubrange(776..<1032, with: tal + [UInt8](repeating: 0, count: 256 - tal.count))
            XCTAssertThrowsError(try EDFReader.read(source)) { error in
                XCTAssertTrue(["edf_timekeeping_invalid", "edf_annotation_invalid", "edf_numeric_invalid"].contains((error as? EDFError)?.code ?? ""))
                XCTAssertFalse(String(describing: error).contains("secret"))
            }
        }
    }

    func testMixedSampleRatesAndAnnotationOnlyFiles() throws {
        let mixed = try EDFReader.read(synthetic(kind: "EDF+C", other: true), startSeconds: 0.5, endSeconds: 1.5)
        XCTAssertEqual(mixed.signals.map { $0.samplingRateHz }, [4, 2])
        XCTAssertEqual(mixed.records[0].signals[1].firstSampleIndex, 1)
        XCTAssertEqual(mixed.records[0].signals[1].physicalSamples, [10])
        XCTAssertEqual(mixed.records[1].signals[1].physicalSamples, [0])
        let annotations = try EDFReader.read(synthetic(kind: "EDF+D", onsets: ["+0", "+3"], duration: "0", annotationOnly: true))
        XCTAssertTrue(annotations.signals.isEmpty)
        XCTAssertEqual(annotations.windowEndSeconds, 4)
        XCTAssertEqual(annotations.report()["annotation_count"] as? Int, 4)
    }

    func testAnnotationOffsetsAndDefaultWindowSelection() throws {
        var source = synthetic(kind: "EDF+C")
        let tal = Array("+0\u{14}\u{14}\0-0.25\u{15}1\u{14}before\u{14}\0+100\u{14}after\u{14}\0".utf8)
        source.replaceSubrange(776..<1032, with: tal + [UInt8](repeating: 0, count: 256 - tal.count))
        let result = try EDFReader.read(source)
        XCTAssertEqual(result.windowEndSeconds, 2)
        XCTAssertEqual(result.annotations.map { $0.onsetSeconds }, [-0.25, 1])
        let noduration = Array("+0\u{14}\u{14}\0+0.75\u{14}event\u{14}\0".utf8)
        source.replaceSubrange(776..<1032, with: noduration + [UInt8](repeating: 0, count: 256 - noduration.count))
        let selected = try EDFReader.read(source, startSeconds: 0.75, endSeconds: 1)
        XCTAssertNil(selected.annotations[0].durationSeconds)
        XCTAssertEqual(selected.annotations[0].count, 1)
        assertCode("edf_annotation_channel_missing") { try EDFReader.read(replaced(source, at: 272, size: 16, value: "ECG I")) }
        assertCode("edf_record_timing_invalid") { try EDFReader.read(synthetic(kind: "EDF+C", onsets: ["+0", "+2"]), startSeconds: 0, endSeconds: 0.5) }
    }

    func testCallerOwnedStream() throws {
        let stream = InputStream(data: synthetic(kind: "EDF+D", onsets: ["+0", "+5"]))
        stream.open()
        defer { stream.close() }
        let result = try EDFReader.read(stream, startSeconds: 5, endSeconds: 6)
        XCTAssertEqual(result.records[0].recordIndex, 1)
        XCTAssertEqual(result.records[0].signals[0].physicalSamples, [1, 0, -0.5, -1])
        XCTAssertEqual(result.gaps[0].endSeconds, 5)
        XCTAssertNotEqual(stream.streamStatus, .closed)
    }
}
