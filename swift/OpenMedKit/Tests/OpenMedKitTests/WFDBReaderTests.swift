import Foundation
import XCTest

@testable import OpenMedKit

final class WFDBReaderTests: XCTestCase {
    private func header(_ format: String = "16", total: Int = 3, gain: String = "200") -> Data {
        Data("SYNTHETIC_RECORD_NAME 1 250 \(total)\n/synthetic/private.dat \(format) \(gain)\n".utf8)
    }

    private func word(_ code: Int, _ interval: Int) -> Data {
        let value = code * 1024 + interval
        return Data([UInt8(value & 255), UInt8(value >> 8)])
    }

    private func assertCode(_ code: String, _ action: () throws -> Void, file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertThrowsError(try action(), file: file, line: line) { error in
            XCTAssertEqual((error as? WFDBError)?.reasonCode, code, file: file, line: line)
            XCTAssertEqual(String(describing: error), code, file: file, line: line)
        }
    }

    func testHandCheckedFormatsCalibrationTimingAndWindowChecksums() throws {
        let fixtures: [(String, Data, [Int], [Int])] = [
            ("16", Data([0, 248, 255, 7, 255, 255, 0, 0, 100, 0, 156, 255]), [-2048, -1, 100], [2047, 0, -100]),
            ("212", Data([0, 120, 255, 255, 15, 0, 100, 240, 156]), [-2048, -1, 100], [2047, 0, -100]),
            ("80", Data([0, 255, 127, 128, 129, 126]), [-128, -1, 1], [127, 0, -2]),
        ]
        for (format, data, first, second) in fixtures {
            let text =
                "SYNTHETIC_RECORD_NAME 2 250 3 12:00:00 01/01/2000\n"
                + "/synthetic/private.dat \(format) 200(17)/mV 12 0 \(first[0]) \(first.reduce(0, +)) 0 II\n"
                + "/synthetic/private.dat \(format) 1000/uV 12 23 \(second[0]) \(second.reduce(0, +)) 0 V1\n"
                + "# SYNTHETIC_COMMENT_NAME age=77\n"
            let record = try WFDBReader.read(header: Data(text.utf8), signals: [WFDBDataSource(data)])
            XCTAssertEqual(record.signals.map(\.samples), [first, second])
            XCTAssertEqual(record.signals.map(\.gain), [200, 1000])
            XCTAssertEqual(record.signals.map(\.baseline), [17, 23])
            XCTAssertEqual(record.signals.map(\.unit), ["mV", "uV"])
            XCTAssertEqual(record.signals.map(\.leadLabel), ["II", "V1"])
            XCTAssertEqual(record.durationSeconds, 0.012)
            XCTAssertEqual(record.sampleRateHz, 250)
            XCTAssertTrue(record.signals.allSatisfy(\.checksumVerified))
            XCTAssertTrue(record.commentsPresent && record.pathFieldsPresent && record.recordNamePresent && record.timingFieldsPresent)
            let window = try WFDBReader.read(header: Data(text.utf8), signals: [WFDBDataSource(data)], startSample: 1, sampleCount: 1)
            XCTAssertEqual(window.signals.map(\.samples), [[first[1]], [second[1]]])
            XCTAssertEqual(window.startSeconds, 0.004)
            assertCode("wfdb_signal_truncated") {
                _ = try WFDBReader.read(header: Data(text.utf8), signals: [WFDBDataSource(data.dropLast())])
            }
            var corrupted = data
            corrupted[corrupted.count - 1] ^= 1
            assertCode("wfdb_checksum_mismatch") {
                _ = try WFDBReader.read(header: Data(text.utf8), signals: [WFDBDataSource(corrupted)], sampleCount: 1)
            }
        }
    }

    func testOddPackedSamplesOffsetAndMultipleFiles() throws {
        let odd = try WFDBReader.read(header: header("212x1:0+4"), signals: [WFDBDataSource(Data([1, 2, 3, 4, 255, 15, 1, 0, 8, 0]))], startSample: 1)
        XCTAssertEqual(odd.signals[0].samples, [1, -2048])
        let unpadded = try WFDBReader.read(
            header: header("212"),
            signals: [WFDBDataSource(Data([255, 15, 1, 0, 8]))], startSample: 1)
        XCTAssertEqual(unpadded.signals[0].samples, [1, -2048])
        assertCode("wfdb_signal_truncated") {
            _ = try WFDBReader.read(header: header("212"), signals: [WFDBDataSource(Data([255, 15, 1, 0]))])
        }
        let multi = Data("r 3 62.5/125(0) 2\na 16 200\na 16 200\nb 80 200\n".utf8)
        let record = try WFDBReader.read(header: multi, signals: [WFDBDataSource(Data([1, 0, 11, 0, 2, 0, 12, 0])), WFDBDataSource(Data([129, 130]))])
        XCTAssertEqual(record.signals.map(\.samples), [[1, 2], [11, 12], [1, 2]])
        XCTAssertEqual(record.durationSeconds, 0.032)
    }

    func testAnnotationPrivacySkipPaddingAndReviewGate() throws {
        var annotation = word(1, 0) + word(63, 3) + Data("PHI\0".utf8)
        annotation += word(60, 1) + word(61, 2) + word(62, 0)
        annotation += word(59, 0) + Data([1, 0, 0, 0]) + word(1, 2) + Data([0, 0])
        let record = try WFDBReader.read(header: header("80", total: 65_540), signals: [WFDBDataSource(Data(repeating: 128, count: 65_540))], startSample: 65_538, sampleCount: 1, annotations: WFDBDataSource(annotation))
        XCTAssertEqual(record.annotations?.totalCount, 2)
        XCTAssertEqual(record.annotations?.samplePositions, [65_538])
        XCTAssertEqual(record.annotations?.auxiliaryTextPresent, true)
        XCTAssertFalse(String(describing: record).contains("PHI"))
        XCTAssertEqual(record.report()["notice"] as? String, WFDBReader.notice)
        assertCode("wfdb_reviewer_confirmation_required") { try record.requireReviewerConfirmation() }
        try record.requireReviewerConfirmation(confirmed: true)
        XCTAssertNil(record.report()["samples"])
        XCTAssertNil(record.report()["sample_positions"])
    }

    func testMultilingualMetadataLeakageControls() throws {
        for marker in ["SYNTHETIC_PRIVATE_NAME", "患者姓名", "patient@example.invalid", "/synthetic/private/path"] {
            let text = "SYNTHETIC_RECORD_NAME 1 250 3\n/synthetic/private.dat 16 200/\(marker) 16 0 0 0 0 \(marker)\n# \(marker)\n"
            let record = try WFDBReader.read(header: Data(text.utf8), signals: [WFDBDataSource(Data(repeating: 0, count: 6))])
            XCTAssertNil(record.signals[0].leadLabel)
            XCTAssertNil(record.signals[0].unit)
            XCTAssertTrue(record.descriptionsWithheld && record.unitsWithheld)
            XCTAssertFalse(String(describing: record).contains(marker))
            let json = try JSONSerialization.data(withJSONObject: record.report())
            XCTAssertFalse(String(decoding: json, as: UTF8.self).contains(marker))
            XCTAssertFalse(String(describing: record).contains("SYNTHETIC_RECORD_NAME"))
            XCTAssertFalse(String(describing: record).contains("/synthetic/private.dat"))
        }
        var text = Data([35, 255, 10])
        text += header()
        XCTAssertTrue(try WFDBReader.read(header: text, signals: [WFDBDataSource(Data(repeating: 0, count: 6))]).commentsPresent)
    }

    func testStableRefusalsAndBudgets() {
        let cases: [(Data, String)] = [
            (Data("private/2 1 250 3\n".utf8), "wfdb_multisegment_unsupported"),
            (header(gain: "0"), "wfdb_gain_invalid"),
            (header(gain: "1e999"), "wfdb_gain_invalid"),
            (header("24"), "wfdb_format_unsupported"),
            (header("16x2"), "wfdb_layout_unsupported"),
            (header("16:1"), "wfdb_layout_unsupported"),
            (header(total: 20_000_001), "wfdb_sample_limit_exceeded"),
            (Data("r 33 250 3\n".utf8), "wfdb_signal_limit_exceeded"),
            (Data("r 1 0 3\na 16\n".utf8), "wfdb_rate_invalid"),
            (Data("r 1\na 16\n".utf8), "wfdb_sample_count_required"),
            (Data("r 2 250 3\na 16\na 80\n".utf8), "wfdb_signal_group_invalid"),
        ]
        for (text, code) in cases {
            assertCode(code) { _ = try WFDBReader.read(header: text, signals: [WFDBDataSource(Data(repeating: 0, count: 6))]) }
        }
        var limits = WFDBLimits()
        limits.maxWindowSamples = 2
        assertCode("wfdb_window_limit_exceeded") { _ = try WFDBReader.read(header: header(), signals: [WFDBDataSource(Data(repeating: 0, count: 6))], limits: limits) }
        limits.maxSamples = 0
        assertCode("wfdb_limits_invalid") { _ = try WFDBReader.read(header: header(), signals: [], limits: limits) }
        assertCode("wfdb_window_invalid") { _ = try WFDBReader.read(header: header(), signals: [], startSample: -1) }
        assertCode("wfdb_source_count_invalid") { _ = try WFDBReader.read(header: header(), signals: []) }
        limits = WFDBLimits()
        limits.maxFileBytes = 5
        assertCode("wfdb_file_limit_exceeded") { _ = try WFDBReader.read(header: header(), signals: [WFDBDataSource(Data(repeating: 0, count: 6))], limits: limits) }
    }

    func testMalformedAnnotationsAndLimits() {
        let cases: [(Data, String)] = [
            (Data(), "wfdb_annotation_truncated"),
            (word(59, 0) + Data([0]), "wfdb_annotation_truncated"),
            (word(59, 1), "wfdb_annotation_invalid"),
            (word(63, 1), "wfdb_annotation_invalid"),
            (word(1, 0) + word(63, 3) + Data("PHI".utf8), "wfdb_annotation_truncated"),
            (word(1, 3) + Data([0, 0]), "wfdb_annotation_position_invalid"),
            (word(50, 0), "wfdb_annotation_format_unsupported"),
        ]
        for (data, code) in cases {
            assertCode(code) { _ = try WFDBReader.read(header: header(), signals: [WFDBDataSource(Data(repeating: 0, count: 6))], annotations: WFDBDataSource(data)) }
        }
        var limits = WFDBLimits()
        limits.maxAnnotations = 1
        assertCode("wfdb_annotation_limit_exceeded") {
            _ = try WFDBReader.read(header: header(), signals: [WFDBDataSource(Data(repeating: 0, count: 6))], annotations: WFDBDataSource(word(1, 0) + word(1, 0) + Data([0, 0])), limits: limits)
        }
    }

    func testLongGeneratedSourceReadsFixedChunksAndRetainsOnlyWindow() throws {
        final class Generated: WFDBByteSource {
            let byteCount = 600_000
            var maximumRead = 0
            func read(offset: Int, count: Int) throws -> Data {
                maximumRead = max(maximumRead, count)
                return Data(repeating: 0, count: min(count, byteCount - offset))
            }
        }
        let source = Generated()
        let record = try WFDBReader.read(header: header(total: 300_000), signals: [source], startSample: 299_990, sampleCount: 10)
        XCTAssertEqual(record.signals[0].samples, Array(repeating: 0, count: 10))
        XCTAssertLessThanOrEqual(source.maximumRead, 8192)
    }

    func testShortReadsAndSanitizedTransportFailures() throws {
        struct ShortSource: WFDBByteSource {
            let byteCount = 6
            func read(offset: Int, count: Int) throws -> Data { Data(repeating: 0, count: min(1, count)) }
        }
        XCTAssertEqual(try WFDBReader.read(header: header(), signals: [ShortSource()]).signals[0].samples, [0, 0, 0])
        struct Broken: WFDBByteSource {
            let byteCount = 6
            func read(offset: Int, count: Int) throws -> Data { throw NSError(domain: "/synthetic/private/path", code: 1) }
        }
        assertCode("wfdb_stream_read_error") { _ = try WFDBReader.read(header: header(), signals: [Broken()]) }
        struct OverRead: WFDBByteSource {
            let byteCount = 6
            func read(offset: Int, count: Int) throws -> Data { Data(repeating: 0, count: count + 1) }
        }
        assertCode("wfdb_stream_contract_error") { _ = try WFDBReader.read(header: header(), signals: [OverRead()]) }
    }

    func testHeaderByteSourceAndSignedChecksumWraparound() throws {
        let text = Data("r 1 250 2\na 16 200 16 0 32767 -2 0 II\n".utf8)
        let record = try WFDBReader.read(
            header: WFDBDataSource(text),
            signals: [WFDBDataSource(Data([255, 127, 255, 127]))])
        XCTAssertEqual(record.signals[0].samples, [32767, 32767])
        XCTAssertTrue(record.signals[0].checksumVerified)
    }

    func testCallerOwnedFileRestoresPosition() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try Data([9, 9, 1, 0, 2, 0, 3, 0]).write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        try handle.seek(toOffset: 2)
        let source = try WFDBFileSource(handle)
        let result = try WFDBReader.read(header: header(), signals: [source], startSample: 1, sampleCount: 1)
        XCTAssertEqual(result.signals[0].samples, [2])
        XCTAssertEqual(try handle.offset(), 2)
        assertCode("wfdb_signal_truncated") { _ = try WFDBReader.read(header: header(total: 4), signals: [source]) }
        XCTAssertEqual(try handle.offset(), 2)
    }
}
