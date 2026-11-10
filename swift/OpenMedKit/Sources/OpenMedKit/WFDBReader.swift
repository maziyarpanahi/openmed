import Foundation

/// Local byte access supplied by the caller. Implementations must not return more
/// than the requested byte count. The reader never resolves header file names.
public protocol WFDBByteSource {
    var byteCount: Int { get }
    func read(offset: Int, count: Int) throws -> Data
}

/// In-memory WFDB input. Window decoding does not copy the complete signal data.
public struct WFDBDataSource: WFDBByteSource {
    private let data: Data
    public var byteCount: Int { data.count }

    public init(_ data: Data) { self.data = data }

    public func read(offset: Int, count: Int) throws -> Data {
        guard offset >= 0, count >= 0, offset <= data.count else {
            throw WFDBError("wfdb_stream_contract_error")
        }
        let end = offset + min(count, data.count - offset)
        return data.subdata(in: offset..<end)
    }
}

/// Caller-owned local file handle, relative to its initial position. Every read
/// restores that position. Do not access the handle concurrently with this source.
public final class WFDBFileSource: WFDBByteSource {
    private let handle: FileHandle
    private let initial: UInt64
    public let byteCount: Int

    public init(_ handle: FileHandle) throws {
        self.handle = handle
        let position: UInt64
        do { position = try handle.offset() } catch { throw WFDBError("wfdb_stream_contract_error") }
        initial = position
        do {
            let end = try handle.seekToEnd()
            try handle.seek(toOffset: initial)
            guard end >= initial, end - initial <= UInt64(Int.max) else {
                throw WFDBError("wfdb_stream_contract_error")
            }
            byteCount = Int(end - initial)
        } catch {
            try? handle.seek(toOffset: position)
            throw WFDBError("wfdb_stream_contract_error")
        }
    }

    public func read(offset: Int, count: Int) throws -> Data {
        guard offset >= 0, count >= 0, offset <= byteCount else {
            throw WFDBError("wfdb_stream_contract_error")
        }
        do {
            try handle.seek(toOffset: initial + UInt64(offset))
            let result = try handle.read(upToCount: min(count, byteCount - offset)) ?? Data()
            try handle.seek(toOffset: initial)
            return result
        } catch {
            try? handle.seek(toOffset: initial)
            throw WFDBError("wfdb_stream_read_error")
        }
    }
}

/// Value-free failure. Underlying transport errors and source text are withheld.
public struct WFDBError: Error, CustomStringConvertible, Equatable {
    public let reasonCode: String
    public var description: String { reasonCode }
    fileprivate init(_ reasonCode: String) { self.reasonCode = reasonCode }
}

/// Positive input/output budgets, aligned with the Python WFDB reader.
public struct WFDBLimits {
    public var maxHeaderBytes = 65_536
    public var maxSignals = 32
    public var maxSamples = 20_000_000
    public var maxWindowSamples = 100_000
    public var maxFileBytes = 256 * 1024 * 1024
    public var maxAnnotationBytes = 4 * 1024 * 1024
    public var maxAnnotations = 100_000
    public init() {}

    fileprivate func validate() throws {
        guard
            [
                maxHeaderBytes, maxSignals, maxSamples, maxWindowSamples,
                maxFileBytes, maxAnnotationBytes, maxAnnotations,
            ].allSatisfy({ $0 > 0 })
        else { throw WFDBError("wfdb_limits_invalid") }
    }
}

/// Declared calibration and raw integer window. No lead normalization is applied.
public struct WFDBSignal: Equatable {
    public let leadLabel: String?
    public let gain: Double
    public let baseline: Int
    public let unit: String?
    public let formatCode: Int
    public let samples: [Int]
    public let checksumVerified: Bool
}

/// Annotation counts and absolute sample positions in the returned window.
public struct WFDBAnnotations: Equatable {
    public let totalCount: Int
    public let samplePositions: [Int]
    public let auxiliaryTextPresent: Bool
}

/// Immutable review-only waveform data. Samples remain potentially sensitive;
/// this is not a de-identification, quality-gate or clinical validation result.
public struct WFDBRecord {
    public let signals: [WFDBSignal]
    public let sampleRateHz: Double
    public let totalSamples: Int
    public let startSample: Int
    public let sampleCount: Int
    public let annotations: WFDBAnnotations?
    public let commentsPresent: Bool
    public let recordNamePresent: Bool
    public let pathFieldsPresent: Bool
    public let timingFieldsPresent: Bool
    public let descriptionsWithheld: Bool
    public let unitsWithheld: Bool
    public var startSeconds: Double { Double(startSample) / sampleRateHz }
    public var durationSeconds: Double { Double(sampleCount) / sampleRateHz }
    public var notice: String { WFDBReader.notice }

    /// Call before consequential downstream use, after explicit human review.
    public func requireReviewerConfirmation(confirmed: Bool = false) throws {
        guard confirmed else { throw WFDBError("wfdb_reviewer_confirmation_required") }
    }

    /// Controlled counts and flags only; omits amplitudes and annotation positions.
    public func report() -> [String: Any] {
        [
            "signal_count": signals.count, "total_samples": totalSamples,
            "start_sample": startSample, "sample_count": sampleCount,
            "annotation_count": annotations?.totalCount ?? 0,
            "checksums_verified": signals.filter(\.checksumVerified).count,
            "comments_present": commentsPresent, "record_name_present": recordNamePresent,
            "path_fields_present": pathFieldsPresent, "timing_fields_present": timingFieldsPresent,
            "descriptions_withheld": descriptionsWithheld, "units_withheld": unitsWithheld,
            "annotation_text_present": annotations?.auxiliaryTextPresent ?? false,
            "review_required": true, "notice": notice,
        ]
    }
}

/// Bounded, dependency-free WFDB formats 16, 212 and 80 and MIT annotation reader.
/// Performs no inference, network calls, source-name resolution or clinical actions.
public enum WFDBReader {
    public static let notice =
        "Non-diagnostic waveform data for human review only. "
        + "No clinical validation or autonomous clinical action is provided. "
        + "Explicit reviewer confirmation is required for consequential use."
    private static let leads = Set([
        "I", "II", "III", "aVR", "aVL", "aVF", "AVR", "AVL", "AVF", "MLI", "MLII", "MLIII",
        "MCL1", "MCL6", "ECG", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9",
    ])
    private static let units = Set(["mV", "uV", "µV", "μV", "V"])
    private static let number = #"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?"#

    private struct Spec {
        let group: Int
        let format: Int
        let offset: Int
        let gain: Double
        let baseline: Int
        let unit: String?
        let label: String?
        let checksum: Int?
        let descriptionWithheld: Bool
        let unitWithheld: Bool
    }

    private final class Cursor {
        let source: any WFDBByteSource
        var offset = 0
        init(_ source: any WFDBByteSource, limit: Int) throws {
            guard source.byteCount >= 0 else { throw WFDBError("wfdb_stream_contract_error") }
            guard source.byteCount <= limit else { throw WFDBError("wfdb_file_limit_exceeded") }
            self.source = source
        }

        func exact(_ count: Int, _ code: String) throws -> Data {
            guard count >= 0, count <= source.byteCount - offset else { throw WFDBError(code) }
            var result = Data()
            while result.count < count {
                let part: Data
                do { part = try source.read(offset: offset, count: count - result.count) } catch { throw WFDBError("wfdb_stream_read_error") }
                guard part.count <= count - result.count else {
                    throw WFDBError("wfdb_stream_contract_error")
                }
                guard !part.isEmpty else { throw WFDBError(code) }
                result.append(part)
                offset += part.count
            }
            return result
        }

        func discard(_ count: Int, _ code: String) throws {
            var remaining = count
            while remaining > 0 {
                let take = min(remaining, 8192)
                _ = try exact(take, code)
                remaining -= take
            }
        }
    }

    private static func match(_ text: String, _ pattern: String) -> [String?]? {
        let regex = try! NSRegularExpression(pattern: "^(?:" + pattern + ")$")
        let range = NSRange(text.startIndex..., in: text)
        guard let result = regex.firstMatch(in: text, range: range), result.range == range else { return nil }
        return (1..<result.numberOfRanges).map { index in
            guard let range = Range(result.range(at: index), in: text) else { return nil }
            return String(text[range])
        }
    }

    private static func integer(_ value: String) throws -> Int {
        guard match(value, #"([+-]?[0-9]{1,12})"#) != nil, let result = Int(value) else {
            throw WFDBError("wfdb_header_invalid")
        }
        return result
    }

    private static func parse(_ header: Data, _ limits: WFDBLimits) throws -> ([Spec], Double, Int, Bool, Bool, Int) {
        var lines: [String] = []
        var comments = false
        // Byte-level comment filtering accepts even non-UTF8 discarded comments.
        for raw in header.split(separator: 10) {
            let trimmed = raw.drop(while: { $0 == 32 || $0 == 9 || $0 == 13 })
            if trimmed.first == 35 {
                comments = true
                continue
            }
            if trimmed.isEmpty { continue }
            guard trimmed.count <= 255, let line = String(data: Data(trimmed), encoding: .utf8) else {
                throw WFDBError("wfdb_header_invalid")
            }
            lines.append(line)
        }
        guard let first = lines.first else { throw WFDBError("wfdb_header_invalid") }
        let record = first.split(whereSeparator: \.isWhitespace).map(String.init)
        guard let name = record.first else { throw WFDBError("wfdb_header_invalid") }
        guard !name.contains("/") else { throw WFDBError("wfdb_multisegment_unsupported") }
        guard record.count >= 4 else { throw WFDBError("wfdb_sample_count_required") }
        let count = try integer(record[1])
        let total = try integer(record[3])
        guard count > 0, count <= limits.maxSignals else { throw WFDBError("wfdb_signal_limit_exceeded") }
        guard total > 0, total <= limits.maxSamples else { throw WFDBError("wfdb_sample_limit_exceeded") }
        guard let rateMatch = match(record[2], "(" + number + ")(?:/" + number + "(?:\\(" + number + "\\))?)?"),
            let rate = Double(rateMatch[0] ?? "")
        else { throw WFDBError("wfdb_header_invalid") }
        guard rate.isFinite, rate > 0 else { throw WFDBError("wfdb_rate_invalid") }
        guard lines.count == count + 1 else { throw WFDBError("wfdb_header_invalid") }
        var specs: [Spec] = []
        var files: [String] = []
        for line in lines.dropFirst() {
            let fields = line.split(maxSplits: 8, whereSeparator: \.isWhitespace).map(String.init)
            guard fields.count >= 2 else { throw WFDBError("wfdb_header_invalid") }
            if fields[0] != files.last {
                guard !files.contains(fields[0]) else { throw WFDBError("wfdb_signal_group_invalid") }
                files.append(fields[0])
            }
            guard let formatMatch = match(fields[1], #"([0-9]+)(?:x([0-9]+))?(?::([0-9]+))?(?:\+([0-9]+))?"#) else {
                throw WFDBError("wfdb_format_unsupported")
            }
            let format = try integer(formatMatch[0]!)
            guard [16, 212, 80].contains(format) else { throw WFDBError("wfdb_format_unsupported") }
            if let frame = formatMatch[1], try integer(frame) != 1 { throw WFDBError("wfdb_layout_unsupported") }
            if let skew = formatMatch[2], try integer(skew) != 0 { throw WFDBError("wfdb_layout_unsupported") }
            let offset = try integer(formatMatch[3] ?? "0")
            guard offset <= limits.maxFileBytes else { throw WFDBError("wfdb_file_limit_exceeded") }
            var gain = 200.0
            var baseline: Int?
            var unit = "mV"
            if fields.count > 2 {
                guard let gainMatch = match(fields[2], "(" + number + #")(?:\(([+-]?[0-9]+)\))?(?:/([^\s]+))?"#),
                    let parsed = Double(gainMatch[0] ?? "")
                else { throw WFDBError("wfdb_header_invalid") }
                gain = parsed
                if let value = gainMatch[1] { baseline = try integer(value) }
                unit = gainMatch[2] ?? "mV"
            }
            guard gain.isFinite, gain != 0 else { throw WFDBError("wfdb_gain_invalid") }
            let numeric = try fields.dropFirst(3).prefix(5).map(integer)
            let resolvedBaseline = baseline ?? (numeric.count > 1 ? numeric[1] : 0)
            let checksum: Int? = numeric.count > 3 ? numeric[3] : nil
            if let checksum, !(-32768...32767).contains(checksum) { throw WFDBError("wfdb_header_invalid") }
            if numeric.count > 4, numeric[4] != 0 { throw WFDBError("wfdb_layout_unsupported") }
            let label = fields.count > 8 ? fields[8].trimmingCharacters(in: .whitespacesAndNewlines) : nil
            let spec = Spec(
                group: files.count - 1, format: format, offset: offset,
                gain: gain, baseline: resolvedBaseline, unit: units.contains(unit) ? unit : nil,
                label: label.flatMap { leads.contains($0) ? $0 : nil }, checksum: checksum,
                descriptionWithheld: label.map { !leads.contains($0) } ?? false,
                unitWithheld: !units.contains(unit))
            if let previous = specs.last, previous.group == spec.group,
                previous.format != format || previous.offset != offset
            {
                throw WFDBError("wfdb_signal_group_invalid")
            }
            specs.append(spec)
        }
        return (specs, rate, total, comments, record.count > 4, files.count)
    }

    private static func annotations(
        _ source: any WFDBByteSource, total: Int, start: Int, end: Int,
        limits: WFDBLimits
    ) throws -> WFDBAnnotations {
        let cursor = try Cursor(source, limit: limits.maxAnnotationBytes)
        var count = 0
        var position = 0
        var auxiliary = false
        var positions: [Int] = []
        while true {
            let pair = [UInt8](try cursor.exact(2, "wfdb_annotation_truncated"))
            let word = Int(pair[0]) | (Int(pair[1]) << 8)
            if word == 0 { break }
            let code = word >> 10
            let interval = word & 1023
            switch code {
            case 59:
                guard interval == 0 else { throw WFDBError("wfdb_annotation_invalid") }
                let bytes = [UInt8](try cursor.exact(4, "wfdb_annotation_truncated"))
                let bits = UInt32(bytes[0]) << 16 | UInt32(bytes[1]) << 24 | UInt32(bytes[2]) | UInt32(bytes[3]) << 8
                let (next, overflow) = position.addingReportingOverflow(Int(Int32(bitPattern: bits)))
                guard !overflow else { throw WFDBError("wfdb_annotation_position_invalid") }
                position = next
            case 63:
                guard count > 0 else { throw WFDBError("wfdb_annotation_invalid") }
                auxiliary = auxiliary || interval > 0
                try cursor.discard(interval + (interval & 1), "wfdb_annotation_truncated")
            case 60...62:
                guard count > 0 else { throw WFDBError("wfdb_annotation_invalid") }
            case 1...49:
                let (next, overflow) = position.addingReportingOverflow(interval)
                guard !overflow, next >= 0, next < total else { throw WFDBError("wfdb_annotation_position_invalid") }
                position = next
                count += 1
                guard count <= limits.maxAnnotations else { throw WFDBError("wfdb_annotation_limit_exceeded") }
                if position >= start, position < end { positions.append(position) }
            default: throw WFDBError("wfdb_annotation_format_unsupported")
            }
        }
        return WFDBAnnotations(totalCount: count, samplePositions: positions, auxiliaryTextPresent: auxiliary)
    }

    /// Read a bounded caller-supplied header stream through byte-range access.
    public static func read(
        header: any WFDBByteSource, signals: [any WFDBByteSource],
        startSample: Int = 0, sampleCount: Int? = nil,
        annotations: (any WFDBByteSource)? = nil,
        limits: WFDBLimits = WFDBLimits()
    ) throws -> WFDBRecord {
        try limits.validate()
        let cursor = try Cursor(header, limit: limits.maxHeaderBytes)
        let bytes = try cursor.exact(header.byteCount, "wfdb_header_invalid")
        return try read(
            header: bytes, signals: signals, startSample: startSample,
            sampleCount: sampleCount, annotations: annotations, limits: limits)
    }

    /// Read one immutable integer window; scan declared samples in fixed chunks
    /// to validate truncation and whole-record checksums, including outside the window.
    /// Sources are supplied in first-file-appearance order. Unknown descriptions
    /// and units are withheld. No wall-clock timing or source names are returned.
    public static func read(
        header: Data, signals: [any WFDBByteSource],
        startSample: Int = 0, sampleCount: Int? = nil,
        annotations annotationSource: (any WFDBByteSource)? = nil,
        limits: WFDBLimits = WFDBLimits()
    ) throws -> WFDBRecord {
        try limits.validate()
        guard header.count <= limits.maxHeaderBytes else { throw WFDBError("wfdb_file_limit_exceeded") }
        let (specs, rate, total, comments, timing, groups) = try parse(header, limits)
        guard startSample >= 0, startSample <= total else { throw WFDBError("wfdb_window_invalid") }
        let count = sampleCount ?? (total - startSample)
        guard count >= 0, count <= total - startSample else { throw WFDBError("wfdb_window_invalid") }
        guard count <= limits.maxWindowSamples else { throw WFDBError("wfdb_window_limit_exceeded") }
        guard signals.count == groups else { throw WFDBError("wfdb_source_count_invalid") }
        var outputs: [WFDBSignal] = []
        for group in 0..<groups {
            let members = specs.filter { $0.group == group }
            let first = members[0]
            let cursor = try Cursor(signals[group], limit: limits.maxFileBytes)
            let (scalars, overflow) = total.multipliedReportingOverflow(by: members.count)
            guard !overflow, scalars < Int.max / 3 else { throw WFDBError("wfdb_file_limit_exceeded") }
            let encoded = first.format == 212 ? ((scalars * 3 + 1) / 2) : scalars * (first.format == 16 ? 2 : 1)
            guard first.offset <= limits.maxFileBytes, encoded <= limits.maxFileBytes - first.offset else {
                throw WFDBError("wfdb_file_limit_exceeded")
            }
            guard encoded <= signals[group].byteCount - first.offset else { throw WFDBError("wfdb_signal_truncated") }
            try cursor.discard(first.offset, "wfdb_signal_truncated")
            var windows = Array(repeating: [Int](), count: members.count)
            var sums = Array(repeating: 0, count: members.count)
            var index = 0
            func append(_ value: Int) {
                let frame = index / members.count
                let channel = index % members.count
                sums[channel] = (sums[channel] + value) & 65535
                if frame >= startSample, frame < startSample + count { windows[channel].append(value) }
                index += 1
            }
            while index < scalars {
                let remaining = scalars - index
                if first.format == 212 {
                    if remaining == 1 {
                        let bytes = [UInt8](try cursor.exact(2, "wfdb_signal_truncated"))
                        let value = Int(bytes[0]) | (Int(bytes[1] & 15) << 8)
                        append(value & 2048 != 0 ? value - 4096 : value)
                        continue
                    }
                    let pairs = min(remaining / 2, 8192 / 3)
                    let bytes = [UInt8](try cursor.exact(pairs * 3, "wfdb_signal_truncated"))
                    for offset in stride(from: 0, to: bytes.count, by: 3) {
                        let a = Int(bytes[offset]) | (Int(bytes[offset + 1] & 15) << 8)
                        let b = Int(bytes[offset + 2]) | (Int(bytes[offset + 1] & 240) << 4)
                        append(a & 2048 != 0 ? a - 4096 : a)
                        if index < scalars { append(b & 2048 != 0 ? b - 4096 : b) }
                    }
                } else {
                    let width = first.format == 16 ? 2 : 1
                    let bytes = [UInt8](try cursor.exact(min(remaining, 8192 / width) * width, "wfdb_signal_truncated"))
                    for offset in stride(from: 0, to: bytes.count, by: width) {
                        if width == 1 {
                            append(Int(bytes[offset]) - 128)
                        } else {
                            let bits = UInt16(bytes[offset]) | UInt16(bytes[offset + 1]) << 8
                            append(Int(Int16(bitPattern: bits)))
                        }
                    }
                }
            }
            for channel in members.indices {
                let spec = members[channel]
                if let checksum = spec.checksum, checksum & 65535 != sums[channel] {
                    throw WFDBError("wfdb_checksum_mismatch")
                }
                outputs.append(
                    WFDBSignal(
                        leadLabel: spec.label, gain: spec.gain, baseline: spec.baseline,
                        unit: spec.unit, formatCode: spec.format, samples: windows[channel], checksumVerified: spec.checksum != nil))
            }
        }
        let annotationResult = try annotationSource.map {
            try annotations($0, total: total, start: startSample, end: startSample + count, limits: limits)
        }
        return WFDBRecord(
            signals: outputs, sampleRateHz: rate, totalSamples: total, startSample: startSample,
            sampleCount: count, annotations: annotationResult, commentsPresent: comments,
            recordNamePresent: true, pathFieldsPresent: true, timingFieldsPresent: timing,
            descriptionsWithheld: specs.contains { $0.descriptionWithheld }, unitsWithheld: specs.contains { $0.unitWithheld })
    }
}
