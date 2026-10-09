import Foundation

/// A controlled EDF failure code. Source bytes and underlying stream errors are withheld.
public struct EDFError: Error, CustomStringConvertible {
    public let code: String
    public var description: String { code }
    fileprivate init(_ code: String) { self.code = code }
}

/// Resource budgets applied before decoding and retaining window samples.
public struct EDFLimits {
    public var maxBytes = 64 * 1024 * 1024
    public var maxSignals = 64
    public var maxRecords = 100_000
    public var maxRecordBytes = 1024 * 1024
    public var maxDurationSeconds = 86_400
    public var maxWindowSeconds = 3600
    public var maxOutputSamples = 1_000_000
    public var maxAnnotationLists = 100_000
    public init() {}
}

/// Header presence and placeholder status, without an anonymization attestation.
public struct EDFIdentityStatus {
    public let present: Bool
    public let anonymizationStatus: String
}

/// Controlled technical labels, ADC ranges and sampling frequency.
public struct EDFSignal {
    public let index: Int
    public let label: String
    public let physicalDimension: String
    public let physicalMinimum: Double
    public let physicalMaximum: Double
    public let digitalMinimum: Int
    public let digitalMaximum: Int
    public let samplesPerRecord: Int
    public let samplingRateHz: Double?
}

/// Samples in one record. No interpolation across gaps is performed.
public struct EDFSignalWindow {
    public let signalIndex: Int
    public let firstSampleIndex: Int
    public let digitalSamples: [Int16]
    public let physicalSamples: [Double]
}

/// A record's offset from the withheld header start second.
public struct EDFRecordWindow {
    public let recordIndex: Int
    public let onsetSeconds: Double
    public let durationSeconds: Double
    public let signals: [EDFSignalWindow]
}

/// One annotation list's offsets and nonempty annotation count; text is withheld.
public struct EDFAnnotation {
    public let recordIndex: Int
    public let signalIndex: Int
    public let onsetSeconds: Double
    public let durationSeconds: Double?
    public let count: Int
}

/// An explicit missing acquisition interval before a discontinuous record.
public struct EDFGap {
    public let beforeRecordIndex: Int
    public let startSeconds: Double
    public let endSeconds: Double
}

/// Non-diagnostic window output carrying a notice and explicit review state.
public struct EDFRecording {
    public let format: String
    public let patient: EDFIdentityStatus
    public let recording: EDFIdentityStatus
    public let signals: [EDFSignal]
    public let recordCount: Int
    public let recordOnsetsSeconds: [Double]
    public let recordDurationSeconds: Double
    public let windowStartSeconds: Double
    public let windowEndSeconds: Double
    public let records: [EDFRecordWindow]
    public let annotations: [EDFAnnotation]
    public let gaps: [EDFGap]
    public static let notice = "Non-diagnostic signals for review only. Explicit reviewer confirmation is required before consequential use. Header withholding does not de-identify waveform samples or establish clinical validity."
    public var notice: String { Self.notice }
    public private(set) var reviewerConfirmed = false

    /// Bind human confirmation before consequential handoff.
    public func reviewed(confirmed: Bool) throws -> EDFRecording {
        guard confirmed else { throw EDFError("edf_review_required") }
        var result = self
        result.reviewerConfirmed = true
        return result
    }

    /// Report controlled counts and statuses, omitting headers, samples and labels.
    public func report() -> [String: Any] {
        [
            "format": format, "signal_count": signals.count, "record_count": recordCount,
            "window_record_count": records.count,
            "annotation_count": annotations.reduce(0) { $0 + $1.count },
            "gap_count": gaps.count,
            "patient": ["present": patient.present, "anonymization_status": patient.anonymizationStatus],
            "recording": ["present": recording.present, "anonymization_status": recording.anonymizationStatus],
            "notice": notice, "review_required": !reviewerConfirmed,
        ]
    }
}

/// Dependency-free on-device EDF/EDF+ parsing. Performs no diagnosis, network or file writes.
public enum EDFReader {
    private static let labels: Set<String> = Set(
        ["ECG", "EEG", "EMG", "EOG", "Temp rectal", "Body temp", "SaO2", "SpO2", "EEG Fpz-Cz", "EEG Pz-Oz"]
            + ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"].map { "ECG \($0)" }
    )
    private static let units: Set<String> = ["V", "mV", "uV", "nV", "degreeC", "%", "Ohm", "mmHg"]

    /// Read bounded in-memory bytes, retaining only the requested half-open sample window.
    public static func read(_ data: Data, startSeconds: Double = 0, endSeconds: Double? = nil, limits: EDFLimits = EDFLimits()) throws -> EDFRecording {
        try validate(limits)
        guard data.count <= limits.maxBytes else { throw EDFError("edf_byte_limit") }
        var position = data.startIndex
        let reader = ByteReader(limit: limits.maxBytes) { size in
            let count = min(size, data.endIndex - position)
            let bytes = Array(data[position..<position + count])
            position += count
            return bytes
        }
        return try parse(reader, startSeconds: startSeconds, endSeconds: endSeconds, limits: limits)
    }

    /// Consume a caller-opened stream from its current position; never open or close it.
    public static func read(_ stream: InputStream, startSeconds: Double = 0, endSeconds: Double? = nil, limits: EDFLimits = EDFLimits()) throws -> EDFRecording {
        try validate(limits)
        let reader = ByteReader(limit: limits.maxBytes) { size in
            var buffer = [UInt8](repeating: 0, count: size)
            let count = stream.read(&buffer, maxLength: size)
            guard count >= 0 else { throw EDFError("edf_stream_read_error") }
            return Array(buffer.prefix(count))
        }
        return try parse(reader, startSeconds: startSeconds, endSeconds: endSeconds, limits: limits)
    }

    private static func validate(_ limits: EDFLimits) throws {
        guard [limits.maxBytes, limits.maxSignals, limits.maxRecords, limits.maxRecordBytes, limits.maxDurationSeconds, limits.maxWindowSeconds, limits.maxOutputSamples, limits.maxAnnotationLists].allSatisfy({ $0 > 0 }) else {
            throw EDFError("edf_limits_invalid")
        }
    }

    private final class ByteReader {
        let limit: Int
        let readChunk: (Int) throws -> [UInt8]
        var count = 0
        var pending: UInt8?
        init(limit: Int, read: @escaping (Int) throws -> [UInt8]) {
            self.limit = limit
            self.readChunk = read
        }
        func read(_ size: Int) throws -> [UInt8] {
            guard size - (pending == nil ? 0 : 1) <= limit - count else { throw EDFError("edf_byte_limit") }
            var bytes: [UInt8] = []
            if let byte = pending {
                bytes.append(byte)
                pending = nil
            }
            while bytes.count < size {
                let chunk = try readChunk(min(65536, size - bytes.count))
                guard !chunk.isEmpty else { throw EDFError("edf_truncated") }
                count += chunk.count
                bytes.append(contentsOf: chunk)
            }
            return bytes
        }
        func atEnd() throws -> Bool {
            if pending != nil { return false }
            let chunk = try readChunk(1)
            if let byte = chunk.first {
                // One EOF probe may exceed the budget; no record will be decoded then.
                guard count < Int.max else { throw EDFError("edf_byte_limit") }
                count += 1
                pending = byte
                return false
            }
            return true
        }
    }

    private static func matches(_ text: String, _ pattern: String) -> Bool {
        text.range(of: "^(?:" + pattern + ")$", options: .regularExpression) != nil
    }
    private static func text(_ bytes: [UInt8]) -> String {
        String(bytes: bytes, encoding: .ascii)?.trimmingCharacters(in: CharacterSet(charactersIn: " ")) ?? ""
    }
    private static func integer(_ bytes: [UInt8]) throws -> Int {
        let value = text(bytes)
        guard matches(value, "-?[0-9]+"), let number = Int(value) else { throw EDFError("edf_header_invalid") }
        return number
    }
    private static func number(_ bytes: [UInt8], pattern: String = "[+-]?(?:[0-9]+(?:\\.[0-9]*)?|\\.[0-9]+)(?:[Ee][+-]?[0-9]+)?") throws -> Decimal {
        let value = pattern == "[+-][0-9]+(?:\\.[0-9]+)?" || pattern == "[0-9]+(?:\\.[0-9]+)?" ? (String(bytes: bytes, encoding: .ascii) ?? "") : text(bytes)
        guard value.count <= 28, matches(value, pattern) else { throw EDFError("edf_numeric_invalid") }
        if let marker = value.lowercased().firstIndex(of: "e") {
            guard let exponent = Int(value[value.index(after: marker)...]), (-12...12).contains(exponent) else { throw EDFError("edf_numeric_invalid") }
        }
        guard let decimal = Decimal(string: value, locale: Locale(identifier: "en_US_POSIX")), !decimal.isNaN else { throw EDFError("edf_numeric_invalid") }
        return decimal
    }
    private static func double(_ value: Decimal) -> Double { NSDecimalNumber(decimal: value).doubleValue }
    private static func identity(_ bytes: [UInt8], placeholder: String) -> EDFIdentityStatus {
        let value = text(bytes)
        return EDFIdentityStatus(present: !value.isEmpty, anonymizationStatus: value.isEmpty ? "absent" : value == placeholder ? "placeholder_only" : "not_verified")
    }
    private static func validDate(_ date: [UInt8], _ time: [UInt8]) -> Bool {
        guard let date = String(bytes: date, encoding: .ascii), let time = String(bytes: time, encoding: .ascii), matches(date, "[0-9]{2}\\.[0-9]{2}\\.(?:[0-9]{2}|yy)"), matches(time, "[0-9]{2}\\.[0-9]{2}\\.[0-9]{2}") else { return false }
        let d = date.split(separator: ".").map(String.init)
        let t = time.split(separator: ".").compactMap { Int($0) }
        guard let day = Int(d[0]), let month = Int(d[1]), (1...12).contains(month) else { return false }
        let y = Int(d[2]) ?? 0
        let year = y + (y >= 85 ? 1900 : 2000)
        let leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)
        let days = [31, leap ? 29 : 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        return (1...days[month - 1]).contains(day) && t[0] < 24 && t[1] < 60 && t[2] < 60
    }

    private struct TAL {
        let onset: Decimal
        let duration: Decimal?
        let count: Int
        let firstEmpty: Bool
    }
    private static func annotations(_ data: [UInt8], maxDuration: Int, maxLists: Int) throws -> [TAL] {
        var results: [TAL] = []
        var cursor = 0
        while cursor < data.count && data[cursor] != 0 {
            guard let stop = data[cursor...].firstIndex(of: 0) else { throw EDFError("edf_annotation_invalid") }
            let parts = data[cursor..<stop].split(separator: 20, omittingEmptySubsequences: false)
            guard parts.count >= 3, parts.last!.isEmpty else { throw EDFError("edf_annotation_invalid") }
            let timing = parts[0].split(separator: 21, omittingEmptySubsequences: false)
            guard timing.count <= 2 else { throw EDFError("edf_annotation_invalid") }
            let onset = try number(Array(timing[0]), pattern: "[+-][0-9]+(?:\\.[0-9]+)?")
            let duration = timing.count == 2 ? try number(Array(timing[1]), pattern: "[0-9]+(?:\\.[0-9]+)?") : nil
            guard abs(onset) <= Decimal(maxDuration), (duration ?? 0) <= Decimal(maxDuration) else { throw EDFError("edf_duration_limit") }
            var count = 0
            for annotation in parts.dropFirst().dropLast() {
                guard String(bytes: annotation, encoding: .utf8) != nil, annotation.allSatisfy({ $0 >= 32 || [9, 10, 13].contains($0) }) else { throw EDFError("edf_annotation_invalid") }
                if !annotation.isEmpty { count += 1 }
            }
            guard results.count < maxLists else { throw EDFError("edf_annotation_limit") }
            results.append(TAL(onset: onset, duration: duration, count: count, firstEmpty: parts[1].isEmpty))
            cursor = stop + 1
        }
        guard data[cursor...].allSatisfy({ $0 == 0 }) else { throw EDFError("edf_annotation_invalid") }
        return results
    }
    private static func sampleBoundary(_ value: Decimal, count: Int) -> Int {
        var value = max(0, min(Decimal(count), value))
        var rounded = Decimal()
        NSDecimalRound(&rounded, &value, 0, .up)
        return NSDecimalNumber(decimal: rounded).intValue
    }

    private static func parse(_ reader: ByteReader, startSeconds: Double, endSeconds: Double?, limits: EDFLimits) throws -> EDFRecording {
        guard startSeconds.isFinite, startSeconds >= 0, startSeconds <= 1e12, endSeconds == nil || (endSeconds!.isFinite && endSeconds! > startSeconds && endSeconds! - startSeconds <= Double(limits.maxWindowSeconds) && endSeconds! <= 1e12) else { throw EDFError("edf_window_invalid") }
        let start = Decimal(string: String(startSeconds))!
        var end = endSeconds.map { Decimal(string: String($0))! } ?? start + Decimal(limits.maxWindowSeconds)
        let header = try reader.read(256)
        guard header.allSatisfy({ (32...126).contains($0) }), Array(header[0..<8]) == Array("0       ".utf8), validDate(Array(header[168..<176]), Array(header[176..<184])) else { throw EDFError("edf_header_invalid") }
        let ns = try integer(Array(header[252..<256]))
        guard ns >= 1 && ns <= limits.maxSignals else { throw EDFError("edf_signal_limit") }
        let headerSize = try integer(Array(header[184..<192]))
        guard headerSize == 256 * (ns + 1) else { throw EDFError("edf_header_size_invalid") }
        let declared = try integer(Array(header[236..<244]))
        guard declared >= -1 && declared != 0 else { throw EDFError("edf_record_count_invalid") }
        guard declared <= limits.maxRecords else { throw EDFError("edf_record_limit") }
        let duration = try number(Array(header[244..<252]))
        guard duration >= 0 && duration <= Decimal(limits.maxDurationSeconds) && duration * Decimal(max(0, declared)) <= Decimal(limits.maxDurationSeconds) else { throw EDFError("edf_duration_limit") }
        let kind = text(Array(header[192..<197]))
        let format = ["EDF+C", "EDF+D"].contains(kind) ? kind : "EDF"
        let fields = try reader.read(headerSize - 256)
        guard fields.allSatisfy({ (32...126).contains($0) }) else { throw EDFError("edf_header_invalid") }
        var columns: [[[UInt8]]] = []
        var cursor = 0
        for width in [16, 80, 8, 8, 8, 8, 8, 80, 8, 32] {
            columns.append((0..<ns).map { Array(fields[cursor + $0 * width..<cursor + ($0 + 1) * width]) })
            cursor += width * ns
        }
        var signals: [EDFSignal] = []
        var sizes: [Int] = []
        var annotationChannels: [Int] = []
        for index in 0..<ns {
            let label = text(columns[0][index])
            let unit = text(columns[2][index])
            let pmin = try number(columns[3][index])
            let pmax = try number(columns[4][index])
            let dmin = try integer(columns[5][index])
            let dmax = try integer(columns[6][index])
            let size = try integer(columns[8][index])
            guard pmin != pmax && dmin >= -32768 && dmin < dmax && dmax <= 32767 && abs(pmin) <= Decimal(1e12) && abs(pmax) <= Decimal(1e12) else { throw EDFError("edf_range_invalid") }
            guard size > 0 else { throw EDFError("edf_samples_invalid") }
            sizes.append(size)
            if label == "EDF Annotations" {
                guard format != "EDF" && dmin == -32768 && dmax == 32767 else { throw EDFError("edf_annotation_header_invalid") }
                annotationChannels.append(index)
            } else {
                guard duration != 0 || (format == "EDF+D" && size == 1) else { throw EDFError("edf_duration_invalid") }
                signals.append(EDFSignal(index: index, label: labels.contains(label) ? label : "withheld", physicalDimension: units.contains(unit) ? unit : "withheld", physicalMinimum: double(pmin), physicalMaximum: double(pmax), digitalMinimum: dmin, digitalMaximum: dmax, samplesPerRecord: size, samplingRateHz: duration == 0 ? nil : double(Decimal(size) / duration)))
            }
        }
        guard format == "EDF" || !annotationChannels.isEmpty else { throw EDFError("edf_annotation_channel_missing") }
        let recordBytes = sizes.reduce(0, +) * 2
        guard recordBytes <= limits.maxRecordBytes else { throw EDFError("edf_record_byte_limit") }
        guard declared <= 0 || recordBytes <= (limits.maxBytes - reader.count) / declared else { throw EDFError("edf_byte_limit") }
        var records: [EDFRecordWindow] = []
        var onsets: [Double] = []
        var events: [(EDFAnnotation, Decimal, Decimal?)] = []
        var gaps: [EDFGap] = []
        var outputCount = 0
        var talCount = 0
        var previousEnd = Decimal(0)
        while declared == -1 || onsets.count < declared {
            if try reader.atEnd() {
                guard declared == -1 else { throw EDFError("edf_record_count_invalid") }
                break
            }
            guard onsets.count < limits.maxRecords else { throw EDFError("edf_record_limit") }
            let data = try reader.read(recordBytes)
            let recordIndex = onsets.count
            var onset = duration * Decimal(recordIndex)
            var channels: [[UInt8]] = []
            cursor = 0
            for size in sizes {
                channels.append(Array(data[cursor..<cursor + size * 2]))
                cursor += size * 2
            }
            for channel in annotationChannels {
                let tals = try annotations(channels[channel], maxDuration: limits.maxDurationSeconds, maxLists: limits.maxAnnotationLists - talCount)
                talCount += tals.count
                if channel == annotationChannels[0] {
                    guard let first = tals.first, first.duration == nil, first.firstEmpty else { throw EDFError("edf_timekeeping_invalid") }
                    onset = first.onset
                }
                for tal in tals where tal.count > 0 {
                    if (start <= tal.onset && tal.onset < end) || ((tal.duration ?? 0) > 0 && tal.onset < end && tal.onset + tal.duration! > start) {
                        events.append((EDFAnnotation(recordIndex: recordIndex, signalIndex: channel, onsetSeconds: double(tal.onset), durationSeconds: tal.duration.map(double), count: tal.count), tal.onset, tal.duration))
                    }
                }
            }
            guard recordIndex != 0 || (onset >= 0 && onset < 1) else { throw EDFError("edf_timekeeping_invalid") }
            guard recordIndex == 0 || (onset >= previousEnd && (format == "EDF+D" || onset == previousEnd)) else { throw EDFError("edf_record_timing_invalid") }
            guard onset + duration <= Decimal(limits.maxDurationSeconds) else { throw EDFError("edf_duration_limit") }
            if recordIndex > 0 && onset > previousEnd {
                gaps.append(EDFGap(beforeRecordIndex: recordIndex, startSeconds: double(previousEnd), endSeconds: double(onset)))
            }
            previousEnd = onset + duration
            onsets.append(double(onset))
            if (duration > 0 && onset < end && onset + duration > start) || (duration == 0 && start <= onset && onset < end) {
                var windows: [EDFSignalWindow] = []
                for signal in signals {
                    let first = duration == 0 ? 0 : sampleBoundary((start - onset) * Decimal(signal.samplesPerRecord) / duration, count: signal.samplesPerRecord)
                    let stop = duration == 0 ? 1 : sampleBoundary((end - onset) * Decimal(signal.samplesPerRecord) / duration, count: signal.samplesPerRecord)
                    guard stop - first <= limits.maxOutputSamples - outputCount else { throw EDFError("edf_output_sample_limit") }
                    outputCount += stop - first
                    let bytes = channels[signal.index]
                    let digital = (first..<stop).map { Int16(bitPattern: UInt16(bytes[2 * $0]) | UInt16(bytes[2 * $0 + 1]) << 8) }
                    guard digital.allSatisfy({ Int($0) >= signal.digitalMinimum && Int($0) <= signal.digitalMaximum }) else { throw EDFError("edf_sample_range_invalid") }
                    let scale = (signal.physicalMaximum - signal.physicalMinimum) / Double(signal.digitalMaximum - signal.digitalMinimum)
                    let physical = digital.map { signal.physicalMinimum + Double(Int($0) - signal.digitalMinimum) * scale }
                    windows.append(EDFSignalWindow(signalIndex: signal.index, firstSampleIndex: first, digitalSamples: digital, physicalSamples: physical))
                }
                records.append(EDFRecordWindow(recordIndex: recordIndex, onsetSeconds: double(onset), durationSeconds: double(duration), signals: windows))
            }
        }
        guard !onsets.isEmpty, try reader.atEnd() else { throw EDFError("edf_record_count_invalid") }
        if endSeconds == nil {
            end = previousEnd + (duration == 0 ? 1 : 0)
            guard end > start && end - start <= Decimal(limits.maxWindowSeconds) else { throw EDFError("edf_window_invalid") }
        }
        return EDFRecording(
            format: format, patient: identity(Array(header[8..<88]), placeholder: "X X X X"), recording: identity(Array(header[88..<168]), placeholder: "Startdate X X X X"), signals: signals, recordCount: onsets.count, recordOnsetsSeconds: onsets, recordDurationSeconds: double(duration), windowStartSeconds: double(start), windowEndSeconds: double(end), records: records,
            annotations: events.filter {
                (start <= $0.1 && $0.1 < end) || (($0.2 ?? 0) > 0 && $0.1 < end && $0.1 + $0.2! > start)
            }.map { $0.0 }, gaps: gaps)
    }
}
