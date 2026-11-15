import Foundation

/// Controlled errors never echo samples, provenance or rejected input.
public enum PhysiologicalWaveformError: String, Error, LocalizedError, Sendable {
    case unknownKind = "unknown_kind"
    case invalidUnit = "invalid_unit"
    case invalidSamplingRate = "invalid_sampling_rate"
    case invalidAcquisitionRange = "invalid_acquisition_range"
    case invalidProvenance = "invalid_provenance"
    case invalidSampleShape = "invalid_sample_shape"
    case invalidSample = "invalid_sample"
    case invalidTiming = "invalid_timing"
    case invalidRecording = "invalid_recording"
    case resourceLimit = "resource_limit"
    case reviewerConfirmationRequired = "reviewer_confirmation_required"

    public var errorDescription: String? { rawValue }
}

/// Non-ECG acquisition semantics, never lead identities or derived vital signs.
public enum PhysiologicalChannelKind: String, CaseIterable, Sendable {
    case ppg
    case respiration
    case invasivePressure = "invasive_pressure"
    case nonInvasivePressure = "non_invasive_pressure"
    case capnography

    /// Version-one encoding bounds, not clinical reference intervals.
    public var constraints: PhysiologicalChannelConstraints {
        switch self {
        case .ppg:
            return .init(unit: "normalized", minimum: 0, maximum: 1, minimumRateHz: 10, maximumRateHz: 2000, motionStepFraction: 0.4)
        case .respiration:
            return .init(unit: "normalized", minimum: -1, maximum: 1, minimumRateHz: 1, maximumRateHz: 200, motionStepFraction: 0.6)
        case .invasivePressure:
            return .init(unit: "mmHg", minimum: -50, maximum: 400, minimumRateHz: 10, maximumRateHz: 2000, motionStepFraction: 0.3)
        case .nonInvasivePressure:
            return .init(unit: "mmHg", minimum: 0, maximum: 400, minimumRateHz: 1, maximumRateHz: 1000, motionStepFraction: 0.3)
        case .capnography:
            return .init(unit: "mmHg", minimum: 0, maximum: 150, minimumRateHz: 1, maximumRateHz: 500, motionStepFraction: 0.5)
        }
    }
}

/// Encoding constraints shared with Python; no device qualification is implied.
public struct PhysiologicalChannelConstraints: Sendable {
    public let unit: String
    public let minimum: Double
    public let maximum: Double
    public let minimumRateHz: Double
    public let maximumRateHz: Double
    public let motionStepFraction: Double
}

/// Immutable protected samples. Description excludes values, clocks and digest.
/// Samples are sensitive, are not de-identified, and have no serialization API.
public struct PhysiologicalWaveformChannel: Sendable, CustomStringConvertible, CustomDebugStringConvertible {
    public static let nonDiagnosticNotice =
        "Acquisition quality only; not vital-sign measurements, diagnosis or alarms. Consequential use requires explicit reviewer confirmation."
    public let kind: PhysiologicalChannelKind
    public let unit: String
    public let sampleRateHz: Double
    public let acquisitionMinimum: Double
    public let acquisitionMaximum: Double
    public let samples: [Double?]
    public let offsetsSeconds: [Double]
    public let sourceSHA256: String

    public var description: String { "PhysiologicalWaveformChannel(kind: \(kind.rawValue), sample_count: \(samples.count))" }
    public var debugDescription: String { description }

    /// Validate exact units, declared rails, source-digest syntax and relative timing.
    /// Nil samples represent dropout on the regular clock; no conversion occurs.
    public init(
        kind: String, unit: String, sampleRateHz: Double,
        acquisitionMinimum: Double, acquisitionMaximum: Double,
        samples: [Double?], offsetsSeconds: [Double], sourceSHA256: String
    ) throws {
        guard let channelKind = PhysiologicalChannelKind(rawValue: kind) else {
            throw PhysiologicalWaveformError.unknownKind
        }
        let limits = channelKind.constraints
        guard unit == limits.unit else { throw PhysiologicalWaveformError.invalidUnit }
        guard sampleRateHz.isFinite, sampleRateHz >= limits.minimumRateHz, sampleRateHz <= limits.maximumRateHz else {
            throw PhysiologicalWaveformError.invalidSamplingRate
        }
        guard acquisitionMinimum.isFinite, acquisitionMaximum.isFinite,
            limits.minimum <= acquisitionMinimum, acquisitionMinimum < acquisitionMaximum, acquisitionMaximum <= limits.maximum
        else {
            throw PhysiologicalWaveformError.invalidAcquisitionRange
        }
        guard sourceSHA256.utf8.count == 64,
            sourceSHA256.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) })
        else {
            throw PhysiologicalWaveformError.invalidProvenance
        }
        guard (2...1_000_000).contains(samples.count), samples.count == offsetsSeconds.count else {
            throw PhysiologicalWaveformError.invalidSampleShape
        }
        for value in samples.compactMap({ $0 }) {
            guard value.isFinite, acquisitionMinimum <= value, value <= acquisitionMaximum else {
                throw PhysiologicalWaveformError.invalidSample
            }
        }
        for (index, offset) in offsetsSeconds.enumerated() {
            guard offset.isFinite, offset >= 0 else { throw PhysiologicalWaveformError.invalidTiming }
            if index > 0 {
                let step = offset - offsetsSeconds[index - 1]
                guard step > 0, abs(step * sampleRateHz - 1) <= 0.01 else { throw PhysiologicalWaveformError.invalidTiming }
            }
        }
        self.kind = channelKind
        self.unit = unit
        self.sampleRateHz = sampleRateHz
        self.acquisitionMinimum = acquisitionMinimum
        self.acquisitionMaximum = acquisitionMaximum
        self.samples = samples
        self.offsetsSeconds = offsetsSeconds
        self.sourceSHA256 = sourceSHA256
    }
}

/// Acquisition usability only, not patient condition.
public enum PhysiologicalQualityState: String, Sendable {
    case pass
    case limitedUse = "limited_use"
    case review
}

/// Counts and controlled codes for one channel in input order.
public struct PhysiologicalChannelQuality: Sendable {
    public let kind: PhysiologicalChannelKind
    public let state: PhysiologicalQualityState
    public let sampleCount: Int
    public let dropoutCount: Int
    public let saturationCount: Int
    public let flatlineCount: Int
    public let motionProxyCount: Int
    public let codes: [String]

    fileprivate var report: [String: Any] {
        [
            "kind": kind.rawValue, "state": state.rawValue, "sample_count": sampleCount,
            "dropout_count": dropoutCount, "saturation_count": saturationCount,
            "flatline_count": flatlineCount, "motion_proxy_count": motionProxyCount, "codes": codes,
        ]
    }
}

/// Mixed kinds retain independent clocks. No resampling or alignment is claimed.
public struct PhysiologicalRecordingQuality: Sendable {
    public let channels: [PhysiologicalChannelQuality]

    /// Mandatory boundary bound to every quality output.
    public var nonDiagnosticNotice: String { PhysiologicalWaveformChannel.nonDiagnosticNotice }

    /// Deterministic metadata-only JSON; excludes all values, clocks and provenance.
    public func reportJSON() throws -> Data {
        try JSONSerialization.data(
            withJSONObject: [
                "channel_count": channels.count,
                "codes": ["non_diagnostic", "reviewer_confirmation_required"],
                "channels": channels.map { $0.report },
            ], options: [.sortedKeys])
    }

    /// Require explicit review before consequential use, including passing signals.
    /// Confirmation does not qualify clinical use or authorize automated decisions.
    public func requireReviewerConfirmation(confirmed: Bool = false) throws {
        guard confirmed else { throw PhysiologicalWaveformError.reviewerConfirmationRequired }
    }

    /// Assess one to 64 regular channels with at most one million total samples.
    /// All computation is local and deterministic; no providers or models are used.
    public static func evaluate(_ channels: [PhysiologicalWaveformChannel]) throws -> Self {
        guard (1...64).contains(channels.count) else { throw PhysiologicalWaveformError.invalidRecording }
        guard channels.reduce(0, { $0 + $1.samples.count }) <= 1_000_000 else { throw PhysiologicalWaveformError.resourceLimit }
        return Self(
            channels: channels.map { channel in
                let values = channel.samples
                let span = channel.acquisitionMaximum - channel.acquisitionMinimum
                let dropout = values.filter { $0 == nil }.count
                let saturation = values.compactMap { $0 }.filter {
                    $0 == channel.acquisitionMinimum || $0 == channel.acquisitionMaximum
                }.count
                var flatline = 0
                var motion = 0
                var run = 0
                let minimumRun = Int(ceil(channel.sampleRateHz))
                for (previous, current) in zip(values, values.dropFirst()) {
                    if let previous, let current {
                        let step = abs(current - previous)
                        if step > span * channel.kind.constraints.motionStepFraction { motion += 1 }
                        if step <= span * 1e-6 {
                            run += 1
                            continue
                        }
                    }
                    if run >= minimumRun { flatline += run }
                    run = 0
                }
                if run >= minimumRun { flatline += run }
                let codes = [("dropout", dropout), ("saturation", saturation), ("flatline", flatline), ("motion_proxy", motion)]
                    .filter { $0.1 > 0 }.map { $0.0 }
                var state: PhysiologicalQualityState = codes.isEmpty ? .pass : .limitedUse
                if flatline > 0 || dropout * 10 >= values.count || saturation * 10 >= values.count || motion * 10 >= values.count - 1 {
                    state = .review
                }
                return PhysiologicalChannelQuality(
                    kind: channel.kind, state: state, sampleCount: values.count,
                    dropoutCount: dropout, saturationCount: saturation,
                    flatlineCount: flatline, motionProxyCount: motion, codes: codes)
            })
    }
}
