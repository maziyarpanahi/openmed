import Foundation

/// Controlled failures; messages never carry sample values or caller diagnostics.
public enum AudioSampleConversionError: String, Error {
    case invalidContract, incompatibleProfile, budgetExceeded, invalidSamples
    case clipping, cancelled, reviewRequired, closed
}

/// An explicit channel policy for local sample conversion.
public enum AudioConversionChannelPolicy: String {
    case preserve, meanMono
}

/// The integer-exact frame plan consumed by the on-device adapter.
public struct AudioSampleConversionPlan {
    public enum Rounding { case floor, nearestEven, ceiling }
    public let sourceRate: Int
    public let targetRate: Int
    public let sourceFrames: Int
    public let targetFrames: Int
    public let rounding: Rounding

    /// Construct an overflow-checked frame plan with explicit rounding.
    public init(sourceRate: Int, targetRate: Int, sourceFrames: Int, rounding: Rounding = .nearestEven) throws {
        guard (1...768_000).contains(sourceRate), (1...768_000).contains(targetRate),
            sourceFrames > 0, sourceFrames <= Int(UInt32.max)
        else { throw AudioSampleConversionError.invalidContract }
        let (product, overflow) = sourceFrames.multipliedReportingOverflow(by: targetRate)
        guard !overflow else { throw AudioSampleConversionError.invalidContract }
        var frames = product / sourceRate
        let remainder = product % sourceRate
        switch rounding {
        case .floor: break
        case .ceiling: if remainder > 0 { frames += 1 }
        case .nearestEven:
            if remainder * 2 > sourceRate || (remainder * 2 == sourceRate && frames % 2 != 0) { frames += 1 }
        }
        self.sourceRate = sourceRate
        self.targetRate = targetRate
        self.sourceFrames = sourceFrames
        self.targetFrames = frames
        self.rounding = rounding
    }
}

/// Local PCM16 provider input limits; no inference or cloud transport is included.
public struct AudioSampleConversionProfile {
    public let rates: Set<Int>
    public let channels: Set<Int>
    public let minDuration: Double
    public let maxDuration: Double
    public let allowResample: Bool
    public let allowDownmix: Bool

    /// Declare accepted output rates, channel counts and duration bounds.
    public init(
        rates: Set<Int>, channels: Set<Int>, minDuration: Double = 0,
        maxDuration: Double = 3600, allowResample: Bool = true, allowDownmix: Bool = true
    ) throws {
        guard !rates.isEmpty, rates.allSatisfy({ (1...768_000).contains($0) }),
            !channels.isEmpty, channels.allSatisfy({ (1...64).contains($0) }),
            minDuration.isFinite, maxDuration.isFinite, minDuration >= 0,
            maxDuration >= minDuration, maxDuration <= 86_400
        else { throw AudioSampleConversionError.invalidContract }
        self.rates = rates
        self.channels = channels
        self.minDuration = minDuration
        self.maxDuration = maxDuration
        self.allowResample = allowResample
        self.allowDownmix = allowDownmix
    }
}

/// Audio-free lineage. Rational frame boundaries remain exact until presentation.
public struct AudioSampleConversionReport {
    public let plan: AudioSampleConversionPlan
    public let sourceChannels: Int
    public let targetChannels: Int
    public let channelPolicy: AudioConversionChannelPolicy
    public let filterIdentity = "hann-sinc-32-zero-pad-v1"
    public let notice = "Converted audio is non-diagnostic. A reviewer must confirm provider handoff; transcripts require separate review before consequential use."

    /// Return an exact source-frame numerator and denominator for an output boundary.
    public func sourcePosition(targetFrame: Int) throws -> (numerator: Int, denominator: Int) {
        guard targetFrame >= 0, targetFrame <= plan.targetFrames else {
            throw AudioSampleConversionError.invalidContract
        }
        return (min(targetFrame * plan.sourceRate, plan.sourceFrames * plan.targetRate), plan.targetRate)
    }
}

/// Protected PCM16 output. Caller-owned copies must be disposed of by the caller.
public final class ConvertedAudioSamples: CustomStringConvertible {
    public let report: AudioSampleConversionReport
    private var samples: [Int16]?
    public var description: String { "ConvertedAudioSamples(protected_samples)" }

    fileprivate init(report: AudioSampleConversionReport, samples: [Int16]) {
        self.report = report
        self.samples = samples
    }

    /// Copy interleaved PCM16 samples after explicit reviewer confirmation.
    public func samplesForReviewedHandoff(reviewerConfirmed: Bool) throws -> [Int16] {
        guard reviewerConfirmed else { throw AudioSampleConversionError.reviewRequired }
        guard let samples else { throw AudioSampleConversionError.closed }
        return samples
    }

    /// Clear owned output; Swift copies already returned remain caller-owned.
    public func close() {
        if samples != nil {
            for index in samples!.indices { samples![index] = 0 }
            samples = nil
        }
    }

    deinit { close() }
}

/// Bounded memory-only, on-device conversion using system Foundation math.
public enum LocalAudioSampleConversion {
    /// Convert caller-decoded normalized PCM16 source chunks with a centered Hann-sinc filter.
    ///
    /// Chunks contain interleaved normalized Double samples. Source metadata is PCM16;
    /// decoding, ASR, capture and consequential transcript use remain separate.
    public static func convert(
        chunks: [[Double]], sourceChannels: Int, plan: AudioSampleConversionPlan,
        profile: AudioSampleConversionProfile, channelPolicy: AudioConversionChannelPolicy,
        maxBufferBytes: Int = 32 * 1024 * 1024, maxChunkFrames: Int = 8192,
        maxFilterOperations: Int = 100_000_000, cancelled: () -> Bool = { false }
    ) throws -> ConvertedAudioSamples {
        guard (1...64).contains(sourceChannels), maxBufferBytes > 0, maxChunkFrames > 0,
            maxFilterOperations > 0
        else { throw AudioSampleConversionError.invalidContract }
        let channels = channelPolicy == .preserve ? sourceChannels : 1
        let sourceDuration = Double(plan.sourceFrames) / Double(plan.sourceRate)
        let duration = Double(plan.targetFrames) / Double(plan.targetRate)
        guard profile.rates.contains(plan.targetRate), profile.channels.contains(channels),
            profile.allowResample || profile.rates.contains(plan.sourceRate),
            profile.channels.contains(sourceChannels) || (profile.allowDownmix && sourceChannels > profile.channels.max()!),
            channels == sourceChannels || profile.allowDownmix,
            sourceDuration >= profile.minDuration, sourceDuration <= profile.maxDuration,
            duration >= profile.minDuration, duration <= profile.maxDuration,
            plan.targetFrames > 0
        else { throw AudioSampleConversionError.incompatibleProfile }
        let cutoff = 0.9 * min(1, Double(plan.targetRate) / Double(plan.sourceRate))
        let radius = Int(ceil(32 / cutoff))
        guard radius <= 512 else { throw AudioSampleConversionError.budgetExceeded }
        let taps = 2 * radius + 1
        // Counts originate in the bounded plan; checked products avoid platform traps.
        let (sourceCount, o1) = plan.sourceFrames.multipliedReportingOverflow(by: channels)
        let (targetCount, o2) = plan.targetFrames.multipliedReportingOverflow(by: channels)
        let (sourceBytes, o3) = sourceCount.multipliedReportingOverflow(by: 8)
        let (targetBytes, o4) = targetCount.multipliedReportingOverflow(by: 2)
        let (bytes, o5) = sourceBytes.addingReportingOverflow(targetBytes)
        let (required, o6) = bytes.addingReportingOverflow(taps * 8)
        let (operations, o7) = targetCount.multipliedReportingOverflow(by: taps)
        guard !o1 && !o2 && !o3 && !o4 && !o5 && !o6 && !o7,
            required <= maxBufferBytes, operations <= maxFilterOperations
        else { throw AudioSampleConversionError.budgetExceeded }
        if cancelled() { throw AudioSampleConversionError.cancelled }
        var source = [Double](repeating: 0, count: sourceCount)
        var output = [Int16](repeating: 0, count: targetCount)
        var weights = [Double](repeating: 0, count: taps)
        var success = false
        defer {
            for i in source.indices { source[i] = 0 }
            for i in weights.indices { weights[i] = 0 }
            if !success { for i in output.indices { output[i] = 0 } }
        }
        var position = 0
        for chunk in chunks {
            guard !chunk.isEmpty, chunk.count % sourceChannels == 0,
                chunk.count / sourceChannels <= maxChunkFrames,
                chunk.count / sourceChannels <= plan.sourceFrames - position
            else { throw AudioSampleConversionError.invalidSamples }
            for frame in 0..<(chunk.count / sourceChannels) {
                if cancelled() { throw AudioSampleConversionError.cancelled }
                var total = 0.0
                for channel in 0..<sourceChannels {
                    let value = chunk[frame * sourceChannels + channel]
                    guard value.isFinite else { throw AudioSampleConversionError.invalidSamples }
                    guard abs(value) <= 1 else { throw AudioSampleConversionError.clipping }
                    if channelPolicy == .preserve { source[position * channels + channel] = value }
                    total += value
                }
                if channelPolicy == .meanMono { source[position] = total / Double(sourceChannels) }
                position += 1
            }
        }
        guard position == plan.sourceFrames else { throw AudioSampleConversionError.invalidSamples }
        for frame in 0..<plan.targetFrames {
            if cancelled() { throw AudioSampleConversionError.cancelled }
            let numerator = frame * plan.sourceRate
            let center = numerator / plan.targetRate
            let fraction = Double(numerator % plan.targetRate) / Double(plan.targetRate)
            let left = center - radius
            var normalization = 0.0
            for tap in 0..<taps {
                let distance = Double(tap - radius) - fraction
                let x = cutoff * distance
                let sinc = x == 0 ? 1 : sin(Double.pi * x) / (Double.pi * x)
                let window = abs(distance) <= Double(radius) ? 0.5 * (1 + cos(Double.pi * distance / Double(radius))) : 0
                weights[tap] = cutoff * sinc * window
                normalization += weights[tap]
            }
            for channel in 0..<channels {
                var value = 0.0
                if plan.sourceRate == plan.targetRate {
                    value = source[frame * channels + channel]
                } else {
                    for index in max(0, left)..<min(plan.sourceFrames, left + taps) {
                        value += source[index * channels + channel] * weights[index - left]
                    }
                    value /= normalization
                }
                guard value.isFinite, abs(value) <= 1 else { throw AudioSampleConversionError.clipping }
                output[frame * channels + channel] = Int16((value * (value < 0 ? 32768 : 32767)).rounded(.toNearestOrEven))
            }
        }
        if cancelled() { throw AudioSampleConversionError.cancelled }
        success = true
        return ConvertedAudioSamples(
            report: AudioSampleConversionReport(
                plan: plan,
                sourceChannels: sourceChannels, targetChannels: channels, channelPolicy: channelPolicy), samples: output)
    }
}
