/// A frame-level activity decision, without speaker or clinical interpretation.
public enum SpeechActivity: Sendable {
    case speech, silence, uncertain
}

/// Caller-supplied local detector. Implementations must release audio on reset.
/// Offline declarations are an integration contract, not model qualification.
public protocol LocalSpeechActivityDetector: AnyObject {
    var requiresNetwork: Bool { get }
    func detect(_ samples: [Float]) throws -> SpeechActivity
    func reset() throws
}

/// Controlled failures that never carry samples or detector exception text.
public enum SpeechEndpointError: Error {
    case invalidLimits, detectorNotLocal, invalidFrame, overlappingFrame
    case cancelled, detectorFailure, detectorResetFailure
}

/// Metadata-only event with half-open source sample bounds.
public struct SpeechEndpointEvent: Equatable, Sendable {
    public let kind: String
    public let startSample: Int
    public let endSample: Int
    public let reason: String
    public var reviewerConfirmationRequired: Bool { true }
    public var notice: String {
        "Non-diagnostic speech segmentation only; no speaker identity, consent or "
            + "clinical meaning. Consequential use requires explicit reviewer confirmation."
    }
}

/// Serial, single-stream speech segmentation with no retained audio or events.
/// Use one source sample clock and consistent normalized mono detector frames.
/// Limits use samples; hangover counts silence and uncertainty. Endpoints include
/// bounded trailing silence. Activity is uniform within each detector frame.
public final class SpeechEndpointAdapter {
    private let detector: any LocalSpeechActivityDetector
    private let hangover: Int
    private let maximum: Int
    private let frameLimit: Int
    private var start: Int?
    private var expected: Int?
    private var silence = 0
    private var cancelled = false

    /// Configure finite sample limits without loading an ASR model.
    public init(
        detector: any LocalSpeechActivityDetector,
        hangoverSamples: Int,
        maxUtteranceSamples: Int,
        maxFrameSamples: Int = 4096
    ) throws {
        guard (1...10_000_000).contains(maxUtteranceSamples),
            (0...maxUtteranceSamples).contains(hangoverSamples),
            (1...min(65_536, maxUtteranceSamples)).contains(maxFrameSamples)
        else { throw SpeechEndpointError.invalidLimits }
        guard !detector.requiresNetwork else { throw SpeechEndpointError.detectorNotLocal }
        self.detector = detector
        hangover = hangoverSamples
        maximum = maxUtteranceSamples
        frameLimit = maxFrameSamples
    }

    /// Audio remains caller-owned: the adapter's buffer count is always zero.
    public var bufferedSampleCount: Int { 0 }

    /// Emit start/end, uncertainty and gap events from one bounded mono frame.
    /// Invalid input leaves the stream untouched. Detector failures discard
    /// stream state and refuse further input until reset.
    public func push(startSample: Int, samples: [Float]) throws -> [SpeechEndpointEvent] {
        guard !cancelled else { throw SpeechEndpointError.cancelled }
        guard !samples.isEmpty, samples.count <= frameLimit,
            startSample >= 0, startSample <= Int.max - samples.count,
            samples.allSatisfy({ $0.isFinite && (-1...1).contains($0) })
        else { throw SpeechEndpointError.invalidFrame }
        if let expected, startSample < expected {
            throw SpeechEndpointError.overlappingFrame
        }
        let activity: SpeechActivity
        do {
            guard !detector.requiresNetwork else { throw SpeechEndpointError.detectorNotLocal }
            if let expected, startSample > expected { try detector.reset() }
            activity = try detector.detect(samples)
        } catch {
            cancelled = true
            clear()
            try? detector.reset()
            throw SpeechEndpointError.detectorFailure
        }

        var events: [SpeechEndpointEvent] = []
        if let expected, startSample > expected {
            end(expected, reason: "discontinuity", events: &events)
            events.append(
                SpeechEndpointEvent(
                    kind: "discontinuity", startSample: expected, endSample: startSample, reason: "gap"))
        }
        let frameEnd = startSample + samples.count
        if activity == .uncertain {
            events.append(
                SpeechEndpointEvent(
                    kind: "uncertain_activity", startSample: startSample, endSample: frameEnd,
                    reason: "uncertain"))
        }
        var cursor = startSample
        while cursor < frameEnd {
            if start == nil {
                guard activity == .speech else { break }
                start = cursor
                events.append(
                    SpeechEndpointEvent(
                        kind: "speech_start", startSample: cursor, endSample: cursor, reason: "detected"))
            }
            guard let start else { break }
            if activity == .speech { silence = 0 }
            // Compare relative lengths to avoid overflowing the absolute clock.
            let remaining = maximum - (cursor - start)
            var amount = remaining
            var reason = "maximum_duration"
            if activity != .speech {
                let silenceRemaining = max(0, hangover - silence)
                if silenceRemaining <= amount {
                    amount = silenceRemaining
                    reason = "hangover"
                }
            }
            let consumed = min(frameEnd - cursor, amount)
            if activity != .speech { silence += consumed }
            cursor += consumed
            if consumed == amount {
                end(cursor, reason: reason, events: &events)
            } else {
                break
            }
        }
        expected = frameEnd
        return events
    }

    /// End the stream and release detector state; permit a fresh sample clock.
    public func finish() throws -> [SpeechEndpointEvent] {
        guard !cancelled else { throw SpeechEndpointError.cancelled }
        return try release(reason: "stream_end", cancelled: false)
    }

    /// Discard temporal state and explicitly reopen a cancelled adapter.
    public func reset() throws -> [SpeechEndpointEvent] {
        try release(reason: "reset", cancelled: false)
    }

    /// Discard detector-owned audio and refuse frames until explicit reset.
    public func cancel() throws -> [SpeechEndpointEvent] {
        try release(reason: "cancelled", cancelled: true)
    }

    private func end(_ sample: Int, reason: String, events: inout [SpeechEndpointEvent]) {
        if let start {
            events.append(
                SpeechEndpointEvent(
                    kind: "speech_end", startSample: start, endSample: sample, reason: reason))
        }
        start = nil
        silence = 0
    }

    private func clear() {
        start = nil
        expected = nil
        silence = 0
    }

    private func release(reason: String, cancelled: Bool) throws -> [SpeechEndpointEvent] {
        var events: [SpeechEndpointEvent] = []
        end(expected ?? 0, reason: reason, events: &events)
        clear()
        self.cancelled = true
        do {
            try detector.reset()
        } catch {
            throw SpeechEndpointError.detectorResetFailure
        }
        self.cancelled = cancelled
        return events
    }
}
