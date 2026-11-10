import Foundation

/// Controlled local speech failures; never includes source text, paths or engine errors.
public enum StreamingSpeechError: String, Error, LocalizedError, Sendable {
    case invalidInput = "speech_input_invalid"
    case busy = "speech_decoder_busy"
    case bufferLimit = "speech_buffer_limit"
    case sequence = "speech_audio_sequence_invalid"
    case unavailableCompute = "speech_compute_unavailable"
    case unverifiedArtifact = "speech_artifact_unverified"
    case artifactMismatch = "speech_artifact_mismatch"
    case unavailableEngine = "speech_engine_unavailable"
    case requiresNetwork = "speech_engine_requires_network"
    case invalidHypothesis = "speech_hypothesis_invalid"
    case cancelled = "speech_session_cancelled"
    case closed = "speech_session_closed"
    public var errorDescription: String? { rawValue }
}

/// Shared speech-contract token timing. Times are milliseconds, confidence is ppm.
/// Text is protected; these values are decoder estimates, not clinical confidence.
public struct SpeechTokenTiming: Sendable, CustomStringConvertible, CustomDebugStringConvertible {
    public let tokenIndex: Int
    public let startMilliseconds: Int64
    public let endMilliseconds: Int64
    public let confidencePPM: Int
    public let text: String
    public var description: String { "SpeechTokenTiming(text=<private>)" }
    public var debugDescription: String { description }

    public init(tokenIndex: Int, startMilliseconds: Int64, endMilliseconds: Int64, confidencePPM: Int, text: String) throws {
        guard (0...4096).contains(tokenIndex), startMilliseconds >= 0,
            endMilliseconds >= startMilliseconds, endMilliseconds <= 86_400_000,
            (0...1_000_000).contains(confidencePPM), text.unicodeScalars.count <= 8192
        else { throw StreamingSpeechError.invalidHypothesis }
        self.tokenIndex = tokenIndex
        self.startMilliseconds = startMilliseconds
        self.endMilliseconds = endMilliseconds
        self.confidencePPM = confidencePPM
        self.text = text
    }

    func shifted(by milliseconds: Int64) throws -> SpeechTokenTiming {
        guard milliseconds >= 0, endMilliseconds <= 86_400_000 - milliseconds else { throw StreamingSpeechError.invalidHypothesis }
        return try SpeechTokenTiming(
            tokenIndex: tokenIndex, startMilliseconds: startMilliseconds + milliseconds,
            endMilliseconds: endMilliseconds + milliseconds, confidencePPM: confidencePPM, text: text)
    }
}

/// Whether word alignment is available and bounded by the actual source audio.
public enum SpeechAlignmentStatus: String, Sendable {
    case unavailable, aligned, outsideWindow, unverifiedTokenization
}

/// One protected decoder result. Token times are relative to the supplied window.
/// Partial results may omit acoustic alignment; window timing is carried separately.
public struct SpeechRecognition: Sendable, CustomStringConvertible, CustomDebugStringConvertible {
    public let text: String
    public let tokenTimes: [SpeechTokenTiming]
    public let decoderConfidencePPM: Int?
    public let alignmentStatus: SpeechAlignmentStatus
    public var description: String { "SpeechRecognition(text=<private>)" }
    public var debugDescription: String { description }

    public init(text: String, tokenTimes: [SpeechTokenTiming] = [], decoderConfidencePPM: Int? = nil, alignmentStatus: SpeechAlignmentStatus? = nil) throws {
        guard text.unicodeScalars.count <= 8192, tokenTimes.count <= 4096,
            decoderConfidencePPM.map({ (0...1_000_000).contains($0) }) ?? true
        else { throw StreamingSpeechError.invalidHypothesis }
        for (index, token) in tokenTimes.enumerated() {
            if index > 0 {
                let previous = tokenTimes[index - 1]
                guard previous.tokenIndex < token.tokenIndex,
                    previous.startMilliseconds <= token.startMilliseconds,
                    previous.endMilliseconds <= token.endMilliseconds
                else { throw StreamingSpeechError.invalidHypothesis }
            }
        }
        self.text = text
        self.tokenTimes = tokenTimes
        self.decoderConfidencePPM = decoderConfidencePPM
        self.alignmentStatus = alignmentStatus ?? (tokenTimes.isEmpty ? .unavailable : .aligned)
        guard (self.alignmentStatus == .aligned) == !tokenTimes.isEmpty else { throw StreamingSpeechError.invalidHypothesis }
    }

    func validate(windowFrames: Int) throws {
        let duration = (Int64(windowFrames) * 1000 + 15_999) / 16_000
        guard tokenTimes.allSatisfy({ $0.endMilliseconds <= duration }) else { throw StreamingSpeechError.invalidHypothesis }
    }
}

/// A protected partial hypothesis, matching the shared speech-provider fields.
/// The window interval locates the input, not individual unaligned words.
public struct SpeechPartialHypothesis: Sendable, CustomStringConvertible, CustomDebugStringConvertible {
    public let revision: Int
    public let text: String
    public let isFinal = false
    public let tokenTimes: [SpeechTokenTiming]
    public let windowStartMilliseconds: Int64
    public let windowEndMilliseconds: Int64
    public let decoderConfidencePPM: Int?
    public let alignmentStatus: SpeechAlignmentStatus
    public var description: String { "SpeechPartialHypothesis(text=<private>)" }
    public var debugDescription: String { description }
}

/// A finalized acoustic segment; this never finalizes a clinical note or releases PHI.
public struct SpeechFinalizedSegment: Sendable, CustomStringConvertible, CustomDebugStringConvertible {
    public let allowsAutomaticClinicalFinalization = false
    public let segmentIndex: Int
    public let text: String
    public let startMilliseconds: Int64
    public let endMilliseconds: Int64
    public let tokenTimes: [SpeechTokenTiming]
    public let decoderConfidencePPM: Int?
    public let alignmentStatus: SpeechAlignmentStatus
    public var description: String { "SpeechFinalizedSegment(text=<private>)" }
    public var debugDescription: String { description }
}

public enum SpeechStreamEvent: Sendable {
    case partial(SpeechPartialHypothesis)
    case finalized(SpeechFinalizedSegment)
}

/// Pinned engine identity. It describes provenance, not source custody or authorization.
public struct SpeechEngineProvenance: Sendable {
    public let providerID: String
    public let providerRevision: String
    public let modelFingerprint: String
    public let language: String

    public init(providerID: String, providerRevision: String, modelFingerprint: String, language: String) throws {
        func matches(_ value: String, _ pattern: String) -> Bool {
            value.range(of: pattern, options: .regularExpression) == value.startIndex..<value.endIndex
        }
        guard matches(providerID, "^[a-z0-9](?:[a-z0-9_.-]{0,62}[a-z0-9])?$"),
            matches(providerRevision, "^[A-Za-z0-9](?:[A-Za-z0-9_.+-]{0,62}[A-Za-z0-9])?$"),
            modelFingerprint.hasPrefix("sha256:"), modelFingerprint.utf8.count == 71,
            modelFingerprint.dropFirst(7).utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }),
            language.utf8.count <= 63, matches(language, "^[a-z]{2,3}(?:-[A-Za-z0-9]{2,8})*$")
        else { throw StreamingSpeechError.unverifiedArtifact }
        self.providerID = providerID
        self.providerRevision = providerRevision
        self.modelFingerprint = modelFingerprint
        self.language = language
    }
}

/// One caller-owned, bounded mono Float32 allocation on the 16 kHz stream grid.
/// Passing it to a session transfers ownership; submission clears it on every path.
public final class SpeechPCMBuffer: @unchecked Sendable, CustomStringConvertible, CustomDebugStringConvertible {
    public static let sampleRateHz = 16_000
    public static let maximumFrames = 480_000
    public let startSample: Int64
    public let frameCount: Int
    private let lock = NSLock()
    private let storage: UnsafeMutablePointer<Float>
    private var cleared = false
    public var description: String { "SpeechPCMBuffer(samples=<private>)" }
    public var debugDescription: String { description }

    public init(samples: UnsafeBufferPointer<Float>, startSample: Int64, sampleRateHz: Int = 16_000) throws {
        guard sampleRateHz == Self.sampleRateHz, (1...Self.maximumFrames).contains(samples.count),
            startSample >= 0, startSample <= 1_382_400_000 - Int64(samples.count),
            samples.allSatisfy({ $0.isFinite && abs($0) <= 1 })
        else { throw StreamingSpeechError.invalidInput }
        self.startSample = startSample
        frameCount = samples.count
        storage = .allocate(capacity: samples.count)
        storage.initialize(from: samples.baseAddress!, count: samples.count)
    }

    deinit {
        clear()
        storage.deinitialize(count: frameCount)
        storage.deallocate()
    }

    /// Synchronous borrowing only. Retaining the pointer is never supported.
    /// A trusted decoder may create a bounded working copy for its prediction call.
    public func withSamples<T>(_ body: (UnsafeBufferPointer<Float>) throws -> T) throws -> T {
        lock.lock()
        defer { lock.unlock() }
        guard !cleared else { throw StreamingSpeechError.invalidInput }
        return try body(UnsafeBufferPointer(start: storage, count: frameCount))
    }

    public func clear() {
        lock.lock()
        defer { lock.unlock() }
        storage.update(repeating: 0, count: frameCount)
        cleared = true
    }

    public var isCleared: Bool {
        lock.lock()
        defer { lock.unlock() }
        return cleared && UnsafeBufferPointer(start: storage, count: frameCount).allSatisfy { $0 == 0 }
    }
}

/// Injected local engine seam. Implementations must honor task cancellation and
/// release their working audio after decode returns. No capture or network is implied.
public protocol LocalSpeechEngine: Sendable {
    var provenance: SpeechEngineProvenance { get }
    var localOnly: Bool { get }
    /// Progress ordinals increase within this call. Returning false requests a stop.
    func decode(
        _ window: SpeechPCMBuffer,
        progress: @escaping @Sendable (_ ordinal: Int, _ recognition: SpeechRecognition) -> Bool
    ) async throws -> SpeechRecognition
    /// Idempotent and safe for concurrent calls. Called after active decoding drains.
    func shutdown() async
}
