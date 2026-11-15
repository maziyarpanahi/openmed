import CoreML
import Foundation
import OpenMedKit
import WhisperKit

/// Allowed Core ML device sets. Accelerator choices require the named device to
/// exist; Core ML still chooses per-operation placement within that set.
public enum SpeechCompute: String, Sendable {
    case cpuOnly, cpuAndGPU, cpuAndNeuralEngine

    func validate(available: Set<String>, simulator: Bool) throws -> MLComputeUnits {
        switch self {
        case .cpuOnly:
            guard available.contains("cpu") else { throw StreamingSpeechError.unavailableCompute }
            return .cpuOnly
        case .cpuAndGPU:
            guard !simulator, available.contains("cpu"), available.contains("gpu") else { throw StreamingSpeechError.unavailableCompute }
            return .cpuAndGPU
        case .cpuAndNeuralEngine:
            guard !simulator, available.contains("cpu"), available.contains("ane") else { throw StreamingSpeechError.unavailableCompute }
            return .cpuAndNeuralEngine
        }
    }

    func units() throws -> MLComputeUnits {
        let devices = Set(
            MLModel.availableComputeDevices.map { device in
                switch device {
                case .cpu: return "cpu"
                case .gpu: return "gpu"
                case .neuralEngine: return "ane"
                @unknown default: return "unknown"
                }
            })
        #if targetEnvironment(simulator)
            return try validate(available: devices, simulator: true)
        #else
            return try validate(available: devices, simulator: false)
        #endif
    }
}

/// Optional, local-only, English tiny.en Core ML adapter. Model and tokenizer files
/// are caller supplied; this type has no downloader or system speech recognizer.
/// Use one engine per session. Close the session with finish() or cancel().
public actor CoreMLSpeechEngine: LocalSpeechEngine {
    public nonisolated let provenance: SpeechEngineProvenance
    public nonisolated let localOnly = true
    private let engine: StrictLocalWhisperKit
    private let tokenizer: LocalEnglishTokenizer
    private var decoding = false
    private var closed = false
    private var drainers = [CheckedContinuation<Void, Never>]()
    private var unloading: Task<Void, Never>?

    private init(engine: StrictLocalWhisperKit, tokenizer: LocalEnglishTokenizer, provenance: SpeechEngineProvenance) {
        self.engine = engine
        self.tokenizer = tokenizer
        self.provenance = provenance
    }

    /// Load only the exact verified bundle. Revalidates before and after Core ML load.
    /// Runtime logging is disabled globally; hosts must not re-enable it during use.
    public static func load(artifact: VerifiedSpeechArtifact?, compute: SpeechCompute = .cpuOnly) async throws -> CoreMLSpeechEngine {
        guard let artifact else { throw StreamingSpeechError.unverifiedArtifact }
        let units = try compute.units()
        var loaded: StrictLocalWhisperKit?
        do {
            try Task.checkCancellation()
            try artifact.revalidate()
            Logging.shared.logLevel = .none
            let config = WhisperKitConfig(
                modelFolder: artifact.directory.path, tokenizerFolder: artifact.directory.appending(path: "tokenizer"),
                computeOptions: ModelComputeOptions(melCompute: units, audioEncoderCompute: units, textDecoderCompute: units),
                verbose: false, prewarm: false, load: false, download: false)
            let engine = try await StrictLocalWhisperKit(config)
            loaded = engine
            let tokenizer = try await LocalEnglishTokenizer(AutoTokenizerWrapper.from(modelFolder: artifact.directory.appending(path: "tokenizer")))
            engine.tokenizer = tokenizer
            engine.textDecoder.isModelMultilingual = false
            try await engine.loadModels()
            guard engine.textDecoder.logitsSize == 51864, engine.audioEncoder.embedSize == 384 else { throw StreamingSpeechError.artifactMismatch }
            try artifact.revalidate()
            try Task.checkCancellation()
            let provenance = try SpeechEngineProvenance(providerID: "apple-coreml-whisper", providerRevision: VerifiedSpeechArtifact.runtimeRevision, modelFingerprint: VerifiedSpeechArtifact.fingerprint, language: "en")
            // Nothing that can throw follows the hand-off, so the catch below never
            // touches the engine after the actor owns it.
            return CoreMLSpeechEngine(engine: engine, tokenizer: tokenizer, provenance: provenance)
        } catch {
            await loaded?.unloadModels()
            if error is CancellationError { throw StreamingSpeechError.cancelled }
            throw error as? StreamingSpeechError ?? .unavailableEngine
        }
    }

    public func decode(_ window: SpeechPCMBuffer, progress: @escaping @Sendable (Int, SpeechRecognition) -> Bool) async throws -> SpeechRecognition {
        guard !closed else { throw StreamingSpeechError.closed }
        guard !decoding else { throw StreamingSpeechError.busy }
        decoding = true
        do {
            let result = try await recognize(window, progress: progress)
            endDecode()
            return result
        } catch {
            endDecode()
            if error is CancellationError { throw StreamingSpeechError.cancelled }
            throw error as? StreamingSpeechError ?? .unavailableEngine
        }
    }

    private func endDecode() {
        decoding = false
        let waiting = drainers
        drainers.removeAll()
        for continuation in waiting { continuation.resume() }
    }

    public func shutdown() async {
        closed = true
        if decoding { await withCheckedContinuation { drainers.append($0) } }
        if unloading == nil {
            // Capture the actor rather than a copy of its non-Sendable engine, so the
            // task inherits this actor's isolation instead of racing it.
            unloading = Task { await self.engine.unloadModels() }
        }
        await unloading?.value
    }

    private func recognize(_ window: SpeechPCMBuffer, progress: @escaping @Sendable (Int, SpeechRecognition) -> Bool) async throws -> SpeechRecognition {
        try Task.checkCancellation()
        guard !Logging.shared.isLoggingEnabled else { throw StreamingSpeechError.unavailableEngine }
        var audio = try window.withSamples { Array($0) }
        defer { audio.withUnsafeMutableBufferPointer { $0.update(repeating: 0) } }
        tokenizer.resetAlignment()
        let gate = SpeechDecoderProgress(progress: progress)
        defer { gate.close() }
        var options = DecodingOptions()
        options.verbose = false
        options.language = "en"
        options.temperature = 0
        options.temperatureFallbackCount = 0
        options.sampleLength = 256
        options.skipSpecialTokens = true
        options.withoutTimestamps = false
        options.wordTimestamps = true
        let results = try await engine.transcribe(audioArray: audio, decodeOptions: options) { value in gate.receive(value) }
        gate.close()
        try Task.checkCancellation()
        guard !closed else { throw StreamingSpeechError.cancelled }
        guard !gate.failed else { throw StreamingSpeechError.invalidHypothesis }
        guard !gate.stopped else { throw StreamingSpeechError.cancelled }
        return try Self.recognition(results: results, frames: window.frameCount, tokenizationVerified: tokenizer.alignmentValid)
    }

    static func recognition(results: [TranscriptionResult], frames: Int, tokenizationVerified: Bool) throws -> SpeechRecognition {
        let duration = (Int64(frames) * 1000 + 15_999) / 16_000
        let words = results.flatMap(\.allWords)
        guard words.count <= 4096 else { throw StreamingSpeechError.invalidHypothesis }
        var outsideWindow = false
        let tokens = try words.enumerated().map { index, word -> SpeechTokenTiming in
            guard word.start.isFinite, word.end.isFinite, word.start >= 0, word.end >= word.start,
                word.end <= 30, word.probability.isFinite, (0...1).contains(word.probability)
            else { throw StreamingSpeechError.invalidHypothesis }
            let start = Int64((Double(word.start) * 1000).rounded(.down))
            let end = Int64((Double(word.end) * 1000).rounded(.up))
            if end > duration { outsideWindow = true }
            return try SpeechTokenTiming(tokenIndex: index, startMilliseconds: start, endMilliseconds: end, confidencePPM: Int((Double(word.probability) * 1_000_000).rounded()), text: word.word)
        }
        var weightedLogProbability = 0.0
        var tokenCount = 0
        for segment in results.flatMap(\.segments) {
            guard segment.avgLogprob.isFinite, segment.avgLogprob <= 0, segment.tokens.count <= 4096 else { throw StreamingSpeechError.invalidHypothesis }
            tokenCount += segment.tokens.count
            guard tokenCount <= 4096 else { throw StreamingSpeechError.invalidHypothesis }
            weightedLogProbability += Double(segment.avgLogprob) * Double(segment.tokens.count)
        }
        let confidence = tokenCount > 0 ? Int((exp(weightedLogProbability / Double(tokenCount)) * 1_000_000).rounded()) : nil
        let status: SpeechAlignmentStatus = !tokenizationVerified ? .unverifiedTokenization : outsideWindow ? .outsideWindow : tokens.isEmpty ? .unavailable : .aligned
        return try SpeechRecognition(text: results.map(\.text).joined(separator: " "), tokenTimes: status == .aligned ? tokens : [], decoderConfidencePPM: confidence, alignmentStatus: status)
    }
}

/// The dependency can deliver progress on detached tasks. Fence those tasks here,
/// including for direct engine users, and again at the session boundary.
final class SpeechDecoderProgress: @unchecked Sendable {
    private let lock = NSLock()
    private let progress: @Sendable (Int, SpeechRecognition) -> Bool
    private var active = true
    private var lastOrdinal = 0
    private var didFail = false
    private var didStop = false
    init(progress: @escaping @Sendable (Int, SpeechRecognition) -> Bool) { self.progress = progress }
    var failed: Bool {
        lock.lock()
        defer { lock.unlock() }
        return didFail
    }
    var stopped: Bool {
        lock.lock()
        defer { lock.unlock() }
        return didStop
    }
    func close() {
        lock.lock()
        defer { lock.unlock() }
        active = false
    }
    func receive(_ value: TranscriptionProgress) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        guard active else { return false }
        let ordinal = value.tokens.count
        guard ordinal > lastOrdinal else { return true }
        do {
            guard ordinal <= 4096 else { throw StreamingSpeechError.invalidHypothesis }
            var confidence: Int?
            if let score = value.avgLogprob {
                guard score.isFinite, score <= 0 else { throw StreamingSpeechError.invalidHypothesis }
                confidence = Int((exp(Double(score)) * 1_000_000).rounded())
            }
            let result = try SpeechRecognition(text: value.text, decoderConfidencePPM: confidence)
            lastOrdinal = ordinal
            guard progress(ordinal, result) else {
                active = false
                didStop = true
                return false
            }
            return true
        } catch {
            active = false
            didFail = true
            return false
        }
    }
}
