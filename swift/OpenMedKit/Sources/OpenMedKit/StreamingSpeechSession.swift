import Foundation

/// Explicit admission and emission bounds. Windows never grow beyond 30 seconds.
public struct SpeechSessionLimits: Sendable {
    public let maximumWindowFrames: Int
    public let decodeStrideFrames: Int
    public let maximumPartialHypotheses: Int
    public let maximumSegments: Int

    public init(maximumWindowFrames: Int = 480_000, decodeStrideFrames: Int = 16_000, maximumPartialHypotheses: Int = 4096, maximumSegments: Int = 4096) throws {
        guard (1...SpeechPCMBuffer.maximumFrames).contains(maximumWindowFrames),
            (1...maximumWindowFrames).contains(decodeStrideFrames),
            (1...4096).contains(maximumPartialHypotheses), (1...4096).contains(maximumSegments)
        else { throw StreamingSpeechError.invalidInput }
        self.maximumWindowFrames = maximumWindowFrames
        self.decodeStrideFrames = decodeStrideFrames
        self.maximumPartialHypotheses = maximumPartialHypotheses
        self.maximumSegments = maximumSegments
    }
}

public enum SpeechSessionState: String, Sendable { case open, finished, cancelled, failed }

/// Content-free ownership and lifecycle accounting.
public struct SpeechSessionSnapshot: Sendable {
    public let state: SpeechSessionState
    public let bufferedFrames: Int
    public let totalFrames: Int64
    public let partialHypotheses: Int
    public let finalizedSegments: Int
    public let decoding: Bool
    public let reason: StreamingSpeechError?
}

/// Bounded, backpressured streaming recognition. Append calls must be awaited.
/// End each utterance explicitly; finish closes the stream. No endpoint, speaker,
/// consent or clinical-finalization policy is inferred by this class.
public final class StreamingSpeechSession: @unchecked Sendable, CustomStringConvertible {
    private struct Work {
        let generation: UInt64
        let window: SpeechPCMBuffer
        let endUtterance: Bool
        let finishStream: Bool
        let task: Task<SpeechRecognition, Error>
    }

    private let lock = NSRecursiveLock()
    private let engine: any LocalSpeechEngine
    private let limits: SpeechSessionLimits
    private let onEvent: @Sendable (SpeechStreamEvent) -> Void
    private let samples: UnsafeMutablePointer<Float>
    private var count = 0
    private var lastDecodedFrames = 0
    private var total: Int64 = 0
    private var origin: Int64 = 0
    private var generation: UInt64 = 0
    private var work: Work?
    private var closing = false
    private var state: SpeechSessionState = .open
    private var reason: StreamingSpeechError?
    private var partials = 0
    private var segments = 0
    private var lastOrdinal = 0
    public let provenance: SpeechEngineProvenance
    public var description: String { "StreamingSpeechSession(content=<private>)" }

    /// The event callback runs synchronously inside the delivery gate. Keep it short;
    /// requestCancellation() is supported there, but never block waiting for cancel().
    public init(engine: any LocalSpeechEngine, limits: SpeechSessionLimits = try! SpeechSessionLimits(), onEvent: @escaping @Sendable (SpeechStreamEvent) -> Void) throws {
        guard engine.localOnly else { throw StreamingSpeechError.requiresNetwork }
        self.engine = engine
        self.limits = limits
        self.onEvent = onEvent
        provenance = engine.provenance
        samples = .allocate(capacity: limits.maximumWindowFrames)
        samples.initialize(repeating: 0, count: limits.maximumWindowFrames)
    }

    deinit {
        requestCancellation()
        samples.deinitialize(count: limits.maximumWindowFrames)
        samples.deallocate()
    }

    private func locked<T>(_ body: () throws -> T) rethrows -> T {
        lock.lock()
        defer { lock.unlock() }
        return try body()
    }

    private func clearAccumulation() {
        samples.update(repeating: 0, count: limits.maximumWindowFrames)
        count = 0
        lastDecodedFrames = 0
    }

    public func snapshot() -> SpeechSessionSnapshot {
        locked {
            SpeechSessionSnapshot(state: state, bufferedFrames: count, totalFrames: total, partialHypotheses: partials, finalizedSegments: segments, decoding: work != nil, reason: reason)
        }
    }

    /// Consumes and clears input even on refusal. Busy means no audio was admitted;
    /// the caller must apply backpressure rather than dropping or retrying this object.
    public func append(_ input: SpeechPCMBuffer, endOfUtterance: Bool = false) async throws {
        try await withTaskCancellationHandler {
            do {
                if Task.isCancelled { requestCancellation() }
                if let item = try prepare(input: input, endUtterance: endOfUtterance, finishStream: false) {
                    try complete(item, result: await item.task.result)
                }
            } catch {
                await drainIfTerminal()
                throw error
            }
        } onCancel: {
            self.requestCancellation()
        }
    }

    /// Finalize any remaining utterance, then release the engine. No later append is allowed.
    public func finish() async throws {
        try await withTaskCancellationHandler {
            do {
                if Task.isCancelled { requestCancellation() }
                let item = try prepare(input: nil, endUtterance: true, finishStream: true)
                if let item { try complete(item, result: await item.task.result) }
                await engine.shutdown()
            } catch {
                await drainIfTerminal()
                throw error
            }
        } onCancel: {
            self.requestCancellation()
        }
    }

    private func drainIfTerminal() async {
        guard locked({ state != .open }) else { return }
        let active = locked { work }
        if let active { _ = await active.task.result }
        await engine.shutdown()
        locked { if work?.generation == active?.generation { work = nil } }
    }

    /// Fence delivery immediately; call cancel() to await decoder drain and cleanup.
    public func requestCancellation() {
        locked {
            guard state != .finished else { return }
            state = .cancelled
            reason = .cancelled
            generation &+= 1
            clearAccumulation()
            work?.task.cancel()
        }
    }

    /// Returns only after active decoding has drained and its owned window was cleared.
    /// Core ML prediction itself is cooperatively cancellable, not forcibly preempted.
    public func cancel() async {
        requestCancellation()
        let item = locked { work }
        if let item { _ = await item.task.result }
        await engine.shutdown()
        locked {
            if work?.generation == item?.generation { work = nil }
        }
    }

    private func prepare(input: SpeechPCMBuffer?, endUtterance: Bool, finishStream: Bool) throws -> Work? {
        defer { input?.clear() }
        return try locked {
            guard state == .open, !closing else { throw reason ?? StreamingSpeechError.closed }
            guard work == nil else { throw StreamingSpeechError.busy }
            guard engine.localOnly else {
                fail(.requiresNetwork)
                throw StreamingSpeechError.requiresNetwork
            }
            if let input {
                guard input.startSample == total else {
                    fail(.sequence)
                    throw StreamingSpeechError.sequence
                }
                guard input.frameCount <= limits.maximumWindowFrames - count else {
                    fail(.bufferLimit)
                    throw StreamingSpeechError.bufferLimit
                }
                do {
                    try input.withSamples { values in samples.advanced(by: count).update(from: values.baseAddress!, count: values.count) }
                } catch {
                    fail(.invalidInput)
                    throw StreamingSpeechError.invalidInput
                }
                count += input.frameCount
                total += Int64(input.frameCount)
            }
            if finishStream { closing = true }
            guard count > 0 else {
                if finishStream { state = .finished }
                return nil
            }
            guard endUtterance || count - lastDecodedFrames >= limits.decodeStrideFrames else { return nil }
            let window = try SpeechPCMBuffer(samples: UnsafeBufferPointer(start: samples, count: count), startSample: origin)
            generation &+= 1
            let ticket = generation
            let engine = engine
            lastOrdinal = 0
            let task = Task<SpeechRecognition, Error> { [weak self] in
                defer { window.clear() }
                try Task.checkCancellation()
                let result = try await engine.decode(window) { [weak self] ordinal, recognition in
                    self?.receive(recognition, ordinal: ordinal, generation: ticket, window: window) ?? false
                }
                try Task.checkCancellation()
                return result
            }
            let item = Work(generation: ticket, window: window, endUtterance: endUtterance, finishStream: finishStream, task: task)
            work = item
            return item
        }
    }

    private func fail(_ value: StreamingSpeechError) {
        state = .failed
        reason = value
        generation &+= 1
        clearAccumulation()
        work?.task.cancel()
    }

    private func receive(_ recognition: SpeechRecognition, ordinal: Int, generation ticket: UInt64, window: SpeechPCMBuffer) -> Bool {
        locked {
            guard state == .open, work?.generation == ticket, generation == ticket else { return false }
            guard engine.localOnly else {
                fail(.requiresNetwork)
                return false
            }
            guard (1...4096).contains(ordinal) else {
                fail(.invalidHypothesis)
                return false
            }
            guard ordinal > lastOrdinal else { return true }
            do {
                try recognition.validate(windowFrames: window.frameCount)
                lastOrdinal = ordinal
                try emitPartial(recognition, window: window)
                return state == .open && generation == ticket
            } catch {
                fail(error as? StreamingSpeechError ?? .invalidHypothesis)
                return false
            }
        }
    }

    private func emitPartial(_ recognition: SpeechRecognition, window: SpeechPCMBuffer) throws {
        guard partials < limits.maximumPartialHypotheses else { throw StreamingSpeechError.bufferLimit }
        let start = window.startSample * 1000 / 16_000
        let end = ((window.startSample + Int64(window.frameCount)) * 1000 + 15_999) / 16_000
        let event = SpeechPartialHypothesis(
            revision: partials, text: recognition.text, tokenTimes: try recognition.tokenTimes.map { try $0.shifted(by: start) },
            windowStartMilliseconds: start, windowEndMilliseconds: end, decoderConfidencePPM: recognition.decoderConfidencePPM, alignmentStatus: recognition.alignmentStatus)
        partials += 1
        onEvent(.partial(event))
    }

    private func complete(_ item: Work, result: Result<SpeechRecognition, Error>) throws {
        try locked {
            guard work?.generation == item.generation else { throw reason ?? StreamingSpeechError.closed }
            work = nil
            guard state == .open, generation == item.generation else { throw reason ?? StreamingSpeechError.closed }
            // Invalidate detached engine callbacks before returning any completed result.
            generation &+= 1
            do {
                guard engine.localOnly else { throw StreamingSpeechError.requiresNetwork }
                let recognition: SpeechRecognition
                switch result {
                case .success(let value): recognition = value
                case .failure(let error):
                    if error is CancellationError { throw StreamingSpeechError.cancelled }
                    throw error as? StreamingSpeechError ?? .unavailableEngine
                }
                try recognition.validate(windowFrames: item.window.frameCount)
                if item.endUtterance {
                    guard segments < limits.maximumSegments else { throw StreamingSpeechError.bufferLimit }
                    let start = item.window.startSample * 1000 / 16_000
                    let end = ((item.window.startSample + Int64(item.window.frameCount)) * 1000 + 15_999) / 16_000
                    let event = SpeechFinalizedSegment(
                        segmentIndex: segments, text: recognition.text, startMilliseconds: start, endMilliseconds: end,
                        tokenTimes: try recognition.tokenTimes.map { try $0.shifted(by: start) }, decoderConfidencePPM: recognition.decoderConfidencePPM, alignmentStatus: recognition.alignmentStatus)
                    segments += 1
                    origin = total
                    clearAccumulation()
                    if item.finishStream { state = .finished }
                    onEvent(.finalized(event))
                } else {
                    lastDecodedFrames = count
                    try emitPartial(recognition, window: item.window)
                }
            } catch {
                let controlled = (error as? StreamingSpeechError) ?? .unavailableEngine
                fail(controlled)
                throw controlled
            }
        }
    }
}
