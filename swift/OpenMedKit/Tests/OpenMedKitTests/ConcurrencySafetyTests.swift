import Foundation
import XCTest
import os

@testable import OpenMedKit

#if canImport(Tokenizers)
    import Tokenizers
#endif

// Concurrent-use coverage for every `@unchecked Sendable` type in OpenMedKit and
// compile-time Sendable checks for public value types. Each test hits one shared
// instance from many tasks or threads and asserts the invariant its lock exists
// to protect. Run them under Thread Sanitizer with:
//
//     swift test --sanitize=thread --filter ConcurrencySafetyTests

private let contention = 64

private func requireSendable<T: Sendable>(_: T.Type) {}

private func speechBuffer(start: Int64, frames: Int = 160) throws -> SpeechPCMBuffer {
    try [Float](repeating: 0.25, count: frames).withUnsafeBufferPointer {
        try SpeechPCMBuffer(samples: $0, startSample: start)
    }
}

private func captureBuffer(start: Int64) throws -> CapturePCMBuffer {
    try [Float(0.125), -0.25, 0.5].withUnsafeBufferPointer {
        try CapturePCMBuffer(samples: $0, sampleRateHz: 16_000, startSample: start)
    }
}

private actor ImmediateSpeechEngine: LocalSpeechEngine {
    nonisolated let provenance = try! SpeechEngineProvenance(
        providerID: "synthetic", providerRevision: "v1",
        modelFingerprint: "sha256:" + String(repeating: "b", count: 64), language: "en")
    nonisolated let localOnly = true

    func decode(
        _ window: SpeechPCMBuffer,
        progress: @escaping @Sendable (Int, SpeechRecognition) -> Bool
    ) async throws -> SpeechRecognition {
        await Task.yield()
        return try SpeechRecognition(text: "synthetic")
    }

    func shutdown() async {}
}

/// Delivers buffers from whichever thread calls `emit`, like an audio tap.
private final class ConcurrentAudioSource: AmbientAudioSource {
    private let receiver = OSAllocatedUnfairLock<(@Sendable (CapturePCMBuffer) -> Void)?>(
        initialState: nil)

    var permission: CapturePermission { .granted }

    func requestPermission(_ completion: @escaping @Sendable (CapturePermission) -> Void) {
        completion(.granted)
    }

    func start(
        receive: @escaping @Sendable (CapturePCMBuffer) -> Void,
        interrupted: @escaping @Sendable (CaptureInterruption) -> Void
    ) throws {
        receiver.withLock { $0 = receive }
    }

    func stop() -> Bool {
        receiver.withLock { $0 = nil }
        return true
    }

    func emit(_ buffer: CapturePCMBuffer) {
        guard let receive = receiver.withLock({ $0 }) else {
            buffer.clear()
            return
        }
        receive(buffer)
    }
}

private final class CountingConsumer: AmbientPCMConsumer {
    let frames = OSAllocatedUnfairLock(initialState: 0)

    func consume(
        _ samples: UnsafeBufferPointer<Float>, sampleRateHz: Int,
        startSample: Int64, captureEpoch: UInt64
    ) throws {
        let count = samples.count
        frames.withLock { $0 += count }
    }
}

private struct NominalThermalSource: MultimodalThermalSource {
    struct Subscription: ThermalObservationSubscription {
        func cancel() {}
    }

    func current() -> MultimodalThermalState? { .nominal }

    func observe(_ change: @escaping @Sendable () -> Void) -> any ThermalObservationSubscription {
        Subscription()
    }
}

private struct YieldingWork: LocalMultimodalWork {
    let requiresNetwork = false
    let value: Int

    func run(profile: MultimodalWorkProfile) async throws -> Int {
        await Task.yield()
        return value
    }
}

final class ConcurrencySafetyTests: XCTestCase {
    func testPublicValueTypesAreSendable() {
        // Compile-time checks: a value type that loses Sendable stops this file building.
        requireSendable(EntityPrediction.self)
        requireSendable(OpenMedBackend.self)
        requireSendable(DeidentificationResult.self)
        requireSendable(Policy.self)
        requireSendable(PostProcessing.TokenPrediction.self)
        requireSendable(PostProcessing.AggregationStrategy.self)
        requireSendable(PostProcessing.SemanticUnitPattern.self)
        requireSendable(PostProcessing.SemanticUnitMatch.self)
        requireSendable(AudioConversionChannelPolicy.self)
        requireSendable(AudioSampleConversionPlan.self)
        requireSendable(AudioSampleConversionPlan.Rounding.self)
        requireSendable(AudioSampleConversionProfile.self)
        requireSendable(AudioSampleConversionReport.self)
        requireSendable(EDFLimits.self)
        requireSendable(EDFIdentityStatus.self)
        requireSendable(EDFSignal.self)
        requireSendable(EDFSignalWindow.self)
        requireSendable(EDFRecordWindow.self)
        requireSendable(EDFAnnotation.self)
        requireSendable(EDFGap.self)
        requireSendable(EDFRecording.self)
        requireSendable(LocalReviewAlternative.self)
        requireSendable(LocalMultimodalReviewContent.self)
        requireSendable(VoiceEmbeddingBoundary.self)
        requireSendable(EmbeddingDestructionReceipt.self)
        requireSendable(VoiceSimilarity.self)
        requireSendable(VoiceEmbeddingDiagnostics.self)
        requireSendable(WFDBDataSource.self)
        requireSendable(WFDBLimits.self)
        requireSendable(WFDBSignal.self)
        requireSendable(WFDBAnnotations.self)
        requireSendable(WFDBRecord.self)
    }

    func testDefaultPIIPatternsAreSharedSafelyAcrossTasks() async {
        let text = "Patient: Mira Vale\nPhone: (555) 010-0199\nVisit notes follow."
        let summary: @Sendable () -> [String] = {
            PostProcessing.findSemanticUnits(in: text).map { "\($0.entityType):\($0.start)-\($0.end)" }
        }
        let expected = summary()
        XCTAssertFalse(expected.isEmpty)
        let observed = await withTaskGroup(of: [String].self, returning: [[String]].self) { group in
            for _ in 0..<contention { group.addTask { summary() } }
            return await group.reduce(into: []) { $0.append($1) }
        }
        XCTAssertEqual(observed.count, contention)
        XCTAssertTrue(observed.allSatisfy { $0 == expected })
    }

    func testSpeechPCMBufferReadersSeeWholeWindowOrRefusal() async throws {
        let buffer = try speechBuffer(start: 0, frames: 1_600)
        let sums = await withTaskGroup(of: Float?.self, returning: [Float?].self) { group in
            for index in 0..<contention {
                group.addTask {
                    if index == contention / 2 {
                        buffer.clear()
                        return nil
                    }
                    return try? buffer.withSamples { $0.reduce(0, +) }
                }
            }
            return await group.reduce(into: []) { $0.append($1) }
        }
        // A clear that raced a reader must never expose a partially zeroed window.
        XCTAssertTrue(sums.compactMap { $0 }.allSatisfy { $0 == 400 })
        XCTAssertTrue(buffer.isCleared)
        XCTAssertThrowsError(try buffer.withSamples { $0.count })
    }

    func testStreamingSpeechSessionClearsEveryContendedInput() async throws {
        let session = try StreamingSpeechSession(
            engine: ImmediateSpeechEngine(), limits: SpeechSessionLimits(decodeStrideFrames: 160)
        ) { _ in }
        // Every append claims the same first sample, so all but one are refused as
        // busy or out of sequence. Each refused input must still be cleared.
        let inputs = try (0..<contention).map { _ in try speechBuffer(start: 0) }
        await withTaskGroup(of: Void.self) { group in
            for input in inputs { group.addTask { try? await session.append(input) } }
            for _ in 0..<contention { group.addTask { _ = session.snapshot() } }
        }
        await session.cancel()
        let snapshot = session.snapshot()
        XCTAssertTrue(inputs.allSatisfy(\.isCleared))
        XCTAssertEqual(snapshot.state, .cancelled)
        XCTAssertFalse(snapshot.decoding)
        XCTAssertEqual(snapshot.bufferedFrames, 0)
    }

    func testStreamingSpeechSessionSnapshotsDuringAValidStream() async throws {
        let finalized = OSAllocatedUnfairLock(initialState: 0)
        let session = try StreamingSpeechSession(
            engine: ImmediateSpeechEngine(), limits: SpeechSessionLimits(decodeStrideFrames: 160)
        ) { event in
            if case .finalized = event { finalized.withLock { $0 += 1 } }
        }
        await withTaskGroup(of: Void.self) { group in
            group.addTask {
                for index in 0..<16 {
                    guard let input = try? speechBuffer(start: Int64(index) * 160) else { return }
                    try? await session.append(input, endOfUtterance: index % 4 == 3)
                }
                try? await session.finish()
            }
            for _ in 0..<contention { group.addTask { _ = session.snapshot() } }
        }
        let snapshot = session.snapshot()
        XCTAssertEqual(snapshot.state, .finished)
        XCTAssertEqual(snapshot.totalFrames, 16 * 160)
        XCTAssertEqual(snapshot.finalizedSegments, 4)
        XCTAssertEqual(finalized.withLock { $0 }, 4)
    }

    func testCapturePCMBufferClearIsAtomicUnderContention() async throws {
        let buffer = try captureBuffer(start: 0)
        let observed = await withTaskGroup(of: Bool.self, returning: [Bool].self) { group in
            for index in 0..<contention {
                group.addTask {
                    if index == contention / 2 { buffer.clear() }
                    return buffer.isCleared
                }
            }
            return await group.reduce(into: []) { $0.append($1) }
        }
        XCTAssertEqual(observed.count, contention)
        XCTAssertTrue(buffer.isCleared)
    }

    func testAmbientAudioCaptureAccountsForEveryConcurrentBuffer() async throws {
        let participant = String(repeating: "a", count: 32)
        let source = ConcurrentAudioSource()
        let consumer = CountingConsumer()
        let capture = try AmbientAudioCapture(
            sessionRef: String(repeating: "c", count: 32), participants: [participant],
            policy: AmbientSessionPolicy(
                revision: String(repeating: "d", count: 64), maxLifetimeMs: 3_600_000,
                maxBytes: 1024),
            source: source, consumer: consumer, clock: { 0 }, automaticExpiry: false)
        try capture.consent(participant)
        try capture.start()
        let buffers = try (0..<contention).map { try captureBuffer(start: Int64($0) * 3) }
        await withTaskGroup(of: Void.self) { group in
            for buffer in buffers { group.addTask { source.emit(buffer) } }
            for _ in 0..<contention { group.addTask { _ = capture.snapshot() } }
        }
        try capture.stop()
        let snapshot = capture.snapshot()
        // Buffers that lose the gate are dropped, never queued, and always cleared.
        XCTAssertEqual(snapshot.receivedBuffers, UInt64(contention))
        XCTAssertEqual(snapshot.ownedBuffers, 0)
        XCTAssertLessThanOrEqual(snapshot.admittedBuffers, snapshot.receivedBuffers)
        XCTAssertEqual(consumer.frames.withLock { $0 }, Int(snapshot.admittedBuffers) * 3)
        XCTAssertTrue(buffers.allSatisfy(\.isCleared))
        XCTAssertFalse(snapshot.running)
    }

    func testAmbientSessionSerializesConcurrentRetainUseAndDelete() async throws {
        let participant = String(repeating: "1", count: 32)
        let session = try AmbientSession(
            sessionRef: String(repeating: "0", count: 32), participants: [participant],
            policy: AmbientSessionPolicy(
                revision: String(repeating: "a", count: 64), maxLifetimeMs: 3_600_000,
                maxArtifacts: 64, maxBytes: 64 * 1024),
            clock: { 0 }, automaticExpiry: false)
        try session.consent(participant)
        try session.transition("start")
        let references = await withTaskGroup(
            of: AmbientReference?.self, returning: [AmbientReference].self
        ) { group in
            for index in 0..<32 {
                group.addTask {
                    try? session.retain(
                        artifactRef: String(format: "%032x", index + 1), kind: .audio,
                        content: Data(repeating: UInt8(index + 1), count: 16))
                }
            }
            for _ in 0..<32 {
                group.addTask {
                    _ = session.snapshot()
                    return nil
                }
            }
            return await group.reduce(into: []) { if let reference = $1 { $0.append(reference) } }
        }
        XCTAssertEqual(references.count, 32)
        XCTAssertEqual(session.snapshot().ownedArtifacts, 32)
        XCTAssertEqual(session.snapshot().ownedBytes, 32 * 16)

        let lengths = await withTaskGroup(of: Int?.self, returning: [Int?].self) { group in
            for (index, reference) in references.enumerated() {
                group.addTask {
                    if index == references.count / 2 {
                        session.delete()
                        return nil
                    }
                    return try? session.use(reference) { bytes in
                        bytes.allSatisfy { $0 != 0 } ? bytes.count : -1
                    }
                }
            }
            return await group.reduce(into: []) { $0.append($1) }
        }
        // Reads either see a whole owned artifact or are refused after deletion.
        XCTAssertTrue(lengths.compactMap { $0 }.allSatisfy { $0 == 16 })
        let deleted = session.snapshot()
        XCTAssertEqual(deleted.state, .deleted)
        XCTAssertEqual(deleted.ownedArtifacts, 0)
        XCTAssertTrue(deleted.deletionVerified)
    }

    func testGeneratedTextGuardNeverReleasesRawTextUnderContention() {
        let asset = String(repeating: "a", count: 64)
        let guardInstance = GeneratedTextGuard(
            detector: { text in
                text.hasPrefix("Mira Vale") ? [GeneratedPHISpan(start: 0, end: 9, label: .name)] : []
            },
            detectorVersion: "synthetic-concurrency-v1")
        let released = OSAllocatedUnfairLock(initialState: [String?]())
        DispatchQueue.concurrentPerform(iterations: contention) { _ in
            let text = guardInstance.release("Mira Vale was seen.", assetDigest: asset).text
            released.withLock { $0.append(text) }
        }
        let texts = released.withLock { $0 }
        // A caller that finds the detector busy is withheld; nobody sees raw text.
        XCTAssertEqual(texts.count, contention)
        XCTAssertTrue(texts.allSatisfy { $0 == nil || $0 == "[REDACTED] was seen." })
        XCTAssertGreaterThanOrEqual(texts.filter { $0 != nil }.count, 1)
        let sequential = guardInstance.release("Mira Vale was seen.", assetDigest: asset)
        XCTAssertEqual(sequential.text, "[REDACTED] was seen.")
    }

    func testReplayStoreClaimsEachDigestOnceUnderContention() {
        let store = AgentInMemoryReceiptReplayStore()
        let digests = (0..<4).map { "sha256:" + String(repeating: String($0), count: 64) }
        let claims = OSAllocatedUnfairLock(initialState: [String: Int]())
        DispatchQueue.concurrentPerform(iterations: contention) { index in
            let digest = digests[index % digests.count]
            if (try? store.claim(tokenDigest: digest, expiresAt: 100, now: 1)) == true {
                claims.withLock { $0[digest, default: 0] += 1 }
            }
        }
        XCTAssertEqual(claims.withLock { $0 }, Dictionary(uniqueKeysWithValues: digests.map { ($0, 1) }))
    }

    func testMultimodalThermalSessionAdmitsOneConcurrentRunAtATime() async throws {
        let budget = try MultimodalThermalBudget(
            maxQueueMs: 1_000, maxLatencyMs: 1_000, maxOutputAgeMs: 1_000, recoveryMs: 10,
            historyAgeMs: 1_000, reducedCadenceMs: 10, normalWindowMs: 100, reducedWindowMs: 50)
        let session = MultimodalThermalSession(
            budget: budget, source: NominalThermalSource(), monotonicMilliseconds: { 1_000 })
        let outcomes = await withTaskGroup(
            of: MultimodalAdmissionOutcome<Int>?.self,
            returning: [MultimodalAdmissionOutcome<Int>].self
        ) { group in
            for index in 0..<contention {
                group.addTask {
                    await session.run(YieldingWork(value: index), queuedAtMilliseconds: 1_000)
                }
            }
            for _ in 0..<contention {
                group.addTask {
                    _ = session.evidence()
                    return nil
                }
            }
            return await group.reduce(into: []) { if let outcome = $1 { $0.append(outcome) } }
        }
        let evidence = session.evidence()
        XCTAssertEqual(outcomes.count, contention)
        XCTAssertEqual(evidence.attempts, contention)
        XCTAssertEqual(evidence.completed + evidence.withheld, contention)
        XCTAssertEqual(outcomes.filter { $0.reason == .ready }.count, evidence.completed)
        XCTAssertGreaterThanOrEqual(evidence.completed, 1)
        XCTAssertTrue(outcomes.allSatisfy { ($0.value != nil) == ($0.reason == .ready) })
        session.close()
        let closed = await session.run(YieldingWork(value: 0), queuedAtMilliseconds: 1_000)
        XCTAssertEqual(closed.reason, .closed)
        XCTAssertNil(closed.value)
    }

    func testThermalSubscriptionCancelsOnceUnderContention() async {
        let calls = OSAllocatedUnfairLock(initialState: 0)
        let subscription = AppleSystemThermalSource().observe { calls.withLock { $0 += 1 } }
        NotificationCenter.default.post(name: ProcessInfo.thermalStateDidChangeNotification, object: nil)
        XCTAssertEqual(calls.withLock { $0 }, 1)
        await withTaskGroup(of: Void.self) { group in
            for _ in 0..<contention { group.addTask { subscription.cancel() } }
        }
        NotificationCenter.default.post(name: ProcessInfo.thermalStateDidChangeNotification, object: nil)
        XCTAssertEqual(calls.withLock { $0 }, 1)
    }

    #if canImport(MLX) && canImport(Tokenizers) && !os(watchOS) && !os(visionOS)
        func testBlockingTokenizerBridgeHandsEachCallerItsOwnResult() {
            // The synchronous tokenizer loaders bridge to async work through a
            // detached task; every blocked caller must receive its own outcome.
            let refusals = OSAllocatedUnfairLock(initialState: 0)
            DispatchQueue.concurrentPerform(iterations: contention) { _ in
                do {
                    _ = try OpenMed.loadTokenizer(
                        tokenizerName: "synthetic/offline", tokenizerFolderURL: nil,
                        allowNetworkAccess: false)
                } catch TokenizerError.missingConfig {
                    refusals.withLock { $0 += 1 }
                } catch {}
            }
            XCTAssertEqual(refusals.withLock { $0 }, contention)
        }
    #endif

    func testAppleMicrophoneSourceStopIsSafeFromManyThreads() {
        // Construction and stop never open the microphone, so this runs headless.
        let source = AppleMicrophoneSource()
        let stopped = OSAllocatedUnfairLock(initialState: 0)
        DispatchQueue.concurrentPerform(iterations: contention) { _ in
            if source.stop() { stopped.withLock { $0 += 1 } }
        }
        XCTAssertEqual(stopped.withLock { $0 }, contention)
    }
}
