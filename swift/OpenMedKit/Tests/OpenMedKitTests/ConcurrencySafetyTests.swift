import Foundation
import XCTest
import os

@testable import OpenMedKit

#if canImport(Tokenizers)
    import Tokenizers
#endif

#if canImport(CoreGraphics)
    import CoreGraphics
#endif

#if canImport(UIKit) && !os(watchOS)
    import UIKit
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

private struct SilentMemoryPressure: OpenMedMemoryPressureSource {
    func observe(
        _ handler: @escaping @Sendable (OpenMedMemoryPressureLevel) -> Void
    ) -> any OpenMedRuntimeObservation {
        OpenMedInactiveObservation()
    }
}

/// Records how many predictions overlap; the runtime must never overlap them.
private final class OverlapRecordingModel: OpenMedRuntimeModel {
    private let counts = OSAllocatedUnfairLock(initialState: (running: 0, peak: 0, calls: 0))

    var peak: Int { counts.withLock { $0.peak } }
    var calls: Int { counts.withLock { $0.calls } }

    func predictEntities(
        in text: String, checkpoint: OpenMedInferenceCheckpoint
    ) async throws -> [EntityPrediction] {
        try checkpoint.check(.tokenization)
        counts.withLock {
            $0.running += 1
            $0.peak = max($0.peak, $0.running)
            $0.calls += 1
        }
        await Task.yield()
        counts.withLock { $0.running -= 1 }
        try checkpoint.check(.modelPrediction)
        return []
    }
}

/// The signed synthetic bundle of `tests/fixtures/bundle_cache/vectors.json`.
private struct SyntheticBundle {
    let bundleID: String
    let generation: String
    let manifest: Data
    let envelope: Data
    let files: [String: Data]
    let trustPolicy: PortableBundleTrustPolicy
    let context: PortableBundleContext

    init() throws {
        var root = URL(fileURLWithPath: #filePath)
        for _ in 0..<5 { root.deleteLastPathComponent() }
        let data = try Data(contentsOf: root.appending(path: "tests/fixtures/bundle_cache/vectors.json"))
        let vectors = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let identities = try XCTUnwrap(vectors["identities"] as? [String: Any])
        let bundles = try XCTUnwrap(vectors["bundles"] as? [String: [String: Any]])
        let bundle = try XCTUnwrap(bundles["v1"])
        let policies = try XCTUnwrap(vectors["trust_policies"] as? [String: Any])
        let context = try XCTUnwrap(vectors["context"] as? [String: Any])
        bundleID = try XCTUnwrap(identities["bundle_id"] as? String)
        generation = try XCTUnwrap(bundle["generation"] as? String)
        manifest = Data(try XCTUnwrap(bundle["manifest"] as? String).utf8)
        envelope = Data(try XCTUnwrap(bundle["envelope"] as? String).utf8)
        var files: [String: Data] = [:]
        for entry in try XCTUnwrap(bundle["files"] as? [[String: Any]]) {
            files[try XCTUnwrap(entry["path"] as? String)] = Data(try XCTUnwrap(entry["text"] as? String).utf8)
        }
        self.files = files
        trustPolicy = try PortableBundleTrustPolicy(
            json: JSONSerialization.data(withJSONObject: try XCTUnwrap(policies["default"])))
        self.context = try PortableBundleContext(now: try XCTUnwrap(context["now"] as? String))
    }

    /// Writes the bundle files under `directory`.
    func write(to directory: URL) throws {
        for (path, data) in files {
            let file = directory.appending(path: path)
            try FileManager.default.createDirectory(
                at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
            try data.write(to: file)
        }
    }

    /// An acquisition that reads the files written to `directory`.
    func request(from directory: URL) -> OpenMedBundleAcquireRequest {
        OpenMedBundleAcquireRequest(
            manifest: manifest, envelope: envelope,
            source: OpenMedDirectoryBundleSource(directory: directory, identity: "source-a"),
            trustPolicy: trustPolicy, context: context)
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
                } catch OpenMedTokenizerError.missingLocalAssets {
                    refusals.withLock { $0 += 1 }
                } catch {}
            }
            XCTAssertEqual(refusals.withLock { $0 }, contention)
        }
    #endif

    func testInferenceLeasesKeepTheirFirstCancellationUnderContention() async {
        let registry = OpenMedLeaseRegistry()
        let interrupts = OSAllocatedUnfairLock(initialState: [UInt64: Int]())
        let leases = (0..<contention).map {
            OpenMedInferenceLease(id: UInt64($0), clock: OpenMedSystemClock(), probe: nil)
        }
        for lease in leases {
            lease.attachInterrupt { interrupts.withLock { $0[lease.id, default: 0] += 1 } }
        }
        await withTaskGroup(of: Void.self) { group in
            for lease in leases { group.addTask { registry.insert(lease) } }
        }
        XCTAssertEqual(registry.count, contention)

        // Per-lease cancels, stage checks and registry-wide cancels race; the
        // first reason must win and each interrupt must fire exactly once.
        await withTaskGroup(of: Void.self) { group in
            for (index, lease) in leases.enumerated() {
                group.addTask {
                    if index.isMultiple(of: 2) { lease.cancel(.backgroundExpired) }
                    do { try lease.check(.windowing) } catch {}
                }
            }
            for _ in 0..<4 { group.addTask { registry.cancelAll(.memoryPressure) } }
        }
        for (index, lease) in leases.enumerated() {
            let reason = lease.cancellation?.reason
            let allowed: [OpenMedRuntimeError] =
                index.isMultiple(of: 2) ? [.backgroundExpired, .memoryPressure] : [.memoryPressure]
            XCTAssertTrue(allowed.contains { $0 == reason }, "lease \(index)")
            XCTAssertThrowsError(try lease.check(.merging)) { error in
                XCTAssertEqual(error as? OpenMedRuntimeError, reason)
            }
        }
        XCTAssertEqual(
            interrupts.withLock { $0 }, Dictionary(uniqueKeysWithValues: leases.map { ($0.id, 1) }))

        await withTaskGroup(of: Void.self) { group in
            for lease in leases { group.addTask { registry.remove(lease) } }
        }
        XCTAssertEqual(registry.count, 0)
    }

    func testRuntimeRunsOneModelSessionAtATimeForConcurrentCallers() async {
        // OpenMedTextModelSession wraps a non-Sendable OpenMed and relies on this
        // serialization, so the runtime is exercised with a recording model.
        let model = OverlapRecordingModel()
        let loads = OSAllocatedUnfairLock(initialState: 0)
        let runtime = OpenMedRuntime(
            configuration: OpenMedRuntimeConfiguration(
                maxQueuedRequests: contention, idleReleaseMilliseconds: nil),
            memoryPressure: SilentMemoryPressure(),
            loader: {
                loads.withLock { $0 += 1 }
                return model
            })
        let completed = await withTaskGroup(of: Bool.self, returning: Int.self) { group in
            for _ in 0..<contention {
                group.addTask { (try? await runtime.analyzeText("Synthetic note.")) != nil }
            }
            return await group.reduce(0) { $0 + ($1 ? 1 : 0) }
        }
        XCTAssertEqual(completed, contention)
        XCTAssertEqual(model.calls, contention)
        XCTAssertEqual(model.peak, 1)
        XCTAssertEqual(loads.withLock { $0 }, 1)
        await runtime.shutdown()
        let snapshot = await runtime.snapshot()
        XCTAssertFalse(snapshot.hasResidentModel)
        XCTAssertEqual(snapshot.activeLeases, 0)
    }

    func testModelBoxHandsItsSessionToExactlyOneTaker() async {
        let box = OpenMedModelBox(OverlapRecordingModel())
        let taken = await withTaskGroup(of: Bool.self, returning: Int.self) { group in
            for _ in 0..<contention { group.addTask { box.take() != nil } }
            return await group.reduce(0) { $0 + ($1 ? 1 : 0) }
        }
        XCTAssertEqual(taken, 1)

        box.store(OverlapRecordingModel())
        let raced = await withTaskGroup(of: Bool.self, returning: Int.self) { group in
            for index in 0..<contention {
                group.addTask {
                    if index == contention / 2 {
                        box.discard()
                        return false
                    }
                    return box.take() != nil
                }
            }
            return await group.reduce(0) { $0 + ($1 ? 1 : 0) }
        }
        // A discard that races takers leaves at most one owner and an empty box.
        XCTAssertLessThanOrEqual(raced, 1)
        XCTAssertNil(box.take())
    }

    func testSignalSequencerIssuesEveryValueOnce() {
        let sequencer = OpenMedSignalSequencer()
        let values = OSAllocatedUnfairLock(initialState: [UInt64]())
        DispatchQueue.concurrentPerform(iterations: contention) { _ in
            let value = sequencer.next()
            values.withLock { $0.append(value) }
        }
        let issued = values.withLock { $0 }
        XCTAssertEqual(issued.count, contention)
        XCTAssertEqual(Set(issued), Set(1...UInt64(contention)))
    }

    func testMemoryPressureObservationCancelsSafelyFromManyThreads() {
        let observation = OpenMedDispatchMemoryPressureSource().observe { _ in }
        // Every caller may cancel; the dispatch source is cancelled exactly once.
        DispatchQueue.concurrentPerform(iterations: contention) { _ in observation.cancel() }
        observation.cancel()
    }

    func testPortableBundlePatternsMatchTheSameFromEveryTask() async {
        let cases: [(PortableBundlePattern, String, Bool)] = [
            (.sha256, "sha256:" + String(repeating: "a", count: 64), true),
            (.sha256, "sha256:" + String(repeating: "A", count: 64), false),
            (.timestamp, "2026-11-07T00:00:00Z", true),
            (.timestamp, "2026-13-07T00:00:00Z", false),
            (.bundleID, "openmed.synthetic.tiny-pii", true),
            (.bundleID, "openmed.synthétic", false),
        ]
        let expected = cases.map(\.2)
        let observed = await withTaskGroup(of: [Bool].self, returning: [[Bool]].self) { group in
            for _ in 0..<contention {
                group.addTask { cases.map { pattern, text, _ in pattern.matches(text) } }
            }
            return await group.reduce(into: []) { $0.append($1) }
        }
        XCTAssertEqual(observed.count, contention)
        XCTAssertTrue(observed.allSatisfy { $0 == expected })
    }

    func testBundleCacheSharesOneVerifiedGenerationAndItsLeasesAcrossThreads() throws {
        let bundle = try SyntheticBundle()
        let root = FileManager.default.temporaryDirectory.appending(
            path: "openmed-bundle-cache-\(UUID().uuidString)", directoryHint: .isDirectory)
        defer { try? FileManager.default.removeItem(at: root) }
        let source = root.appending(path: "source")
        try bundle.write(to: source)
        let clockReads = OSAllocatedUnfairLock(initialState: 0)
        // One options value whose clock every caller thread reads.
        let options = OpenMedBundleCacheOptions(
            chunkBytes: 16, lockTimeoutMilliseconds: 30_000,
            now: {
                clockReads.withLock { $0 += 1 }
                return 1_793_577_600_000
            })
        var otherOptions = options
        otherOptions.ownerID = UUID().uuidString
        // Two instances over one directory contend like two processes, through
        // flock and the process-wide mutex registry.
        let caches = [
            try OpenMedBundleCache(rootDirectory: root.appending(path: "cache"), options: options),
            try OpenMedBundleCache(rootDirectory: root.appending(path: "cache"), options: otherOptions),
        ]

        let acquired = OSAllocatedUnfairLock(initialState: [OpenMedBundleAcquireResult]())
        DispatchQueue.concurrentPerform(iterations: 8) { index in
            let result = caches[index % caches.count].acquire(bundle.request(from: source))
            acquired.withLock { $0.append(result) }
        }
        let results = acquired.withLock { $0 }
        XCTAssertEqual(results.count, 8)
        XCTAssertTrue(results.allSatisfy(\.ok), "\(results.map { $0.reason?.rawValue ?? "ok" })")
        XCTAssertEqual(results.filter { !$0.fromCache }.count, 1)
        XCTAssertEqual(Set(results.compactMap(\.generation)), [bundle.generation])

        let reads = OSAllocatedUnfairLock(initialState: [Bool]())
        DispatchQueue.concurrentPerform(iterations: contention) { index in
            guard let lease = try? caches[index % caches.count].open(bundleID: bundle.bundleID) else {
                reads.withLock { $0.append(false) }
                return
            }
            // A leased generation is complete and stays readable until released.
            let intact = bundle.files.allSatisfy { path, data in
                (try? Data(contentsOf: lease.payloadDirectory.appending(path: path))) == data
            }
            reads.withLock { $0.append(intact && lease.generation == bundle.generation) }
            lease.release()
            lease.release()
        }
        XCTAssertEqual(reads.withLock { $0 }.count, contention)
        XCTAssertTrue(reads.withLock { $0 }.allSatisfy { $0 })

        // One OpenMedBundleLease released from many threads at once ends once.
        let lease: OpenMedBundleLease = try caches[0].lease(bundleID: bundle.bundleID)
        DispatchQueue.concurrentPerform(iterations: contention) { _ in lease.release() }
        let snapshot = try caches[1].snapshot(bundleID: bundle.bundleID)
        XCTAssertEqual(snapshot.active, bundle.generation)
        XCTAssertFalse(snapshot.generations.contains { $0.protections.contains(.leased) })
        XCTAssertGreaterThan(clockReads.withLock { $0 }, 0)
    }

    func testManagedBundleImportersSerializeTheirSharedStateDirectory() throws {
        let suite = "org.openmed.tests.concurrency-\(UUID().uuidString)"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        let configuration = OpenMedManagedAppConfiguration(defaults: defaults)
        let root = FileManager.default.temporaryDirectory.appending(
            path: "openmed-managed-import-\(UUID().uuidString)", directoryHint: .isDirectory)
        defer { try? FileManager.default.removeItem(at: root) }
        let cache = try OpenMedBundleCache(rootDirectory: root.appending(path: "cache"))
        // Two importers over one state directory share one process-wide mutex.
        let importers = try (0..<2).map { _ in
            try OpenMedManagedBundleImporter(cache: cache, stateDirectory: root.appending(path: "state"))
        }
        let context = try PortableBundleContext(now: "2026-11-07T00:00:00Z")
        let attempts = OpenMedManagedBundleImporter.receiptCapacity
        let reasons = OSAllocatedUnfairLock(initialState: [OpenMedManagedImportReason?]())
        DispatchQueue.concurrentPerform(iterations: attempts) { index in
            let receipt = importers[index % 2].importManaged(from: configuration, context: context)
            reasons.withLock { $0.append(receipt.reason) }
            _ = importers[(index + 1) % 2].receipts()
        }
        // No managed configuration is installed, so every import defers; the
        // serialized read-modify-write of the state file keeps every receipt.
        let deferred = reasons.withLock { $0 }
        XCTAssertEqual(deferred.count, attempts)
        XCTAssertTrue(deferred.allSatisfy { $0 == .managedConfigAbsent })
        let receipts = importers[0].receipts()
        XCTAssertEqual(receipts.count, attempts)
        XCTAssertTrue(receipts.allSatisfy { $0.reason == .managedConfigAbsent })
        XCTAssertEqual(importers[1].receipts(), receipts)
    }

    #if canImport(UIKit) && !os(watchOS)
        @MainActor
        func testUIApplicationBackgroundExecutionEndsAndCancelsFromManyThreads() throws {
            // Unhosted test bundles have no UIApplication, and one must never be created.
            guard
                let application = (UIApplication.self as AnyObject).value(forKey: "sharedApplication")
                    as? UIApplication
            else {
                throw XCTSkip("This test bundle has no host UIApplication.")
            }
            let execution = OpenMedUIApplicationBackgroundExecution(application: application)
            let expirations = OSAllocatedUnfairLock(initialState: 0)
            let foregrounds = OSAllocatedUnfairLock(initialState: 0)
            let assertion = execution.beginBackgroundExecution { expirations.withLock { $0 += 1 } }
            let observation = execution.observeForeground { foregrounds.withLock { $0 += 1 } }
            DispatchQueue.concurrentPerform(iterations: contention) { _ in
                assertion.end()
                observation.cancel()
            }
            NotificationCenter.default.post(name: UIApplication.willEnterForegroundNotification, object: nil)
            XCTAssertEqual(foregrounds.withLock { $0 }, 0)
            XCTAssertEqual(expirations.withLock { $0 }, 0)
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

    #if canImport(CoreGraphics) && canImport(ImageIO)
        func testDocumentIntakeImageReleaseIsAtomicUnderContention() async throws {
            let context = try XCTUnwrap(
                CGContext(
                    data: nil, width: 4, height: 4, bitsPerComponent: 8, bytesPerRow: 16,
                    space: CGColorSpaceCreateDeviceRGB(),
                    bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue))
            let image = CGDocumentIntakeImage(try XCTUnwrap(context.makeImage()))
            await withTaskGroup(of: Void.self) { group in
                for index in 0..<contention {
                    group.addTask {
                        if index.isMultiple(of: 8) { image.release() }
                        // A reader sees the whole image or nothing, never a torn value.
                        if let pixels = image.cgImage { XCTAssertEqual(pixels.width, 4) }
                        _ = image.isReleased
                    }
                }
            }
            XCTAssertTrue(image.isReleased)
            XCTAssertNil(image.cgImage)
            XCTAssertEqual(image.pixelWidth, 4)
        }

        func testDocumentRedactionSessionRegistryKeepsEveryLiveSessionUnderContention() async throws {
            let root = FileManager.default.temporaryDirectory.appending(
                path: "openmed-registry-\(UUID().uuidString)", directoryHint: .isDirectory)
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: root) }
            let registry = DocumentRedactionSessionRegistry()
            let names = (0..<contention).map { "session-\($0)" }
            await withTaskGroup(of: Void.self) { group in
                for name in names {
                    group.addTask {
                        // Register before creating the directory, like the exporter.
                        registry.register(name, sweeping: root)
                        try? FileManager.default.createDirectory(
                            at: root.appending(path: name), withIntermediateDirectories: true)
                    }
                }
            }
            let live = Set(try FileManager.default.contentsOfDirectory(atPath: root.path))
            XCTAssertEqual(live, Set(names))

            let removed = Set(names.prefix(contention / 2))
            await withTaskGroup(of: Void.self) { group in
                for name in removed { group.addTask { registry.remove(name) } }
            }
            registry.register("session-sweeper", sweeping: root)
            let remaining = Set(try FileManager.default.contentsOfDirectory(atPath: root.path))
            XCTAssertEqual(remaining, Set(names).subtracting(removed))
        }

        func testDocumentRedactionExportersShareTheSessionRegistryUnderContention() async throws {
            let root = FileManager.default.temporaryDirectory.appending(
                path: "openmed-exporters-\(UUID().uuidString)", directoryHint: .isDirectory)
            defer { try? FileManager.default.removeItem(at: root) }
            let exporters = try await withThrowingTaskGroup(of: DocumentRedactionExporter.self) { group in
                for _ in 0..<16 {
                    group.addTask {
                        try DocumentRedactionExporter(recognizer: RefusingTextRecognizer(), directory: root)
                    }
                }
                var made: [DocumentRedactionExporter] = []
                for try await exporter in group { made.append(exporter) }
                return made
            }
            // Concurrent sweeps never remove another live exporter's session.
            let sessions = try FileManager.default.contentsOfDirectory(atPath: root.path)
            XCTAssertEqual(sessions.count, exporters.count)
        }
    #endif
}

#if canImport(CoreGraphics) && canImport(ImageIO)
    /// Recognizer double for exporter construction; exports are never run.
    private struct RefusingTextRecognizer: DocumentTextRecognizer {
        func recognize(_ image: any DocumentIntakeImage) async throws -> DocumentRecognition {
            throw CancellationError()
        }
    }
#endif
