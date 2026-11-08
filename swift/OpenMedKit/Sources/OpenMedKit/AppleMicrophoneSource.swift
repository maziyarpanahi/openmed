import AVFAudio
import Foundation

#if os(iOS) || os(visionOS)
    import UIKit
#elseif canImport(AppKit)
    import AppKit
#endif

/// Explicitly started, bounded local microphone source. Construction and permission
/// requests do not create an engine, install a tap, or start recording.
public final class AppleMicrophoneSource: AmbientAudioSource, @unchecked Sendable {
    private let lock = NSLock()
    private let engineQueue = DispatchQueue(label: "org.openmed.capture.engine")
    private let deliveryQueue = DispatchQueue(label: "org.openmed.capture.delivery")
    private var engine: AVAudioEngine?
    private var active = false
    private var generation: UInt64 = 0
    private var pending: CapturePCMBuffer?
    private var delivering = false
    private var event: (@Sendable (CaptureInterruption) -> Void)?
    private var observers: [NSObjectProtocol] = []

    public init() {
        let center = NotificationCenter.default
        observers.append(
            center.addObserver(forName: .AVAudioEngineConfigurationChange, object: nil, queue: nil) { [weak self] note in
                guard let self else { return }
                self.lock.lock()
                let matches = self.engine === (note.object as? AVAudioEngine)
                self.lock.unlock()
                if matches { self.notify(.routeChanged) }
            })
        #if os(iOS) || os(visionOS) || os(watchOS)
            observers.append(
                center.addObserver(forName: AVAudioSession.interruptionNotification, object: nil, queue: nil) { [weak self] _ in
                    // Both begin and end remain stopped; never honor shouldResume automatically.
                    self?.notify(.interruption)
                })
            observers.append(
                center.addObserver(forName: AVAudioSession.routeChangeNotification, object: nil, queue: nil) { [weak self] _ in
                    self?.notify(.routeChanged)
                })
        #endif
        #if os(iOS) || os(visionOS)
            observers.append(
                center.addObserver(forName: UIApplication.willResignActiveNotification, object: nil, queue: nil) { [weak self] _ in
                    self?.notify(.background)
                })
        #elseif canImport(AppKit)
            observers.append(
                center.addObserver(forName: NSApplication.didResignActiveNotification, object: nil, queue: nil) { [weak self] _ in
                    self?.notify(.background)
                })
        #endif
    }

    deinit {
        for observer in observers { NotificationCenter.default.removeObserver(observer) }
        _ = stop()
    }

    public var permission: CapturePermission {
        switch AVAudioApplication.shared.recordPermission {
        case .granted: return .granted
        case .denied: return .denied
        case .undetermined: return .undetermined
        @unknown default: return .unavailable
        }
    }

    public func requestPermission(_ completion: @escaping @Sendable (CapturePermission) -> Void) {
        guard let purpose = Bundle.main.object(forInfoDictionaryKey: "NSMicrophoneUsageDescription") as? String,
            !purpose.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else {
            completion(.unavailable)
            return
        }
        AVAudioApplication.requestRecordPermission { granted in completion(granted ? .granted : .denied) }
    }

    private func notify(_ reason: CaptureInterruption) {
        lock.lock()
        guard active else {
            lock.unlock()
            return
        }
        active = false
        let callback = event
        let revoked = pending
        pending = nil
        lock.unlock()
        revoked?.clear()
        deliveryQueue.async { callback?(reason) }
    }

    public func start(
        receive: @escaping @Sendable (CapturePCMBuffer) -> Void,
        interrupted: @escaping @Sendable (CaptureInterruption) -> Void
    ) throws {
        guard permission == .granted,
            let purpose = Bundle.main.object(forInfoDictionaryKey: "NSMicrophoneUsageDescription") as? String,
            !purpose.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else { throw AmbientAudioCaptureError.permissionRequired }
        try engineQueue.sync {
            lock.lock()
            guard engine == nil, !delivering else {
                lock.unlock()
                throw AmbientAudioCaptureError.invalidTransition
            }
            generation += 1
            let current = generation
            event = interrupted
            lock.unlock()
            #if os(iOS) || os(visionOS) || os(watchOS)
                // The host owns AVAudioSession category/activation and restores its own policy.
                guard AVAudioSession.sharedInstance().isInputAvailable else { throw AmbientAudioCaptureError.sourceFailed }
            #endif
            let newEngine = AVAudioEngine()
            let input = newEngine.inputNode
            let format = input.outputFormat(forBus: 0)
            guard format.commonFormat == .pcmFormatFloat32, !format.isInterleaved,
                (1...8).contains(format.channelCount), format.sampleRate.isFinite,
                (8000...192_000).contains(format.sampleRate), format.sampleRate.rounded() == format.sampleRate
            else { throw AmbientAudioCaptureError.sourceFailed }
            lock.lock()
            engine = newEngine
            active = true
            lock.unlock()
            input.installTap(onBus: 0, bufferSize: 1024, format: format) { [weak self] pcm, time in
                self?.offer(pcm, time: time, generation: current, receive: receive)
            }
            do {
                newEngine.prepare()
                try newEngine.start()
            } catch {
                newEngine.stop()
                input.removeTap(onBus: 0)
                lock.lock()
                engine = nil
                active = false
                lock.unlock()
                throw AmbientAudioCaptureError.sourceFailed
            }
        }
    }

    // Shared by the actual microphone tap and synthetic native-buffer tests.
    static func copyInput(_ pcm: AVAudioPCMBuffer, time: AVAudioTime) throws -> CapturePCMBuffer {
        let frames = Int(pcm.frameLength)
        guard (1...CapturePCMBuffer.maximumFrames).contains(frames), time.isSampleTimeValid,
            time.sampleRate == pcm.format.sampleRate,
            let channels = pcm.floatChannelData, !pcm.format.isInterleaved,
            (1...8).contains(pcm.format.channelCount), pcm.format.sampleRate.isFinite,
            (8000...192_000).contains(pcm.format.sampleRate), pcm.format.sampleRate.rounded() == pcm.format.sampleRate
        else { throw AmbientAudioCaptureError.invalidBuffer }
        let mono = UnsafeMutablePointer<Float>.allocate(capacity: frames)
        mono.initialize(repeating: 0, count: frames)
        defer {
            mono.update(repeating: 0, count: frames)
            mono.deinitialize(count: frames)
            mono.deallocate()
        }
        for channel in 0..<Int(pcm.format.channelCount) {
            for frame in 0..<frames { mono[frame] += channels[channel][frame] / Float(pcm.format.channelCount) }
        }
        return try CapturePCMBuffer(
            samples: UnsafeBufferPointer(start: mono, count: frames),
            sampleRateHz: Int(pcm.format.sampleRate), startSample: time.sampleTime)
    }

    private func offer(
        _ pcm: AVAudioPCMBuffer, time: AVAudioTime, generation expected: UInt64,
        receive: @escaping @Sendable (CapturePCMBuffer) -> Void
    ) {
        // One pending/borrowed buffer, no unbounded dispatch backlog or audio files.
        guard lock.try() else { return }
        guard active, generation == expected, !delivering else {
            lock.unlock()
            return
        }
        let buffer: CapturePCMBuffer
        do { buffer = try Self.copyInput(pcm, time: time) } catch {
            lock.unlock()
            notify(.sourceFailure)
            return
        }
        pending = buffer
        delivering = true
        lock.unlock()
        deliveryQueue.async { [weak self] in
            guard let self else {
                buffer.clear()
                return
            }
            self.lock.lock()
            let permitted = self.active && self.generation == expected && self.pending === buffer
            self.lock.unlock()
            if permitted { receive(buffer) }
            buffer.clear()
            self.lock.lock()
            if self.pending === buffer { self.pending = nil }
            self.delivering = false
            self.lock.unlock()
        }
    }

    public func stop() -> Bool {
        lock.lock()
        active = false
        generation += 1
        let revoked = pending
        pending = nil
        event = nil
        lock.unlock()
        revoked?.clear()
        let stopped = engineQueue.sync {
            lock.lock()
            let old = engine
            engine = nil
            lock.unlock()
            old?.stop()
            old?.inputNode.removeTap(onBus: 0)
            return old?.isRunning != true
        }
        // Even a callback dispatched before stop now holds an overwritten buffer.
        return stopped && (revoked?.isCleared ?? true)
    }
}
