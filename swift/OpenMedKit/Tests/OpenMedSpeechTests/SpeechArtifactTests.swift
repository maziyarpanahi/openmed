import CryptoKit
import Foundation
import OpenMedKit
import WhisperKit
import XCTest

@testable import OpenMedSpeech

final class SpeechArtifactTests: XCTestCase {
    func testWordAlignmentOutsideSourceIsWithheldWithoutClipping() throws {
        let segment = TranscriptionSegment(text: "synthetic", tokens: [1], avgLogprob: -0.5, words: [WordTiming(word: "synthetic", tokens: [1], start: 0, end: 0.12, probability: 0.75)])
        let raw = TranscriptionResult(text: "synthetic", segments: [segment], language: "en", timings: TranscriptionTimings())
        let withheld = try CoreMLSpeechEngine.recognition(results: [raw], frames: 1600, tokenizationVerified: true)
        XCTAssertEqual(withheld.text, "synthetic")
        XCTAssertEqual(withheld.alignmentStatus, .outsideWindow)
        XCTAssertTrue(withheld.tokenTimes.isEmpty)
        XCTAssertEqual(withheld.decoderConfidencePPM, 606_531)
        let aligned = try CoreMLSpeechEngine.recognition(results: [raw], frames: 3200, tokenizationVerified: true)
        XCTAssertEqual(aligned.alignmentStatus, .aligned)
        XCTAssertEqual(aligned.tokenTimes.count, 1)
        let unverified = try CoreMLSpeechEngine.recognition(results: [raw], frames: 3200, tokenizationVerified: false)
        XCTAssertEqual(unverified.alignmentStatus, .unverifiedTokenization)
        XCTAssertTrue(unverified.tokenTimes.isEmpty)
    }

    func testMissingVerificationRefusesBeforeModelLoad() async {
        do {
            _ = try await CoreMLSpeechEngine.load(artifact: nil)
            XCTFail("unverified artifact admitted")
        } catch { XCTAssertEqual(error as? StreamingSpeechError, .unverifiedArtifact) }
    }

    func testMissingBundleAndNonFileURLRefuseWithControlledReason() throws {
        XCTAssertThrowsError(try VerifiedSpeechArtifact.verifyTinyEnglish(at: URL(string: "https://example.invalid/model")!)) {
            XCTAssertEqual($0 as? StreamingSpeechError, .unverifiedArtifact)
        }
        XCTAssertThrowsError(try VerifiedSpeechArtifact.verifyTinyEnglish(at: URL(fileURLWithPath: "/nonexistent/synthetic-model"))) {
            XCTAssertEqual($0 as? StreamingSpeechError, .artifactMismatch)
        }
    }

    func testHashVerifierRejectsChangedTruncatedAndSymlinkedContent() throws {
        let directory = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let file = directory.appending(path: "synthetic.bin")
        let data = Data("synthetic".utf8)
        let pin = SpeechArtifactFile(path: "synthetic.bin", sha256: SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined(), byteCount: data.count)
        try data.write(to: file)
        try VerifiedSpeechArtifact.verifyFile(file, expected: pin)
        try Data("different".utf8).write(to: file)
        XCTAssertThrowsError(try VerifiedSpeechArtifact.verifyFile(file, expected: pin))
        try Data("short".utf8).write(to: file)
        XCTAssertThrowsError(try VerifiedSpeechArtifact.verifyFile(file, expected: pin))
        try data.write(to: file)
        let link = directory.appending(path: "link.bin")
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: file)
        XCTAssertThrowsError(try VerifiedSpeechArtifact.verifyFile(link, expected: pin))
    }

    func testComputeNeverSilentlySubstitutesUnavailableAccelerator() throws {
        XCTAssertThrowsError(try SpeechCompute.cpuOnly.validate(available: [], simulator: false))
        XCTAssertThrowsError(try SpeechCompute.cpuAndGPU.validate(available: ["cpu"], simulator: false))
        XCTAssertThrowsError(try SpeechCompute.cpuAndNeuralEngine.validate(available: ["cpu", "gpu"], simulator: false))
        XCTAssertThrowsError(try SpeechCompute.cpuAndGPU.validate(available: ["cpu", "gpu"], simulator: true))
        XCTAssertEqual(try SpeechCompute.cpuOnly.validate(available: ["cpu"], simulator: true), .cpuOnly)
        XCTAssertEqual(try SpeechCompute.cpuAndGPU.validate(available: ["cpu", "gpu"], simulator: false), .cpuAndGPU)
    }

    /// Opt-in real local inference. No microphone or download is used by this test.
    func testPinnedNativeSyntheticSpeechWhenProvided() async throws {
        guard let path = ProcessInfo.processInfo.environment["OPENMED_TEST_SPEECH_MODEL"],
            let audioPath = ProcessInfo.processInfo.environment["OPENMED_TEST_SYNTHETIC_PCM"]
        else { throw XCTSkip("Optional caller-supplied native speech fixture absent") }
        let artifact = try VerifiedSpeechArtifact.verifyTinyEnglish(at: URL(fileURLWithPath: path))
        let engine = try await CoreMLSpeechEngine.load(artifact: artifact)
        let data = try Data(contentsOf: URL(fileURLWithPath: audioPath))
        guard data.count % 4 == 0 else {
            XCTFail("invalid synthetic PCM fixture")
            return
        }
        let samples = data.withUnsafeBytes { values in
            stride(from: 0, to: data.count, by: 4).map { Float(bitPattern: UInt32(littleEndian: values.loadUnaligned(fromByteOffset: $0, as: UInt32.self))) }
        }
        let input = try samples.withUnsafeBufferPointer { try SpeechPCMBuffer(samples: $0, startSample: 0) }
        let output = NativeSpeechEvents()
        let session = try StreamingSpeechSession(engine: engine) { output.append($0) }
        do {
            try await session.append(input, endOfUtterance: true)
            try await session.finish()
        } catch {
            await session.cancel()
            throw error
        }
        XCTAssertTrue(input.isCleared)
        let events = output.values
        let final = try XCTUnwrap(
            events.compactMap {
                if case .finalized(let value) = $0 { return value }
                return nil
            }.first)
        XCTAssertTrue(final.text.lowercased().contains("package"))
        XCTAssertNotNil(final.decoderConfidencePPM)
        if final.alignmentStatus == .aligned {
            XCTAssertFalse(final.tokenTimes.isEmpty)
        } else {
            XCTAssertEqual(final.alignmentStatus, .outsideWindow)
            XCTAssertTrue(final.tokenTimes.isEmpty)
        }
        XCTAssertFalse(final.allowsAutomaticClinicalFinalization)
        XCTAssertEqual(session.snapshot().state, .finished)
        XCTAssertEqual(session.snapshot().bufferedFrames, 0)
    }
}

private final class NativeSpeechEvents: @unchecked Sendable {
    private let lock = NSLock()
    private var stored = [SpeechStreamEvent]()
    var values: [SpeechStreamEvent] {
        lock.lock()
        defer { lock.unlock() }
        return stored
    }
    func append(_ event: SpeechStreamEvent) {
        lock.lock()
        defer { lock.unlock() }
        stored.append(event)
    }
}
