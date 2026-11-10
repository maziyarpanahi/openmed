import XCTest

@testable import OpenMedKit

private final class SyntheticActivityDetector: LocalSpeechActivityDetector {
    var requiresNetwork = false
    var owned: [Float]?
    var resets = 0
    var failDetect = false
    var failReset = false

    func detect(_ samples: [Float]) throws -> SpeechActivity {
        owned = samples
        if failDetect { throw SpeechEndpointError.invalidFrame }
        let peak = samples.map { abs($0) }.max() ?? 0
        if peak > 0.5 { return .speech }
        return peak == 0 ? .silence : .uncertain
    }

    func reset() throws {
        owned = nil
        resets += 1
        if failReset { throw SpeechEndpointError.invalidFrame }
    }
}

final class SpeechEndpointTests: XCTestCase {
    private func adapter(
        _ detector: SyntheticActivityDetector = SyntheticActivityDetector(),
        hangover: Int = 3, maximum: Int = 10
    ) throws -> SpeechEndpointAdapter {
        try SpeechEndpointAdapter(
            detector: detector, hangoverSamples: hangover,
            maxUtteranceSamples: maximum, maxFrameSamples: 4)
    }

    private func rows(_ events: [SpeechEndpointEvent]) -> [String] {
        events.map { "\($0.kind):\($0.startSample):\($0.endSample):\($0.reason)" }
    }

    func testShortSpeechAndLongSilence() throws {
        let stream = try adapter()
        XCTAssertEqual(try stream.push(startSample: 100, samples: [0, 0, 0, 0]), [])
        XCTAssertEqual(
            rows(try stream.push(startSample: 104, samples: [0.8, 0.8])),
            ["speech_start:104:104:detected"])
        XCTAssertEqual(try stream.push(startSample: 106, samples: [0, 0]), [])
        XCTAssertEqual(
            rows(try stream.push(startSample: 108, samples: [0, 0, 0, 0])),
            ["speech_end:104:109:hangover"])
        for start in stride(from: 112, to: 20_112, by: 4) {
            XCTAssertEqual(try stream.push(startSample: start, samples: [0, 0, 0, 0]), [])
            XCTAssertEqual(stream.bufferedSampleCount, 0)
        }
        XCTAssertEqual(try stream.finish(), [])
    }

    func testLongSpeechConservesCoverage() throws {
        let stream = try adapter()
        var events: [SpeechEndpointEvent] = []
        for start in stride(from: 71, to: 2071, by: 4) {
            let result = try stream.push(startSample: start, samples: [0.8, 0.8, 0.8, 0.8])
            XCTAssertLessThanOrEqual(result.count, 4)
            events += result
        }
        events += try stream.finish()
        XCTAssertEqual(
            rows(events.filter { $0.kind == "speech_end" }),
            stride(from: 71, to: 2071, by: 10).map {
                "speech_end:\($0):\($0 + 10):maximum_duration"
            })
    }

    func testNoiseGapAndSourceClock() throws {
        let detector = SyntheticActivityDetector()
        let stream = try adapter(detector)
        XCTAssertEqual(
            rows(try stream.push(startSample: 0, samples: [0.1, -0.1, 0.1, -0.1])),
            ["uncertain_activity:0:4:uncertain"])
        _ = try stream.push(startSample: 4, samples: [0.8, 0.8, 0.8, 0.8])
        XCTAssertEqual(
            rows(try stream.push(startSample: 8, samples: [0.1, 0.1, 0.1, 0.1])),
            ["uncertain_activity:8:12:uncertain", "speech_end:4:11:hangover"])
        _ = try stream.reset()
        _ = try stream.push(startSample: 30, samples: [0.8, 0.8, 0.8, 0.8])
        XCTAssertEqual(
            rows(try stream.push(startSample: 40, samples: [0.8, 0.8, 0.8, 0.8])),
            [
                "speech_end:30:34:discontinuity", "discontinuity:34:40:gap",
                "speech_start:40:40:detected",
            ])
        XCTAssertEqual(detector.resets, 2)
        XCTAssertThrowsError(try stream.push(startSample: 43, samples: [0.8]))
        _ = try stream.reset()
        _ = try stream.push(startSample: Int.max - 4, samples: [0.8, 0.8, 0.8, 0.8])
        XCTAssertEqual(try stream.finish().first?.endSample, Int.max)
    }

    func testResetCancelFinishDiscardAudio() throws {
        for operation in ["reset", "cancelled", "stream_end"] {
            let detector = SyntheticActivityDetector()
            let stream = try adapter(detector)
            _ = try stream.push(startSample: 50, samples: [0.8, 0.8, 0.8, 0.8])
            XCTAssertNotNil(detector.owned)
            let events: [SpeechEndpointEvent]
            switch operation {
            case "reset": events = try stream.reset()
            case "cancelled": events = try stream.cancel()
            default: events = try stream.finish()
            }
            XCTAssertEqual(rows(events), ["speech_end:50:54:\(operation)"])
            XCTAssertNil(detector.owned)
            XCTAssertEqual(stream.bufferedSampleCount, 0)
            if operation == "cancelled" {
                XCTAssertThrowsError(try stream.push(startSample: 0, samples: [0.8]))
                XCTAssertThrowsError(try stream.finish())
                XCTAssertEqual(try stream.reset(), [])
            }
            XCTAssertEqual(
                rows(try stream.push(startSample: 0, samples: [0.8])),
                ["speech_start:0:0:detected"])
        }
    }

    func testHangoverBoundariesAndResumedSpeech() throws {
        let zero = try adapter(hangover: 0)
        _ = try zero.push(startSample: 0, samples: [0.8, 0.8, 0.8, 0.8])
        XCTAssertEqual(
            rows(try zero.push(startSample: 4, samples: [0, 0, 0, 0])),
            ["speech_end:0:4:hangover"])
        let capped = try adapter(hangover: 10)
        _ = try capped.push(startSample: 0, samples: [0.8, 0.8, 0.8, 0.8])
        _ = try capped.push(startSample: 4, samples: [0, 0, 0, 0])
        XCTAssertEqual(
            rows(try capped.push(startSample: 8, samples: [0, 0, 0, 0])),
            ["speech_end:0:10:maximum_duration"])
        let resumed = try adapter(maximum: 20)
        _ = try resumed.push(startSample: 0, samples: [0.8, 0.8, 0.8, 0.8])
        _ = try resumed.push(startSample: 4, samples: [0, 0])
        _ = try resumed.push(startSample: 6, samples: [0.8, 0.8])
        _ = try resumed.push(startSample: 8, samples: [0, 0])
        XCTAssertEqual(
            rows(try resumed.push(startSample: 10, samples: [0, 0])),
            ["speech_end:0:11:hangover"])
    }

    func testInvalidInputsLocalityAndDetectorFailures() throws {
        for samples: [Float] in [[], [.nan], [.infinity], [1.1], [0, 0, 0, 0, 0]] {
            XCTAssertThrowsError(try adapter().push(startSample: 0, samples: samples))
        }
        XCTAssertThrowsError(try adapter().push(startSample: -1, samples: [0]))
        XCTAssertThrowsError(try adapter().push(startSample: Int.max, samples: [0]))
        XCTAssertThrowsError(try adapter(hangover: -1))
        XCTAssertThrowsError(try adapter(hangover: 11))
        XCTAssertThrowsError(try adapter(maximum: 0))
        let detector = SyntheticActivityDetector()
        detector.requiresNetwork = true
        XCTAssertThrowsError(try adapter(detector))
        detector.requiresNetwork = false
        let stream = try adapter(detector)
        detector.failDetect = true
        XCTAssertThrowsError(try stream.push(startSample: 0, samples: [0.8])) { error in
            XCTAssertEqual(String(describing: error), "detectorFailure")
        }
        XCTAssertNil(detector.owned)
        XCTAssertThrowsError(try stream.push(startSample: 1, samples: [0.8]))
        detector.failDetect = false
        _ = try stream.reset()
        _ = try stream.push(startSample: 0, samples: [0.8])
        detector.failReset = true
        XCTAssertThrowsError(try stream.reset())
        XCTAssertThrowsError(try stream.push(startSample: 1, samples: [0.8]))
    }

    func testEventsRequireReviewAndBindNonDiagnosticNotice() throws {
        let stream = try adapter()
        let events = try stream.push(startSample: 0, samples: [0.812345]) + stream.finish()
        for event in events {
            XCTAssertTrue(event.reviewerConfirmationRequired)
            XCTAssertTrue(event.notice.contains("Non-diagnostic"))
            XCTAssertTrue(event.notice.contains("no speaker identity, consent or clinical meaning"))
            XCTAssertFalse(String(describing: event).contains("0.812345"))
        }
    }
}
