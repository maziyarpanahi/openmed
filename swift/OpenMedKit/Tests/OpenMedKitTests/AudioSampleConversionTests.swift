import XCTest

@testable import OpenMedKit

final class AudioSampleConversionTests: XCTestCase {
    func testSilenceRoundingAndLineage() throws {
        let profile = try AudioSampleConversionProfile(rates: [16000], channels: [1])
        for rounding in [AudioSampleConversionPlan.Rounding.floor, .nearestEven, .ceiling] {
            let plan = try AudioSampleConversionPlan(sourceRate: 44100, targetRate: 16000, sourceFrames: 101, rounding: rounding)
            let result = try LocalAudioSampleConversion.convert(chunks: [[Double](repeating: 0, count: 101)], sourceChannels: 1, plan: plan, profile: profile, channelPolicy: .preserve)
            XCTAssertEqual(try result.samplesForReviewedHandoff(reviewerConfirmed: true).count, plan.targetFrames)
            XCTAssertTrue(try result.samplesForReviewedHandoff(reviewerConfirmed: true).allSatisfy { $0 == 0 })
            let position = try result.report.sourcePosition(targetFrame: 1)
            XCTAssertEqual(position.numerator, 44100)
            XCTAssertEqual(position.denominator, 16000)
            XCTAssertTrue(result.report.notice.contains("non-diagnostic"))
            result.close()
            XCTAssertThrowsError(try result.samplesForReviewedHandoff(reviewerConfirmed: true))
        }
    }

    func testChunkContinuityAndAliasing() throws {
        let plan = try AudioSampleConversionPlan(sourceRate: 48000, targetRate: 16000, sourceFrames: 4800)
        let profile = try AudioSampleConversionProfile(rates: [16000], channels: [1])
        for frequency in [1000.0, 12000.0] {
            let signal = (0..<4800).map { 0.5 * sin(2 * Double.pi * frequency * Double($0) / 48000) }
            let whole = try LocalAudioSampleConversion.convert(chunks: [signal], sourceChannels: 1, plan: plan, profile: profile, channelPolicy: .preserve)
            let split = try LocalAudioSampleConversion.convert(chunks: [Array(signal[..<1701]), Array(signal[1701...])], sourceChannels: 1, plan: plan, profile: profile, channelPolicy: .preserve)
            let samples = try whole.samplesForReviewedHandoff(reviewerConfirmed: true)
            XCTAssertEqual(samples, try split.samplesForReviewedHandoff(reviewerConfirmed: true))
            let rms = sqrt(samples[100..<1500].reduce(0.0) { $0 + pow(Double($1) / 32767, 2) } / 1400)
            if frequency == 1000 {
                XCTAssertGreaterThan(rms, 0.34)
                XCTAssertLessThan(rms, 0.36)
            } else {
                XCTAssertLessThan(rms, 0.002)
            }
            whole.close()
            split.close()
        }
    }

    func testImpulseChannelPolicyAndReview() throws {
        let plan = try AudioSampleConversionPlan(sourceRate: 16000, targetRate: 16000, sourceFrames: 3)
        let profile = try AudioSampleConversionProfile(rates: [16000], channels: [1, 2])
        for policy in [AudioConversionChannelPolicy.preserve, .meanMono] {
            let result = try LocalAudioSampleConversion.convert(chunks: [[0, 0, 1, -1], [0, 0]], sourceChannels: 2, plan: plan, profile: profile, channelPolicy: policy)
            XCTAssertThrowsError(try result.samplesForReviewedHandoff(reviewerConfirmed: false))
            XCTAssertEqual(try result.samplesForReviewedHandoff(reviewerConfirmed: true), policy == .preserve ? [0, 0, 32767, -32768, 0, 0] : [0, 0, 0])
            XCTAssertEqual(result.description, "ConvertedAudioSamples(protected_samples)")
            result.close()
        }
    }

    func testInvalidSamplesBudgetsAndCancellation() throws {
        let plan = try AudioSampleConversionPlan(sourceRate: 48000, targetRate: 16000, sourceFrames: 100)
        let profile = try AudioSampleConversionProfile(rates: [16000], channels: [1])
        for value in [Double.nan, Double.infinity, 1.01, -1.01] {
            var signal = [Double](repeating: 0, count: 100)
            signal[99] = value
            XCTAssertThrowsError(try LocalAudioSampleConversion.convert(chunks: [signal], sourceChannels: 1, plan: plan, profile: profile, channelPolicy: .preserve))
        }
        XCTAssertThrowsError(try LocalAudioSampleConversion.convert(chunks: [], sourceChannels: 1, plan: plan, profile: profile, channelPolicy: .preserve, maxBufferBytes: 1))
        XCTAssertThrowsError(try LocalAudioSampleConversion.convert(chunks: [[0]], sourceChannels: 1, plan: plan, profile: profile, channelPolicy: .preserve))
        var calls = 0
        XCTAssertThrowsError(
            try LocalAudioSampleConversion.convert(
                chunks: [[Double](repeating: 0.25, count: 100)], sourceChannels: 1, plan: plan, profile: profile, channelPolicy: .preserve,
                cancelled: {
                    calls += 1
                    return calls > 110
                }))
    }
}
