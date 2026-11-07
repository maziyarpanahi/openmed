import Foundation
import XCTest

@testable import OpenMedKit

final class MultimodalContractParityTests: XCTestCase {
    private func fixture() throws -> [String: Any] {
        var root = URL(fileURLWithPath: #filePath)
        for _ in 0..<5 { root.deleteLastPathComponent() }
        let data = try Data(contentsOf: root.appending(path: "tests/fixtures/parity/multimodal_contract_v1.json"))
        return try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    private func packets() throws -> [String: MultimodalPacket] {
        var result = [String: MultimodalPacket]()
        for row in try XCTUnwrap(fixture()["cases"] as? [[String: Any]]) {
            let id = try XCTUnwrap(row["id"] as? String)
            result[id] = try MultimodalPacket.fromJSON(Data(try XCTUnwrap(row["wire"] as? String).utf8))
        }
        return result
    }

    func testSharedSourceProjectionsAndCanonicalIdentity() throws {
        let corpus = try fixture()
        XCTAssertEqual(corpus["version"] as? Int, 1)
        XCTAssertEqual(corpus["synthetic_only"] as? Bool, true)
        for row in try XCTUnwrap(corpus["cases"] as? [[String: Any]]) {
            let id = try XCTUnwrap(row["id"] as? String)
            let wire = Data(try XCTUnwrap(row["wire"] as? String).utf8)
            let packet = try MultimodalPacket.fromJSON(wire)
            let expected = try XCTUnwrap(row["expected"] as? [String: Any])
            XCTAssertEqual(packet.evidenceDigest, row["digest"] as? String, id)
            XCTAssertEqual(packet.toJSON(), wire, id)
            XCTAssertEqual(packet.projection.kind.rawValue, expected["kind"] as? String, id)
            switch packet.projection {
            case .asset(let asset):
                if let width = expected["width"] as? Int64 { XCTAssertEqual(asset.width, width, id) }
                if let duration = expected["duration"] as? Double { XCTAssertEqual(asset.durationSeconds, duration, id) }
                if let bytes = expected["bytes"] as? String { XCTAssertEqual(String(asset.byteSize), bytes, id) }
            case .visual(let visual):
                if let box = expected["box"] as? [Double] {
                    let actual = try XCTUnwrap(visual.displayBox, id)
                    XCTAssertEqual([actual.left, actual.top, actual.right, actual.bottom], box, id)
                } else {
                    XCTAssertNil(visual.displayBox, id)
                }
                if let text = expected["synthetic_text"] as? String {
                    let span = try XCTUnwrap(visual.scalarSpan, id)
                    let scalars = Array(text.unicodeScalars)
                    XCTAssertEqual(String(String.UnicodeScalarView(scalars[Int(span.start)..<Int(span.end)])), "😀é", id)
                }
            case .transcript(let transcript):
                XCTAssertEqual(transcript.finalized, expected["finalized"] as? Bool, id)
                if let bounds = expected["sample_bounds"] as? [Int64] {
                    let actual = try XCTUnwrap(transcript.sampleBounds, id)
                    XCTAssertEqual([actual.start, actual.end], bounds, id)
                } else {
                    XCTAssertNil(transcript.sampleBounds, id)
                    XCTAssertNil(transcript.currentScalarSpan, id)
                }
                if let bounds = expected["current_span"] as? [Int64] {
                    let actual = try XCTUnwrap(transcript.currentScalarSpan, id)
                    XCTAssertEqual([actual.start, actual.end], bounds, id)
                }
            case .provider(let result):
                XCTAssertEqual(result.outcome.rawValue, expected["outcome"] as? String, id)
                XCTAssertEqual(result.abstention?.rawValue, expected["reason"] as? String, id)
                if result.outcome != .success { XCTAssertNil(result.outputDigest, id) }
            case .clock(let clock):
                XCTAssertEqual(clock.state.rawValue, expected["state"] as? String, id)
                if let bounds = expected["overlap"] as? [Int64] {
                    let actual = try XCTUnwrap(clock.guaranteedOverlap, id)
                    XCTAssertEqual([actual.start, actual.end], bounds, id)
                } else if let bounds = expected["overlap_strings"] as? [String] {
                    let actual = try XCTUnwrap(clock.guaranteedOverlap, id)
                    XCTAssertEqual([String(actual.start), String(actual.end)], bounds, id)
                } else {
                    XCTAssertNil(clock.guaranteedOverlap, id)
                    XCTAssertNil(clock.targetStart, id)
                    XCTAssertNil(clock.targetEnd, id)
                }
            }
        }
    }

    func testSharedNegativeControlsFailBeforeConsumption() throws {
        for row in try XCTUnwrap(fixture()["negative_cases"] as? [[String: Any]]) {
            let id = try XCTUnwrap(row["id"] as? String)
            let wire = Data(try XCTUnwrap(row["wire"] as? String).utf8)
            XCTAssertThrowsError(try MultimodalPacket.fromJSON(wire), id) { error in
                XCTAssertEqual((error as? MultimodalContractRefusal)?.rawValue, row["reason"] as? String, id)
            }
        }
    }

    func testCompleteReasonVocabulariesMatchPython() throws {
        let corpus = try fixture()
        XCTAssertEqual(MultimodalProviderOutcome.allCases.map(\.rawValue), corpus["provider_outcomes"] as? [String])
        XCTAssertEqual(MultimodalAbstention.allCases.map(\.rawValue), corpus["abstention_codes"] as? [String])
        XCTAssertEqual(MultimodalClockReason.allCases.map(\.rawValue), corpus["clock_reasons"] as? [String])
    }

    private actor Counter {
        var value = 0
        func increment() { value += 1 }
    }

    private struct Adapter: LocalMultimodalAdapter {
        let localOnly: Bool
        let capabilities: Set<MultimodalProjectionKind>
        let response: MultimodalPacket
        let counter: Counter
        func run(_ request: MultimodalPacket) async throws -> MultimodalPacket {
            await counter.increment()
            return response
        }
    }

    func testLocalCapabilityAndSourceBindingBeforeConsumption() async throws {
        let rows = try packets()
        let request = try XCTUnwrap(rows["asset_image"])
        let response = try XCTUnwrap(rows["provider_success_none"])
        let counter = Counter()
        let adapter = Adapter(localOnly: true, capabilities: [.asset], response: response, counter: counter)
        let actual = try await LocalMultimodalExecution.run(adapter, request: request)
        XCTAssertEqual(actual, response)
        for refused in [
            Adapter(localOnly: false, capabilities: [.asset], response: response, counter: counter),
            Adapter(localOnly: true, capabilities: [], response: response, counter: counter),
        ] {
            do {
                _ = try await LocalMultimodalExecution.run(refused, request: request)
                XCTFail("unsupported adapter ran")
            } catch { XCTAssertEqual(error as? MultimodalContractRefusal, .capability) }
        }
        let calls = await counter.value
        XCTAssertEqual(calls, 1)
        do {
            _ = try await LocalMultimodalExecution.run(adapter, request: XCTUnwrap(rows["asset_audio"]))
            XCTFail("unbound result consumed")
        } catch { XCTAssertEqual(error as? MultimodalContractRefusal, .provenance) }
    }
}
