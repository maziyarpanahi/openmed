import Foundation
import XCTest

@testable import OpenMedKit

final class WaveformNoticeParityTests: XCTestCase {
    func testNoticesMatchSharedFixture() throws {
        var root = URL(fileURLWithPath: #filePath)
        for _ in 0..<5 { root.deleteLastPathComponent() }
        let data = try Data(contentsOf: root.appending(path: "tests/fixtures/parity/waveform_notices_v1.json"))
        let fixture = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])

        XCTAssertEqual(Set(fixture.keys), ["version", "edf", "wfdb", "aecg"])
        XCTAssertEqual(try XCTUnwrap(fixture["version"] as? Int), 1)
        XCTAssertEqual(EDFRecording.notice, try XCTUnwrap(fixture["edf"] as? String))
        XCTAssertEqual(WFDBReader.notice, try XCTUnwrap(fixture["wfdb"] as? String))
        XCTAssertEqual(OpenMedAECG.notice, try XCTUnwrap(fixture["aecg"] as? String))
    }
}
