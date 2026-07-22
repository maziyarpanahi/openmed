import Foundation
import XCTest

@testable import OpenMedKit

final class SegmenterParityTests: XCTestCase {
    private struct Fixture: Decodable {
        struct Segment: Decodable, Equatable {
            let text: String
            let start: Int
            let end: Int
        }

        let text: String
        let segments: [Segment]
    }

    func testSwiftMatchesSharedZhHiUTF8OffsetFixture() throws {
        let root = repositoryRoot()
        let fixtureURL = root.appending(path: "tests/fixtures/segmenter/zh_hi_parity.json")
        let fixture = try JSONDecoder().decode(
            Fixture.self,
            from: Data(contentsOf: fixtureURL)
        )
        let bundleURL = FileManager.default.temporaryDirectory
            .appending(path: UUID().uuidString, directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: bundleURL, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: bundleURL) }

        let resourceDirectory = bundleURL.appending(path: "segmenter", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(
            at: resourceDirectory,
            withIntermediateDirectories: true
        )
        let sourceDirectory = root.appending(path: "openmed/processing/resources/segmenter")
        let resources = [
            ("han_words.txt", "han_dictionary", "MIT"),
            ("indic_rules.json", "indic_break_rules", "ICU-1.8.1"),
        ]
        var resourceDescriptors: [[String: Any]] = []
        var totalSize = 0
        for (filename, role, license) in resources {
            let data = try Data(contentsOf: sourceDirectory.appending(path: filename))
            try data.write(to: resourceDirectory.appending(path: filename))
            totalSize += data.count
            resourceDescriptors.append([
                "path": "segmenter/\(filename)",
                "role": role,
                "license": license,
                "size_bytes": data.count,
                "sha256": "sha256:test-fixture",
            ])
        }

        let manifest: [String: Any] = [
            "segmenter": [
                "id": "openmed-cjk-indic-v1",
                "format_version": 1,
                "scripts": ["Han", "Devanagari"],
                "license": "MIT AND ICU-1.8.1",
                "resource_files": resourceDescriptors,
                "total_size_bytes": totalSize,
                "size_budget_bytes": 65_536,
            ]
        ]
        let manifestData = try JSONSerialization.data(withJSONObject: manifest)
        try manifestData.write(to: bundleURL.appending(path: "openmed-mlx.json"))

        let segmenter = try OpenMedSegmenter(bundleURL: bundleURL)
        let actual = segmenter.segment(fixture.text).map {
            Fixture.Segment(text: $0.text, start: $0.start, end: $0.end)
        }

        XCTAssertEqual(actual, fixture.segments)
    }

    private func repositoryRoot() -> URL {
        var url = URL(fileURLWithPath: #filePath)
        for _ in 0..<5 { url.deleteLastPathComponent() }
        return url
    }
}
