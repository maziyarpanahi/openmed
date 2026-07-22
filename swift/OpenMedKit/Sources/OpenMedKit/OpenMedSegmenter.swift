import Foundation

struct OpenMedSegmenterDescriptor: Decodable, Sendable {
    struct ResourceFile: Decodable, Sendable {
        let path: String
        let role: String
        let license: String
        let sizeBytes: Int
        let sha256: String

        enum CodingKeys: String, CodingKey {
            case path
            case role
            case license
            case sizeBytes = "size_bytes"
            case sha256
        }
    }

    let id: String
    let formatVersion: Int
    let scripts: [String]
    let license: String
    let resourceFiles: [ResourceFile]
    let totalSizeBytes: Int
    let sizeBudgetBytes: Int

    enum CodingKeys: String, CodingKey {
        case id
        case formatVersion = "format_version"
        case scripts
        case license
        case resourceFiles = "resource_files"
        case totalSizeBytes = "total_size_bytes"
        case sizeBudgetBytes = "size_budget_bytes"
    }
}

/// One segment expressed with UTF-8 byte offsets for Python/Swift parity.
public struct OpenMedSegment: Equatable, Sendable {
    public let text: String
    public let start: Int
    public let end: Int

    public init(text: String, start: Int, end: Int) {
        self.text = text
        self.start = start
        self.end = end
    }
}

public enum OpenMedSegmenterError: LocalizedError {
    case missingDescriptor(URL)
    case invalidDescriptor(String)
    case missingResource(URL)

    public var errorDescription: String? {
        switch self {
        case .missingDescriptor(let url):
            return "No segmenter descriptor was found in \(url.path)"
        case .invalidDescriptor(let reason):
            return "Invalid segmenter descriptor: \(reason)"
        case .missingResource(let url):
            return "Segmenter resource is missing: \(url.path)"
        }
    }
}

/// Dependency-free Han and Devanagari segmentation from artifact resources.
public struct OpenMedSegmenter: Sendable {
    private struct ManifestEnvelope: Decodable {
        let segmenter: OpenMedSegmenterDescriptor?
    }

    private struct IndicRules: Decodable {
        struct Script: Decodable {
            let ranges: [[UInt32]]
            let viramas: [UInt32]
            let joiners: [UInt32]
        }

        let scripts: [String: Script]
    }

    private let scripts: Set<String>
    private let hanWords: Set<String>
    private let maximumHanWordScalars: Int
    private let devanagariRanges: [ClosedRange<UInt32>]
    private let viramas: Set<UInt32>
    private let joiners: Set<UInt32>

    /// Load the common segmenter descriptor from an OpenMed artifact manifest.
    public init(
        bundleURL: URL,
        manifestFilename: String = "openmed-mlx.json"
    ) throws {
        let manifestURL = bundleURL.appending(path: manifestFilename)
        let data = try Data(contentsOf: manifestURL)
        let envelope = try JSONDecoder().decode(ManifestEnvelope.self, from: data)
        guard let descriptor = envelope.segmenter else {
            throw OpenMedSegmenterError.missingDescriptor(manifestURL)
        }
        try self.init(bundleURL: bundleURL, descriptor: descriptor)
    }

    init(bundleURL: URL, descriptor: OpenMedSegmenterDescriptor) throws {
        guard descriptor.formatVersion == 1 else {
            throw OpenMedSegmenterError.invalidDescriptor(
                "unsupported format version \(descriptor.formatVersion)"
            )
        }
        guard !descriptor.id.isEmpty, !descriptor.scripts.isEmpty, !descriptor.license.isEmpty else {
            throw OpenMedSegmenterError.invalidDescriptor("missing id, scripts, or license")
        }
        guard descriptor.sizeBudgetBytes == 64 * 1024 else {
            throw OpenMedSegmenterError.invalidDescriptor("size budget must be 64 KiB")
        }

        let expectedResources: [String: [String: String]]
        let expectedScripts: [String]
        let expectedLicense: String
        switch descriptor.id {
        case "openmed-han-v1":
            expectedScripts = ["Han"]
            expectedLicense = "MIT"
            expectedResources = [
                "han_words.txt": ["role": "han_dictionary", "license": "MIT"]
            ]
        case "openmed-indic-v1":
            expectedScripts = ["Devanagari"]
            expectedLicense = "ICU-1.8.1"
            expectedResources = [
                "indic_rules.json": [
                    "role": "indic_break_rules", "license": "ICU-1.8.1",
                ]
            ]
        case "openmed-cjk-indic-v1":
            expectedScripts = ["Han", "Devanagari"]
            expectedLicense = "MIT AND ICU-1.8.1"
            expectedResources = [
                "han_words.txt": ["role": "han_dictionary", "license": "MIT"],
                "indic_rules.json": [
                    "role": "indic_break_rules", "license": "ICU-1.8.1",
                ],
            ]
        default:
            throw OpenMedSegmenterError.invalidDescriptor(
                "unsupported segmenter id \(descriptor.id)"
            )
        }
        guard descriptor.scripts == expectedScripts, descriptor.license == expectedLicense else {
            throw OpenMedSegmenterError.invalidDescriptor(
                "scripts or license do not match the segmenter id"
            )
        }
        guard descriptor.resourceFiles.count == expectedResources.count else {
            throw OpenMedSegmenterError.invalidDescriptor("unexpected segmenter resources")
        }

        var loadedHanWords: Set<String> = []
        var loadedRanges: [ClosedRange<UInt32>] = []
        var loadedViramas: Set<UInt32> = []
        var loadedJoiners: Set<UInt32> = []
        var seenResources: Set<String> = []
        var actualTotalSize = 0
        let rootURL = bundleURL.standardizedFileURL
        let rootPrefix = rootURL.path.hasSuffix("/") ? rootURL.path : rootURL.path + "/"

        for resource in descriptor.resourceFiles {
            guard
                !resource.path.isEmpty,
                !resource.license.isEmpty,
                resource.sha256.hasPrefix("sha256:")
            else {
                throw OpenMedSegmenterError.invalidDescriptor("resource path or license is empty")
            }
            let filename = URL(fileURLWithPath: resource.path).lastPathComponent
            guard
                let expected = expectedResources[filename],
                resource.role == expected["role"],
                resource.license == expected["license"]
            else {
                throw OpenMedSegmenterError.invalidDescriptor(
                    "resource role or license does not match the segmenter id"
                )
            }
            seenResources.insert(filename)
            let resourceURL = rootURL.appending(path: resource.path).standardizedFileURL
            guard resourceURL.path.hasPrefix(rootPrefix) else {
                throw OpenMedSegmenterError.invalidDescriptor(
                    "resource escapes the artifact bundle"
                )
            }
            guard FileManager.default.fileExists(atPath: resourceURL.path) else {
                throw OpenMedSegmenterError.missingResource(resourceURL)
            }
            let data = try Data(contentsOf: resourceURL)
            guard data.count == resource.sizeBytes else {
                throw OpenMedSegmenterError.invalidDescriptor(
                    "resource size mismatch for \(resource.path)"
                )
            }
            actualTotalSize += data.count

            if resource.role == "han_dictionary" {
                guard let contents = String(data: data, encoding: .utf8) else {
                    throw OpenMedSegmenterError.invalidDescriptor(
                        "Han dictionary is not UTF-8"
                    )
                }
                for line in contents.split(whereSeparator: \Character.isNewline) {
                    let trimmed = line.trimmingCharacters(in: .whitespaces)
                    guard !trimmed.isEmpty, !trimmed.hasPrefix("#") else { continue }
                    if let word = trimmed.split(whereSeparator: \Character.isWhitespace).first {
                        loadedHanWords.insert(String(word))
                    }
                }
            } else if resource.role == "indic_break_rules" {
                let rules = try JSONDecoder().decode(IndicRules.self, from: data)
                if let devanagari = rules.scripts["Devanagari"] {
                    loadedRanges = devanagari.ranges.compactMap { values in
                        guard values.count == 2 else { return nil }
                        return values[0]...values[1]
                    }
                    loadedViramas = Set(devanagari.viramas)
                    loadedJoiners = Set(devanagari.joiners)
                }
            }
        }

        guard seenResources == Set(expectedResources.keys) else {
            throw OpenMedSegmenterError.invalidDescriptor("unexpected segmenter resources")
        }

        guard actualTotalSize == descriptor.totalSizeBytes else {
            throw OpenMedSegmenterError.invalidDescriptor("total resource size mismatch")
        }
        guard actualTotalSize <= descriptor.sizeBudgetBytes else {
            throw OpenMedSegmenterError.invalidDescriptor("resource size exceeds budget")
        }

        scripts = Set(descriptor.scripts)
        hanWords = loadedHanWords
        maximumHanWordScalars = loadedHanWords.map { $0.unicodeScalars.count }.max() ?? 1
        devanagariRanges = loadedRanges
        viramas = loadedViramas
        joiners = loadedJoiners
    }

    /// Segment text without importing or embedding jieba or ICU runtimes.
    public func segment(_ text: String) -> [OpenMedSegment] {
        guard !text.isEmpty else { return [] }
        let scalars = Array(text.unicodeScalars)
        let utf8 = Array(text.utf8)
        var byteOffsets = [0]
        byteOffsets.reserveCapacity(scalars.count + 1)
        for scalar in scalars {
            byteOffsets.append(byteOffsets.last! + String(scalar).utf8.count)
        }

        var segments: [OpenMedSegment] = []
        var cursor = 0
        while cursor < scalars.count {
            if CharacterSet.whitespacesAndNewlines.contains(scalars[cursor]) {
                cursor += 1
                continue
            }
            if scripts.contains("Han"), isHan(scalars[cursor].value) {
                var end = cursor + 1
                while end < scalars.count, isHan(scalars[end].value) { end += 1 }
                appendHanSegments(
                    to: &segments,
                    utf8: utf8,
                    byteOffsets: byteOffsets,
                    start: cursor,
                    end: end
                )
                cursor = end
                continue
            }
            if scripts.contains("Devanagari"), isDevanagari(scalars[cursor].value) {
                var end = cursor + 1
                while end < scalars.count,
                    isDevanagari(scalars[end].value) || joiners.contains(scalars[end].value)
                {
                    end += 1
                }
                appendDevanagariSegments(
                    to: &segments,
                    scalars: scalars,
                    utf8: utf8,
                    byteOffsets: byteOffsets,
                    start: cursor,
                    end: end
                )
                cursor = end
                continue
            }

            var end = cursor + 1
            while end < scalars.count,
                !CharacterSet.whitespacesAndNewlines.contains(scalars[end]),
                !(scripts.contains("Han") && isHan(scalars[end].value)),
                !(scripts.contains("Devanagari")
                    && (isDevanagari(scalars[end].value) || joiners.contains(scalars[end].value)))
            {
                end += 1
            }
            segments.append(
                makeSegment(utf8: utf8, byteOffsets: byteOffsets, start: cursor, end: end)
            )
            cursor = end
        }
        return segments
    }

    private func appendHanSegments(
        to segments: inout [OpenMedSegment],
        utf8: [UInt8],
        byteOffsets: [Int],
        start: Int,
        end: Int
    ) {
        var cursor = start
        while cursor < end {
            let upper = min(end, cursor + maximumHanWordScalars)
            var matchEnd = cursor + 1
            if upper > cursor {
                for candidateEnd in stride(from: upper, through: cursor + 1, by: -1) {
                    let candidate = decode(
                        utf8: utf8,
                        start: byteOffsets[cursor],
                        end: byteOffsets[candidateEnd]
                    )
                    if hanWords.contains(candidate) {
                        matchEnd = candidateEnd
                        break
                    }
                }
            }
            segments.append(
                makeSegment(
                    utf8: utf8,
                    byteOffsets: byteOffsets,
                    start: cursor,
                    end: matchEnd
                )
            )
            cursor = matchEnd
        }
    }

    private func appendDevanagariSegments(
        to segments: inout [OpenMedSegment],
        scalars: [Unicode.Scalar],
        utf8: [UInt8],
        byteOffsets: [Int],
        start: Int,
        end: Int
    ) {
        var boundaries = [start]
        if start + 1 < end {
            for cursor in (start + 1)..<end {
                let codepoint = scalars[cursor].value
                let previous = scalars[cursor - 1].value
                let previousPrevious = cursor - 2 >= start ? scalars[cursor - 2].value : nil
                let joinsPrevious =
                    isMark(scalars[cursor])
                    || joiners.contains(codepoint)
                    || viramas.contains(previous)
                    || (joiners.contains(previous)
                        && previousPrevious.map(viramas.contains) == true)
                if !joinsPrevious { boundaries.append(cursor) }
            }
        }
        boundaries.append(end)
        for index in 0..<boundaries.count - 1 {
            segments.append(
                makeSegment(
                    utf8: utf8,
                    byteOffsets: byteOffsets,
                    start: boundaries[index],
                    end: boundaries[index + 1]
                )
            )
        }
    }

    private func makeSegment(
        utf8: [UInt8],
        byteOffsets: [Int],
        start: Int,
        end: Int
    ) -> OpenMedSegment {
        let byteStart = byteOffsets[start]
        let byteEnd = byteOffsets[end]
        return OpenMedSegment(
            text: decode(utf8: utf8, start: byteStart, end: byteEnd),
            start: byteStart,
            end: byteEnd
        )
    }

    private func decode(utf8: [UInt8], start: Int, end: Int) -> String {
        String(decoding: utf8[start..<end], as: UTF8.self)
    }

    private func isHan(_ value: UInt32) -> Bool {
        (0x3400...0x4DBF).contains(value)
            || (0x4E00...0x9FFF).contains(value)
            || (0xF900...0xFAFF).contains(value)
            || (0x20000...0x323AF).contains(value)
    }

    private func isDevanagari(_ value: UInt32) -> Bool {
        devanagariRanges.contains { $0.contains(value) }
    }

    private func isMark(_ scalar: Unicode.Scalar) -> Bool {
        switch scalar.properties.generalCategory {
        case .nonspacingMark, .spacingMark, .enclosingMark:
            return true
        default:
            return false
        }
    }
}
