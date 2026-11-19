import Foundation
import OpenMedExtensionSupport
import XCTest

/// Sentinel gate for the extension modules' errors.
///
/// The extension sources must declare no error type besides
/// ``ExtensionRedactionError``, every declared case needs a sample here, and no
/// rendering may contain host text, asset names or the asset directory.
final class ExtensionErrorSentinelTests: XCTestCase {
    private static let sentinel = "OMKSENTINEL7Q4Z"
    private static let extensionModules = ["OpenMedExtensionSupport", "ShareExtension", "ActionExtension"]

    private var root: URL!

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory.appending(
            path: "openmed-extension-\(Self.sentinel)-\(UUID().uuidString)",
            directoryHint: .isDirectory
        )
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        if let root {
            try? FileManager.default.removeItem(at: root)
        }
    }

    func testEveryDeclaredExtensionErrorCaseIsSwept() throws {
        let declared = try Self.declaredErrorCases()
        XCTAssertEqual(Set(declared.keys), ["ExtensionRedactionError"])

        let samples: [ExtensionRedactionError] =
            [
                .emptyInput,
                .inputTooLarge(actual: 4_097, limit: 4_096),
                .inputBytesTooLarge(actual: 16_385, limit: 16_384),
                .tooManyInputItems(actual: 9, limit: 8),
                .aggregateInputTooLarge(actual: 131_073, limit: 131_072),
                .missingPlainTextInput,
                .tooManyAssetFiles(limit: 4_096),
                .unsupportedModelFormat,
                .modelAssetsTooLarge(actual: 41_943_041, limit: 41_943_040),
                .invalidRedactionOutput,
            ]
            + ExtensionAssetRole.allCases.flatMap { role -> [ExtensionRedactionError] in
                [.nonLocalAsset(role), .missingAsset(role), .invalidAsset(role)]
            }
        XCTAssertEqual(
            Set(samples.map(Self.caseLabel)),
            Set(declared["ExtensionRedactionError"] ?? []),
            "ExtensionRedactionError has cases without a sentinel sample"
        )
        for sample in samples {
            assertValueFree(sample)
        }
    }

    func testAssetFailuresFromSentinelDirectoryNameRolesOnly() throws {
        let model = root.appending(path: "\(Self.sentinel).mlmodelc", directoryHint: .isDirectory)
        let labels = root.appending(path: "\(Self.sentinel)-id2label.json")
        let tokenizer = root.appending(path: "\(Self.sentinel)-tokenizer", directoryHint: .isDirectory)

        assertFailure(.nonLocalAsset(.model)) {
            try NanoModelConfiguration(
                modelURL: try XCTUnwrap(URL(string: "https://example.invalid/\(Self.sentinel).mlmodelc")),
                id2labelURL: labels,
                tokenizerFolderURL: tokenizer
            )
        }
        assertFailure(.unsupportedModelFormat) {
            try NanoModelConfiguration(
                modelURL: self.root.appending(path: "\(Self.sentinel).mlpackage"),
                id2labelURL: labels,
                tokenizerFolderURL: tokenizer
            )
        }
        assertFailure(.missingAsset(.model)) {
            try NanoModelConfiguration(modelURL: model, id2labelURL: labels, tokenizerFolderURL: tokenizer)
        }

        try FileManager.default.createDirectory(at: model, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: tokenizer, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: labels)
        assertFailure(.missingAsset(.tokenizerFile)) {
            try NanoModelConfiguration(modelURL: model, id2labelURL: labels, tokenizerFolderURL: tokenizer)
        }

        try Data("{}".utf8).write(to: tokenizer.appending(path: "tokenizer.json"))
        try Data("{}".utf8).write(to: tokenizer.appending(path: "tokenizer_config.json"))
        try Data("weights".utf8).write(to: model.appending(path: "weights.bin"))
        XCTAssertNoThrow(
            try NanoModelConfiguration(modelURL: model, id2labelURL: labels, tokenizerFolderURL: tokenizer),
            "negative control must load"
        )

        try FileManager.default.createSymbolicLink(
            at: model.appending(path: "\(Self.sentinel)-link"),
            withDestinationURL: labels
        )
        assertFailure(.invalidAsset(.model)) {
            try NanoModelConfiguration(modelURL: model, id2labelURL: labels, tokenizerFolderURL: tokenizer)
        }
    }

    func testHostInputFailuresCarryCountsOnly() {
        let oversized = String(repeating: "\(Self.sentinel) ", count: 2_048)
        XCTAssertThrowsError(try ExtensionRedactionHandler.validateInput(oversized)) { assertValueFree($0) }
        XCTAssertThrowsError(
            try ExtensionRedactionHandler.validateInputs(Array(repeating: Self.sentinel, count: 9))
        ) { assertValueFree($0) }
        XCTAssertThrowsError(try ExtensionRedactionHandler.validateInputs([])) { assertValueFree($0) }
    }

    // MARK: - Helpers

    private func assertFailure(
        _ expected: ExtensionRedactionError,
        file: StaticString = #filePath,
        line: UInt = #line,
        _ body: () throws -> NanoModelConfiguration
    ) {
        XCTAssertThrowsError(try body(), file: file, line: line) { error in
            XCTAssertEqual(error as? ExtensionRedactionError, expected, file: file, line: line)
            assertValueFree(error, file: file, line: line)
        }
    }

    private func assertValueFree(_ error: Error, file: StaticString = #filePath, line: UInt = #line) {
        var forbidden = [Self.sentinel, "example.invalid", FileManager.default.temporaryDirectory.standardizedFileURL.path]
        if let root {
            forbidden += [root.path, root.resolvingSymlinksInPath().path, root.lastPathComponent]
        }
        let bridged = error as NSError
        var renderings = [
            error.localizedDescription,
            String(describing: error),
            String(reflecting: error),
            bridged.debugDescription,
            String(describing: bridged.userInfo),
        ]
        if let localized = error as? LocalizedError {
            renderings += [localized.errorDescription, localized.failureReason, localized.recoverySuggestion]
                .compactMap { $0 }
        }
        if renderings.contains(where: { rendering in forbidden.contains { rendering.contains($0) } }) {
            XCTFail("\(Self.caseLabel(of: error)) renders host text or an asset path", file: file, line: line)
        }
    }

    private static func caseLabel(of value: Any) -> String {
        let mirror = Mirror(reflecting: value)
        if mirror.displayStyle == .enum, let label = mirror.children.first?.label {
            return label
        }
        return String(describing: value)
    }

    /// Error types declared in the extension modules, with their case names.
    private static func declaredErrorCases() throws -> [String: [String]] {
        let sources = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appending(path: "Sources", directoryHint: .isDirectory)
        let declaration = try NSRegularExpression(
            pattern: #"(?m)^[ \t]*(?:(?:public|package|internal|fileprivate|private|final)[ \t]+)*(?:enum|struct|class)[ \t]+([A-Za-z_][A-Za-z0-9_]*)[ \t]*:([^{]*)\{"#
        )
        var result: [String: [String]] = [:]
        for module in extensionModules {
            let files = try XCTUnwrap(
                FileManager.default.enumerator(
                    at: sources.appending(path: module, directoryHint: .isDirectory),
                    includingPropertiesForKeys: nil
                )
            )
            for case let url as URL in files where url.pathExtension == "swift" {
                let contents = try String(contentsOf: url, encoding: .utf8)
                let text = contents as NSString
                for match in declaration.matches(in: contents, range: NSRange(location: 0, length: text.length)) {
                    let conformances = text.substring(with: match.range(at: 2))
                        .split { !$0.isLetter && !$0.isNumber && $0 != "_" }
                    guard conformances.contains("Error") || conformances.contains("LocalizedError") else {
                        continue
                    }
                    let start = match.range.location + match.range.length
                    let rest = text.substring(from: start)
                    result[text.substring(with: match.range(at: 1))] = caseNames(inBodyStartingWith: rest)
                }
            }
        }
        return result
    }

    /// Case names declared directly in a type body (depth zero, outside strings).
    private static func caseNames(inBodyStartingWith text: String) -> [String] {
        var names: [String] = []
        var depth = 0
        for line in text.split(separator: "\n", omittingEmptySubsequences: false) {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if depth == 0, trimmed.hasPrefix("case ") {
                var current = ""
                var parentheses = 0
                for character in trimmed.dropFirst(5) {
                    if character == "(" { parentheses += 1 }
                    if character == ")" { parentheses -= 1 }
                    if character == "," && parentheses == 0 {
                        names.append(current)
                        current = ""
                    } else {
                        current.append(character)
                    }
                }
                names.append(current)
            }
            var inString = false
            for character in line {
                if character == "\"" { inString.toggle() }
                guard !inString else { continue }
                if character == "{" { depth += 1 }
                if character == "}" { depth -= 1 }
            }
            if depth < 0 { break }
        }
        return names.map { element in
            String(element.trimmingCharacters(in: .whitespaces).prefix { $0.isLetter || $0.isNumber || $0 == "_" })
        }
    }
}
