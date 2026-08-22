import ActionExtension
import Foundation
import OpenMedExtensionSupport
import ShareExtension
import XCTest

@testable import OpenMedKit

final class ShareRedactionTests: XCTestCase {
    #if os(iOS) || os(macOS)
        func testSyntheticTextMatchesOpenMedKitReferenceSpans() throws {
            let text = "Name Ada DOB 04/01/2026"
            let policy = try Policy(named: "gdpr")
            let entities = [
                entity(label: "first_name", value: "Ada", in: text),
                entity(label: "date_of_birth", value: "04/01/2026", in: text),
            ]
            let reference = OpenMed.deidentify(text, entities: entities, policy: policy)
            let handler = ExtensionRedactionHandler { receivedText, receivedPolicy in
                XCTAssertEqual(receivedText, text)
                XCTAssertEqual(receivedPolicy, policy)
                return reference
            }

            let output = try handler.redact(text, policyName: "gdpr")

            XCTAssertEqual(output.redactedText, reference.redactedText)
            XCTAssertEqual(output.policyName, reference.policyName)
            XCTAssertEqual(output.spans.count, reference.actions.count)
            for (span, expected) in zip(output.spans, reference.actions) {
                XCTAssertLessThanOrEqual(abs(span.start - expected.start), 0)
                XCTAssertLessThanOrEqual(abs(span.end - expected.end), 0)
                XCTAssertEqual(span.canonicalLabel, expected.canonicalLabel)
                XCTAssertEqual(span.action, expected.action)
                XCTAssertEqual(span.replacement, expected.replacement)
            }
        }
    #endif

    func testExtensionHasNoNetworkCapabilityOrNetworkingAPIs() throws {
        XCTAssertFalse(ExtensionSecurityPolicy.allowsNetworkAccess)
        XCTAssertEqual(ExtensionSecurityPolicy.modelAssetURLScheme, "file")

        let packageRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let sourceDirectories = [
            "Sources/OpenMedExtensionSupport",
            "Sources/ShareExtension",
            "Sources/ActionExtension",
        ]
        let forbiddenTokens = [
            "URLSession",
            "NWConnection",
            "import Network",
            "com.apple.security.network.client",
        ]

        for directory in sourceDirectories {
            let directoryURL = packageRoot.appending(path: directory, directoryHint: .isDirectory)
            let files = try FileManager.default.contentsOfDirectory(
                at: directoryURL,
                includingPropertiesForKeys: nil
            )
            for file in files where file.pathExtension == "swift" {
                let source = try String(contentsOf: file, encoding: .utf8)
                for token in forbiddenTokens {
                    XCTAssertFalse(source.contains(token), "\(file.lastPathComponent) contains \(token)")
                }
            }
        }
    }

    func testRemoteModelAssetsAreRejectedBeforeLoading() {
        let remote = URL(string: "https://example.invalid/OpenMedPIINano.mlmodelc")!
        let local = URL(fileURLWithPath: "/tmp/openmed-extension-fixture")

        XCTAssertThrowsError(
            try NanoModelConfiguration(
                modelURL: remote,
                id2labelURL: local.appending(path: "id2label.json"),
                tokenizerFolderURL: local.appending(path: "tokenizer")
            )
        ) { error in
            guard case ExtensionRedactionError.nonLocalAsset(let url) = error else {
                return XCTFail("Unexpected error: \(error)")
            }
            XCTAssertEqual(url, remote)
        }
    }

    #if os(iOS) || os(macOS)
        func testLocalOnlyTokenizerLoadingFailsClosedWithoutAssets() {
            let missingFolder = FileManager.default.temporaryDirectory.appending(
                path: UUID().uuidString,
                directoryHint: .isDirectory
            )

            XCTAssertThrowsError(
                try OpenMed.loadTokenizer(
                    tokenizerName: "OpenMed/remote-fallback-must-not-run",
                    tokenizerFolderURL: missingFolder,
                    allowNetworkAccess: false
                )
            )
        }
    #endif

    func testNanoBudgetStaysBelowExtensionEnvelope() throws {
        let budget = try NanoModelMemoryBudget(
            modelAssetBytes: NanoModelMemoryBudget.maximumModelAssetBytes
        )

        XCTAssertEqual(
            budget.estimatedPeakBytes,
            NanoModelMemoryBudget.maximumEstimatedPeakBytes
        )
        XCTAssertLessThan(
            budget.estimatedPeakBytes,
            NanoModelMemoryBudget.extensionWorkingSetEnvelopeBytes
        )
        XCTAssertThrowsError(
            try NanoModelMemoryBudget(
                modelAssetBytes: NanoModelMemoryBudget.maximumModelAssetBytes + 1
            )
        )
    }

    func testOversizedExtensionInputIsRejectedBeforeInference() throws {
        let handler = ExtensionRedactionHandler { _, _ in
            throw ExtensionRedactionError.invalidRedactionOutput
        }
        let oversized = String(
            repeating: "x",
            count: ExtensionRedactionHandler.maximumInputCharacters + 1
        )

        XCTAssertThrowsError(try handler.redact(oversized)) { error in
            guard case ExtensionRedactionError.inputTooLarge(let actual, let limit) = error else {
                return XCTFail("Unexpected error: \(error)")
            }
            XCTAssertEqual(actual, oversized.count)
            XCTAssertEqual(limit, ExtensionRedactionHandler.maximumInputCharacters)
        }
    }

    func testUTF8AndBatchBudgetsCannotBeBypassedByGraphemeClustering() {
        let oversizedGrapheme =
            "a"
            + String(
                repeating: "\u{0301}",
                count: ExtensionRedactionHandler.maximumInputUTF8Bytes
            )
        XCTAssertLessThanOrEqual(
            oversizedGrapheme.count,
            ExtensionRedactionHandler.maximumInputCharacters
        )
        XCTAssertThrowsError(
            try ExtensionRedactionHandler.validateInput(oversizedGrapheme)
        ) { error in
            guard case ExtensionRedactionError.inputBytesTooLarge = error else {
                return XCTFail("Unexpected error: \(error)")
            }
        }

        XCTAssertThrowsError(
            try ExtensionRedactionHandler.validateInputs(
                Array(
                    repeating: "synthetic",
                    count: ExtensionRedactionHandler.maximumInputItems + 1
                )
            )
        ) { error in
            guard case ExtensionRedactionError.tooManyInputItems = error else {
                return XCTFail("Unexpected error: \(error)")
            }
        }

        let boundedItem = String(repeating: "é", count: 16_000)
        XCTAssertThrowsError(
            try ExtensionRedactionHandler.validateInputs(
                Array(repeating: boundedItem, count: 5)
            )
        ) { error in
            guard case ExtensionRedactionError.aggregateInputTooLarge = error else {
                return XCTFail("Unexpected error: \(error)")
            }
        }
    }

    func testInvalidRuntimeOutputFailsClosed() throws {
        let wrongPolicyHandler = ExtensionRedactionHandler { text, _ in
            PolicyDeidentificationResult(
                redactedText: text,
                policyName: "unexpected_policy",
                actions: []
            )
        }

        XCTAssertThrowsError(try wrongPolicyHandler.redact("Synthetic text")) { error in
            XCTAssertEqual(error as? ExtensionRedactionError, .invalidRedactionOutput)
        }

        let invalidSpanHandler = ExtensionRedactionHandler { text, policy in
            PolicyDeidentificationResult(
                redactedText: text,
                policyName: policy.name,
                actions: [
                    DeidentifiedSpanAction(
                        label: "NAME",
                        canonicalLabel: "PERSON",
                        action: .mask,
                        start: 0,
                        end: text.unicodeScalars.count + 1,
                        confidence: .nan,
                        replacement: "[NAME]"
                    )
                ]
            )
        }

        XCTAssertThrowsError(try invalidSpanHandler.redact("Synthetic text")) { error in
            XCTAssertEqual(error as? ExtensionRedactionError, .invalidRedactionOutput)
        }
    }

    func testNanoAssetConfigurationRejectsSymlinksAndWrongTypes() throws {
        let root = FileManager.default.temporaryDirectory.appending(
            path: UUID().uuidString,
            directoryHint: .isDirectory
        )
        let model = root.appending(path: "OpenMedPIINano.mlmodelc", directoryHint: .isDirectory)
        let labels = root.appending(path: "id2label.json")
        let tokenizer = root.appending(path: "tokenizer", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: model, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: tokenizer, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        try Data("model".utf8).write(to: model.appending(path: "weights.bin"))
        try Data("{}".utf8).write(to: labels)
        let tokenizerJSON = tokenizer.appending(path: "tokenizer.json")
        try Data("{}".utf8).write(to: tokenizerJSON)
        try Data("{}".utf8).write(to: tokenizer.appending(path: "tokenizer_config.json"))

        XCTAssertNoThrow(
            try NanoModelConfiguration(
                modelURL: model,
                id2labelURL: labels,
                tokenizerFolderURL: tokenizer
            )
        )

        try FileManager.default.removeItem(at: tokenizerJSON)
        try FileManager.default.createSymbolicLink(at: tokenizerJSON, withDestinationURL: labels)
        XCTAssertThrowsError(
            try NanoModelConfiguration(
                modelURL: model,
                id2labelURL: labels,
                tokenizerFolderURL: tokenizer
            )
        ) { error in
            guard case ExtensionRedactionError.invalidAsset = error else {
                return XCTFail("Unexpected error: \(error)")
            }
        }

        try FileManager.default.removeItem(at: tokenizerJSON)
        try Data("{}".utf8).write(to: tokenizerJSON)
        try FileManager.default.removeItem(at: labels)
        try FileManager.default.createDirectory(at: labels, withIntermediateDirectories: false)
        XCTAssertThrowsError(
            try NanoModelConfiguration(
                modelURL: model,
                id2labelURL: labels,
                tokenizerFolderURL: tokenizer
            )
        ) { error in
            guard case ExtensionRedactionError.invalidAsset = error else {
                return XCTFail("Unexpected error: \(error)")
            }
        }
    }

    func testPolicyPickerUsesEveryBundledProfile() throws {
        XCTAssertTrue(Policy.bundledProfileNames.contains(Policy.defaultName))
        for profile in Policy.bundledProfileNames {
            XCTAssertNoThrow(try Policy(named: profile))
        }
    }

    private func entity(
        label: String,
        value: String,
        in text: String
    ) -> EntityPrediction {
        let range = text.range(of: value)!
        return EntityPrediction(
            label: label,
            text: value,
            confidence: 0.99,
            start: text.distance(from: text.startIndex, to: range.lowerBound),
            end: text.distance(from: text.startIndex, to: range.upperBound)
        )
    }
}
