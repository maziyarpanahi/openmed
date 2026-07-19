import ActionExtension
import Foundation
import OpenMedExtensionSupport
import ShareExtension
import XCTest

@testable import OpenMedKit

final class ShareRedactionTests: XCTestCase {
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
        let text = "x"
        let policy = try Policy(named: Policy.defaultName)
        let reference = OpenMed.deidentify(text, entities: [], policy: policy)
        let handler = ExtensionRedactionHandler { _, _ in reference }
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
