import Foundation
import XCTest

#if canImport(MLX) && canImport(Tokenizers) && !os(watchOS) && !os(visionOS)
    import CryptoKit
    import Tokenizers

    @testable import OpenMedKit

    /// Runs swift-transformers and the OpenMedKit tokenizer adapter against the
    /// shared synthetic vectors in `tests/fixtures/parity/tokenizer` before any
    /// inference comparison. Diagnostics carry case ids, field names, indices
    /// and reason codes only.
    final class TokenizerConformanceTests: XCTestCase {
        private static let runtime = "swift-transformers"
        private static let contractVersion = "openmed.tokenizer-conformance.v1"
        private static let fields = [
            "normalized",
            "tokens",
            "input_ids",
            "attention_mask",
            "token_type_ids",
            "special_tokens_mask",
            "offsets",
            "truncated_input_ids",
            "window_ranges",
            "window_input_ids",
        ]

        private struct Vectors: Decodable {
            let contractVersion: String
            let tokenizer: TokenizerInfo
            let cases: [Case]

            enum CodingKeys: String, CodingKey {
                case contractVersion = "contract_version"
                case tokenizer
                case cases
            }
        }

        private struct TokenizerInfo: Decodable {
            let directory: String
            let sha256: String
            let templateSpecialTokens: Int

            enum CodingKeys: String, CodingKey {
                case directory
                case sha256
                case templateSpecialTokens = "template_special_tokens"
            }
        }

        private struct Window: Decodable {
            let maxLength: Int
            let stride: Int

            enum CodingKeys: String, CodingKey {
                case maxLength = "max_length"
                case stride
            }
        }

        private struct Expected: Decodable {
            let normalized: String
            let tokens: [String]
            let inputIDs: [Int]
            let attentionMask: [Int]
            let tokenTypeIDs: [Int]
            let specialTokensMask: [Int]
            let offsets: [[Int]]
            let truncatedInputIDs: [Int]?
            let windowRanges: [[Int]]?
            let windowInputIDs: [[Int]]?

            enum CodingKeys: String, CodingKey {
                case normalized
                case tokens
                case inputIDs = "input_ids"
                case attentionMask = "attention_mask"
                case tokenTypeIDs = "token_type_ids"
                case specialTokensMask = "special_tokens_mask"
                case offsets
                case truncatedInputIDs = "truncated_input_ids"
                case windowRanges = "window_ranges"
                case windowInputIDs = "window_input_ids"
            }

            var byField: [String: [String]] {
                var values: [String: [String]] = [
                    "normalized": normalized.unicodeScalars.map { String($0) },
                    "tokens": tokens,
                    "input_ids": inputIDs.map(String.init),
                    "attention_mask": attentionMask.map(String.init),
                    "token_type_ids": tokenTypeIDs.map(String.init),
                    "special_tokens_mask": specialTokensMask.map(String.init),
                    "offsets": offsets.map { $0.map(String.init).joined(separator: ",") },
                ]
                if let truncatedInputIDs {
                    values["truncated_input_ids"] = truncatedInputIDs.map(String.init)
                }
                if let windowRanges {
                    values["window_ranges"] = windowRanges.map {
                        $0.map(String.init).joined(separator: ",")
                    }
                }
                if let windowInputIDs {
                    values["window_input_ids"] = windowInputIDs.map {
                        $0.map(String.init).joined(separator: ",")
                    }
                }
                return values
            }
        }

        private struct Case: Decodable {
            let id: String
            let features: [String]
            let text: String
            let window: Window?
            let expected: Expected
        }

        private struct FieldStatus: Decodable {
            let status: String
            let reason: String?
        }

        private struct Divergence: Decodable {
            let feature: String
            let fields: [String]
            let reason: String
        }

        private struct RuntimeEntry: Decodable {
            let fields: [String: FieldStatus]
            let divergences: [Divergence]
        }

        private struct Support: Decodable {
            let contractVersion: String
            let runtimes: [String: RuntimeEntry]

            enum CodingKeys: String, CodingKey {
                case contractVersion = "contract_version"
                case runtimes
            }
        }

        private struct ManifestFile: Decodable {
            let path: String
            let sha256: String
        }

        private struct Manifest: Decodable {
            let contractVersion: String
            let files: [ManifestFile]

            enum CodingKeys: String, CodingKey {
                case contractVersion = "contract_version"
                case files
            }
        }

        private struct Mismatch: CustomStringConvertible {
            let caseID: String
            let field: String
            let index: Int?
            let code: String
            var feature: String? = nil

            var description: String {
                let position = index.map(String.init) ?? "-"
                let suffix = feature.map { " feature=\($0)" } ?? ""
                return "\(code) case=\(caseID) field=\(field) index=\(position)\(suffix)"
            }
        }

        private struct Report {
            var checked = 0
            var unsupported: [String] = []
            var declared: [(caseID: String, field: String, reason: String)] = []
            var mismatches: [Mismatch] = []

            var diagnostics: String {
                mismatches.map(\.description).joined(separator: "\n")
            }
        }

        func testConformanceAssetsMatchVectorsAndManifestDigests() throws {
            let vectors = try loadVectors()
            let manifest = try JSONDecoder().decode(
                Manifest.self,
                from: Data(contentsOf: fixtureDirectory().appending(path: "manifest.json"))
            )
            let tokenizerData = try Data(
                contentsOf: fixtureDirectory()
                    .appending(path: vectors.tokenizer.directory)
                    .appending(path: "tokenizer.json")
            )

            XCTAssertEqual(manifest.contractVersion, Self.contractVersion)
            XCTAssertEqual(sha256(tokenizerData), vectors.tokenizer.sha256)
            for file in manifest.files {
                let data = try Data(contentsOf: fixtureDirectory().appending(path: file.path))
                XCTAssertEqual(sha256(data), file.sha256, "TOKENIZER_ASSET_DIGEST_MISMATCH")
            }
        }

        func testSwiftTransformersMatchesSharedVectorsBeforeInference() throws {
            let vectors = try loadVectors()
            let support = try loadSupport()
            let tokenizer = try OpenMed.loadTokenizerFromDirectory(
                fixtureDirectory().appending(path: vectors.tokenizer.directory),
                fallbackTokenizerName: nil
            )

            let report = verify(vectors: vectors, support: support, tokenizer: tokenizer)
            print(
                "TOKENIZER_CONFORMANCE runtime=\(Self.runtime) "
                    + "status=\(report.mismatches.isEmpty ? "pass" : "fail") "
                    + "checked=\(report.checked) "
                    + "unsupported=\(report.unsupported.joined(separator: ",")) "
                    + "declared_divergences=\(report.declared.count)"
            )

            XCTAssertTrue(report.mismatches.isEmpty, report.diagnostics)
            XCTAssertGreaterThan(report.checked, 0)
            XCTAssertEqual(
                report.unsupported,
                [
                    "normalized:normalizer_not_public",
                    "special_tokens_mask:special_tokens_mask_not_exposed",
                ]
            )
        }

        func testSpecialTokenAndNormalizationChangesFailBeforeModelExecution() throws {
            let vectors = try loadVectors()
            let support = try loadSupport()
            let source = fixtureDirectory().appending(path: vectors.tokenizer.directory)
            var modelCalls = 0

            let withoutCLS = try mutatedTokenizerDirectory(from: source) { tokenizer, _ in
                var postProcessor = tokenizer["post_processor"] as? [String: Any] ?? [:]
                let single = postProcessor["single"] as? [[String: Any]] ?? []
                postProcessor["single"] = single.filter { item in
                    (item["SpecialToken"] as? [String: Any])?["id"] as? String != "[CLS]"
                }
                tokenizer["post_processor"] = postProcessor
            }
            let cased = try mutatedTokenizerDirectory(from: source) { tokenizer, config in
                var normalizer = tokenizer["normalizer"] as? [String: Any] ?? [:]
                normalizer["lowercase"] = false
                normalizer["strip_accents"] = false
                tokenizer["normalizer"] = normalizer
                config["do_lower_case"] = false
            }

            for (directory, index) in [(withoutCLS, 0), (cased, 1)] {
                let tokenizer = try OpenMed.loadTokenizerFromDirectory(
                    directory,
                    fallbackTokenizerName: nil
                )
                let report = verify(vectors: vectors, support: support, tokenizer: tokenizer)
                if report.mismatches.isEmpty {
                    modelCalls += 1
                }
                XCTAssertTrue(
                    report.mismatches.contains {
                        $0.caseID == "ascii-basic-01" && $0.field == "input_ids"
                            && $0.index == index
                    },
                    report.diagnostics
                )
                for vectorCase in vectors.cases where vectorCase.text.count > 3 {
                    XCTAssertFalse(report.diagnostics.contains(vectorCase.text))
                }
            }
            XCTAssertEqual(modelCalls, 0)
        }

        // MARK: - Runner

        private func run(
            _ vectorCase: Case,
            tokenizer: any Tokenizer,
            templateSpecialTokens: Int
        ) -> [String: [String]] {
            let (inputIDs, attentionMask, tokenTypeIDs, offsets) = OpenMed.encodeForModel(
                tokenizer: tokenizer,
                text: vectorCase.text,
                maxSeqLength: 512
            )
            var produced: [String: [String]] = [
                "tokens": tokenizer.convertIdsToTokens(inputIDs).map { $0 ?? "" },
                "input_ids": inputIDs.map(String.init),
                "attention_mask": attentionMask.map(String.init),
                "token_type_ids": tokenTypeIDs.map(String.init),
                "offsets": offsets.map { "\($0.0),\($0.1)" },
            ]
            if let window = vectorCase.window {
                // Single-pass model input at maxSeqLength, as analyzeText uses it.
                produced["truncated_input_ids"] = OpenMed.encodeForModel(
                    tokenizer: tokenizer,
                    text: vectorCase.text,
                    maxSeqLength: window.maxLength
                ).0.map(String.init)
                let chunks = OpenMed.makeTokenChunks(
                    tokenOffsets: OpenMed.contentTokenOffsets(
                        tokenizer: tokenizer,
                        text: vectorCase.text
                    ),
                    text: vectorCase.text,
                    chunkTokenLimit: window.maxLength - templateSpecialTokens,
                    tokenOverlap: window.stride
                )
                produced["window_ranges"] = chunks.map { "\($0.tokenStart),\($0.tokenEnd)" }
                produced["window_input_ids"] = chunks.map { chunk in
                    let chunkText = PostProcessing.scalarSubstring(
                        vectorCase.text,
                        start: chunk.start,
                        end: chunk.end
                    )
                    return OpenMed.encodeForModel(
                        tokenizer: tokenizer,
                        text: chunkText,
                        maxSeqLength: 512
                    ).0.map(String.init).joined(separator: ",")
                }
            }
            return produced
        }

        private func verify(
            vectors: Vectors,
            support: Support,
            tokenizer: any Tokenizer
        ) -> Report {
            var report = Report()
            guard vectors.contractVersion == Self.contractVersion,
                support.contractVersion == Self.contractVersion
            else {
                report.mismatches.append(
                    Mismatch(
                        caseID: "*",
                        field: "runtime",
                        index: nil,
                        code: "TOKENIZER_CONTRACT_VERSION_MISMATCH"
                    )
                )
                return report
            }
            guard let entry = support.runtimes[Self.runtime] else {
                report.mismatches.append(
                    Mismatch(
                        caseID: "*",
                        field: "runtime",
                        index: nil,
                        code: "TOKENIZER_RUNTIME_UNDECLARED"
                    )
                )
                return report
            }
            var unsupported = Set<String>()
            for field in Self.fields where entry.fields[field]?.status != "supported" {
                unsupported.insert(field)
                report.unsupported.append("\(field):\(entry.fields[field]?.reason ?? "undeclared")")
            }
            var observed = Set<String>()
            for vectorCase in vectors.cases {
                let expected = vectorCase.expected.byField
                let produced = run(
                    vectorCase,
                    tokenizer: tokenizer,
                    templateSpecialTokens: vectors.tokenizer.templateSpecialTokens
                )
                let features = Set(vectorCase.features)
                for field in Self.fields {
                    guard let expectedValues = expected[field], !unsupported.contains(field) else {
                        continue
                    }
                    guard let producedValues = produced[field] else {
                        report.mismatches.append(
                            Mismatch(
                                caseID: vectorCase.id,
                                field: field,
                                index: nil,
                                code: "TOKENIZER_FIELD_MISSING"
                            )
                        )
                        continue
                    }
                    guard let finding = compare(expectedValues, producedValues) else {
                        report.checked += 1
                        continue
                    }
                    let matching = entry.divergences.filter {
                        features.contains($0.feature) && $0.fields.contains(field)
                    }
                    if let first = matching.first {
                        for item in matching {
                            observed.insert("\(item.feature)|\(field)")
                        }
                        report.declared.append((vectorCase.id, field, first.reason))
                        continue
                    }
                    report.checked += 1
                    report.mismatches.append(
                        Mismatch(
                            caseID: vectorCase.id,
                            field: field,
                            index: finding.index,
                            code: finding.code
                        )
                    )
                }
            }
            for item in entry.divergences {
                for field in item.fields where !observed.contains("\(item.feature)|\(field)") {
                    report.mismatches.append(
                        Mismatch(
                            caseID: "*",
                            field: field,
                            index: nil,
                            code: "TOKENIZER_DECLARED_DIVERGENCE_STALE",
                            feature: item.feature
                        )
                    )
                }
            }
            return report
        }

        private func compare(_ expected: [String], _ produced: [String]) -> (code: String, index: Int)? {
            for (index, pair) in zip(expected, produced).enumerated() where pair.0 != pair.1 {
                return ("TOKENIZER_FIELD_VALUE_MISMATCH", index)
            }
            if expected.count != produced.count {
                return ("TOKENIZER_FIELD_LENGTH_MISMATCH", min(expected.count, produced.count))
            }
            return nil
        }

        // MARK: - Fixtures

        private func mutatedTokenizerDirectory(
            from source: URL,
            mutate: (inout [String: Any], inout [String: Any]) -> Void
        ) throws -> URL {
            let directory = try FileManager.default.url(
                for: .itemReplacementDirectory,
                in: .userDomainMask,
                appropriateFor: FileManager.default.temporaryDirectory,
                create: true
            )
            addTeardownBlock { try? FileManager.default.removeItem(at: directory) }
            var tokenizer =
                try JSONSerialization.jsonObject(
                    with: Data(contentsOf: source.appending(path: "tokenizer.json"))
                ) as? [String: Any] ?? [:]
            var config =
                try JSONSerialization.jsonObject(
                    with: Data(contentsOf: source.appending(path: "tokenizer_config.json"))
                ) as? [String: Any] ?? [:]
            mutate(&tokenizer, &config)
            try JSONSerialization.data(withJSONObject: tokenizer).write(
                to: directory.appending(path: "tokenizer.json")
            )
            try JSONSerialization.data(withJSONObject: config).write(
                to: directory.appending(path: "tokenizer_config.json")
            )
            try Data(contentsOf: source.appending(path: "config.json")).write(
                to: directory.appending(path: "config.json")
            )
            return directory
        }

        private func loadVectors() throws -> Vectors {
            try JSONDecoder().decode(
                Vectors.self,
                from: Data(
                    contentsOf: fixtureDirectory().appending(path: "tokenizer_conformance_v1.json")
                )
            )
        }

        private func loadSupport() throws -> Support {
            try JSONDecoder().decode(
                Support.self,
                from: Data(
                    contentsOf: fixtureDirectory().appending(
                        path: "tokenizer_runtime_support_v1.json"
                    )
                )
            )
        }

        private func sha256(_ data: Data) -> String {
            SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        }

        private func fixtureDirectory() -> URL {
            var url = URL(fileURLWithPath: #filePath)
            for _ in 0..<5 { url.deleteLastPathComponent() }
            return url.appending(path: "tests/fixtures/parity/tokenizer")
        }
    }
#else
    /// swift-transformers is unavailable on this platform, so the tokenizer
    /// conformance runner reports an explicit unsupported status.
    final class TokenizerConformanceTests: XCTestCase {
        func testSwiftTransformersMatchesSharedVectorsBeforeInference() throws {
            print(
                "TOKENIZER_CONFORMANCE runtime=swift-transformers status=unsupported "
                    + "reason=tokenizers_unavailable_on_platform"
            )
            throw XCTSkip("tokenizers_unavailable_on_platform")
        }
    }
#endif
