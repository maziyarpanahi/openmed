import Foundation
import XCTest

@testable import OpenMedKit

#if canImport(ZIPFoundation)
    import ZIPFoundation
#endif

/// Sentinel gate for value-free OpenMedKit errors.
///
/// Every error type declared under `swift/OpenMedKit/Sources` is enumerated
/// from source. Each declared case needs a sample here, and every sample and
/// every driven failure is rendered through `localizedDescription`,
/// `description`, `String(reflecting:)`, the bridged `NSError` and the
/// `LocalizedError` fields, then searched for synthetic sentinels, the
/// sentinel's byte-level BPE form and the sentinel-named temporary directory.
final class ValueFreeErrorSentinelTests: XCTestCase {
    private static let sentinel = "OMKSENTINEL7Q4Z"
    /// GPT-2 byte-level form of `" " + sentinel`, as the Privacy Filter BPE sees it.
    private static let byteLevelSentinel = "\u{0120}" + sentinel

    /// Error types whose sweep lives elsewhere, with the reason.
    private static let delegatedTypes: [String: String] = [
        "OpenMedModelStoreError": "model store errors are owned by #3644",
        "ExtensionRedactionError": "swept by ExtensionTests/ExtensionErrorSentinelTests",
    ]

    /// Struct errors whose codes reach `init(_:)` through one forwarding helper.
    private static let structCodeForwarders: [String: String] = [
        "OpenMedAECGError": "reject(parser, "
    ]

    private var sentinelDirectory: URL!

    override func setUpWithError() throws {
        let directory = FileManager.default.temporaryDirectory.appending(
            path: "openmed-\(Self.sentinel)-\(UUID().uuidString)",
            directoryHint: .isDirectory
        )
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        sentinelDirectory = directory
    }

    override func tearDownWithError() throws {
        if let sentinelDirectory {
            try? FileManager.default.removeItem(at: sentinelDirectory)
        }
    }

    // MARK: - Catalogue

    func testEveryDeclaredErrorTypeAndCaseIsSwept() throws {
        let declared = try Self.declaredErrors()
        XCTAssertGreaterThanOrEqual(declared.count, 25, "the source scan found too few error types")
        let catalogue = Dictionary(declared.map { ($0.name, $0) }, uniquingKeysWith: { first, _ in first })
        XCTAssertEqual(catalogue.count, declared.count, "error type names must be unique for the sweep")

        var samples = try Self.samples(catalogue: catalogue)
        for (name, errors) in try drivenStructErrors() {
            samples[name, default: []] += errors
        }

        let expected = Set(catalogue.keys)
            .subtracting(Self.delegatedTypes.keys)
            .subtracting(Self.typesCompiledOutOfThisBuild)
        XCTAssertEqual(
            Set(samples.keys).subtracting(expected),
            [],
            "samples exist for types the source scan did not find"
        )
        XCTAssertEqual(
            expected.subtracting(samples.keys),
            [],
            "declared error types without a sentinel sweep; add samples here"
        )

        for item in declared where item.kind != "struct" {
            guard let errors = samples[item.name] else { continue }
            XCTAssertEqual(
                Set(errors.map(Self.caseLabel)),
                Set(item.cases.map(\.name)),
                "\(item.name) has cases without a sentinel sample"
            )
        }
        for errors in samples.values {
            for error in errors {
                assertValueFree(error)
            }
        }
    }

    func testStructErrorCodesAreFixedLiterals() throws {
        let structs = try Self.declaredErrors().filter { $0.kind == "struct" }
        XCTAssertEqual(
            Set(structs.map(\.name)),
            ["BigTiffPreflightError", "EDFError", "OpenMedAECGError"]
        )
        for item in structs {
            XCTAssertTrue(
                item.body.contains("fileprivate init(_ code: String)"),
                "\(item.name) must keep its code initializer file-private"
            )
            let constructions = Self.count(of: #"\b\#(item.name)\("#, in: item.fileContents)
            let literals = Self.count(of: #"\b\#(item.name)\("[a-z0-9_]+"\)"#, in: item.fileContents)
            if let forwarder = Self.structCodeForwarders[item.name] {
                let escaped = NSRegularExpression.escapedPattern(for: forwarder)
                XCTAssertEqual(constructions, literals + 1, "\(item.name) may forward codes only once")
                let forwarded = Self.count(of: escaped, in: item.fileContents)
                XCTAssertGreaterThan(forwarded, 0)
                XCTAssertEqual(
                    Self.count(of: escaped + "\"[a-z0-9_]+\"\\)", in: item.fileContents),
                    forwarded,
                    "\(item.name) forwarder call sites must pass fixed literals"
                )
            } else {
                XCTAssertEqual(constructions, literals, "\(item.name) codes must be fixed literals")
            }
            XCTAssertGreaterThan(literals, 0)
        }
    }

    func testSweepFlagsTheErrorShapesItReplaces() throws {
        // Negative controls: a raw Foundation read error from the sentinel
        // directory and a case carrying the byte-level piece both leak.
        XCTAssertThrowsError(try Data(contentsOf: sentinelDirectory.appending(path: "config.json"))) {
            XCTAssertTrue(leaks($0), "a Foundation read error carries the path")
        }
        XCTAssertTrue(leaks(ValueBearingProbe.piece(Self.byteLevelSentinel)))
        XCTAssertTrue(leaks(ValueBearingProbe.location(sentinelDirectory)))
        XCTAssertFalse(leaks(ValueBearingProbe.index(7)))
    }

    // MARK: - Driven failures

    func testPolicyLookupDoesNotEchoTheRequestedName() {
        XCTAssertThrowsError(try Policy(named: "profile-\(Self.sentinel)")) { error in
            guard case PolicyError.unknownProfile(let allowed) = error else {
                return XCTFail("Unexpected error type")
            }
            XCTAssertTrue(allowed.contains(Policy.defaultName))
            assertValueFree(error)
        }
    }

    func testMapleParserFailuresDoNotQuoteGeneratedText() {
        let document = "Synthetic note for \(Self.sentinel)."
        XCTAssertThrowsError(
            try OpenMedMapleOutputParser.parse(
                "</think>{\"entities\": \"\(Self.sentinel)\"}",
                task: .entityExtraction,
                sourceDocument: document
            )
        ) { error in
            guard case OpenMedMapleOutputParser.Error.invalidJSONObject(let code) = error else {
                return XCTFail("Unexpected error type")
            }
            XCTAssertEqual(code, "type_mismatch")
            assertValueFree(error)
        }
        XCTAssertThrowsError(
            try OpenMedMapleOutputParser.parse(
                "</think>{\"entities\": [{\"label\": \"\(Self.sentinel)\"}]}",
                task: .entityExtraction,
                sourceDocument: document
            )
        ) { error in
            guard case OpenMedMapleOutputParser.Error.invalidJSONObject(let code) = error else {
                return XCTFail("Unexpected error type")
            }
            XCTAssertEqual(code, "key_not_found")
            assertValueFree(error)
        }
        XCTAssertThrowsError(
            try OpenMedMapleOutputParser.parse(
                "</think>\(Self.sentinel) without JSON",
                task: .entityExtraction,
                sourceDocument: document
            )
        ) { assertValueFree($0) }
    }

    func testArtifactErrorsFromSentinelDirectoryOmitThePath() throws {
        let cases: [(String, OpenMedMLXArtifactError, (URL) throws -> Void)] = [
            ("missing config", .unreadableFile(.config), { _ in }),
            (
                "malformed config", .malformedFile(.config),
                { try Data("{\"model_type\": \(Self.sentinel)".utf8).write(to: $0.appending(path: "config.json")) }
            ),
            (
                "malformed manifest", .malformedFile(.manifest),
                { try Data("{\"format\": \"\(Self.sentinel)\"".utf8).write(to: $0.appending(path: "openmed-mlx.json")) }
            ),
            (
                "foreign manifest format", .invalidManifestFormat,
                { try self.writeArtifact(at: $0, manifest: ["format": Self.sentinel]) }
            ),
            (
                "unrecognized family", .unsupportedArchitecture(task: .tokenClassification, family: nil),
                { try self.writeArtifact(at: $0, manifest: ["family": Self.sentinel]) }
            ),
            (
                "unrecognized task", .unsupportedArchitecture(task: nil, family: .bert),
                { try self.writeArtifact(at: $0, manifest: ["task": Self.sentinel]) }
            ),
            (
                "missing listed tokenizer", .missingTokenizerAssets,
                {
                    try self.writeArtifact(
                        at: $0,
                        manifest: ["tokenizer": ["path": ".", "files": ["\(Self.sentinel)-tokenizer.json"]]]
                    )
                }
            ),
            (
                "no tokenizer reference", .missingTokenizerReference,
                { try self.writeConfig(at: $0, sourceModel: nil) }
            ),
            (
                "missing label map", .unreadableFile(.labelMap),
                { try self.writeArtifact(at: $0, manifest: ["label_map_path": "\(Self.sentinel)-labels.json"]) }
            ),
            (
                "malformed label map", .malformedFile(.labelMap),
                {
                    try self.writeArtifact(at: $0, manifest: [:])
                    try Data("[\"\(Self.sentinel)\"]".utf8).write(to: $0.appending(path: "id2label.json"))
                }
            ),
        ]

        for (index, (name, expected, prepare)) in cases.enumerated() {
            let directory = sentinelDirectory.appending(
                path: "artifact-\(index)-\(Self.sentinel)",
                directoryHint: .isDirectory
            )
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            try prepare(directory)
            XCTAssertThrowsError(try OpenMedMLXArtifact(modelDirectoryURL: directory), name) { error in
                XCTAssertEqual(error as? OpenMedMLXArtifactError, expected, name)
                assertValueFree(error)
            }
        }

        let valid = sentinelDirectory.appending(path: "artifact-valid-\(Self.sentinel)", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: valid, withIntermediateDirectories: true)
        try writeArtifact(at: valid, manifest: [:])
        XCTAssertNoThrow(try OpenMedMLXArtifact(modelDirectoryURL: valid), "negative control must load")
    }

    func testSegmenterErrorsFromSentinelDirectoryOmitThePath() throws {
        let resourceFolder = "\(Self.sentinel)-resources"
        let cases: [(String, OpenMedSegmenterError, (URL) throws -> Void)] = [
            ("missing manifest", .unreadableManifest, { _ in }),
            (
                "malformed manifest", .invalidDescriptor(.malformedManifest),
                { try Data("{\"segmenter\": \(Self.sentinel)".utf8).write(to: $0.appending(path: "openmed-mlx.json")) }
            ),
            ("no descriptor", .missingDescriptor, { try self.writeJSON(["family": Self.sentinel], to: $0.appending(path: "openmed-mlx.json")) }),
            (
                "unsupported id", .invalidDescriptor(.unsupportedSegmenterID),
                { try self.writeSegmenterManifest(at: $0, id: Self.sentinel, resourcePath: "han_words.txt") }
            ),
            (
                "escaping resource", .invalidDescriptor(.resourceEscapesBundle),
                { try self.writeSegmenterManifest(at: $0, resourcePath: "../\(resourceFolder)/han_words.txt") }
            ),
            (
                "missing resource", .missingResource(index: 0),
                { try self.writeSegmenterManifest(at: $0, resourcePath: "\(resourceFolder)/han_words.txt") }
            ),
            (
                "unreadable resource", .unreadableResource(index: 0),
                {
                    try self.writeSegmenterManifest(at: $0, resourcePath: "\(resourceFolder)/han_words.txt")
                    try FileManager.default.createDirectory(
                        at: $0.appending(path: "\(resourceFolder)/han_words.txt"),
                        withIntermediateDirectories: true
                    )
                }
            ),
            (
                "size mismatch", .invalidDescriptor(.resourceSizeMismatch),
                {
                    try self.writeSegmenterManifest(at: $0, resourcePath: "\(resourceFolder)/han_words.txt")
                    try FileManager.default.createDirectory(
                        at: $0.appending(path: resourceFolder),
                        withIntermediateDirectories: true
                    )
                    try Data("\(Self.sentinel)\n".utf8).write(to: $0.appending(path: "\(resourceFolder)/han_words.txt"))
                }
            ),
        ]

        for (index, (name, expected, prepare)) in cases.enumerated() {
            let directory = sentinelDirectory.appending(
                path: "segmenter-\(index)-\(Self.sentinel)",
                directoryHint: .isDirectory
            )
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            try prepare(directory)
            XCTAssertThrowsError(try OpenMedSegmenter(bundleURL: directory), name) { error in
                XCTAssertEqual(error as? OpenMedSegmenterError, expected, name)
                assertValueFree(error)
            }
        }
    }

    func testStructErrorsDrivenWithSentinelInputsStayValueFree() throws {
        for errors in try drivenStructErrors().values {
            XCTAssertFalse(errors.isEmpty)
            for error in errors {
                assertValueFree(error)
            }
        }
    }

    #if canImport(MLX) && !os(watchOS) && !os(visionOS)
        func testUnknownTokenOmitsSentinelAndItsByteLevelPiece() throws {
            let text = "Patient \(Self.sentinel)"
            let directory = sentinelDirectory.appending(path: "tokenizer-\(Self.sentinel)", directoryHint: .isDirectory)
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

            // Negative control: with the merged piece in the vocabulary the
            // fixture encodes, proving the unknown piece is the byte-level
            // sentinel that the old error message would have quoted.
            try writePrivacyFilterTokenizer(at: directory, includeSentinelPiece: true)
            let control = try OpenMedPrivacyFilterTokenizer(directoryURL: directory).encode(text, maxTokens: 64)
            XCTAssertEqual(control.tokenIDs.count, 8)

            try writePrivacyFilterTokenizer(at: directory, includeSentinelPiece: false)
            let tokenizer = try OpenMedPrivacyFilterTokenizer(directoryURL: directory)
            XCTAssertThrowsError(try tokenizer.encode(text, maxTokens: 64)) { error in
                XCTAssertEqual(error as? OpenMedPrivacyFilterError, .unknownToken(index: 7))
                XCTAssertTrue(error.localizedDescription.contains("index 7"))
                assertValueFree(error)
            }
        }

        func testPrivacyFilterTokenizerFileErrorsOmitThePath() throws {
            let cases: [(String, OpenMedPrivacyFilterError, (URL) throws -> Void)] = [
                ("missing", .missingTokenizer, { _ in }),
                (
                    "unreadable", .unreadableTokenizer,
                    { try FileManager.default.createDirectory(at: $0.appending(path: "tokenizer.json"), withIntermediateDirectories: true) }
                ),
                (
                    "malformed", .invalidTokenizer,
                    { try Data("{\"model\": \"\(Self.sentinel)\"}".utf8).write(to: $0.appending(path: "tokenizer.json")) }
                ),
                (
                    "not BPE", .invalidTokenizer,
                    {
                        try self.writeJSON(
                            ["model": ["type": Self.sentinel, "vocab": [Self.sentinel: 0], "merges": [String]()]],
                            to: $0.appending(path: "tokenizer.json")
                        )
                    }
                ),
            ]
            for (index, (name, expected, prepare)) in cases.enumerated() {
                let directory = sentinelDirectory.appending(
                    path: "pf-\(index)-\(Self.sentinel)",
                    directoryHint: .isDirectory
                )
                try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
                try prepare(directory)
                XCTAssertThrowsError(try OpenMedPrivacyFilterTokenizer(directoryURL: directory), name) { error in
                    XCTAssertEqual(error as? OpenMedPrivacyFilterError, expected, name)
                    assertValueFree(error)
                }
            }
        }

        func testPipelineArtifactMismatchUsesClosedVocabulary() throws {
            let directory = sentinelDirectory.appending(path: "mismatch-\(Self.sentinel)", directoryHint: .isDirectory)
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            try writeArtifact(at: directory, manifest: ["task": "Token_Classification", "family": "BERT"])

            XCTAssertThrowsError(try OpenMedPrivacyFilterPipeline(modelDirectoryURL: directory)) { error in
                if let error = error as? OpenMedPrivacyFilterError {
                    XCTAssertEqual(error, .unsupportedArtifact(task: .tokenClassification, family: .bert))
                } else {
                    XCTAssertTrue(error is OpenMedMLXRuntimeError, "Unexpected error type")
                }
                assertValueFree(error)
            }

            #if canImport(Tokenizers)
                XCTAssertThrowsError(try OpenMedZeroShotNER(modelDirectoryURL: directory)) { error in
                    if case OpenMedZeroShotError.unsupportedArtifact(_, _, let task, let family) = error {
                        XCTAssertEqual(task, "token-classification")
                        XCTAssertEqual(family, "bert", "raw manifest spelling must not be echoed")
                    } else {
                        XCTAssertTrue(error is OpenMedMLXRuntimeError, "Unexpected error type")
                    }
                    assertValueFree(error)
                }
            #endif
        }
    #endif

    #if canImport(MLX) && canImport(ZIPFoundation) && !os(watchOS) && !os(visionOS)
        func testWeightErrorsFromSentinelDirectoryOmitPathsAndEntryNames() throws {
            let directory: URL = sentinelDirectory
            let missing = directory.appending(path: "\(Self.sentinel)-missing.safetensors")
            let unsupported = directory.appending(path: "\(Self.sentinel).bin")
            try Data([1]).write(to: unsupported)
            let garbageArchive = directory.appending(path: "\(Self.sentinel)-garbage.npz")
            try Data("PK\(Self.sentinel)".utf8).write(to: garbageArchive)

            // `unreadableWeights` wraps `MLX.loadArrays`, which needs the Metal
            // library the SwiftPM CLI bundle lacks; its sample covers the case.
            let cases: [([URL], OpenMedMLXWeightError)] = [
                ([missing], .missingWeights(candidateCount: 1)),
                ([missing, unsupported], .unsupportedWeightFile(candidateIndex: 1)),
                ([garbageArchive], .invalidNPZArchive(candidateIndex: 0)),
                ([try makeArchive(npy: Data("\(Self.sentinel)".utf8))], .invalidNPYHeader(entryIndex: 0)),
                ([try makeArchive(npy: makeNPY(descriptor: ">f4"))], .unsupportedNPYEndianness(entryIndex: 0)),
                ([try makeArchive(npy: makeNPY(descriptor: "<c16"))], .unsupportedNPYDType(entryIndex: 0)),
                ([try makeArchive(npy: makeNPY(descriptor: "<f4", fortranOrder: true))], .unsupportedFortranOrder(entryIndex: 0)),
            ]
            for (candidates, expected) in cases {
                XCTAssertThrowsError(try OpenMedMLXWeightArchive.loadWeights(from: candidates)) { error in
                    XCTAssertEqual(error as? OpenMedMLXWeightError, expected)
                    assertValueFree(error)
                }
            }
        }
    #endif

    #if canImport(MLX) && canImport(MLXLMCommon) && canImport(MLXLLM) && canImport(MLXNN) && canImport(Tokenizers) && !os(watchOS) && !os(visionOS)
        func testMapleRuntimeErrorsOmitPathAndConfigurationValues() async throws {
            do {
                _ = try await OpenMedMaple(modelDirectoryURL: sentinelDirectory)
                XCTFail("an empty directory must not load")
            } catch {
                guard case OpenMedMapleRuntimeError.missingFiles(let files) = error else {
                    return XCTFail("Unexpected error type")
                }
                XCTAssertEqual(files, OpenMedMaple.requiredModelFiles)
                assertValueFree(error)
            }

            let configuration = try JSONDecoder().decode(
                OpenMedMapleConfiguration.self,
                from: Data(#"{"model_type": "\#(Self.sentinel)", "num_hidden_layers": 1}"#.utf8)
            )
            XCTAssertThrowsError(try configuration.validate()) { error in
                guard case OpenMedMapleRuntimeError.unsupportedArchitecture = error else {
                    return XCTFail("Unexpected error type")
                }
                assertValueFree(error)
            }
        }
    #endif

    // MARK: - Samples

    /// One sample per declared case. Payloads are opaque references, counts or
    /// closed vocabularies; the driven tests above show input never reaches them.
    private static func samples(catalogue: [String: DeclaredError]) throws -> [String: [Error]] {
        var samples: [String: [Error]] = [:]
        func add<E: Error>(_ values: [E]) {
            samples[String(describing: E.self), default: []] += values.map { $0 as Error }
        }
        func addRawCases<E: Error & RawRepresentable>(_ type: E.Type) throws where E.RawValue == String {
            let name = String(describing: type)
            let declared = try XCTUnwrap(catalogue[name], "\(name) was not found in Sources")
            add(
                try declared.cases.map { item in
                    try XCTUnwrap(E(rawValue: item.rawValue ?? item.name), "\(name).\(item.name)")
                }
            )
        }

        try addRawCases(AgentGovernanceError.self)
        try addRawCases(AmbientAudioCaptureError.self)
        try addRawCases(AmbientSessionError.self)
        try addRawCases(AudioSampleConversionError.self)
        try addRawCases(ClinicalBriefError.self)
        try addRawCases(ClinicalBriefFHIRDocumentError.self)
        try addRawCases(MultimodalContractRefusal.self)
        try addRawCases(MultimodalReviewError.self)
        try addRawCases(PreprocessingConformanceError.self)
        try addRawCases(ReviewAdmissionRefusal.self)
        try addRawCases(StreamingSpeechError.self)
        try addRawCases(TranscriptLanguageError.self)

        add([
            SpeechEndpointError.invalidLimits, .detectorNotLocal, .invalidFrame, .overlappingFrame,
            .cancelled, .detectorFailure, .detectorResetFailure,
        ])
        add([MultimodalThermalError.invalidBudget])
        add(
            OpenMedApplePlatform.allCases.map { PlatformModelError.noCompatibleModel($0) }
                + [.invalidInputShape, .sequenceTooLong(actual: 4_097, maximum: 512)]
        )
        add([
            PolicyError.blankName,
            .unknownProfile(allowed: Policy.bundledProfileNames + Policy.aliases.keys.sorted()),
            .missingResource(Policy.defaultName),
            .invalidProfile("unreadable_resource"),
            .invalidProfile("malformed_resource"),
            .invalidProfile("unsupported_schema_version"),
        ])
        add([NERPipelineError.missingOutput("logits"), .uncompiledModelUnsupported])
        add(
            [
                OpenMedSegmenterError.unreadableManifest, .missingDescriptor,
                .missingResource(index: 0), .unreadableResource(index: 2),
            ]
                + OpenMedSegmenterError.DescriptorReason.allCases.map {
                    OpenMedSegmenterError.invalidDescriptor($0)
                }
        )
        add(
            [
                OpenMedMLXArtifactError.invalidManifestFormat, .missingTokenizerAssets, .missingTokenizerReference,
                .unsupportedArchitecture(task: .tokenClassification, family: .debertaV2),
                .unsupportedArchitecture(task: nil, family: nil),
            ]
                + OpenMedMLXArtifactError.File.allCases.flatMap { file -> [OpenMedMLXArtifactError] in
                    [.unreadableFile(file), .malformedFile(file)]
                }
        )
        add(
            [OpenMedMapleOutputParser.Error.missingJSONObject]
                + ["type_mismatch", "value_not_found", "key_not_found", "data_corrupted"].map {
                    OpenMedMapleOutputParser.Error.invalidJSONObject($0)
                }
        )

        #if canImport(MLX) && !os(watchOS) && !os(visionOS)
            add([OpenMedMLXRuntimeError.unsupportedPlatform])
            add([
                OpenMedPrivacyFilterError.unsupportedArtifact(task: .tokenClassification, family: .bert),
                .missingTokenizer, .unreadableTokenizer, .invalidTokenizer, .unknownToken(index: 7),
            ])
        #endif
        #if canImport(MLX) && canImport(ZIPFoundation) && !os(watchOS) && !os(visionOS)
            add([
                OpenMedMLXWeightError.unsupportedWeightFile(candidateIndex: 0), .missingWeights(candidateCount: 2),
                .unreadableWeights(candidateIndex: 0), .invalidNPZArchive(candidateIndex: 1),
                .invalidNPYHeader(entryIndex: 3), .unsupportedNPYDType(entryIndex: 3),
                .unsupportedNPYEndianness(entryIndex: 3), .unsupportedFortranOrder(entryIndex: 3),
            ])
        #endif
        #if canImport(MLX) && canImport(Tokenizers) && !os(watchOS) && !os(visionOS)
            add([
                OpenMedZeroShotError.unsupportedArtifact(
                    expectedTask: OpenMedMLXTask.zeroShotNER.rawValue,
                    expectedFamily: OpenMedMLXFamily.glinerUniEncoderSpan.rawValue,
                    actualTask: OpenMedMLXTask.tokenClassification.rawValue,
                    actualFamily: OpenMedMLXFamily.bert.rawValue
                ),
                .missingPromptSpec("entity_token"),
                .emptyInput,
            ])
        #endif
        #if canImport(MLXLMCommon) && canImport(Tokenizers) && !os(watchOS) && !os(visionOS)
            add([
                OpenMedCompassError.invalidAspectRatio(201), .invalidDimensions(0, 4), .invalidGenerationLimit(0),
                .imageTokenMissing, .imageTokenCountMismatch, .videoUnsupported,
            ])
        #endif
        #if canImport(MLX) && canImport(MLXLMCommon) && canImport(MLXLLM) && canImport(MLXNN) && canImport(Tokenizers) && !os(watchOS) && !os(visionOS)
            add([
                OpenMedMapleRuntimeError.missingFiles(OpenMedMaple.requiredModelFiles),
                .unsupportedArchitecture,
                .invalidConfiguration("the runtime has been unloaded"),
            ])
        #endif
        return samples
    }

    /// Error types declared in source but excluded by this build's conditions.
    private static var typesCompiledOutOfThisBuild: Set<String> {
        var names: Set<String> = []
        #if !(canImport(MLX) && !os(watchOS) && !os(visionOS))
            names.formUnion(["OpenMedMLXRuntimeError", "OpenMedPrivacyFilterError"])
        #endif
        #if !(canImport(MLX) && canImport(ZIPFoundation) && !os(watchOS) && !os(visionOS))
            names.insert("OpenMedMLXWeightError")
        #endif
        #if !(canImport(MLX) && canImport(Tokenizers) && !os(watchOS) && !os(visionOS))
            names.insert("OpenMedZeroShotError")
        #endif
        #if !(canImport(MLXLMCommon) && canImport(Tokenizers) && !os(watchOS) && !os(visionOS))
            names.insert("OpenMedCompassError")
        #endif
        #if !(canImport(MLX) && canImport(MLXLMCommon) && canImport(MLXLLM) && canImport(MLXNN) && canImport(Tokenizers) && !os(watchOS) && !os(visionOS))
            names.insert("OpenMedMapleRuntimeError")
        #endif
        return names
    }

    /// Struct errors only expose file-private initializers, so drive them.
    private func drivenStructErrors() throws -> [String: [Error]] {
        let payload = Data(String(repeating: "\(Self.sentinel) ", count: 32).utf8)
        var errors: [String: [Error]] = [:]
        func capture(_ name: String, _ body: () throws -> Void) {
            do {
                try body()
                XCTFail("\(name) accepted sentinel input")
            } catch {
                errors[String(describing: type(of: error)), default: []].append(error)
            }
        }
        capture("EDFReader") { _ = try EDFReader.read(payload) }
        capture("OpenMedAECG") { _ = try OpenMedAECG.read(Data("<\(Self.sentinel)>\(Self.sentinel)</\(Self.sentinel)>".utf8)) }
        capture("OpenMedAECG XML") { _ = try OpenMedAECG.read(Data("<\(Self.sentinel)".utf8)) }
        capture("BigTiffPreflight") { _ = try BigTiffPreflight.read(payload, profile: .ifdChain) }
        capture("BigTiffPreflight size") { _ = try BigTiffPreflight.read(Data("\(Self.sentinel)".utf8), profile: .subIFDs) }
        return errors
    }

    // MARK: - Assertions

    private var forbiddenValues: [String] {
        var values = [Self.sentinel, Self.byteLevelSentinel]
        if let sentinelDirectory {
            values += [
                sentinelDirectory.path,
                sentinelDirectory.resolvingSymlinksInPath().path,
                sentinelDirectory.lastPathComponent,
            ]
        }
        let temporary = FileManager.default.temporaryDirectory
        values += [temporary.standardizedFileURL.path, temporary.resolvingSymlinksInPath().path]
        let home = NSHomeDirectory()
        if home.count > 1 { values.append(home) }
        return values
    }

    private func assertValueFree(_ error: Error, file: StaticString = #filePath, line: UInt = #line) {
        if leaks(error) {
            let label = "\(type(of: error)).\(Self.caseLabel(of: error))"
            XCTFail("\(label) renders a sentinel, path or home directory", file: file, line: line)
        }
    }

    private func leaks(_ error: Error) -> Bool {
        let forbidden = forbiddenValues
        return Self.renderings(of: error).contains { rendering in
            forbidden.contains { rendering.contains($0) }
        }
    }

    private static func renderings(of error: Error) -> [String] {
        let bridged = error as NSError
        var renderings = [
            error.localizedDescription,
            String(describing: error),
            String(reflecting: error),
            bridged.description,
            bridged.debugDescription,
            String(describing: bridged.userInfo),
        ]
        if let localized = error as? LocalizedError {
            renderings += [
                localized.errorDescription,
                localized.failureReason,
                localized.recoverySuggestion,
                localized.helpAnchor,
            ].compactMap { $0 }
        }
        return renderings
    }

    private static func caseLabel(of value: Any) -> String {
        let mirror = Mirror(reflecting: value)
        if mirror.displayStyle == .enum, let label = mirror.children.first?.label {
            return label
        }
        return String(describing: value)
    }

    // MARK: - Source scan

    private struct DeclaredCase {
        let name: String
        let rawValue: String?
    }

    private struct DeclaredError {
        let name: String
        let kind: String
        let body: String
        let fileContents: String
        let cases: [DeclaredCase]
    }

    private static var sourcesRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appending(path: "Sources", directoryHint: .isDirectory)
    }

    /// Every `enum`, `struct` or `class` under `Sources` that conforms to
    /// `Error` or `LocalizedError`, with its directly declared cases.
    private static func declaredErrors() throws -> [DeclaredError] {
        let declaration = try NSRegularExpression(
            pattern: #"(?m)^[ \t]*(?:(?:public|package|internal|fileprivate|private|final)[ \t]+)*(enum|struct|class)[ \t]+([A-Za-z_][A-Za-z0-9_]*)[ \t]*:([^{]*)\{"#
        )
        let files = try XCTUnwrap(
            FileManager.default.enumerator(at: sourcesRoot, includingPropertiesForKeys: nil)
        )
        var result: [DeclaredError] = []
        for case let url as URL in files where url.pathExtension == "swift" {
            let contents = try String(contentsOf: url, encoding: .utf8)
            let text = contents as NSString
            for match in declaration.matches(in: contents, range: NSRange(location: 0, length: text.length)) {
                let conformances = text.substring(with: match.range(at: 3))
                    .split { !$0.isLetter && !$0.isNumber && $0 != "_" }
                guard conformances.contains("Error") || conformances.contains("LocalizedError") else {
                    continue
                }
                let openingBrace = match.range.location + match.range.length - 1
                let body = Self.body(in: text, openingBrace: openingBrace)
                result.append(
                    DeclaredError(
                        name: text.substring(with: match.range(at: 2)),
                        kind: text.substring(with: match.range(at: 1)),
                        body: body,
                        fileContents: contents,
                        cases: Self.cases(in: body)
                    )
                )
            }
        }
        return result
    }

    /// Text between a declaration's braces, skipping braces in string literals.
    private static func body(in text: NSString, openingBrace: Int) -> String {
        var depth = 0
        var inString = false
        var escaped = false
        var index = openingBrace
        while index < text.length {
            let character = text.character(at: index)
            if !inString, character == 0x2F, index + 1 < text.length, text.character(at: index + 1) == 0x2F {
                // Skip a line comment so quotes or braces in prose do not count.
                while index < text.length, text.character(at: index) != 0x0A { index += 1 }
                continue
            }
            if inString {
                if escaped {
                    escaped = false
                } else if character == 0x5C {
                    escaped = true
                } else if character == 0x22 {
                    inString = false
                }
            } else if character == 0x22 {
                inString = true
            } else if character == 0x7B {
                depth += 1
            } else if character == 0x7D {
                depth -= 1
                if depth == 0 {
                    return text.substring(with: NSRange(location: openingBrace + 1, length: index - openingBrace - 1))
                }
            }
            index += 1
        }
        return ""
    }

    /// `case` declarations at the type's own nesting level.
    private static func cases(in body: String) -> [DeclaredCase] {
        var depth = 0
        var cases: [DeclaredCase] = []
        for line in body.split(separator: "\n", omittingEmptySubsequences: false) {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if depth == 0, trimmed.hasPrefix("case ") {
                cases += Self.splitCaseList(String(trimmed.dropFirst(5)))
            }
            for character in Self.codeOutsideStrings(String(line)) {
                if character == "{" { depth += 1 }
                if character == "}" { depth -= 1 }
            }
        }
        return cases
    }

    private static func splitCaseList(_ list: String) -> [DeclaredCase] {
        var elements: [String] = []
        var current = ""
        var depth = 0
        var inString = false
        for character in list {
            if character == "\"" { inString.toggle() }
            if !inString {
                if character == "(" || character == "[" { depth += 1 }
                if character == ")" || character == "]" { depth -= 1 }
                if character == "/" && current.hasSuffix("/") {
                    current.removeLast()
                    break
                }
                if character == "," && depth == 0 {
                    elements.append(current)
                    current = ""
                    continue
                }
            }
            current.append(character)
        }
        elements.append(current)
        return elements.compactMap { element in
            let trimmed = element.trimmingCharacters(in: .whitespaces)
            let name = String(trimmed.prefix { $0.isLetter || $0.isNumber || $0 == "_" })
            guard !name.isEmpty else { return nil }
            var rawValue: String?
            if let equals = trimmed.firstIndex(of: "=") {
                rawValue = trimmed[trimmed.index(after: equals)...]
                    .trimmingCharacters(in: .whitespaces)
                    .trimmingCharacters(in: CharacterSet(charactersIn: "\""))
            }
            return DeclaredCase(name: name, rawValue: rawValue)
        }
    }

    private static func codeOutsideStrings(_ line: String) -> String {
        var code = ""
        var inString = false
        var escaped = false
        for character in line {
            if inString {
                if escaped {
                    escaped = false
                } else if character == "\\" {
                    escaped = true
                } else if character == "\"" {
                    inString = false
                }
            } else if character == "\"" {
                inString = true
            } else if character == "/", code.hasSuffix("/") {
                code.removeLast()
                break
            } else {
                code.append(character)
            }
        }
        return code
    }

    private static func count(of pattern: String, in text: String) -> Int {
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return -1 }
        return regex.numberOfMatches(in: text, range: NSRange(text.startIndex..., in: text))
    }

    // MARK: - Fixtures

    private func writeJSON(_ object: Any, to url: URL) throws {
        try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys]).write(to: url)
    }

    private func writeConfig(at directory: URL, sourceModel: String?) throws {
        var config: [String: Any] = [
            "model_type": "bert",
            "vocab_size": 32,
            "hidden_size": 8,
            "num_attention_heads": 2,
            "num_hidden_layers": 1,
            "intermediate_size": 16,
            "max_position_embeddings": 32,
            "num_labels": 2,
            "id2label": ["0": "O", "1": "B-NAME"],
        ]
        if let sourceModel { config["_name_or_path"] = sourceModel }
        try writeJSON(config, to: directory.appending(path: "config.json"))
    }

    /// A manifest-only BERT artifact; `manifest` overrides top-level fields.
    private func writeArtifact(at directory: URL, manifest overrides: [String: Any]) throws {
        try writeConfig(at: directory, sourceModel: nil)
        try Data("{}".utf8).write(to: directory.appending(path: "tokenizer.json"))
        try writeJSON(["0": "O", "1": "B-NAME"], to: directory.appending(path: "id2label.json"))
        try Data().write(to: directory.appending(path: "weights.safetensors"))
        var manifest: [String: Any] = [
            "format": "openmed-mlx",
            "format_version": 1,
            "task": "token-classification",
            "family": "bert",
            "source_model_id": "OpenMed/\(Self.sentinel)",
            "config_path": "config.json",
            "label_map_path": "id2label.json",
            "preferred_weights": "weights.safetensors",
            "fallback_weights": [String](),
            "available_weights": ["weights.safetensors"],
            "weights_format": "safetensors",
            "max_sequence_length": 32,
            "tokenizer": ["path": ".", "files": ["tokenizer.json"]],
        ]
        manifest.merge(overrides) { _, new in new }
        try writeJSON(manifest, to: directory.appending(path: "openmed-mlx.json"))
    }

    private func writeSegmenterManifest(
        at directory: URL,
        id: String = "openmed-han-v1",
        resourcePath: String
    ) throws {
        let descriptor: [String: Any] = [
            "id": id,
            "format_version": 1,
            "scripts": ["Han"],
            "license": "MIT",
            "resource_files": [
                [
                    "path": resourcePath,
                    "role": "han_dictionary",
                    "license": "MIT",
                    "size_bytes": 4,
                    "sha256": "sha256:\(Self.sentinel)",
                ]
            ],
            "total_size_bytes": 4,
            "size_budget_bytes": 65_536,
        ]
        try writeJSON(["segmenter": descriptor], to: directory.appending(path: "openmed-mlx.json"))
    }

    #if canImport(MLX) && !os(watchOS) && !os(visionOS)
        /// Byte-level BPE whose merges fold `" " + sentinel` into one piece.
        private func writePrivacyFilterTokenizer(at directory: URL, includeSentinelPiece: Bool) throws {
            var vocab: [String: Int] = [:]
            for character in "Patient" where vocab[String(character)] == nil {
                vocab[String(character)] = vocab.count
            }
            if includeSentinelPiece {
                vocab[Self.byteLevelSentinel] = vocab.count
            }
            let pieces = Self.byteLevelSentinel.map(String.init)
            var merges: [[String]] = []
            var prefix = pieces[0]
            for piece in pieces.dropFirst() {
                merges.append([prefix, piece])
                prefix += piece
            }
            try writeJSON(
                ["model": ["type": "BPE", "vocab": vocab, "merges": merges]],
                to: directory.appending(path: "tokenizer.json")
            )
        }
    #endif

    #if canImport(MLX) && canImport(ZIPFoundation) && !os(watchOS) && !os(visionOS)
        private func makeNPY(descriptor: String, fortranOrder: Bool = false) -> Data {
            var header = "{'descr': '\(descriptor)', 'fortran_order': \(fortranOrder ? "True" : "False"), 'shape': (1,), }"
            let padded = ((10 + header.utf8.count + 1 + 15) / 16) * 16
            header += String(repeating: " ", count: padded - 10 - header.utf8.count - 1) + "\n"
            var data = Data([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 0x01, 0x00])
            data.append(UInt8(header.utf8.count & 0xff))
            data.append(UInt8((header.utf8.count >> 8) & 0xff))
            data.append(Data(header.utf8))
            data.append(Data(repeating: 0, count: 16))
            return data
        }

        /// An NPZ whose only entry is named after the sentinel.
        private func makeArchive(npy: Data) throws -> URL {
            let folder = sentinelDirectory.appending(path: UUID().uuidString, directoryHint: .isDirectory)
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            let entryName = "\(Self.sentinel).npy"
            try npy.write(to: folder.appending(path: entryName))
            let archiveURL = folder.appending(path: "weights-\(Self.sentinel).npz")
            let archive = try Archive(url: archiveURL, accessMode: .create)
            try archive.addEntry(with: entryName, relativeTo: folder)
            return archiveURL
        }
    #endif
}

/// The value-bearing shapes this gate replaced, kept as negative controls.
private enum ValueBearingProbe: Error {
    case piece(String)
    case location(URL)
    case index(Int)
}
