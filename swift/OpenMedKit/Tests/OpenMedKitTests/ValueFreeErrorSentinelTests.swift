import Foundation
import OpenMedPrivacySweep
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

    /// Private, payload-free markers caught inside their own file. The sweep
    /// cannot construct them, so the scan proves they stay private and empty.
    private static let privateMarkerTypes: Set<String> = [
        "DocumentRedactionPDF.Malformed",
        "ExecutionPlanResolver.MetadataInvalid",
    ]

    /// Struct errors that store a `String` code: each keeps a file-private code
    /// initializer and is constructed only from fixed literals, unless it is
    /// one of the `pathCodeStructErrors`.
    private static let codeStructErrors: Set<String> = [
        "BigTiffPreflightError", "EDFError", "KitParityError", "OpenMedAECGError",
        "PolicyError", "SlideDeidError", "WFDBError",
    ]

    /// Struct errors whose stored strings never render, by those fields. Each
    /// conforms to the description and reflection protocols with fixed
    /// renderings and mentions the fields only where it declares them.
    private static let unrenderedStructErrors: [String: [String]] = [
        "OpenMedBundleCacheCrash": ["checkpoint"]
    ]

    /// Struct errors with a public `(code, fieldPath)` initializer. Every
    /// construction across Sources passes a fixed code and a structural JSON path.
    private static let pathCodeStructErrors: Set<String> = ["KitParityError", "PolicyError"]

    /// Reviewed path segments that are neither path variables nor integer
    /// offsets: the helpers that bind them, and why they stay value-free. Every
    /// helper call passes a fixed literal or relays the segment unchanged.
    private static let reviewedPathSegments: [String: (segment: String, helpers: [String], reason: String)] = [
        "KitParityError": (
            "key",
            ["field", "requiredString", "optionalString", "requiredInteger", "optionalInteger", "requiredArray"],
            "fixed schema keys; unknown keys are reported at the enclosing object path"
        )
    ]

    /// Struct errors that store only closed reason enums, or nothing at all.
    private static let reasonStructErrors: Set<String> = [
        "DecisionCalibrationError", "DocumentRedactionPDF.Malformed", "ExecutionPlanResolver.MetadataInvalid",
        "OpenMedBundleArchive.Failure", "OpenMedBundleCacheError", "OpenMedManagedImportError",
        "PortableBundle.Refusal", "PortableBundleCalibration.Malformed", "PortableBundleError",
        "PortableBundleJSON.Invalid",
    ]

    /// How a struct error's codes reach its initializer through helpers.
    private struct CodeForwarder {
        /// Helpers that take a `code` parameter and pass it on.
        let helpers: [String]
        /// Literal the forwarding construction prepends to `code`.
        var prefix = ""
        /// Constructions and helper calls that pass `code` on instead of a literal.
        let forwardingSites: Int
        /// Helper calls that pass a fixed literal; pinned so new sites get reviewed.
        let literalCallers: Int
    }

    /// Struct errors whose codes reach their initializer through forwarding helpers.
    private static let structCodeForwarders: [String: CodeForwarder] = [
        "OpenMedAECGError": CodeForwarder(helpers: ["reject"], forwardingSites: 1, literalCallers: 7),
        "SlideDeidError": CodeForwarder(helpers: ["fail"], prefix: "slide_", forwardingSites: 1, literalCallers: 38),
        "WFDBError": CodeForwarder(helpers: ["exact", "discard"], forwardingSites: 3, literalCallers: 8),
    ]

    /// A stored `let` or `var` whose declared type mentions `String`.
    private static let storedStringPattern =
        #"(?m)^[ \t]*(?:(?:public|package|internal|fileprivate|private)[ \t]+)*(?:let|var)[ \t]+\w+[ \t]*:[^={\n]*\bString\b[^={\n]*(?:=[^{\n]*)?$"#

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
        let catalogue = Dictionary(declared.map { ($0.qualifiedName, $0) }, uniquingKeysWith: { first, _ in first })
        XCTAssertEqual(catalogue.count, declared.count, "error type names must be unique for the sweep")

        for name in Self.privateMarkerTypes {
            let item = try XCTUnwrap(catalogue[name], "\(name) was not found in Sources")
            XCTAssertTrue(item.isPrivate, "\(name) must stay private to its file")
            XCTAssertTrue(item.body.allSatisfy(\.isWhitespace), "\(name) must stay payload-free")
        }

        var samples = try Self.samples(catalogue: catalogue)
        for (name, errors) in try drivenStructErrors() {
            samples[name, default: []] += errors
        }

        let expected = Set(catalogue.keys)
            .subtracting(Self.delegatedTypes.keys)
            .subtracting(Self.privateMarkerTypes)
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
            guard let errors = samples[item.qualifiedName] else { continue }
            XCTAssertEqual(
                Set(errors.map { Self.caseLabel(of: $0, in: item) }),
                Set(item.cases.map(\.name)),
                "\(item.qualifiedName) has cases without a sentinel sample"
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
            Set(structs.map(\.qualifiedName)),
            Self.codeStructErrors.union(Self.reasonStructErrors).union(Self.unrenderedStructErrors.keys)
        )
        for item in structs {
            let storesString = Self.count(of: Self.storedStringPattern, in: item.body) > 0
            if let fields = Self.unrenderedStructErrors[item.qualifiedName] {
                XCTAssertTrue(storesString, "\(item.qualifiedName) no longer stores a string; move it to the reason structs")
                try assertFixedRendering(item, hiding: fields)
                continue
            }
            guard Self.codeStructErrors.contains(item.qualifiedName) else {
                XCTAssertFalse(storesString, "\(item.qualifiedName) must store only closed reason enums")
                continue
            }
            XCTAssertTrue(storesString, "\(item.qualifiedName) no longer stores a code")
            if Self.pathCodeStructErrors.contains(item.qualifiedName) {
                try assertStructuralPathCodes(item)
                continue
            }
            XCTAssertGreaterThan(
                Self.count(of: #"fileprivate init\((?:_ )?\w+: String\)"#, in: item.body),
                0,
                "\(item.name) must keep its code initializer file-private"
            )
            let constructions = Self.count(of: #"\b\#(item.name)\("#, in: item.fileContents)
            let literals = Self.count(
                of: #"\b\#(item.name)\(\s*(?:\w+:\s*)?"[a-z0-9_]+"\s*\)"#,
                in: item.fileContents
            )
            if let forwarder = Self.structCodeForwarders[item.qualifiedName] {
                let code =
                    forwarder.prefix.isEmpty
                    ? "code" : #""\#(NSRegularExpression.escapedPattern(for: forwarder.prefix))"\s*\+\s*code"#
                let forwarded = Self.count(of: #"\b\#(item.name)\((?:\w+:\s*)?\#(code)\)"#, in: item.fileContents)
                XCTAssertEqual(constructions, literals + forwarded, "\(item.name) may forward only a `code` parameter")
                var relayed = 0
                var literalCallers = 0
                for helper in forwarder.helpers {
                    let call = #"(?<!func )\b\#(helper)\("#
                    let calls = Self.count(of: call, in: item.fileContents)
                    let fixed = Self.count(of: call + #"[^"\n]*"[a-z0-9_]+"\)"#, in: item.fileContents)
                    let relays = Self.count(of: call + #"[^"\n]*\bcode\)"#, in: item.fileContents)
                    XCTAssertGreaterThan(fixed, 0, "\(item.name) forwarder \(helper) is unused")
                    XCTAssertEqual(calls, fixed + relays, "\(item.name) forwarder call sites must pass fixed literals")
                    relayed += relays
                    literalCallers += fixed
                }
                XCTAssertEqual(forwarded + relayed, forwarder.forwardingSites, "\(item.name) changed how often it forwards codes")
                XCTAssertEqual(
                    literalCallers,
                    forwarder.literalCallers,
                    "\(item.name) forwarder call sites changed; review them and update the count"
                )
            } else {
                XCTAssertEqual(constructions, literals, "\(item.name) codes must be fixed literals")
            }
            XCTAssertGreaterThan(literals, 0, "\(item.name) has no fixed-literal construction in its file")
        }
    }

    /// A struct error whose stored strings never render: it conforms to the
    /// description and reflection protocols with fixed literals and an empty
    /// mirror, gains no extension, and names the hidden fields only once.
    private func assertFixedRendering(_ item: DeclaredError, hiding fields: [String]) throws {
        let name = item.qualifiedName
        for conformance in ["CustomStringConvertible", "CustomDebugStringConvertible", "CustomReflectable"] {
            XCTAssertTrue(item.conformances.contains(conformance), "\(name) must conform to \(conformance)")
        }
        for property in ["description", "debugDescription"] {
            XCTAssertEqual(
                Self.count(of: #"\bvar \#(property): String \{\s*"[A-Za-z0-9_ ]*"\s*\}"#, in: item.body),
                1,
                "\(name).\(property) must be a fixed literal"
            )
        }
        XCTAssertEqual(
            Self.count(of: #"\bvar customMirror: Mirror \{\s*Mirror\(self, children: \[:\]\)\s*\}"#, in: item.body),
            1,
            "\(name).customMirror must have no children"
        )
        for field in fields {
            XCTAssertEqual(
                Self.count(of: #"\b\#(field)\b"#, in: item.body),
                1,
                "\(name) may mention \(field) only in its declaration"
            )
        }
        let extensions = try Self.sourceFiles().reduce(0) { total, file in
            total + Self.count(of: #"\bextension\s+(?:\w+\.)*\#(item.name)\b"#, in: file.contents)
        }
        XCTAssertEqual(extensions, 0, "\(name) must not gain renderings through an extension")
    }

    /// Every construction of a public `(code, fieldPath)` struct error across
    /// Sources passes a fixed code and a structural JSON path: `$`-rooted or
    /// path-variable strings built from literal segments, `enumerated()`
    /// offsets and reviewed segments whose helpers only receive literals.
    private func assertStructuralPathCodes(_ item: DeclaredError) throws {
        let reviewed = Self.reviewedPathSegments[item.qualifiedName]
        let path = Self.structuralPath(reviewed: reviewed.map { [$0.segment] } ?? [])
        let allowed = #"\b\#(item.name)\(\s*"[a-z0-9_]+"\s*(?:,\s*\#(path))?\s*\)"#
        var constructions = 0
        for (file, contents) in try Self.sourceFiles() {
            let sites = Self.count(of: #"\b\#(item.name)(?:\.init)?\("#, in: contents)
            guard sites > 0 else { continue }
            constructions += sites
            XCTAssertEqual(
                Self.count(of: allowed, in: contents),
                sites,
                "\(item.name) in \(file) needs a fixed code and a structural path"
            )
            let assignment = #"\b(?:let|var)\s+(?:path|fieldPath)\s*="#
            XCTAssertEqual(
                Self.count(of: #"(?m)\#(assignment)\s*\#(path)[ \t]*$"#, in: contents),
                Self.count(of: assignment, in: contents),
                "\(file) assigns a path variable that is not a structural path"
            )
            XCTAssertTrue(
                Self.indexInterpolationsAreOffsets(in: contents),
                "\(file) interpolates an index that is not an enumerated() offset"
            )
            guard let reviewed, Self.count(of: #"\\\(\#(reviewed.segment)\)"#, in: contents) > 0 else { continue }
            let segment = reviewed.segment
            XCTAssertEqual(
                Self.count(of: #"\b\#(segment)\s*:"#, in: contents),
                reviewed.helpers.count,
                "\(file) binds `\(segment)` outside its reviewed helpers"
            )
            XCTAssertEqual(
                Self.count(of: #"\b(?:let|var|for)\s+\(?\s*\#(segment)\b|\(\s*\#(segment)\s*,"#, in: contents),
                0,
                "\(file) binds `\(segment)` from a local value"
            )
            var literalKeys = 0
            for helper in reviewed.helpers {
                let call = #"(?<!func )\b\#(helper)\([^,()\n]+,\s*"#
                let fixed = Self.count(of: call + #""[a-z0-9_]+","#, in: contents)
                let relays = Self.count(of: call + #"\#(segment),"#, in: contents)
                XCTAssertEqual(
                    Self.count(of: #"(?<!func )\b\#(helper)\("#, in: contents),
                    fixed + relays,
                    "\(file) calls \(helper) with a `\(segment)` that is not a fixed literal"
                )
                literalKeys += fixed
            }
            XCTAssertGreaterThan(literalKeys, 0, "\(file) no longer passes fixed `\(segment)` literals")
        }
        XCTAssertGreaterThan(constructions, 0, "\(item.name) is never constructed in Sources")
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
            XCTAssertEqual(error as? PolicyError, PolicyError("profile_unknown", "$.name"))
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

    func testBundleCacheCrashNeverRendersItsCheckpoint() {
        let crash = OpenMedBundleCacheCrash(checkpoint: "checkpoint-\(Self.sentinel)")
        XCTAssertEqual(crash.checkpoint, "checkpoint-\(Self.sentinel)", "the test seam still carries the name")
        let renderings = [
            ("String(describing:)", String(describing: crash)),
            ("String(reflecting:)", String(reflecting: crash)),
            ("NSError description", (crash as NSError).description),
        ]
        for (label, rendering) in renderings {
            XCTAssertFalse(rendering.contains(Self.sentinel), "\(label) renders the checkpoint")
        }
        assertValueFree(crash)
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
                if OpenMedMLXRuntimeSupport.isSupported {
                    guard case OpenMedMapleRuntimeError.missingFiles(let files) = error else {
                        return XCTFail("Unexpected error type")
                    }
                    XCTAssertEqual(files, OpenMedMaple.requiredModelFiles)
                } else {
                    // The iOS Simulator refuses MLX before any model file is read.
                    XCTAssertEqual(error as? OpenMedMLXRuntimeError, .unsupportedPlatform)
                }
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
            samples[Self.typeName(E.self), default: []] += values.map { $0 as Error }
        }
        func addRawCases<E: Error & RawRepresentable>(_ type: E.Type) throws where E.RawValue == String {
            let name = Self.typeName(type)
            let declared = try XCTUnwrap(catalogue[name], "\(name) was not found in Sources")
            add(
                try declared.cases.map { item in
                    try XCTUnwrap(E(rawValue: item.rawValue ?? item.name), "\(name).\(item.name)")
                }
            )
        }

        try addRawCases(AgentGovernanceError.self)
        try addRawCases(AmbientAudioCaptureError.self)
        try addRawCases(AmbientFHIRDocumentError.self)
        try addRawCases(AmbientReviewConflict.self)
        try addRawCases(AmbientSessionError.self)
        try addRawCases(AudioSampleConversionError.self)
        try addRawCases(BMFFPreflightError.self)
        try addRawCases(ClinicalBriefError.self)
        try addRawCases(ClinicalBriefFHIRDocumentError.self)
        try addRawCases(DeviceCapabilityError.self)
        try addRawCases(DocumentIntakeReason.self)
        try addRawCases(DocumentRedactionExportError.self)
        try addRawCases(MultimodalContractRefusal.self)
        try addRawCases(MultimodalReviewError.self)
        try addRawCases(OpenMedDecisionEncoderError.self)
        try addRawCases(OpenMedSecretProviderError.self)
        try addRawCases(PHISentinelSweepError.self)
        try addRawCases(PhysiologicalWaveformError.self)
        try addRawCases(PreprocessingConformanceError.self)
        try addRawCases(ReviewAdmissionRefusal.self)
        try addRawCases(StreamingSpeechError.self)
        try addRawCases(TranscriptHoldError.self)
        try addRawCases(TranscriptLanguageError.self)
        try addRawCases(VoiceEmbeddingError.self)

        add([
            SpeechEndpointError.invalidLimits, .detectorNotLocal, .invalidFrame, .overlappingFrame,
            .cancelled, .detectorFailure, .detectorResetFailure,
        ])
        add([MultimodalThermalError.invalidBudget])
        add([MultimodalNoticeError.invalidNotice, .invalidResult, .reviewerConfirmationRequired])
        add(OpenMedRuntimeError.allCases)
        add([
            OpenMedDecisionError.invalidRequest(reason: "request_json_invalid"),
            .unsupportedContract(reason: "schema_major_unsupported"),
            .invalidConfiguration(reason: "access_policy_invalid"),
        ])
        add([DecisionRefusal.unsupported, .conflict])
        add(
            [
                OpenMedTokenizerError.missingLocalAssets, .pinnedRevisionRequired, .identityMismatch,
                .downloadFailed(statusCode: 404), .downloadFailed(statusCode: nil), .sizeMismatch,
                .digestMismatch, .cacheUnavailable, .invalidAssets,
            ]
                + [
                    OpenMedTokenizerPinIssue.repositoryID, .revision, .digest, .fileName, .duplicateFile,
                    .missingRequiredFile, .byteCount,
                ].map { OpenMedTokenizerError.invalidPin($0) }
        )
        add(
            OpenMedApplePlatform.allCases.map { PlatformModelError.noCompatibleModel($0) }
                + [.invalidInputShape, .sequenceTooLong(actual: 4_097, maximum: 512)]
        )
        add([
            PolicyError("profile_name_invalid", "$.name"),
            PolicyError("profile_unknown", "$.name"),
            PolicyError("profile_not_portable", "$.name"),
            PolicyError("digest_mismatch", "$.digest"),
            PolicyError("resource_missing", "$.profiles"),
            PolicyError("profile_invalid", "$.actions"),
            PolicyError("manifest_version_unsupported", "$.schema_version"),
        ])
        add([
            KitParityError("json_invalid"),
            KitParityError("field_unknown", "$"),
            KitParityError("offset_out_of_bounds", "$.spans[0].end"),
        ])
        add(PortableBundleReason.allCases.map(PortableBundleError.init))
        add(PortableBundleReason.allCases.map(PortableBundle.Refusal.init))
        add(DecisionCalibrationReason.allCases.map(DecisionCalibrationError.init))
        add(
            OpenMedBundleCacheReason.allCases.map { OpenMedBundleCacheError($0) }
                + [OpenMedBundleCacheError(.bundleRejected, bundleReason: .signatureInvalid)]
        )
        add(
            OpenMedManagedImportReason.allCases.map { OpenMedManagedImportError($0) }
                + [OpenMedManagedImportError(.verificationFailed, bundleReason: .contentIDMismatch)]
        )
        add(OpenMedManagedImportReason.allCases.map(OpenMedBundleArchive.Failure.init(reason:)))
        // The checkpoint never renders, so even a sentinel one stays out.
        add([OpenMedBundleCacheCrash(checkpoint: "checkpoint-\(Self.sentinel)")])
        add([PortableBundleCalibration.Malformed()])
        add([PortableBundleJSON.Invalid()])
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

    /// Struct errors driven from sentinel inputs; most expose only file-private
    /// initializers, so driving them is the only way to build them.
    private func drivenStructErrors() throws -> [String: [Error]] {
        let payload = Data(String(repeating: "\(Self.sentinel) ", count: 32).utf8)
        var errors: [String: [Error]] = [:]
        func capture(_ name: String, _ body: () throws -> Void) {
            do {
                try body()
                XCTFail("\(name) accepted sentinel input")
            } catch {
                errors[Self.typeName(type(of: error)), default: []].append(error)
            }
        }
        capture("EDFReader") { _ = try EDFReader.read(payload) }
        capture("OpenMedAECG") { _ = try OpenMedAECG.read(Data("<\(Self.sentinel)>\(Self.sentinel)</\(Self.sentinel)>".utf8)) }
        capture("OpenMedAECG XML") { _ = try OpenMedAECG.read(Data("<\(Self.sentinel)".utf8)) }
        capture("BigTiffPreflight") { _ = try BigTiffPreflight.read(payload, profile: .ifdChain) }
        capture("BigTiffPreflight size") { _ = try BigTiffPreflight.read(Data("\(Self.sentinel)".utf8), profile: .subIFDs) }
        capture("WFDBReader") { _ = try WFDBReader.read(header: payload, signals: []) }
        capture("WFDBReader stream") { _ = try WFDBReader.read(header: WFDBDataSource(payload), signals: []) }
        capture("SlideDeidentification") { _ = try SlideDeidentification.deidentifyTIFF(payload) }
        capture("SlideDeidentification DICOM") {
            _ = try SlideDeidentification.validateDICOMInstance(
                sopClassUID: "1.2.840.10008.5.1.4.1.1.77.1.6",
                imageType: ["ORIGINAL", "PRIMARY", Self.sentinel]
            )
        }
        capture("KitParity") { _ = try KitParity.parse(payload) }
        capture("KitParity profile") { _ = try KitParity.parse(#"{"profile":"\#(Self.sentinel)"}"#) }
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

    /// The declared case a sample stands for. Types that customize
    /// `description` map back through their raw value or `CaseIterable` order.
    private static func caseLabel(of value: Any, in item: DeclaredError) -> String {
        let label = caseLabel(of: value)
        if item.cases.contains(where: { $0.name == label }) {
            return label
        }
        if let raw = (value as? any RawRepresentable)?.rawValue as? String,
            let declared = item.cases.first(where: { ($0.rawValue ?? $0.name) == raw })
        {
            return declared.name
        }
        if let value = value as? any CaseIterable & Equatable, let index = caseIndex(of: value),
            item.cases.indices.contains(index)
        {
            return item.cases[index].name
        }
        return label
    }

    private static func caseIndex<E: CaseIterable & Equatable>(of value: E) -> Int? {
        E.allCases.firstIndex(of: value).map { E.allCases.distance(from: E.allCases.startIndex, to: $0) }
    }

    /// A type's name below its module, qualified the way the source scan is.
    private static func typeName(_ type: Any.Type) -> String {
        String(String(reflecting: type).drop { $0 != "." }.dropFirst())
    }

    // MARK: - Source scan

    private struct DeclaredCase {
        let name: String
        let rawValue: String?
    }

    private struct DeclaredError {
        let name: String
        /// `name` prefixed by its enclosing types, e.g. `PortableBundle.Refusal`.
        let qualifiedName: String
        let kind: String
        let isPrivate: Bool
        let conformances: [String]
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
            pattern: #"(?m)^[ \t]*((?:(?:public|package|internal|fileprivate|private|final)[ \t]+)*)(enum|struct|class)[ \t]+([A-Za-z_][A-Za-z0-9_]*)[ \t]*:([^{]*)\{"#
        )
        // Any type or extension scope, so nested errors get qualified names.
        let scope = try NSRegularExpression(
            pattern: #"(?m)^[ \t]*(?:(?:public|package|internal|fileprivate|private|final|indirect)[ \t]+)*(?:enum|struct|class|actor|extension)[ \t]+(?!(?:func|var|let|subscript|init)\b)([A-Za-z_][A-Za-z0-9_.]*)[^{]*\{"#
        )
        var result: [DeclaredError] = []
        for (_, contents) in try sourceFiles() {
            let text = contents as NSString
            let whole = NSRange(location: 0, length: text.length)
            let scopes = scope.matches(in: contents, range: whole).map { match in
                (
                    name: text.substring(with: match.range(at: 1)),
                    body: Self.bodyRange(in: text, openingBrace: match.range.location + match.range.length - 1)
                )
            }
            for match in declaration.matches(in: contents, range: whole) {
                let conformances = text.substring(with: match.range(at: 4))
                    .split { !$0.isLetter && !$0.isNumber && $0 != "_" }
                guard conformances.contains("Error") || conformances.contains("LocalizedError") else {
                    continue
                }
                let name = text.substring(with: match.range(at: 3))
                let enclosing =
                    scopes
                    .filter { NSLocationInRange(match.range.location, $0.body) }
                    .sorted { $0.body.location < $1.body.location }
                    .map { $0.name }
                let openingBrace = match.range.location + match.range.length - 1
                let body = text.substring(with: Self.bodyRange(in: text, openingBrace: openingBrace))
                result.append(
                    DeclaredError(
                        name: name,
                        qualifiedName: (enclosing + [name]).joined(separator: "."),
                        kind: text.substring(with: match.range(at: 2)),
                        isPrivate: text.substring(with: match.range(at: 1)).contains("private"),
                        conformances: conformances.map(String.init),
                        body: body,
                        fileContents: contents,
                        cases: Self.cases(in: body)
                    )
                )
            }
        }
        return result
    }

    /// Every Swift file under `Sources`, by file name.
    private static func sourceFiles() throws -> [(name: String, contents: String)] {
        let files = try XCTUnwrap(
            FileManager.default.enumerator(at: sourcesRoot, includingPropertiesForKeys: nil)
        )
        var result: [(name: String, contents: String)] = []
        for case let url as URL in files where url.pathExtension == "swift" {
            result.append((url.lastPathComponent, try String(contentsOf: url, encoding: .utf8)))
        }
        return result
    }

    /// A structural JSON path argument: `path` or `fieldPath`, or a string
    /// rooted at `$` or a path variable that adds only literal segments,
    /// `[\(index)]` offsets and the given reviewed `.\(segment)` keys.
    private static func structuralPath(reviewed: [String]) -> String {
        let segments = [#"[A-Za-z0-9_.\[\]]"#, #"\[\\\(index\)\]"#] + reviewed.map { #"\.\\\(\#($0)\)"# }
        return #"(?:path|fieldPath|"(?:\$|\\\((?:path|fieldPath)\))(?:\#(segments.joined(separator: "|")))*")"#
    }

    /// Whether every `[\(index)]` follows an `enumerated()` binding of `index`
    /// as its nearest binding, so the interpolated value is an integer offset.
    private static func indexInterpolationsAreOffsets(in text: String) -> Bool {
        guard
            let binding = try? NSRegularExpression(
                pattern:
                    #"(for\s*\(\s*index\s*,\s*\w+\s*\)\s+in\s[^\n{]*\.enumerated\(\))|\b(?:let|var)\s+index\b|\bfor\s+index\b|\bindex\s*:|\(\s*index\s*,"#
            ),
            let use = try? NSRegularExpression(pattern: #"\[\\\(index\)\]"#)
        else { return false }
        let range = NSRange(text.startIndex..., in: text)
        let bindings = binding.matches(in: text, range: range)
        return use.matches(in: text, range: range).allSatisfy { use in
            guard let nearest = bindings.last(where: { $0.range.location < use.range.location }) else {
                return false
            }
            return nearest.range(at: 1).location != NSNotFound
        }
    }

    /// Range between a declaration's braces, skipping braces in string literals.
    private static func bodyRange(in text: NSString, openingBrace: Int) -> NSRange {
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
                    return NSRange(location: openingBrace + 1, length: index - openingBrace - 1)
                }
            }
            index += 1
        }
        return NSRange(location: openingBrace + 1, length: 0)
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
