import Foundation
import XCTest

@testable import OpenMedPrivacySweep

#if canImport(OSLog)
    import OSLog
#endif

final class PHISentinelSweepTests: XCTestCase {
    private var scratch: URL!

    override func setUpWithError() throws {
        scratch = FileManager.default.temporaryDirectory
            .appending(path: "phi-sweep-\(UUID().uuidString)", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: scratch)
    }

    // MARK: Positive controls

    func testPlantedSentinelIsFoundInTemporaryCacheAndLogWithInjectedRootsAndReaders() throws {
        let sentinel = try PHISentinel(identifier: "person.name", value: "Vexorin Qualtrane")
        let tmp = try directory("tmp")
        let caches = try directory("caches")
        try write(Data("draft: \(sentinel.plantedValue)\n".utf8), to: tmp.appending(path: "draft.txt"))
        try write(
            Data([0x62, 0x70, 0x6C, 0x69, 0x73, 0x74]) + utf16BE(sentinel.plantedValue),
            to: caches.appending(path: "state.plist")
        )
        let reader = CapturedLogReader(lines: ["started", "lookup \(sentinel.plantedValue) ok"])

        let report = try PHISentinelSweep(
            sentinels: [sentinel],
            roots: [
                SweepRoot(url: tmp, storageClass: .temporary),
                SweepRoot(url: caches, storageClass: .caches),
            ],
            logReaders: [reader]
        ).run()

        XCTAssertFalse(report.clean)
        XCTAssertTrue(report.exhaustive, report.description)
        XCTAssertEqual(
            report.findings.map { [$0.location.rawValue, $0.surface, $0.fileKind?.rawValue ?? "-", $0.encoding.rawValue] },
            [
                ["file_content", "caches", "plist", "utf16be"],
                ["file_content", "tmp", "text", "utf8"],
                ["log_entry", "captured", "-", "utf8"],
            ]
        )
        XCTAssertTrue(report.findings.allSatisfy { $0.sentinelIdentifier == "person.name" && $0.itemCount == 1 })
        XCTAssertEqual(report.filesScanned, 2)
        XCTAssertEqual(report.logEntriesScanned, 2)
        XCTAssertEqual(report.logs.map(\.status), [.read])
    }

    func testEveryEncodingIsDetectedAtEveryByteAlignment() throws {
        let value = "Zoë Quinterra-Ångström 41"
        let sentinel = try PHISentinel(identifier: "person.full", value: value)
        let raw = Data(value.utf8)
        let plants: [(SentinelEncoding, Data)] = [
            (.utf8, raw),
            (.utf16le, utf16LE(value)),
            (.utf16be, utf16BE(value)),
            (.percent, Data(value.addingPercentEncoding(withAllowedCharacters: unreserved)!.utf8)),
            (.percent, Data(lowercaseHex(value.addingPercentEncoding(withAllowedCharacters: unreserved)!).utf8)),
            (.percent, Data(formEncoded(value).utf8)),
        ]
        for (encoding, payload) in plants {
            let found = try findings(for: sentinel, fileContents: Data("prefix:".utf8) + payload + Data(";".utf8))
            XCTAssertEqual(found.map(\.encoding), [encoding], "\(encoding)")
        }
        for offset in 0..<3 {
            let prefix = Data(repeating: 0x41, count: offset)
            let standard = (prefix + raw + Data("tail".utf8)).base64EncodedString()
            XCTAssertEqual(
                try findings(for: sentinel, fileContents: Data(standard.utf8)).map(\.encoding),
                [.base64],
                "offset \(offset)"
            )
            let urlSafe = standard.replacingOccurrences(of: "+", with: "-")
                .replacingOccurrences(of: "/", with: "_")
            let urlFindings = try findings(for: sentinel, fileContents: Data(urlSafe.utf8)).map(\.encoding)
            XCTAssertFalse(urlFindings.isEmpty, "offset \(offset)")
            XCTAssertTrue(Set(urlFindings).isSubset(of: [.base64, .base64url]), "offset \(offset)")
        }
    }

    func testSentinelInFileOrDirectoryNameIsReportedWithoutTheName() throws {
        let sentinel = try PHISentinel(identifier: "person.name", value: "Vexorin Qualtrane")
        let root = try directory("documents")
        let named = try directory("documents/\(sentinel.plantedValue)")
        try write(Data("ok".utf8), to: named.appending(path: "notes.json"))
        let encoded = sentinel.plantedValue.addingPercentEncoding(withAllowedCharacters: unreserved)!
        try write(Data("ok".utf8), to: root.appending(path: "export-\(encoded).json"))

        let report = try PHISentinelSweep(
            sentinels: [sentinel],
            roots: [SweepRoot(url: root, storageClass: .documents)]
        ).run()

        XCTAssertEqual(
            report.findings.map { [$0.location.rawValue, $0.fileKind?.rawValue ?? "-", $0.encoding.rawValue] },
            [["file_name", "json", "percent"], ["file_name", "other", "utf8"]]
        )
        XCTAssertFalse(try reportText(report).contains("export-"))
    }

    func testMatchSpanningReadChunksIsFound() throws {
        let sentinel = try PHISentinel(identifier: "record.id", value: "OMS-CHUNK-BOUNDARY-77")
        var contents = Data(repeating: 0x2E, count: PHISentinelSweep.chunkSize - 7)
        contents += Data(sentinel.plantedValue.utf8)
        contents += Data(repeating: 0x2E, count: 4_096)

        XCTAssertEqual(try findings(for: sentinel, fileContents: contents).map(\.encoding), [.utf8])
    }

    // MARK: Negative controls

    func testNegativeControlsStayCleanAndExhaustive() throws {
        let sentinel = try PHISentinel(identifier: "person.name", value: "Vexorin Qualtrane")
        let other = try PHISentinel(identifier: "person.other", value: "Morvessa Tillcrane")
        let root = try directory("caches")
        try write(Data("Vexorin Qualtran".utf8), to: root.appending(path: "prefix.txt"))
        try write(Data("vexorin qualtrane".utf8), to: root.appending(path: "case.txt"))
        try write(Data("Vexorin\nQualtrane".utf8), to: root.appending(path: "split.txt"))
        try write(Data(other.plantedValue.utf8), to: root.appending(path: "other.txt"))
        try write(Data(), to: root.appending(path: "empty.bin"))
        let missing = scratch.appending(path: "missing", directoryHint: .isDirectory)

        let report = try PHISentinelSweep(
            sentinels: [sentinel],
            roots: [
                SweepRoot(url: root, storageClass: .caches),
                SweepRoot(url: missing, storageClass: .documents),
            ],
            logReaders: [CapturedLogReader(lines: ["redacted [NAME] seen", other.plantedValue])]
        ).run()

        XCTAssertTrue(report.clean, report.description)
        XCTAssertTrue(report.exhaustive, report.description)
        XCTAssertEqual(report.filesScanned, 5)
        XCTAssertEqual(report.roots.map(\.present), [true, false])
        XCTAssertEqual(report.roots.map(\.storageClass), [.caches, .documents])
    }

    // MARK: Bounds

    func testPerFileAndTotalByteBoundsAreEnforcedAndReported() throws {
        let sentinel = try PHISentinel(identifier: "record.id", value: "OMS-LATE-VALUE-0042")
        let root = try directory("caches")
        try write(Data(repeating: 0x2E, count: 64) + Data(sentinel.plantedValue.utf8), to: root.appending(path: "a.log"))
        try write(Data(sentinel.plantedValue.utf8), to: root.appending(path: "b.log"))

        var bounds = SweepBounds.standard
        bounds.maxBytesPerFile = 32
        let perFile = try PHISentinelSweep(sentinels: [sentinel], roots: [SweepRoot(url: root, storageClass: .caches)], bounds: bounds).run()
        XCTAssertEqual(perFile.boundsReached, [.maxBytesPerFile])
        XCTAssertEqual(perFile.truncatedFiles, 1)
        XCTAssertEqual(perFile.bytesScanned, 32 + 19)
        XCTAssertFalse(perFile.exhaustive)
        XCTAssertEqual(perFile.findings.map(\.itemCount), [1])

        bounds = .standard
        bounds.maxTotalBytes = 40
        let total = try PHISentinelSweep(sentinels: [sentinel], roots: [SweepRoot(url: root, storageClass: .caches)], bounds: bounds).run()
        XCTAssertEqual(total.boundsReached, [.maxTotalBytes])
        XCTAssertEqual(total.bytesScanned, 40)
        XCTAssertEqual(total.filesScanned, 1)
        XCTAssertTrue(total.clean)
        XCTAssertFalse(total.exhaustive)
    }

    func testFileCountAndDepthBoundsAreEnforcedAndReported() throws {
        let sentinel = try PHISentinel(identifier: "record.id", value: "OMS-DEEP-VALUE-0042")
        let root = try directory("documents")
        for name in ["a.txt", "b.txt", "c.txt"] {
            try write(Data("filler".utf8), to: root.appending(path: name))
        }
        let deep = try directory("documents/one/two")
        try write(Data(sentinel.plantedValue.utf8), to: deep.appending(path: "deep.txt"))

        var bounds = SweepBounds.standard
        bounds.maxFiles = 2
        let files = try PHISentinelSweep(sentinels: [sentinel], roots: [SweepRoot(url: root, storageClass: .documents)], bounds: bounds).run()
        XCTAssertEqual(files.boundsReached, [.maxFiles])
        XCTAssertEqual(files.filesScanned, 2)
        XCTAssertFalse(files.exhaustive)

        bounds = .standard
        bounds.maxDepth = 1
        let shallow = try PHISentinelSweep(sentinels: [sentinel], roots: [SweepRoot(url: root, storageClass: .documents)], bounds: bounds).run()
        XCTAssertEqual(shallow.boundsReached, [.maxDepth])
        XCTAssertTrue(shallow.clean)
        XCTAssertEqual(shallow.filesScanned, 3)

        bounds.maxDepth = 2
        let deepEnough = try PHISentinelSweep(sentinels: [sentinel], roots: [SweepRoot(url: root, storageClass: .documents)], bounds: bounds).run()
        XCTAssertEqual(deepEnough.boundsReached, [])
        XCTAssertEqual(deepEnough.findings.map(\.location), [.fileContent])
    }

    func testTimeBoundUsesInjectedClockAndSkipsRemainingReaders() throws {
        let sentinel = try PHISentinel(identifier: "record.id", value: "OMS-SLOW-VALUE-0042")
        let root = try directory("tmp")
        for index in 0..<10 {
            try write(Data("filler".utf8), to: root.appending(path: "f\(index).txt"))
        }
        var ticks = 0.0
        var bounds = SweepBounds.standard
        bounds.maxDuration = 5
        let report = try PHISentinelSweep(
            sentinels: [sentinel],
            roots: [SweepRoot(url: root, storageClass: .temporary)],
            logReaders: [CapturedLogReader(lines: [sentinel.plantedValue])],
            bounds: bounds,
            clock: {
                ticks += 1
                return ticks
            }
        ).run()

        XCTAssertEqual(report.boundsReached, [.maxDuration])
        XCTAssertLessThan(report.filesScanned, 10)
        XCTAssertEqual(report.logs.map(\.status), [.skipped])
        XCTAssertTrue(report.clean)
        XCTAssertFalse(report.exhaustive)
    }

    func testLogEntryAndByteBoundsAreEnforcedAndReported() throws {
        let sentinel = try PHISentinel(identifier: "record.id", value: "OMS-LOG-VALUE-0042")
        var bounds = SweepBounds.standard
        bounds.maxLogEntries = 2
        let entries = try PHISentinelSweep(
            sentinels: [sentinel],
            roots: [],
            logReaders: [CapturedLogReader(lines: ["one", "two", sentinel.plantedValue])],
            bounds: bounds
        ).run()
        XCTAssertEqual(entries.boundsReached, [.maxLogEntries])
        XCTAssertEqual(entries.logEntriesScanned, 2)
        XCTAssertTrue(entries.clean)

        bounds = .standard
        bounds.maxLogBytes = 10
        let bytes = try PHISentinelSweep(
            sentinels: [sentinel],
            roots: [],
            logReaders: [CapturedLogReader(lines: ["0123456", sentinel.plantedValue])],
            bounds: bounds
        ).run()
        XCTAssertEqual(bytes.boundsReached, [.maxLogBytes])
        XCTAssertEqual(bytes.logBytesScanned, 10)
        XCTAssertTrue(bytes.clean)
        XCTAssertFalse(bytes.exhaustive)
    }

    func testInvalidBoundsAndSentinelsAreRefusedWithValueFreeCodes() throws {
        let sentinel = try PHISentinel(identifier: "record.id", value: "OMS-VALID-VALUE")
        let mutations: [(inout SweepBounds) -> Void] = [
            { $0.maxFiles = 0 },
            { $0.maxDepth = -1 },
            { $0.maxBytesPerFile = SweepBounds.ceiling.maxBytesPerFile + 1 },
            { $0.maxDuration = 0 },
            { $0.maxLogEntries = 0 },
        ]
        for mutate in mutations {
            var bounds = SweepBounds.standard
            mutate(&bounds)
            XCTAssertThrowsError(try PHISentinelSweep(sentinels: [sentinel], roots: [], bounds: bounds)) {
                XCTAssertEqual($0 as? PHISentinelSweepError, .boundsInvalid)
            }
        }
        let refusals: [(() throws -> Any, PHISentinelSweepError)] = [
            ({ try PHISentinel(identifier: "Person", value: "OMS-VALID-VALUE") }, .sentinelIdentifierInvalid),
            ({ try PHISentinel(identifier: ".dot", value: "OMS-VALID-VALUE") }, .sentinelIdentifierInvalid),
            ({ try PHISentinel(identifier: "short", value: "Ada Lee") }, .sentinelLengthOutOfRange),
            ({ try PHISentinelSweep(sentinels: [], roots: []) }, .sentinelsEmpty),
            ({ try PHISentinelSweep(sentinels: [sentinel, sentinel], roots: []) }, .sentinelIdentifierDuplicate),
            (
                {
                    try PHISentinelSweep(
                        sentinels: (0...PHISentinel.maximumCount).map { try PHISentinel.synthetic(identifier: "s\($0)") },
                        roots: []
                    )
                }, .sentinelsTooMany
            ),
        ]
        for (action, code) in refusals {
            XCTAssertThrowsError(try action()) { XCTAssertEqual($0 as? PHISentinelSweepError, code) }
        }
        XCTAssertEqual(PHISentinelSweepError.sentinelLengthOutOfRange.rawValue, "sentinel_length_out_of_range")
    }

    // MARK: Allow-list, links and container layout

    func testAllowListSkipsModelArtifactContentsButStillSweepsNames() throws {
        let sentinel = try PHISentinel(identifier: "person.name", value: "Vexorin Qualtrane")
        let caches = try directory("caches")
        let documents = try directory("documents")
        try write(Data(sentinel.plantedValue.utf8), to: caches.appending(path: "weights.onnx"))
        let bundle = try directory("caches/Model.mlmodelc")
        try write(Data(sentinel.plantedValue.utf8), to: bundle.appending(path: "coremldata.bin"))
        try write(Data("weights".utf8), to: bundle.appending(path: "\(sentinel.plantedValue).bin"))
        let cached = try directory("caches/OpenMed/MLXModels/repo")
        try write(Data(sentinel.plantedValue.utf8), to: cached.appending(path: "tokenizer.json"))
        let lookalike = try directory("documents/OpenMed/MLXModels/repo")
        try write(Data(sentinel.plantedValue.utf8), to: lookalike.appending(path: "tokenizer.json"))

        let report = try PHISentinelSweep(
            sentinels: [sentinel],
            roots: [
                SweepRoot(url: caches, storageClass: .caches),
                SweepRoot(url: documents, storageClass: .documents),
            ],
            allowList: .modelArtifacts
        ).run()

        XCTAssertEqual(report.allowListedFiles, 4)
        XCTAssertEqual(
            report.findings.map { [$0.location.rawValue, $0.surface, $0.fileKind?.rawValue ?? "-"] },
            [["file_content", "documents", "json"], ["file_name", "caches", "other"]]
        )
        let unlisted = try PHISentinelSweep(sentinels: [sentinel], roots: [SweepRoot(url: caches, storageClass: .caches)]).run()
        XCTAssertEqual(unlisted.allowListedFiles, 0)
        XCTAssertEqual(unlisted.findings.filter { $0.location == .fileContent }.reduce(0) { $0 + $1.itemCount }, 3)
    }

    func testSymbolicLinksAreCountedButNotFollowed() throws {
        let sentinel = try PHISentinel(identifier: "person.name", value: "Vexorin Qualtrane")
        let root = try directory("caches")
        let outside = try directory("outside")
        try write(Data(sentinel.plantedValue.utf8), to: outside.appending(path: "secret.txt"))
        try FileManager.default.createSymbolicLink(at: root.appending(path: "link"), withDestinationURL: outside)
        try FileManager.default.createSymbolicLink(
            at: root.appending(path: "file-link.txt"),
            withDestinationURL: outside.appending(path: "secret.txt")
        )

        let report = try PHISentinelSweep(sentinels: [sentinel], roots: [SweepRoot(url: root, storageClass: .caches)]).run()

        XCTAssertTrue(report.clean, report.description)
        XCTAssertEqual(report.skippedSymbolicLinks, 2)
        XCTAssertEqual(report.filesScanned, 0)
    }

    func testContainerRootsCoverEachStorageClassExactlyOnce() throws {
        let sentinel = try PHISentinel(identifier: "person.name", value: "Vexorin Qualtrane")
        let home = try directory("container")
        let planted = [
            "Documents/a.txt", "Library/Application Support/b.json", "Library/Caches/c.db",
            "Library/Preferences/org.openmed.plist", "Library/Cookies/d.binarycookies", "tmp/e.txt",
        ]
        for path in planted {
            let url = home.appending(path: path)
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try write(Data(sentinel.plantedValue.utf8), to: url)
        }
        let roots = AppContainerRoots.roots(
            containerHome: home,
            temporaryDirectory: home.appending(path: "tmp", directoryHint: .isDirectory)
        )

        let report = try PHISentinelSweep(sentinels: [sentinel], roots: roots).run()

        XCTAssertEqual(roots.map(\.storageClass), [.documents, .applicationSupport, .caches, .preferences, .library, .temporary])
        XCTAssertEqual(report.filesScanned, planted.count)
        XCTAssertEqual(
            report.findings.map { "\($0.surface)/\($0.fileKind!.rawValue)" },
            ["application_support/json", "caches/sqlite", "documents/text", "library/other", "preferences/plist", "tmp/text"]
        )
        XCTAssertTrue(report.findings.allSatisfy { $0.itemCount == 1 })
    }

    func testHostScopeDetectionAndGateSemantics() throws {
        let uuid = "0F7E2B9A-6C1D-4E8F-9A3B-2C5D7E9F1A3B"
        XCTAssertTrue(AppContainerRoots.isPerAppContainer(homePath: "/private/var/mobile/Containers/Data/Application/\(uuid)"))
        XCTAssertTrue(AppContainerRoots.isPerAppContainer(homePath: "/sim/data/Containers/Data/PluginKitPlugin/\(uuid)/"))
        XCTAssertFalse(AppContainerRoots.isPerAppContainer(homePath: "/sim/Devices/\(uuid)/data"))
        XCTAssertFalse(AppContainerRoots.isPerAppContainer(homePath: "/sim/data/Containers/Data/Application/not-a-uuid"))
        XCTAssertFalse(AppContainerRoots.isPerAppContainer(homePath: "/"))
        #if os(macOS)
            if ProcessInfo.processInfo.environment["APP_SANDBOX_CONTAINER_ID"] == nil {
                XCTAssertEqual(HostStorageScope.current, .unsandboxedHost)
            }
        #endif

        let sentinel = try PHISentinel(identifier: "person.name", value: "Vexorin Qualtrane")
        let root = try directory("caches")
        for index in 0..<3 {
            try write(Data("filler".utf8), to: root.appending(path: "f\(index).txt"))
        }
        var bounds = SweepBounds.standard
        bounds.maxFiles = 1
        let bounded = try PHISentinelSweep(sentinels: [sentinel], roots: [SweepRoot(url: root, storageClass: .caches)], bounds: bounds).run()
        XCTAssertTrue(bounded.clean)
        XCTAssertFalse(bounded.passesGate(for: .appContainer))
        XCTAssertFalse(bounded.passesGate(for: .unsandboxedHost))
        XCTAssertTrue(bounded.passesGate(for: .simulatorDeviceHome))

        try write(Data(sentinel.plantedValue.utf8), to: root.appending(path: "leak.txt"))
        let leaked = try PHISentinelSweep(sentinels: [sentinel], roots: [SweepRoot(url: root, storageClass: .caches)]).run()
        let unreadableLog = try PHISentinelSweep(
            sentinels: [sentinel],
            roots: [],
            logReaders: [ThrowingLogReader(message: "denied")]
        ).run()
        for scope in [HostStorageScope.appContainer, .simulatorDeviceHome, .unsandboxedHost] {
            XCTAssertFalse(leaked.passesGate(for: scope), scope.rawValue)
            XCTAssertFalse(unreadableLog.passesGate(for: scope), scope.rawValue)
        }
    }

    // MARK: Value-free reporting

    func testReportsAndDescriptionsContainNoSentinelValuesContentsNamesOrPaths() throws {
        let sentinels = [
            try PHISentinel(identifier: "person.name", value: "Vexorin Qualtrane"),
            try PHISentinel(identifier: "record.id", value: "MRN 4471-OMS-90"),
        ]
        let root = try directory("caches")
        for (index, sentinel) in sentinels.enumerated() {
            let value = sentinel.plantedValue
            try write(Data("context-\(index) \(value)".utf8), to: root.appending(path: "plain-\(index).txt"))
            try write(utf16LE(value), to: root.appending(path: "wide-\(index).bin"))
            try write(Data(Data(value.utf8).base64EncodedString().utf8), to: root.appending(path: "b64-\(index).txt"))
            try write(Data("x".utf8), to: root.appending(path: "\(value)-\(index).txt"))
        }
        let failing = ThrowingLogReader(message: sentinels[0].plantedValue)

        let report = try PHISentinelSweep(
            sentinels: sentinels,
            roots: [SweepRoot(url: root, storageClass: .caches)],
            logReaders: [CapturedLogReader(lines: sentinels.map(\.plantedValue)), failing]
        ).run()
        let text = try reportText(report)

        XCTAssertFalse(report.clean)
        XCTAssertFalse(report.exhaustive)
        XCTAssertEqual(report.logs.map(\.status), [.read, .unreadable])
        XCTAssertEqual(Set(report.findings.map(\.sentinelIdentifier)), ["person.name", "record.id"])
        for sentinel in sentinels {
            for needle in SentinelNeedles.make(for: sentinel.plantedValue, sentinelIndex: 0) {
                XCTAssertNil(Data(text.utf8).range(of: needle.bytes), "\(needle.encoding)")
            }
            let descriptions = [
                sentinel.description, String(reflecting: sentinel), String(describing: Mirror(reflecting: sentinel).children.map(\.value)),
            ]
            XCTAssertFalse(descriptions.joined().contains(sentinel.plantedValue))
        }
        for forbidden in [scratch.path, root.path, "context-", "plain-", "wide-", "b64-", "-0.txt", NSHomeDirectory()] {
            XCTAssertFalse(text.contains(forbidden), forbidden)
        }
        let decoded = try JSONSerialization.jsonObject(with: report.jsonData()) as? [String: Any]
        XCTAssertEqual(decoded?["schema"] as? String, "openmed.phi_sentinel_sweep.v1")
        XCTAssertNotNil(decoded?["bounds_reached"])
    }

    func testSyntheticSentinelsAreUniqueAndValid() throws {
        let values = try (0..<32).map { try PHISentinel.synthetic(identifier: "s\($0)").plantedValue }
        XCTAssertEqual(Set(values).count, values.count)
        XCTAssertTrue(values.allSatisfy { $0.hasPrefix("OMS") && $0.utf8.count == 16 })
    }

    // MARK: Unified log

    #if canImport(OSLog)
        func testOSLogStoreReaderFindsPublicValuesLoggedByThisProcess() throws {
            let start = Date().addingTimeInterval(-1)
            let visible = try PHISentinel.synthetic(identifier: "log.public")
            let hidden = try PHISentinel.synthetic(identifier: "log.private")
            let logger = Logger(subsystem: "org.openmed.tests.phi-sweep", category: "positive-control")
            logger.log("positive control \(visible.plantedValue, privacy: .public)")
            logger.log("private control \(hidden.plantedValue, privacy: .private)")

            var report = try osLogReport(sentinels: [visible, hidden], since: start)
            for _ in 0..<20 where report.findings.isEmpty {
                Thread.sleep(forTimeInterval: 0.1)
                report = try osLogReport(sentinels: [visible, hidden], since: start)
            }

            // Private arguments are redacted by the OS unless a logging profile
            // enables private data, so only the public value is asserted.
            XCTAssertEqual(report.logs.map(\.status), [.read], report.description)
            let publicFindings = report.findings.filter { $0.sentinelIdentifier == "log.public" }
            XCTAssertEqual(publicFindings.map(\.surface), ["os_log"], report.description)
            XCTAssertEqual(publicFindings.map(\.location), [.logEntry])
        }

        private func osLogReport(sentinels: [PHISentinel], since: Date) throws -> PHISentinelSweepReport {
            try PHISentinelSweep(
                sentinels: sentinels,
                roots: [],
                logReaders: [OSLogStoreProcessLogReader(since: since)]
            ).run()
        }
    #endif

    // MARK: Helpers

    private struct ThrowingLogReader: ProcessLogReader {
        struct Failure: Error, CustomStringConvertible {
            let description: String
        }

        let message: String
        var source: ProcessLogSource { .osLog }

        func readEntries(_ visit: (String) -> Bool) throws {
            throw Failure(description: message)
        }
    }

    private let unreserved = CharacterSet(charactersIn: "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~")

    private func formEncoded(_ value: String) -> String {
        value.split(separator: " ", omittingEmptySubsequences: false)
            .map { String($0).addingPercentEncoding(withAllowedCharacters: unreserved)! }
            .joined(separator: "+")
    }

    private func lowercaseHex(_ encoded: String) -> String {
        var output = ""
        var pending = 0
        for character in encoded {
            if character == "%" {
                pending = 2
                output.append(character)
            } else if pending > 0 {
                pending -= 1
                output.append(Character(character.lowercased()))
            } else {
                output.append(character)
            }
        }
        return output
    }

    private func findings(for sentinel: PHISentinel, fileContents: Data) throws -> [SweepFinding] {
        let root = scratch.appending(path: UUID().uuidString, directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try write(fileContents, to: root.appending(path: "payload.bin"))
        return try PHISentinelSweep(sentinels: [sentinel], roots: [SweepRoot(url: root, storageClass: .caches)])
            .run().findings
    }

    private func reportText(_ report: PHISentinelSweepReport) throws -> String {
        String(decoding: try report.jsonData(), as: UTF8.self) + report.description
    }

    private func directory(_ relativePath: String) throws -> URL {
        let url = scratch.appending(path: relativePath, directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func write(_ data: Data, to url: URL) throws {
        try data.write(to: url)
    }

    private func utf16LE(_ value: String) -> Data {
        Data(value.utf16.flatMap { [UInt8(truncatingIfNeeded: $0), UInt8($0 >> 8)] })
    }

    private func utf16BE(_ value: String) -> Data {
        Data(value.utf16.flatMap { [UInt8($0 >> 8), UInt8(truncatingIfNeeded: $0)] })
    }
}
