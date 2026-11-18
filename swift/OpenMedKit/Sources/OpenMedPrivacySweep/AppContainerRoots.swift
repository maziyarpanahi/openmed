import Foundation

#if canImport(OSLog)
    import OSLog
#endif

/// How much of the current process's storage a sweep can meaningfully cover.
public enum HostStorageScope: String, Codable, Sendable {
    /// A per-app data container (an app or app-hosted test on a device or
    /// Simulator, or a sandboxed macOS app): sweep it and require an exhaustive report.
    case appContainer = "app_container"
    /// A Simulator process without an app container, such as SwiftPM package tests
    /// in the `xctest` agent, whose home is the simulated device's shared data
    /// directory: sweep it within bounds, but other processes' files make an
    /// exhaustive report unlikely.
    case simulatorDeviceHome = "simulator_device_home"
    /// An unsandboxed macOS process, such as `swift test`, whose home and temporary
    /// directories belong to the user: do not sweep them; use injected roots.
    case unsandboxedHost = "unsandboxed_host"

    /// The scope of this process.
    public static var current: HostStorageScope {
        #if os(macOS)
            let sandboxed = ProcessInfo.processInfo.environment["APP_SANDBOX_CONTAINER_ID"] != nil
            return sandboxed ? .appContainer : .unsandboxedHost
        #elseif targetEnvironment(simulator)
            let home = NSHomeDirectory()
            return AppContainerRoots.isPerAppContainer(homePath: home) ? .appContainer : .simulatorDeviceHome
        #else
            return .appContainer
        #endif
    }
}

/// App-writable roots of the current process's container.
public enum AppContainerRoots {
    /// Library children swept under their own storage class.
    static let classifiedLibraryChildren: Set<String> = [
        "Application Support", "Caches", "Preferences",
    ]

    /// Whether a home path is a per-app data container
    /// (`…/Containers/Data/Application/<UUID>` or `…/Containers/Data/PluginKitPlugin/<UUID>`).
    static func isPerAppContainer(homePath: String) -> Bool {
        let components = Array(URL(fileURLWithPath: homePath).standardizedFileURL.pathComponents.suffix(4))
        guard components.count == 4 else { return false }
        return components[0] == "Containers" && components[1] == "Data"
            && ["Application", "PluginKitPlugin"].contains(components[2])
            && UUID(uuidString: components[3]) != nil
    }

    /// Roots for this process: Documents, Application Support, Caches, Preferences,
    /// the rest of Library, tmp and any requested App Group containers.
    public static func current(appGroupIdentifiers: [String] = []) -> [SweepRoot] {
        var roots = roots(
            containerHome: URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true),
            temporaryDirectory: FileManager.default.temporaryDirectory
        )
        for identifier in appGroupIdentifiers {
            if let url = FileManager.default.containerURL(
                forSecurityApplicationGroupIdentifier: identifier
            ) {
                roots.append(SweepRoot(url: url, storageClass: .appGroup))
            }
        }
        return roots
    }

    /// Non-overlapping roots under an injected container home and temporary directory.
    public static func roots(containerHome: URL, temporaryDirectory: URL) -> [SweepRoot] {
        let library = containerHome.appending(path: "Library", directoryHint: .isDirectory)
        return [
            SweepRoot(
                url: containerHome.appending(path: "Documents", directoryHint: .isDirectory),
                storageClass: .documents
            ),
            SweepRoot(
                url: library.appending(path: "Application Support", directoryHint: .isDirectory),
                storageClass: .applicationSupport
            ),
            SweepRoot(
                url: library.appending(path: "Caches", directoryHint: .isDirectory),
                storageClass: .caches
            ),
            SweepRoot(
                url: library.appending(path: "Preferences", directoryHint: .isDirectory),
                storageClass: .preferences
            ),
            SweepRoot(
                url: library,
                storageClass: .library,
                excludedChildNames: classifiedLibraryChildren
            ),
            SweepRoot(url: temporaryDirectory, storageClass: .temporary),
        ]
    }
}

extension PHISentinelSweepReport {
    /// Whether this report passes the sweep gate for a host scope: always clean with
    /// every log read, and exhaustive unless the scope is a shared Simulator device home.
    public func passesGate(for scope: HostStorageScope) -> Bool {
        let logsRead = logs.allSatisfy { $0.status == .read }
        switch scope {
        case .appContainer, .unsandboxedHost:
            return clean && exhaustive && logsRead
        case .simulatorDeviceHome:
            return clean && unreadableItems == 0 && logsRead
        }
    }
}

#if canImport(OSLog)
    @available(macOS 12.0, iOS 15.0, tvOS 15.0, watchOS 8.0, *)
    extension PHISentinelSweep {
        /// A sweep of this process's storage, as far as `scope` allows, and of its
        /// unified log since `start`.
        ///
        /// An ``HostStorageScope/unsandboxedHost`` scope sweeps the log only.
        public static func currentProcess(
            sentinels: [PHISentinel],
            logSince start: Date,
            scope: HostStorageScope = .current,
            appGroupIdentifiers: [String] = [],
            bounds: SweepBounds = .standard,
            allowList: ArtifactAllowList = .modelArtifacts
        ) throws -> PHISentinelSweep {
            let roots =
                scope == .unsandboxedHost
                ? [] : AppContainerRoots.current(appGroupIdentifiers: appGroupIdentifiers)
            return try PHISentinelSweep(
                sentinels: sentinels,
                roots: roots,
                logReaders: [OSLogStoreProcessLogReader(since: start)],
                bounds: bounds,
                allowList: allowList
            )
        }
    }

    /// Reads the current process's unified-log entries through
    /// `OSLogStore(scope: .currentProcessIdentifier)`.
    ///
    /// Arguments logged as private are returned redacted by the OS, so this reader
    /// detects values that reached the log as public or literal text.
    @available(macOS 12.0, iOS 15.0, tvOS 15.0, watchOS 8.0, *)
    public struct OSLogStoreProcessLogReader: ProcessLogReader {
        public let source = ProcessLogSource.osLog
        /// Earliest entry date; `nil` reads every retained entry for this process.
        public let since: Date?

        public init(since: Date?) {
            self.since = since
        }

        public func readEntries(_ visit: (String) -> Bool) throws {
            let store = try OSLogStore(scope: .currentProcessIdentifier)
            let entries =
                try since.map { try store.getEntries(at: store.position(date: $0)) }
                ?? store.getEntries()
            for entry in entries {
                let line: String
                if let payload = entry as? OSLogEntryLog {
                    line = "\(payload.subsystem) \(payload.category) \(payload.composedMessage)"
                } else {
                    line = entry.composedMessage
                }
                guard visit(line) else { return }
            }
        }
    }
#endif
