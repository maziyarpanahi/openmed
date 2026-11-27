import AppIntents
import OpenMedIntents
import SwiftUI

@main
struct OpenMedDemoApp: App {
    init() {
        // Shortcuts redact with the demo's default model once the app has
        // downloaded it; the intents never download a model themselves.
        OpenMedIntentsConfiguration.redactor = .cachedMLXModel(
            repoID: OpenMedIntentsConfiguration.defaultModelRepoID
        )
    }

    var body: some Scene {
        #if os(macOS)
        WindowGroup {
            ContentView()
                .frame(minWidth: 600, minHeight: 500)
        }
        #else
        WindowGroup {
            ContentView()
        }
        #endif
    }
}

/// Makes the intents that ship in the OpenMedIntents package visible to the
/// system from this app.
struct OpenMedDemoAppIntents: AppIntentsPackage {
    static var includedPackages: [any AppIntentsPackage.Type] {
        [OpenMedIntentsPackage.self]
    }
}

/// Shortcuts phrases for the on-device OpenMed intents. Both run without
/// opening the app and return only redacted text or label counts.
struct OpenMedDemoShortcuts: AppShortcutsProvider {
    static var appShortcuts: [AppShortcut] {
        AppShortcut(
            intent: RedactTextIntent(),
            phrases: [
                "Redact text with \(.applicationName)",
                "Remove health information with \(.applicationName)",
            ],
            shortTitle: "Redact Text",
            systemImageName: "eye.slash"
        )
        AppShortcut(
            intent: PIISummaryIntent(),
            phrases: [
                "Summarize PII with \(.applicationName)",
                "Count identifiers with \(.applicationName)",
            ],
            shortTitle: "PII Summary",
            systemImageName: "list.bullet.rectangle"
        )
    }
}
