import SwiftUI

@main
struct OpenMedMedicalReasoningDemoApp: App {
    @StateObject private var store = MedicalReasoningStore()
    @Environment(\.scenePhase) private var scenePhase

    var body: some Scene {
        WindowGroup {
            ContentView(store: store)
                .tint(MedicalReasoningTheme.teal)
                .preferredColorScheme(.light)
                #if os(macOS)
                    .frame(minWidth: 420, idealWidth: 560, minHeight: 640, idealHeight: 850)
                #endif
        }
        .onChange(of: scenePhase) { _, newValue in
            // On iOS stop before entering the background, where Metal submission
            // is prohibited. A Mac window can continue while another app is active.
            #if os(iOS)
                guard newValue != .active else { return }
                Task { await store.releaseRuntime() }
            #endif
        }
    }
}
