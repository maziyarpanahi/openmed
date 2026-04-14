import SwiftUI

@main
struct OpenMedDemoApp: App {
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
