import SwiftUI

@main
struct OpenMedScanDemoApp: App {
    @StateObject private var flow: ScanFlowViewModel
    @StateObject private var downloads: ModelDownloadManager

    init() {
        OMTypography.verifyRegistration()
        let downloads = ModelDownloadManager.shared
        _downloads = StateObject(wrappedValue: downloads)
        _flow = StateObject(
            wrappedValue: ScanFlowViewModel(
                downloads: downloads
            ))
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(flow)
                .environmentObject(downloads)
                .tint(Color.omTealAccent)
        }
    }
}
