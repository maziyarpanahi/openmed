import SwiftUI

@main
struct OpenMedScanDemoApp: App {
    @StateObject private var flow: ScanFlowViewModel
    @StateObject private var downloads: ModelDownloadManager
    @StateObject private var presets: ClinicalPresetsStore
    private let isSimulatorDemoReel: Bool

    init() {
        OMTypography.verifyRegistration()
        isSimulatorDemoReel = ProcessInfo.processInfo.arguments.contains(
            "--maple-simulator-demo"
        )
        let downloads = ModelDownloadManager.shared
        let presets = ClinicalPresetsStore()
        _downloads = StateObject(wrappedValue: downloads)
        _presets = StateObject(wrappedValue: presets)
        _flow = StateObject(
            wrappedValue: ScanFlowViewModel(
                downloads: downloads,
                presets: presets
            ))
    }

    var body: some Scene {
        WindowGroup {
            if isSimulatorDemoReel {
                MapleSimulatorDemoReelView()
                    .tint(Color.omTealAccent)
            } else {
                ContentView()
                    .environmentObject(flow)
                    .environmentObject(downloads)
                    .environmentObject(presets)
                    .tint(Color.omTealAccent)
            }
        }
    }
}
