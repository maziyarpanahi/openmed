import SwiftUI
import OpenMedKit
#if canImport(UIKit)
import UIKit
#endif

/// Root layout. Hosts the stage screen in the shared chrome and mediates
/// every sheet presentation the flow needs.
struct ContentView: View {
    @EnvironmentObject private var flow: ScanFlowViewModel
    @EnvironmentObject private var downloads: ModelDownloadManager
    @EnvironmentObject private var presets: ClinicalPresetsStore

    @State private var isShowingScanner = false
    @State private var isShowingModelSheet = false
    @State private var isShowingLabelEditor = false
    @State private var isShowingComparison = false
    @State private var furthestReached: ScanStage = .input

    var body: some View {
        ScanScreenChrome(
            currentStage: flow.stage,
            furthestReached: furthestReached,
            onJump: handleJump,
            content: { screen(for: flow.stage) },
            actionBar: { actionBar }
        )
        .onChange(of: flow.stage) { _, new in
            if new.rawValue > furthestReached.rawValue {
                furthestReached = new
            }
        }
        .onAppear {
            downloads.refreshAll()
        }
        #if canImport(UIKit) && canImport(VisionKit)
        .sheet(isPresented: $isShowingScanner) {
            ScannerSheet(
                onComplete: { pages in
                    isShowingScanner = false
                    flow.usePages(pages)
                    HapticsCenter.notify(.success)
                },
                onCancel: { isShowingScanner = false },
                onError: { error in
                    isShowingScanner = false
                    flow.errorMessage = error.localizedDescription
                }
            )
            .ignoresSafeArea()
        }
        #endif
        .sheet(isPresented: $isShowingModelSheet) {
            ModelDownloadSheet(downloads: downloads)
                .presentationDetents([.medium, .large])
                .presentationDragIndicator(.visible)
        }
        .sheet(isPresented: $isShowingLabelEditor) {
            LabelEditorSheet(presets: presets)
                .presentationDetents([.large])
        }
        .sheet(isPresented: $isShowingComparison) {
            PIIComparisonSheet(
                sourceText: flow.trimmedText,
                leftOutput: flow.openMedPIIOutput,
                rightOutput: flow.multilingualPIIOutput ?? flow.privacyFilterPIIOutput,
                leftEngine: .openMed,
                rightEngine: flow.multilingualPIIOutput == nil ? .privacyFilter : .multilingual
            )
        }
        .alert(
            "Something went wrong",
            isPresented: Binding(
                get: { flow.errorMessage != nil },
                set: { if !$0 { flow.errorMessage = nil } }
            ),
            actions: {
                Button("OK", role: .cancel) { flow.errorMessage = nil }
            },
            message: {
                if let message = flow.errorMessage { Text(message) }
            }
        )
    }

    // MARK: - Stage → Screen

    @ViewBuilder
    private func screen(for stage: ScanStage) -> some View {
        switch stage {
        case .input:
            InputScreen(
                flow: flow,
                downloads: downloads,
                onShowScanner: { isShowingScanner = true },
                onShowModelSheet: { isShowingModelSheet = true }
            )
        case .review:
            ReviewScreen(flow: flow, downloads: downloads)
        case .deidentify:
            DeIDScreen(
                flow: flow,
                downloads: downloads,
                onShowComparison: { isShowingComparison = true }
            )
        case .clinical:
            ClinicalScreen(
                flow: flow,
                downloads: downloads,
                presets: presets,
                onSaveAsNewPreset: { isShowingLabelEditor = true }
            )
        case .summary:
            SummaryScreen(
                flow: flow,
                onShowComparison: { isShowingComparison = true },
                onStartOver: {
                    flow.restart()
                    furthestReached = .input
                }
            )
        }
    }

    // MARK: - Action bar

    @ViewBuilder
    private var actionBar: some View {
        HStack(spacing: OM.Space.s3) {
            if flow.stage != .input {
                Button("Back", action: { flow.goBack() })
                    .buttonStyle(.omGhost(.ink))
                    .frame(width: 64)
            }
            Button {
                Task { await primaryAction() }
            } label: {
                HStack {
                    Text(primaryTitle)
                    if flow.isWorking {
                        ProgressView().tint(Color.omPaper)
                    }
                }
            }
            .buttonStyle(.omPrimary(.md))
            .disabled(!primaryEnabled)
        }
    }

    private var primaryTitle: String {
        switch flow.stage {
        case .input:      return flow.hasText ? "Review document" : "Add document or text"
        case .review:     return flow.needsOCR ? "Run OCR" : "Continue to de-identification"
        case .deidentify:
            if flow.currentPIIOutput == nil { return "Run PII redaction" }
            return "Continue to clinical"
        case .clinical:
            if flow.clinicalOutput == nil { return "Extract clinical signals" }
            return "Review summary"
        case .summary:    return "Start a new scan"
        }
    }

    private var primaryEnabled: Bool {
        if flow.isWorking { return false }
        switch flow.stage {
        case .input:
            return flow.hasText || flow.currentSource == .scan
        case .review:
            if flow.needsOCR { return !flow.documentImages.isEmpty }
            return flow.hasText
        case .deidentify:
            if flow.currentPIIOutput == nil {
                return downloads.state(for: flow.piiEngine.modelID) == .ready && flow.hasText
            }
            return true
        case .clinical:
            if flow.clinicalOutput == nil {
                return downloads.state(for: .glinerRelex) == .ready
                    && flow.currentPIIOutput != nil
                    && !flow.activeLabels.isEmpty
            }
            return true
        case .summary:
            return true
        }
    }

    private func primaryAction() async {
        switch flow.stage {
        case .input:
            flow.advance()
        case .review:
            if flow.needsOCR {
                #if canImport(UIKit)
                await flow.runOCRIfNeeded()
                #endif
            } else {
                flow.advance()
            }
        case .deidentify:
            if flow.currentPIIOutput == nil {
                await flow.runPIIForAllEngines()
            } else {
                flow.advance()
            }
        case .clinical:
            if flow.clinicalOutput == nil {
                await flow.runClinical()
            } else {
                flow.advance()
            }
        case .summary:
            flow.restart()
            furthestReached = .input
        }
    }

    // MARK: - Navigation helpers

    private func handleJump(_ stage: ScanStage) {
        guard stage.rawValue <= furthestReached.rawValue else {
            HapticsCenter.impact(.rigid)
            return
        }
        flow.jump(to: stage)
    }
}

#Preview {
    ContentView()
        .environmentObject(ScanFlowViewModel())
        .environmentObject(ModelDownloadManager.shared)
        .environmentObject(ClinicalPresetsStore())
}
