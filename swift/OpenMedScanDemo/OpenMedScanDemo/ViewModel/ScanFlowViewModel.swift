import Combine
import Foundation
import OpenMedKit
import SwiftUI
import os.log

#if canImport(UIKit)
    import UIKit
#endif

/// Single source of truth for the redesigned flow. Owns every piece of
/// state that was previously scattered across ContentView's `@State`s.
@MainActor
public final class ScanFlowViewModel: ObservableObject {

    // MARK: Stage + navigation

    @Published public var stage: ScanStage = .input

    // MARK: Input

    #if canImport(UIKit)
        @Published public var documentImages: [UIImage] = []
    #endif
    @Published public var pastedOrScannedText: String = "" {
        didSet {
            guard oldValue != pastedOrScannedText else { return }
            invalidateResultsForDocumentChange()
        }
    }
    @Published public var needsOCR: Bool = false
    @Published public var pageCount: Int = 0
    @Published public var currentSource: InputSource = .none

    public enum InputSource: Sendable, Hashable {
        case none, paste, scan, sample
        public var label: String {
            switch self {
            case .none: return ""
            case .paste: return "Pasted"
            case .scan: return "Scanned"
            case .sample: return "Sample"
            }
        }
    }

    // MARK: De-identification

    @Published public var piiEngine: PIIEngine = ScanFlowViewModel.loadPersistedEngine() {
        didSet {
            ScanFlowViewModel.persistEngine(piiEngine)
            if oldValue != piiEngine {
                selectedPIINeedsRun = true
            }
        }
    }
    @Published public private(set) var selectedPIINeedsRun = true
    @Published public var openMedPIIOutput: PIIOutput?
    @Published public var privacyFilterPIIOutput: PIIOutput?
    @Published public var multilingualPIIOutput: PIIOutput?

    private static let engineKey = "com.openmed.scan.pii-engine"

    private static func loadPersistedEngine() -> PIIEngine {
        if let raw = UserDefaults.standard.string(forKey: engineKey),
            let engine = PIIEngine(rawValue: raw)
        {
            return engine
        }
        return .openMed
    }

    private static func persistEngine(_ engine: PIIEngine) {
        UserDefaults.standard.set(engine.rawValue, forKey: engineKey)
    }

    public enum PIIEngine: String, CaseIterable, Identifiable, Hashable, Sendable {
        case openMed, privacyFilter, multilingual

        public var id: String { rawValue }
        public var modelID: ScanModelID {
            switch self {
            case .openMed: return .piiLiteClinical
            case .privacyFilter: return .openaiPrivacyFilter
            case .multilingual: return .multilingualPrivacyFilter
            }
        }
        public var displayName: String {
            switch self {
            case .openMed: return "OpenMed PII"
            case .privacyFilter: return "OpenAI Nemotron Privacy Filter"
            case .multilingual: return "OpenMed Multilingual Privacy Filter"
            }
        }
        public var eyebrow: String {
            switch self {
            case .openMed: return "OPENMED · LOCAL"
            case .privacyFilter: return "OPENAI NEMOTRON · 8-BIT"
            case .multilingual: return "OPENMED MULTILINGUAL · 8-BIT"
            }
        }
    }

    public var currentPIIOutput: PIIOutput? {
        output(for: piiEngine)
    }

    public var selectedPIIRequiresRun: Bool {
        selectedPIINeedsRun || currentPIIOutput == nil
    }

    public func output(for engine: PIIEngine) -> PIIOutput? {
        switch engine {
        case .openMed: return openMedPIIOutput
        case .privacyFilter: return privacyFilterPIIOutput
        case .multilingual: return multilingualPIIOutput
        }
    }

    public var completedPIIEngines: [PIIEngine] {
        PIIEngine.allCases.filter { output(for: $0) != nil }
    }

    public var comparisonPIIEngines: (left: PIIEngine, right: PIIEngine)? {
        let completed = completedPIIEngines
        guard completed.count >= 2 else { return nil }
        return (completed[0], completed[1])
    }

    // MARK: Clinical NER

    @Published public var nerModel: NERModel = .disease {
        didSet {
            if oldValue != nerModel {
                selectedNERNeedsRun = true
            }
        }
    }
    @Published public private(set) var selectedNERNeedsRun = true
    @Published public private(set) var nerOutputs: [NERModel: NEROutput] = [:]

    public enum NERModel: String, CaseIterable, Identifiable, Hashable, Sendable {
        case disease
        case medication
        case anatomy

        public var id: String { rawValue }

        public var modelID: ScanModelID {
            switch self {
            case .disease: return .nerDisease
            case .medication: return .nerMedication
            case .anatomy: return .nerAnatomy
            }
        }

        public var displayName: String {
            switch self {
            case .disease: return "Disease NER"
            case .medication: return "Medication NER"
            case .anatomy: return "Anatomy NER"
            }
        }

        public var detail: String {
            switch self {
            case .disease: return "Disease and condition spans"
            case .medication: return "Medication and chemical spans"
            case .anatomy: return "Anatomical structure spans"
            }
        }
    }

    public var currentNEROutput: NEROutput? {
        nerOutputs[nerModel]
    }

    public var selectedNERRequiresRun: Bool {
        selectedNERNeedsRun || currentNEROutput == nil
    }

    public var completedNERModels: [NERModel] {
        NERModel.allCases.filter { nerOutputs[$0] != nil }
    }

    public var allNEREntities: [DetectedEntity] {
        NERModel.allCases.flatMap { nerOutputs[$0]?.entities ?? [] }
    }

    // MARK: Status + errors

    @Published public var status: PipelineProgress?
    @Published public var errorMessage: String?
    @Published public var isWorking: Bool = false
    @Published public var hasRunAnalysis: Bool = false

    // MARK: Filters (summary)

    @Published public var summaryCategoryFilter: EntityCategory?

    // MARK: Dependencies

    public let downloads: ModelDownloadManager
    private let runtime: OMPipelineRuntime
    private let log = Logger(subsystem: "com.openmed.scan", category: "flow")
    private var piiRevision: Int = 0

    public init(
        downloads: ModelDownloadManager? = nil,
        runtime: OMPipelineRuntime = .shared
    ) {
        self.downloads = downloads ?? ModelDownloadManager.shared
        self.runtime = runtime
    }

    // MARK: - Derived

    public var trimmedText: String {
        pastedOrScannedText.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    public var hasText: Bool { !trimmedText.isEmpty }

    // MARK: - Input actions

    public func useSample(text: String) {
        reset(clearing: .all)
        pastedOrScannedText = text
        currentSource = .sample
        needsOCR = false
        pageCount = 1
    }

    public func useText(_ text: String) {
        reset(clearing: .all)
        pastedOrScannedText = text
        currentSource = .paste
        needsOCR = false
    }

    #if canImport(UIKit)
        public func usePages(_ images: [UIImage]) {
            reset(clearing: .all)
            documentImages = images
            currentSource = .scan
            pastedOrScannedText = ""
            needsOCR = true
            pageCount = images.count
        }
    #endif

    // MARK: - Stage transitions

    public func advance() {
        guard let next = stage.next else { return }
        stage = next
        HapticsCenter.selection()
    }

    public func goBack() {
        guard let previous = stage.previous else { return }
        stage = previous
        HapticsCenter.selection()
    }

    public func jump(to stage: ScanStage) {
        self.stage = stage
        HapticsCenter.selection()
    }

    public func restart() {
        reset(clearing: .all)
        stage = .input
        HapticsCenter.notify(.success)
    }

    // MARK: - Pipeline actions

    #if canImport(UIKit)
        public func runOCRIfNeeded() async {
            guard needsOCR, !documentImages.isEmpty, !isWorking else { return }
            isWorking = true
            status = PipelineProgress(phase: .recognizing, detail: "Vision OCR on \(documentImages.count) page(s)")
            defer {
                isWorking = false
                status = nil
            }
            do {
                let result = try await runtime.recognizeText(in: documentImages)
                pastedOrScannedText = result.text
                needsOCR = false
                pageCount = result.pageCount
                documentImages.removeAll(keepingCapacity: false)
                HapticsCenter.notify(.success)
            } catch {
                errorMessage = error.localizedDescription
                HapticsCenter.notify(.error)
                log.error("OCR failed: \(error.localizedDescription, privacy: .public)")
            }
        }
    #endif

    public func runPIIForCurrentEngine() async {
        let engine = piiEngine
        let text = trimmedText
        let revision = piiRevision
        guard !text.isEmpty, !isWorking else { return }
        guard downloads.state(for: engine.modelID) == .ready else {
            errorMessage = "Model not ready yet — start the download first."
            return
        }
        isWorking = true
        status = PipelineProgress(phase: .inferencing, detail: "Running \(engine.displayName) on-device")
        defer {
            isWorking = false
            status = nil
        }
        do {
            let output = try await runtime.runPII(
                text: text,
                modelID: engine.modelID
            )
            guard revision == piiRevision, text == trimmedText else { return }
            setPIIOutput(output, for: engine)
            invalidateNERResults()
            if engine == piiEngine {
                selectedPIINeedsRun = false
            }
            hasRunAnalysis = true
            HapticsCenter.impact(.soft)
        } catch {
            guard revision == piiRevision, text == trimmedText else { return }
            errorMessage = error.localizedDescription
            HapticsCenter.notify(.error)
            log.error("PII run failed: \(error.localizedDescription, privacy: .public)")
        }
    }

    public func runNERForCurrentModel() async {
        guard let masked = currentPIIOutput?.maskedText, !isWorking else { return }
        let model = nerModel
        let modelID = model.modelID
        let revision = piiRevision
        guard downloads.state(for: modelID) == .ready else {
            errorMessage = "NER model not ready — download it first."
            return
        }
        isWorking = true
        status = PipelineProgress(
            phase: .inferencing,
            detail: "Loading \(model.displayName)"
        )
        defer {
            isWorking = false
            status = nil
        }
        do {
            status = PipelineProgress(
                phase: .inferencing,
                detail: "Running \(model.displayName), then unloading it"
            )
            let output = try await runtime.runNER(
                text: masked,
                modelID: modelID
            )
            guard revision == piiRevision,
                masked == currentPIIOutput?.maskedText
            else { return }
            nerOutputs[model] = output
            if model == nerModel {
                selectedNERNeedsRun = false
            }
            HapticsCenter.impact(.soft)
        } catch {
            guard revision == piiRevision,
                masked == currentPIIOutput?.maskedText
            else { return }
            errorMessage = error.localizedDescription
            HapticsCenter.notify(.error)
            log.error("NER run failed: \(error.localizedDescription, privacy: .public)")
        }
    }

    // MARK: - Reset

    public enum ResetScope {
        case all, piiOnly, nerOnly
    }

    public func reset(clearing scope: ResetScope = .all) {
        errorMessage = nil
        switch scope {
        case .all:
            piiRevision += 1
            selectedPIINeedsRun = true
            #if canImport(UIKit)
                documentImages = []
            #endif
            pastedOrScannedText = ""
            needsOCR = false
            pageCount = 0
            currentSource = .none
            openMedPIIOutput = nil
            privacyFilterPIIOutput = nil
            multilingualPIIOutput = nil
            selectedNERNeedsRun = true
            nerOutputs = [:]
            hasRunAnalysis = false
            status = nil
            summaryCategoryFilter = nil
        case .piiOnly:
            piiRevision += 1
            selectedPIINeedsRun = true
            openMedPIIOutput = nil
            privacyFilterPIIOutput = nil
            multilingualPIIOutput = nil
            selectedNERNeedsRun = true
            nerOutputs = [:]
            hasRunAnalysis = false
            status = nil
            summaryCategoryFilter = nil
        case .nerOnly:
            selectedNERNeedsRun = true
            nerOutputs = [:]
        }
    }

    private func setPIIOutput(_ output: PIIOutput, for engine: PIIEngine) {
        switch engine {
        case .openMed: openMedPIIOutput = output
        case .privacyFilter: privacyFilterPIIOutput = output
        case .multilingual: multilingualPIIOutput = output
        }
    }

    private func invalidateResultsForDocumentChange() {
        piiRevision += 1
        selectedPIINeedsRun = true
        openMedPIIOutput = nil
        privacyFilterPIIOutput = nil
        multilingualPIIOutput = nil
        selectedNERNeedsRun = true
        nerOutputs = [:]
        hasRunAnalysis = false
        summaryCategoryFilter = nil
    }

    private func invalidateNERResults() {
        selectedNERNeedsRun = true
        nerOutputs = [:]
        summaryCategoryFilter = nil
    }
}
