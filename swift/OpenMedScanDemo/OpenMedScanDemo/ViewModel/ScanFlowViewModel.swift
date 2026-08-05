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
    @Published public var pastedOrScannedText: String = ""
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
        didSet { ScanFlowViewModel.persistEngine(piiEngine) }
    }
    @Published public var openMedPIIOutput: PIIOutput?
    @Published public var privacyFilterPIIOutput: PIIOutput?
    @Published public var multilingualPIIOutput: PIIOutput?
    @Published public var maplePIIOutput: PIIOutput?

    private static let engineKey = "com.openmed.scan.pii-engine"

    private static func loadPersistedEngine() -> PIIEngine {
        if let raw = UserDefaults.standard.string(forKey: engineKey),
            let engine = PIIEngine(rawValue: raw)
        {
            return engine
        }
        return .maple
    }

    private static func persistEngine(_ engine: PIIEngine) {
        UserDefaults.standard.set(engine.rawValue, forKey: engineKey)
    }

    public enum PIIEngine: String, CaseIterable, Identifiable, Hashable, Sendable {
        case maple, openMed, privacyFilter, multilingual

        public var id: String { rawValue }
        public var modelID: ScanModelID {
            switch self {
            case .maple: return .maplePreview
            case .openMed: return .piiLiteClinical
            case .privacyFilter: return .openaiPrivacyFilter
            case .multilingual: return .multilingualPrivacyFilter
            }
        }
        public var displayName: String {
            switch self {
            case .maple: return "Maple Preview"
            case .openMed: return "OpenMed PII"
            case .privacyFilter: return "OpenAI Nemotron Privacy Filter"
            case .multilingual: return "OpenMed Multilingual Privacy Filter"
            }
        }
        public var eyebrow: String {
            switch self {
            case .maple: return "DEEPGROVE MAPLE · 2-BIT"
            case .openMed: return "OPENMED · LOCAL"
            case .privacyFilter: return "OPENAI NEMOTRON · 8-BIT"
            case .multilingual: return "OPENMED MULTILINGUAL · 8-BIT"
            }
        }
    }

    public var currentPIIOutput: PIIOutput? {
        output(for: piiEngine)
    }

    public func output(for engine: PIIEngine) -> PIIOutput? {
        switch engine {
        case .maple: return maplePIIOutput
        case .openMed: return openMedPIIOutput
        case .privacyFilter: return privacyFilterPIIOutput
        case .multilingual: return multilingualPIIOutput
        }
    }

    // MARK: Clinical extraction

    @Published public var clinicalThreshold: Double = 0.6
    @Published public var clinicalOutput: ClinicalOutput?

    // MARK: Maple reasoning + chat

    public struct MapleChatTurn: Identifiable, Hashable, Sendable {
        public enum Role: String, Sendable {
            case user
            case assistant
        }

        public let id: UUID
        public let role: Role
        public let content: String

        public init(id: UUID = UUID(), role: Role, content: String) {
            self.id = id
            self.role = role
            self.content = content
        }
    }

    @Published public var mapleBrief: String?
    @Published public var mapleChatTurns: [MapleChatTurn] = []
    @Published public var mapleChatDraft: String = ""

    // MARK: Status + errors

    @Published public var status: PipelineProgress?
    @Published public var errorMessage: String?
    @Published public var isWorking: Bool = false
    @Published public var hasRunAnalysis: Bool = false

    // MARK: Filters (summary)

    @Published public var summaryCategoryFilter: EntityCategory?

    // MARK: Dependencies

    public let downloads: ModelDownloadManager
    public let presets: ClinicalPresetsStore
    private let runtime: OMPipelineRuntime
    private let log = Logger(subsystem: "com.openmed.scan", category: "flow")
    private var piiRevision: Int = 0

    public init(
        downloads: ModelDownloadManager? = nil,
        presets: ClinicalPresetsStore? = nil,
        runtime: OMPipelineRuntime = .shared
    ) {
        self.downloads = downloads ?? ModelDownloadManager.shared
        self.presets = presets ?? ClinicalPresetsStore()
        self.runtime = runtime
    }

    // MARK: - Derived

    public var trimmedText: String {
        pastedOrScannedText.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    public var hasText: Bool { !trimmedText.isEmpty }

    public var activeLabels: [String] { presets.selectedPreset.labels }

    public var clinicalModelID: ScanModelID { .maplePreview }

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
            hasRunAnalysis = true
            HapticsCenter.impact(.soft)
        } catch {
            guard revision == piiRevision, text == trimmedText else { return }
            errorMessage = error.localizedDescription
            HapticsCenter.notify(.error)
            log.error("PII run failed: \(error.localizedDescription, privacy: .public)")
        }
    }

    public func runPIIForAllEngines() async {
        // Run current engine first (so user sees their selection complete),
        // then opportunistically run the other if its model is ready.
        let selectedEngine = piiEngine
        let text = trimmedText
        let revision = piiRevision
        guard !text.isEmpty, !isWorking else { return }
        guard downloads.state(for: selectedEngine.modelID) == .ready else {
            errorMessage = "Model not ready yet — start the download first."
            return
        }

        isWorking = true
        status = PipelineProgress(phase: .inferencing, detail: "Running \(selectedEngine.displayName) on-device")
        defer {
            isWorking = false
            status = nil
        }

        do {
            let output = try await runtime.runPII(text: text, modelID: selectedEngine.modelID)
            guard revision == piiRevision, text == trimmedText else { return }
            setPIIOutput(output, for: selectedEngine)
            hasRunAnalysis = true
            HapticsCenter.impact(.soft)
        } catch {
            guard revision == piiRevision, text == trimmedText else { return }
            errorMessage = error.localizedDescription
            HapticsCenter.notify(.error)
            log.error("PII run failed: \(error.localizedDescription, privacy: .public)")
            return
        }

        guard revision == piiRevision, text == trimmedText else { return }
        if selectedEngine == .maple { return }
        for engine in PIIEngine.allCases where engine != selectedEngine {
            guard engine != .maple else { continue }
            guard downloads.state(for: engine.modelID) == .ready else { continue }
            status = PipelineProgress(phase: .inferencing, detail: "Running \(engine.displayName) on-device")
            do {
                let output = try await runtime.runPII(text: text, modelID: engine.modelID)
                guard revision == piiRevision, text == trimmedText else { return }
                setPIIOutput(output, for: engine)
            } catch {
                guard revision == piiRevision, text == trimmedText else { return }
                log.error("Secondary PII engine failed: \(error.localizedDescription, privacy: .public)")
            }
        }
    }

    public func runClinical() async {
        guard let masked = currentPIIOutput?.maskedText, !isWorking else { return }
        let modelID = clinicalModelID
        guard downloads.state(for: modelID) == .ready else {
            errorMessage = "Clinical model not ready — download it first."
            return
        }
        isWorking = true
        status = PipelineProgress(phase: .inferencing, detail: "Preparing Maple for extraction")
        defer {
            isWorking = false
            status = nil
        }
        do {
            status = PipelineProgress(phase: .inferencing, detail: "Extracting entities + relations with Maple")
            let output = try await runtime.runClinical(
                maskedText: masked,
                labels: activeLabels,
                threshold: Float(clinicalThreshold),
                modelID: modelID
            )
            clinicalOutput = output
            HapticsCenter.impact(.soft)
        } catch {
            errorMessage = error.localizedDescription
            HapticsCenter.notify(.error)
            log.error("Clinical run failed: \(error.localizedDescription, privacy: .public)")
        }
    }

    public func generateMapleBrief() async {
        guard let masked = currentPIIOutput?.maskedText, !isWorking else { return }
        guard downloads.state(for: .maplePreview) == .ready else {
            errorMessage = "Maple is not ready — download it first."
            return
        }
        isWorking = true
        status = PipelineProgress(phase: .inferencing, detail: "Maple is reviewing the de-identified note")
        defer {
            isWorking = false
            status = nil
        }
        do {
            mapleBrief = try await runtime.reason(
                maskedText: masked,
                question: "What are the key clinical facts, relationships, uncertainties, and follow-up items?",
                messages: []
            )
            HapticsCenter.impact(.soft)
        } catch {
            errorMessage = error.localizedDescription
            HapticsCenter.notify(.error)
            log.error("Maple brief failed: \(error.localizedDescription, privacy: .public)")
        }
    }

    public func askMaple() async {
        let question = mapleChatDraft.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let masked = currentPIIOutput?.maskedText,
            !question.isEmpty,
            !isWorking
        else { return }
        guard downloads.state(for: .maplePreview) == .ready else {
            errorMessage = "Maple is not ready — download it first."
            return
        }

        let history = mapleChatTurns.suffix(8).map { turn in
            OpenMedMapleMessage(
                role: turn.role == .user ? .user : .assistant,
                content: turn.content
            )
        }
        let pendingTurn = MapleChatTurn(role: .user, content: question)
        mapleChatTurns.append(pendingTurn)
        mapleChatDraft = ""
        isWorking = true
        status = PipelineProgress(phase: .inferencing, detail: "Maple is answering on-device")
        defer {
            isWorking = false
            status = nil
        }
        do {
            let answer = try await runtime.reason(
                maskedText: masked,
                question: question,
                messages: history
            )
            mapleChatTurns.append(MapleChatTurn(role: .assistant, content: answer))
            HapticsCenter.impact(.soft)
        } catch {
            mapleChatTurns.removeAll { $0.id == pendingTurn.id }
            mapleChatDraft = question
            errorMessage = error.localizedDescription
            HapticsCenter.notify(.error)
            log.error("Maple chat failed: \(error.localizedDescription, privacy: .public)")
        }
    }

    // MARK: - Reset

    public enum ResetScope {
        case all, piiOnly, clinicalOnly
    }

    public func reset(clearing scope: ResetScope = .all) {
        errorMessage = nil
        switch scope {
        case .all:
            piiRevision += 1
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
            maplePIIOutput = nil
            clinicalOutput = nil
            mapleBrief = nil
            mapleChatTurns = []
            mapleChatDraft = ""
            hasRunAnalysis = false
            status = nil
            summaryCategoryFilter = nil
        case .piiOnly:
            piiRevision += 1
            openMedPIIOutput = nil
            privacyFilterPIIOutput = nil
            multilingualPIIOutput = nil
            maplePIIOutput = nil
            clinicalOutput = nil
            mapleBrief = nil
            mapleChatTurns = []
            mapleChatDraft = ""
            hasRunAnalysis = false
            status = nil
            summaryCategoryFilter = nil
        case .clinicalOnly:
            clinicalOutput = nil
            mapleBrief = nil
            mapleChatTurns = []
            mapleChatDraft = ""
        }
    }

    private func setPIIOutput(_ output: PIIOutput, for engine: PIIEngine) {
        switch engine {
        case .maple: maplePIIOutput = output
        case .openMed: openMedPIIOutput = output
        case .privacyFilter: privacyFilterPIIOutput = output
        case .multilingual: multilingualPIIOutput = output
        }
    }
}
