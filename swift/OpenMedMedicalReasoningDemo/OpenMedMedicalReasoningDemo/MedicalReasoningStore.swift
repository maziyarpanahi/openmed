import Foundation
import OpenMedKit
import os.log

@MainActor
final class MedicalReasoningStore: ObservableObject {
    @Published var route: MedicalReasoningRoute
    @Published var downloadState: LFMDownloadState
    @Published var activeDownloadFile = ""
    @Published var clinicalContext: String
    @Published var messages: [MedicalConversationMessage]
    @Published var draft = ""
    @Published var isGenerating = false
    @Published var isLoadingModel = false
    @Published private(set) var isReleasingModel = false
    @Published var errorMessage: String?

    private let downloader: LFMModelDownloader
    private let log = Logger(
        subsystem: "life.openmed.medicalreasoningdemo",
        category: "reasoning"
    )
    private var downloadTask: Task<Void, Never>?
    private var downloadID: UUID?
    private var generationTask: Task<Void, Never>?
    private var runtime: OpenMedLFM?

    init(
        route: MedicalReasoningRoute = .modelSetup,
        downloadState: LFMDownloadState? = nil,
        clinicalContext: String = SyntheticClinicalCase.text,
        messages: [MedicalConversationMessage] = [],
        downloader: LFMModelDownloader = .shared
    ) {
        self.route = route
        self.clinicalContext = clinicalContext
        self.messages = messages
        self.downloader = downloader
        if let downloadState {
            self.downloadState = downloadState
        } else if let directory = try? LFMModelDownloader.cachedDirectory(),
            LFMModelDownloader.isCachedArtifactValid(at: directory)
        {
            self.downloadState = .ready
        } else {
            let bytes = LFMModelDownloader.cachedBytes()
            self.downloadState = bytes > 0 ? .partial(bytesOnDisk: bytes) : .missing
        }
    }

    var contextIsUsable: Bool {
        !clinicalContext.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    var modelIsReady: Bool {
        downloadState == .ready
    }

    func startDownload() {
        guard downloadTask == nil, !modelIsReady else { return }
        let existingBytes = LFMModelDownloader.cachedBytes()
        downloadState = .downloading(
            bytesDownloaded: existingBytes,
            bytesExpected: OpenMedLFM.estimatedDownloadBytes
        )
        activeDownloadFile = "Preparing pinned artifact"
        let id = UUID()
        downloadID = id

        downloadTask = Task { [weak self] in
            guard let self else { return }
            do {
                _ = try await downloader.prepare { [weak self] file, downloaded, expected in
                    Task { @MainActor [weak self] in
                        guard let self, downloadID == id, downloadTask != nil else { return }
                        activeDownloadFile = file
                        downloadState = .downloading(
                            bytesDownloaded: downloaded,
                            bytesExpected: expected
                        )
                    }
                }
                guard !Task.isCancelled, downloadID == id else { return }
                downloadState = .ready
                activeDownloadFile = "Verified and ready"
                downloadTask = nil
            } catch is CancellationError {
                guard downloadID == id else { return }
                downloadState = .cancelled
                activeDownloadFile = "Download paused"
                downloadTask = nil
            } catch {
                guard downloadID == id else { return }
                downloadState = .failed(message: error.localizedDescription)
                activeDownloadFile = "Download failed"
                downloadTask = nil
                log.error("Pinned LFM download failed: \(error.localizedDescription, privacy: .public)")
            }
        }
    }

    func cancelDownload() {
        downloadTask?.cancel()
        // Keep admission closed until URLSession has actually stopped writing.
        let cancelledTask = downloadTask
        let cancelledID = downloadID
        downloadID = nil
        Task {
            await cancelledTask?.value
            if downloadID == nil || downloadID == cancelledID { downloadTask = nil }
        }
        downloadState = .cancelled
        activeDownloadFile = "Download paused"
    }

    func showClinicalContext() {
        guard modelIsReady else { return }
        route = .clinicalContext
    }

    func restoreSyntheticCase() {
        clinicalContext = SyntheticClinicalCase.text
    }

    func startConversation() {
        guard contextIsUsable, modelIsReady, !isGenerating, !isReleasingModel else { return }
        messages = []
        draft = ""
        route = .conversation
    }

    func backToModelSetup() {
        guard !isGenerating else { return }
        route = .modelSetup
        Task { await releaseRuntime() }
    }

    func editClinicalContext() {
        guard !isGenerating else { return }
        route = .clinicalContext
    }

    func startNewCase() {
        guard !isGenerating else { return }
        messages = []
        draft = ""
        route = .clinicalContext
        Task { await releaseRuntime() }
    }

    func sendMessage() async {
        let question = draft.trimmingCharacters(in: .whitespacesAndNewlines)
        let context = clinicalContext.trimmingCharacters(in: .whitespacesAndNewlines)
        guard
            !question.isEmpty,
            !context.isEmpty,
            modelIsReady,
            !isGenerating,
            !isReleasingModel
        else { return }

        let history = Self.completedHistory(from: messages)
        let userMessage = MedicalConversationMessage(role: .user, content: question)
        let assistantMessage = MedicalConversationMessage(
            role: .assistant,
            content: "",
            activity: .reasoning
        )
        messages.append(userMessage)
        messages.append(assistantMessage)
        draft = ""
        isGenerating = true
        errorMessage = nil

        let task = Task {
            await generate(question: question, context: context, history: history, messageID: assistantMessage.id)
        }
        generationTask = task
        await withTaskCancellationHandler {
            await task.value
        } onCancel: {
            task.cancel()
        }
        generationTask = nil
        isGenerating = false
        isLoadingModel = false
    }

    private func generate(question: String, context: String, history: [OpenMedLFMMessage], messageID: UUID) async {
        do {
            try Task.checkCancellation()
            let runtime = try await loadRuntimeIfNeeded()
            let response = try await runtime.complete(
                OpenMedLFMRequest(
                    task: .chat,
                    document: context,
                    messages: history,
                    question: question,
                    maximumTokens: 1_536
                ),
                onReasoningChunk: { [weak self, id = messageID] chunk in
                    await self?.appendReasoning(chunk, to: id)
                },
                onFinalAnswerChunk: { [weak self, id = messageID] chunk in
                    await self?.appendAnswer(chunk, to: id)
                }
            )
            guard let answer = response.answer, !answer.isEmpty else {
                throw MedicalReasoningError.missingAnswer
            }
            try Task.checkCancellation()
            if let index = messages.firstIndex(where: { $0.id == messageID }) {
                messages[index].content = answer
                messages[index].activity = .complete
            }
        } catch is CancellationError {
            if let index = messages.firstIndex(where: { $0.id == messageID }) {
                messages[index].activity = .stopped
            }
        } catch {
            markFailed(messageID: messageID)
            errorMessage = error.localizedDescription
            log.error("Local LFM generation failed; details are shown only in the app.")
        }
    }

    func stopGenerating() {
        generationTask?.cancel()
    }

    /// Only complete user/assistant pairs enter future prompts. Failed or stopped
    /// responses remain visible but never create orphaned user turns in history.
    static func completedHistory(from messages: [MedicalConversationMessage]) -> [OpenMedLFMMessage] {
        var pairs: [[OpenMedLFMMessage]] = []
        for index in messages.indices where index > 0 {
            let answer = messages[index]
            let question = messages[index - 1]
            guard answer.role == .assistant, answer.activity == .complete,
                !answer.content.isEmpty, question.role == .user
            else { continue }
            pairs.append([
                .init(role: .user, content: question.content),
                .init(role: .assistant, content: answer.content),
            ])
        }
        return pairs.suffix(8).flatMap { $0 }
    }

    func releaseRuntime() async {
        guard !isReleasingModel else { return }
        isReleasingModel = true
        defer { isReleasingModel = false }
        let task = generationTask
        task?.cancel()
        await task?.value
        let loadedRuntime = runtime
        runtime = nil
        await loadedRuntime?.unload()
    }

    private func loadRuntimeIfNeeded() async throws -> OpenMedLFM {
        if let runtime { return runtime }
        isLoadingModel = true
        let directory = try LFMModelDownloader.cachedDirectory()
        guard LFMModelDownloader.isCachedArtifactValid(at: directory) else {
            downloadState = .partial(bytesOnDisk: LFMModelDownloader.cachedBytes())
            throw MedicalReasoningError.modelNotReady
        }
        let loadedRuntime = try await OpenMedLFM(modelDirectoryURL: directory)
        if Task.isCancelled {
            await loadedRuntime.unload()
            throw CancellationError()
        }
        runtime = loadedRuntime
        isLoadingModel = false
        return loadedRuntime
    }

    private func appendReasoning(_ chunk: String, to messageID: UUID) {
        guard let index = messages.firstIndex(where: { $0.id == messageID }) else { return }
        messages[index].reasoning.append(chunk)
        messages[index].activity = .reasoning
    }

    private func appendAnswer(_ chunk: String, to messageID: UUID) {
        guard let index = messages.firstIndex(where: { $0.id == messageID }) else { return }
        messages[index].content.append(chunk)
        messages[index].activity = .answering
    }

    private func markFailed(messageID: UUID) {
        guard let index = messages.firstIndex(where: { $0.id == messageID }) else { return }
        messages[index].activity = .failed
    }
}

extension MedicalReasoningStore {
    static func preview(
        route: MedicalReasoningRoute,
        downloadState: LFMDownloadState
    ) -> MedicalReasoningStore {
        MedicalReasoningStore(route: route, downloadState: downloadState)
    }

    static var previewConversation: MedicalReasoningStore {
        MedicalReasoningStore(
            route: .conversation,
            downloadState: .ready,
            messages: [
                MedicalConversationMessage(
                    role: .user,
                    content: "What follow-up is documented?"
                ),
                MedicalConversationMessage(
                    role: .assistant,
                    content: "The note documents **primary-care follow-up within 48 hours** and cardiology review if symptoms recur.",
                    reasoning: "I located the explicit plan and separated scheduled follow-up from conditional escalation."
                ),
            ]
        )
    }
}

enum MedicalReasoningError: LocalizedError {
    case modelNotReady
    case missingAnswer

    var errorDescription: String? {
        switch self {
        case .modelNotReady:
            return "The pinned LFM2.5 model cache is incomplete. Return to model setup and resume the download."
        case .missingAnswer:
            return "LFM2.5 did not return a final answer. Try a more specific question."
        }
    }
}
