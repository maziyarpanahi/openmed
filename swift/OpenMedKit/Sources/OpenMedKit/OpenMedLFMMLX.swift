#if canImport(MLX) && canImport(MLXLMCommon) && canImport(MLXLLM) && canImport(Tokenizers) && !os(watchOS) && !os(visionOS)
    import Foundation
    import MLXLMCommon
    import MLXLLM
    import Tokenizers

    private struct OpenMedLFMTokenizerAdapter: MLXLMCommon.Tokenizer, @unchecked Sendable {
        let tokenizer: any Tokenizers.Tokenizer

        func encode(text: String, addSpecialTokens: Bool) -> [Int] {
            tokenizer.encode(text: text, addSpecialTokens: addSpecialTokens)
        }

        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
            tokenizer.decode(tokens: tokenIds, skipSpecialTokens: skipSpecialTokens)
        }

        func convertTokenToId(_ token: String) -> Int? {
            tokenizer.convertTokenToId(token)
        }

        func convertIdToToken(_ id: Int) -> String? {
            tokenizer.convertIdToToken(id)
        }

        var bosToken: String? { tokenizer.bosToken }
        var eosToken: String? { tokenizer.eosToken }
        var unknownToken: String? { tokenizer.unknownToken }

        func applyChatTemplate(
            messages: [[String: any Sendable]],
            tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] {
            try tokenizer.applyChatTemplate(
                messages: messages,
                tools: tools,
                additionalContext: additionalContext
            )
        }
    }

    private struct OpenMedLFMTokenizerLoader: TokenizerLoader {
        func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
            let tokenizer = try await OpenMed.loadTokenizerAsync(
                tokenizerName: directory.path,
                tokenizerFolderURL: directory
            )
            return OpenMedLFMTokenizerAdapter(tokenizer: tokenizer)
        }
    }

    /// Errors raised before or during local LFM2.5 inference.
    public enum OpenMedLFMRuntimeError: LocalizedError {
        case missingFiles([String])
        case unsupportedArchitecture(String)
        case invalidConfiguration(String)
        case generationInProgress

        public var errorDescription: String? {
            switch self {
            case .missingFiles(let files):
                return "The LFM2.5 model cache is incomplete. Missing: \(files.joined(separator: ", "))."
            case .unsupportedArchitecture(let type):
                return "Unsupported LFM2.5 model_type: \(type)."
            case .invalidConfiguration(let detail):
                return "Invalid LFM2.5 model configuration: \(detail)."
            case .generationInProgress:
                return "LFM2.5 is already generating a response. Stop it before starting another."
            }
        }
    }

    /// Local-only LiquidAI LFM2.5 2.6B generation backed by MLX on Apple silicon.
    ///
    /// The runtime never downloads, logs, or persists prompts. Callers must
    /// prepare the pinned official 4-bit model directory before initialization.
    public actor OpenMedLFM {
        public static let repositoryID = "LiquidAI/LFM2.5-2.6B-MLX-4bit"
        public static let pinnedRevision = "04efa23776ce61ec34ec95ec34c859854c89542b"
        public static let estimatedDownloadBytes: Int64 = 1_601_108_840

        public static let requiredModelFiles = [
            "chat_template.jinja",
            "config.json",
            "generation_config.json",
            "model.safetensors",
            "model.safetensors.index.json",
            "tokenizer.json",
            "tokenizer_config.json",
        ]

        /// Paths relative to the dedicated Hugging Face repository root.
        public static let requiredRepositoryFiles = requiredModelFiles

        private struct ConfigurationHeader: Decodable {
            let modelType: String

            enum CodingKeys: String, CodingKey {
                case modelType = "model_type"
            }
        }

        private var container: ModelContainer?
        private var generationTask: Task<OpenMedLFMResponse, Error>?

        public init(modelDirectoryURL: URL) async throws {
            let directory = Self.modelDirectory(in: modelDirectoryURL)
            let missing = Self.missingRequiredFiles(in: directory)
            guard missing.isEmpty else {
                throw OpenMedLFMRuntimeError.missingFiles(missing)
            }

            do {
                let configurationData = try Data(
                    contentsOf: directory.appending(path: "config.json")
                )
                let header = try JSONDecoder().decode(
                    ConfigurationHeader.self,
                    from: configurationData
                )
                guard header.modelType == "lfm2" else {
                    throw OpenMedLFMRuntimeError.unsupportedArchitecture(header.modelType)
                }
                container = try await LLMModelFactory.shared.loadContainer(
                    from: directory,
                    using: OpenMedLFMTokenizerLoader()
                )
            } catch let error as OpenMedLFMRuntimeError {
                throw error
            } catch {
                throw OpenMedLFMRuntimeError.invalidConfiguration(error.localizedDescription)
            }
        }

        /// Resolves the dedicated repository-root cache.
        public nonisolated static func modelDirectory(in directory: URL) -> URL {
            directory
        }

        /// Returns whether the pinned official 4-bit files are present and non-empty.
        public nonisolated static func isModelDirectoryReady(_ directory: URL) -> Bool {
            missingRequiredFiles(in: modelDirectory(in: directory)).isEmpty
        }

        /// Runs a local clinical task. Structured output remains buffered until
        /// validated; reasoning/chat expose separate reasoning and answer streams.
        public func complete(
            _ request: OpenMedLFMRequest,
            onReasoningChunk: (@Sendable (String) async -> Void)? = nil,
            onFinalAnswerChunk: (@Sendable (String) async -> Void)? = nil
        ) async throws -> OpenMedLFMResponse {
            guard generationTask == nil else {
                throw OpenMedLFMRuntimeError.generationInProgress
            }
            guard let container else {
                throw OpenMedLFMRuntimeError.invalidConfiguration(
                    "the runtime has been unloaded"
                )
            }

            try Task.checkCancellation()
            let task = Task {
                try await Self.generate(
                    request, container: container,
                    onReasoningChunk: onReasoningChunk,
                    onFinalAnswerChunk: onFinalAnswerChunk
                )
            }
            generationTask = task
            defer { generationTask = nil }
            return try await withTaskCancellationHandler {
                try await task.value
            } onCancel: {
                task.cancel()
            }
        }

        private nonisolated static func generate(
            _ request: OpenMedLFMRequest,
            container: ModelContainer,
            onReasoningChunk: (@Sendable (String) async -> Void)?,
            onFinalAnswerChunk: (@Sendable (String) async -> Void)?
        ) async throws -> OpenMedLFMResponse {
            try Task.checkCancellation()
            let messages = OpenMedLFMPrompt.messages(for: request).map { message in
                switch message.role {
                case .system:
                    return Chat.Message.system(message.content)
                case .user:
                    return Chat.Message.user(message.content)
                case .assistant:
                    return Chat.Message.assistant(message.content)
                }
            }
            let input = try await container.prepare(
                input: UserInput(prompt: .chat(messages))
            )
            let isStructured =
                request.task == .deidentify
                || request.task == .entityExtraction
                || request.task == .relationExtraction
            let parameters = GenerateParameters(
                maxTokens: min(request.maximumTokens, 4_096),
                temperature: isStructured ? 0 : 0.1,
                topP: 1,
                topK: isStructured ? 0 : 50,
                repetitionPenalty: 1.1,
                prefill: PrefillParameters(stepSize: 256),
                seed: 0
            )
            try Task.checkCancellation()
            // Retain the producer task: cancelling only the stream consumer does
            // not guarantee that in-flight Metal work has finished before unload.
            let (stream, producer) = try await container.perform(nonSendable: input) { context, input in
                try Task.checkCancellation()
                let iterator = try TokenIterator(
                    input: input, model: context.model, parameters: parameters
                )
                return MLXLMCommon.generateTask(
                    promptTokenCount: input.text.tokens.size,
                    modelConfiguration: context.configuration,
                    tokenizer: context.tokenizer,
                    iterator: iterator
                )
            }
            return try await withTaskCancellationHandler {
                do {
                    let response = try await consume(
                        stream, request: request,
                        onReasoningChunk: onReasoningChunk,
                        onFinalAnswerChunk: onFinalAnswerChunk
                    )
                    await producer.value
                    try Task.checkCancellation()
                    return response
                } catch {
                    producer.cancel()
                    await producer.value
                    throw error
                }
            } onCancel: {
                producer.cancel()
            }
        }

        private nonisolated static func consume(
            _ stream: AsyncStream<Generation>,
            request: OpenMedLFMRequest,
            onReasoningChunk: (@Sendable (String) async -> Void)?,
            onFinalAnswerChunk: (@Sendable (String) async -> Void)?
        ) async throws -> OpenMedLFMResponse {
            var generatedText = ""
            var streamSplitter = OpenMedLFMStreamSplitter()
            let streamsText = request.task == .reasoning || request.task == .chat
            for await event in stream {
                try Task.checkCancellation()
                if case .chunk(let text) = event {
                    generatedText.append(text)
                    if streamsText {
                        for chunk in streamSplitter.consume(text) {
                            switch chunk {
                            case .reasoning(let reasoning):
                                await onReasoningChunk?(reasoning)
                            case .finalAnswer(let answer):
                                await onFinalAnswerChunk?(answer)
                            }
                        }
                    }
                }
            }
            try Task.checkCancellation()
            streamSplitter.finish()
            return try OpenMedLFMOutputParser.parse(
                generatedText,
                task: request.task,
                sourceDocument: request.document,
                allowedEntityLabels: request.entityLabels,
                allowedRelationLabels: request.relationLabels
            )
        }

        /// Releases model ownership. Call before switching to another large
        /// on-device model.
        public func unload() async {
            // Close admission before awaiting: actors are reentrant during awaits.
            container = nil
            let task = generationTask
            task?.cancel()
            _ = await task?.result
            OpenMed.clearRuntimeMemoryCache()
        }

        private nonisolated static func missingRequiredFiles(in directory: URL) -> [String] {
            requiredModelFiles.filter { file in
                let url = directory.appending(path: file)
                guard FileManager.default.fileExists(atPath: url.path) else { return true }
                let values = try? url.resourceValues(forKeys: [.fileSizeKey])
                return (values?.fileSize ?? 0) <= 0
            }
        }
    }
#endif
