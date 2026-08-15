#if canImport(MLXLMCommon) && canImport(Tokenizers) && !os(watchOS) && !os(visionOS)
    import CoreImage
    import Foundation
    import Hub
    import MLX
    import MLXLMCommon
    import Tokenizers

    /// Generation result returned by ``OpenMedVisionLanguageModel``.
    public struct OpenMedVisionLanguageGeneration: Sendable {
        public let text: String
        public let tokenIDs: [Int]
        public let promptTokenCount: Int
        public let generationTokenCount: Int
        public let promptTime: TimeInterval
        public let generationTime: TimeInterval

        public init(
            text: String,
            tokenIDs: [Int] = [],
            promptTokenCount: Int,
            generationTokenCount: Int,
            promptTime: TimeInterval,
            generationTime: TimeInterval
        ) {
            self.text = text
            self.tokenIDs = tokenIDs
            self.promptTokenCount = promptTokenCount
            self.generationTokenCount = generationTokenCount
            self.promptTime = promptTime
            self.generationTime = generationTime
        }
    }

    /// First-class OpenMedKit runtime for OpenMed MLX vision-language models.
    ///
    /// Model weights are downloaded once and all inference remains on-device.
    /// A local-directory initializer is provided for fully offline and PHI-safe
    /// deployments.
    public final class OpenMedVisionLanguageModel: Sendable {
        public static let defaultModelID =
            "OpenMed/North-Micro-Vision-Instruct-4bit-mlx"

        private let container: ModelContainer

        private init(container: ModelContainer) {
            self.container = container
        }

        /// Load a public or authenticated OpenMed model from Hugging Face Hub.
        public static func load(
            modelID: String = defaultModelID,
            revision: String = "main",
            progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
        ) async throws -> OpenMedVisionLanguageModel {
            let modelDirectory = try await OpenMedHubDownloader().download(
                id: modelID,
                revision: revision,
                matching: ["*.json", "*.jinja", "*.safetensors"],
                useLatest: false,
                progressHandler: progressHandler
            )
            let container = try await OpenMedCompassLoader.loadContainer(
                from: modelDirectory
            )
            return OpenMedVisionLanguageModel(container: container)
        }

        /// Strictly load an already downloaded OpenMed model directory.
        public static func load(
            modelDirectory: URL
        ) async throws -> OpenMedVisionLanguageModel {
            let container = try await OpenMedCompassLoader.loadContainer(
                from: modelDirectory
            )
            return OpenMedVisionLanguageModel(container: container)
        }

        /// Generate from text only.
        public func generate(
            _ prompt: String,
            maxTokens: Int = 256,
            temperature: Float = 0
        ) async throws -> OpenMedVisionLanguageGeneration {
            try await generate(
                prompt,
                images: [],
                maxTokens: maxTokens,
                temperature: temperature
            )
        }

        /// Generate from text and one local image URL.
        public func generate(
            _ prompt: String,
            imageURL: URL,
            maxTokens: Int = 256,
            temperature: Float = 0
        ) async throws -> OpenMedVisionLanguageGeneration {
            try await generate(
                prompt,
                images: [.url(imageURL)],
                maxTokens: maxTokens,
                temperature: temperature
            )
        }

        /// Generate from text and one in-memory Core Image value.
        public func generate(
            _ prompt: String,
            image: CIImage,
            maxTokens: Int = 256,
            temperature: Float = 0
        ) async throws -> OpenMedVisionLanguageGeneration {
            try await generate(
                prompt,
                images: [.ciImage(image)],
                maxTokens: maxTokens,
                temperature: temperature
            )
        }

        /// Generate from text and zero or more images.
        public func generate(
            _ prompt: String,
            images: [UserInput.Image],
            maxTokens: Int = 256,
            temperature: Float = 0
        ) async throws -> OpenMedVisionLanguageGeneration {
            guard maxTokens > 0 else {
                throw OpenMedCompassError.invalidGenerationLimit(maxTokens)
            }
            let input = UserInput(prompt: prompt, images: images)
            let prepared = try await container.prepare(input: input)
            return try await generate(
                prepared: prepared,
                maxTokens: maxTokens,
                temperature: temperature
            )
        }

        private func generate(
            prepared: consuming sending LMInput,
            maxTokens: Int,
            temperature: Float
        ) async throws -> OpenMedVisionLanguageGeneration {
            let parameters = GenerateParameters(
                maxTokens: maxTokens,
                temperature: temperature,
                topP: temperature == 0 ? 1 : 0.8,
                topK: temperature == 0 ? 0 : 20,
                prefillStepSize: 2_048
            )
            let stream = try await container.perform(nonSendable: prepared) {
                context,
                input in
                try MLXLMCommon.generateTokens(
                    input: input,
                    parameters: parameters,
                    context: context
                )
            }
            var tokenIDs = [Int]()
            var completion: GenerateCompletionInfo?
            for await event in stream {
                switch event {
                case .token(let token):
                    tokenIDs.append(token)
                case .info(let info):
                    completion = info
                }
            }
            let text = await container.decode(tokenIds: tokenIDs)
            return OpenMedVisionLanguageGeneration(
                text: text.trimmingCharacters(in: .whitespacesAndNewlines),
                tokenIDs: tokenIDs,
                promptTokenCount: completion?.promptTokenCount ?? 0,
                generationTokenCount: completion?.generationTokenCount ?? 0,
                promptTime: completion?.promptTime ?? 0,
                generationTime: completion?.generateTime ?? 0
            )
        }

    }

    enum OpenMedCompassError: LocalizedError {
        case invalidAspectRatio(Double)
        case invalidDimensions(Int, Int)
        case invalidGenerationLimit(Int)
        case imageTokenMissing
        case imageTokenCountMismatch
        case videoUnsupported

        var errorDescription: String? {
            switch self {
            case .invalidAspectRatio(let ratio):
                "Image aspect ratio must not exceed 200 (got \(ratio))."
            case .invalidDimensions(let height, let width):
                "Image dimensions must be positive (got \(width)×\(height))."
            case .invalidGenerationLimit(let value):
                "maxTokens must be positive (got \(value))."
            case .imageTokenMissing:
                "The Compass tokenizer does not define <|IMAGE_PAD|>."
            case .imageTokenCountMismatch:
                "The number of image placeholders does not match the images."
            case .videoUnsupported:
                "OpenMed Compass currently accepts still images, not video."
            }
        }
    }

    struct OpenMedCompassProcessorConfiguration: Codable, Sendable {
        let processorClass: String
        let imageMean: [CGFloat]
        let imageStd: [CGFloat]
        let mergeSize: Int
        let patchSize: Int
        let temporalPatchSize: Int
        let minPixels: Int
        let maxPixels: Int

        enum CodingKeys: String, CodingKey {
            case processorClass = "processor_class"
            case imageMean = "image_mean"
            case imageStd = "image_std"
            case mergeSize = "merge_size"
            case patchSize = "patch_size"
            case temporalPatchSize = "temporal_patch_size"
            case minPixels = "min_pixels"
            case maxPixels = "max_pixels"
        }

        init(from decoder: Swift.Decoder) throws {
            let values = try decoder.container(keyedBy: CodingKeys.self)
            processorClass = try values.decode(String.self, forKey: .processorClass)
            imageMean =
                try values.decodeIfPresent([CGFloat].self, forKey: .imageMean)
                ?? [0.5, 0.5, 0.5]
            imageStd =
                try values.decodeIfPresent([CGFloat].self, forKey: .imageStd)
                ?? [0.5, 0.5, 0.5]
            mergeSize = try values.decodeIfPresent(Int.self, forKey: .mergeSize) ?? 2
            patchSize = try values.decodeIfPresent(Int.self, forKey: .patchSize) ?? 16
            temporalPatchSize =
                try values.decodeIfPresent(Int.self, forKey: .temporalPatchSize) ?? 2
            // The exported preprocessor's legacy `size` values are byte-like
            // bounds. Compass uses these architecture defaults unless explicit
            // min_pixels/max_pixels values exist.
            minPixels = try values.decodeIfPresent(Int.self, forKey: .minPixels) ?? 16_384
            maxPixels =
                try values.decodeIfPresent(Int.self, forKey: .maxPixels) ?? 3_868_706
        }
    }

    struct OpenMedCompassProcessor: UserInputProcessor, @unchecked Sendable {
        private let configuration: OpenMedCompassProcessorConfiguration
        private let tokenizer: any MLXLMCommon.Tokenizer

        init(
            _ configuration: OpenMedCompassProcessorConfiguration,
            tokenizer: any MLXLMCommon.Tokenizer
        ) {
            self.configuration = configuration
            self.tokenizer = tokenizer
        }

        static func targetSize(
            height: Int,
            width: Int,
            factor: Int,
            minPixels: Int = 16_384,
            maxPixels: Int = 3_868_706
        ) throws -> (Int, Int) {
            guard height > 0, width > 0, factor > 0 else {
                throw OpenMedCompassError.invalidDimensions(height, width)
            }
            let ratio = Double(max(height, width)) / Double(min(height, width))
            guard ratio <= 200 else {
                throw OpenMedCompassError.invalidAspectRatio(ratio)
            }
            var resizedHeight =
                Int((Double(height) / Double(factor)).rounded(.toNearestOrEven)) * factor
            var resizedWidth =
                Int((Double(width) / Double(factor)).rounded(.toNearestOrEven)) * factor
            if resizedHeight * resizedWidth > maxPixels {
                let scale = sqrt(Double(height * width) / Double(maxPixels))
                resizedHeight = max(
                    factor,
                    Int(floor(Double(height) / scale / Double(factor))) * factor
                )
                resizedWidth = max(
                    factor,
                    Int(floor(Double(width) / scale / Double(factor))) * factor
                )
            } else if resizedHeight * resizedWidth < minPixels {
                let scale = sqrt(Double(minPixels) / Double(height * width))
                resizedHeight =
                    Int(ceil(Double(height) * scale / Double(factor))) * factor
                resizedWidth =
                    Int(ceil(Double(width) * scale / Double(factor))) * factor
            }
            return (resizedHeight, resizedWidth)
        }

        private var mean: (CGFloat, CGFloat, CGFloat) {
            (configuration.imageMean[0], configuration.imageMean[1], configuration.imageMean[2])
        }

        private var standardDeviation: (CGFloat, CGFloat, CGFloat) {
            (configuration.imageStd[0], configuration.imageStd[1], configuration.imageStd[2])
        }

        private func patchify(_ image: MLXArray) -> (MLXArray, THW) {
            let height = image.dim(2)
            let width = image.dim(3)
            let patch = configuration.patchSize
            let merge = configuration.mergeSize
            let temporal = configuration.temporalPatchSize
            let gridHeight = height / patch
            let gridWidth = width / patch
            let channels = image.dim(1)
            var values = repeated(
                image.expandedDimensions(axis: 1),
                count: temporal,
                axis: 1
            )
            values = values.reshaped(
                1,
                1,
                temporal,
                channels,
                gridHeight / merge,
                merge,
                patch,
                gridWidth / merge,
                merge,
                patch
            )
            values = values.transposed(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
            values = values.reshaped(
                gridHeight * gridWidth,
                channels * temporal * patch * patch
            )
            return (values, THW(1, gridHeight, gridWidth))
        }

        private func preprocess(
            _ image: UserInput.Image,
            processing: UserInput.Processing
        ) throws -> (MLXArray, THW) {
            var ciImage = try image.asCIImage()
            ciImage = OpenMedImageProcessing.apply(ciImage, processing: processing)
            let factor = configuration.patchSize * configuration.mergeSize
            let size = ciImage.extent.size
            let (height, width) = try Self.targetSize(
                height: Int(size.height.rounded()),
                width: Int(size.width.rounded()),
                factor: factor,
                minPixels: processing.minPixels ?? configuration.minPixels,
                maxPixels: processing.maxPixels ?? configuration.maxPixels
            )
            ciImage = OpenMedImageProcessing.inSRGBToneCurveSpace(ciImage)
            ciImage = OpenMedImageProcessing.resampleBicubic(
                ciImage,
                to: CGSize(width: width, height: height)
            )
            // PIL's bicubic resize, used by the reference processor, clips its
            // uint8 result before normalization. Core Image's cubic kernel can
            // overshoot the [0, 1] component range unless we clamp explicitly.
            ciImage = ciImage.applyingFilter(
                "CIColorClamp",
                parameters: [
                    "inputMinComponents": CIVector(x: 0, y: 0, z: 0, w: 0),
                    "inputMaxComponents": CIVector(x: 1, y: 1, z: 1, w: 1),
                ]
            )
            ciImage = OpenMedImageProcessing.normalize(
                ciImage,
                mean: mean,
                std: standardDeviation
            )
            return patchify(OpenMedImageProcessing.asMLXArray(ciImage))
        }

        private func messages(for input: UserInput) -> [MLXLMCommon.Message] {
            var messages = DefaultMessageGenerator().generate(from: input)
            guard !input.images.isEmpty else { return messages }
            guard
                let userIndex = messages.lastIndex(where: {
                    ($0["role"] as? String) == "user"
                })
            else { return messages }
            let text =
                messages[userIndex]["content"] as? String
                ?? input.prompt.description
            var content = input.images.map { _ -> [String: any Sendable] in
                ["type": "image"]
            }
            content.append(["type": "text", "text": text, "content": text])
            messages[userIndex]["content"] = content
            return messages
        }

        func prepare(input: UserInput) async throws -> LMInput {
            guard input.videos.isEmpty else {
                throw OpenMedCompassError.videoUnsupported
            }
            var promptTokens = try tokenizer.applyChatTemplate(
                messages: messages(for: input),
                tools: input.tools,
                additionalContext: input.additionalContext
            )
            guard !input.images.isEmpty else {
                let tokens = MLXArray(promptTokens).expandedDimensions(axis: 0)
                return LMInput(
                    text: .init(tokens: tokens, mask: ones(like: tokens).asType(.int8))
                )
            }
            guard let imageToken = tokenizer.convertTokenToId("<|IMAGE_PAD|>") else {
                throw OpenMedCompassError.imageTokenMissing
            }
            let processed = try input.images.map {
                try preprocess($0, processing: input.processing)
            }
            var searchStart = 0
            for (_, frame) in processed {
                guard
                    let index = promptTokens[searchStart...].firstIndex(of: imageToken)
                else {
                    throw OpenMedCompassError.imageTokenCountMismatch
                }
                let count = frame.product / (configuration.mergeSize * configuration.mergeSize)
                promptTokens.replaceSubrange(
                    index...index,
                    with: repeatElement(imageToken, count: count)
                )
                searchStart = index + count
            }
            if promptTokens[searchStart...].contains(imageToken) {
                throw OpenMedCompassError.imageTokenCountMismatch
            }
            let tokens = MLXArray(promptTokens).expandedDimensions(axis: 0)
            return LMInput(
                text: .init(tokens: tokens, mask: ones(like: tokens).asType(.int8)),
                image: .init(
                    pixels: concatenated(processed.map(\.0), axis: 0),
                    frames: processed.map(\.1)
                )
            )
        }
    }

    private enum OpenMedImageProcessing {
        static func apply(
            _ image: CIImage,
            processing: UserInput.Processing
        ) -> CIImage {
            guard let target = processing.resize else { return image }
            let scale = min(
                target.width / image.extent.width,
                target.height / image.extent.height
            )
            return image.transformed(
                by: CGAffineTransform(scaleX: scale, y: scale)
            )
        }

        static func inSRGBToneCurveSpace(_ image: CIImage) -> CIImage {
            image.applyingFilter("CILinearToSRGBToneCurve")
        }

        static func resampleBicubic(_ image: CIImage, to size: CGSize) -> CIImage {
            let verticalScale = size.height / image.extent.height
            let horizontalScale = size.width / image.extent.width
            return image.applyingFilter(
                "CIBicubicScaleTransform",
                parameters: [
                    "inputScale": verticalScale,
                    "inputAspectRatio": horizontalScale / verticalScale,
                ]
            ).cropped(
                to: CGRect(origin: .zero, size: size)
            )
        }

        static func normalize(
            _ image: CIImage,
            mean: (CGFloat, CGFloat, CGFloat),
            std: (CGFloat, CGFloat, CGFloat)
        ) -> CIImage {
            image.applyingFilter(
                "CIColorMatrix",
                parameters: [
                    "inputRVector": CIVector(x: 1 / std.0, y: 0, z: 0, w: 0),
                    "inputGVector": CIVector(x: 0, y: 1 / std.1, z: 0, w: 0),
                    "inputBVector": CIVector(x: 0, y: 0, z: 1 / std.2, w: 0),
                    "inputAVector": CIVector(x: 0, y: 0, z: 0, w: 1),
                    "inputBiasVector": CIVector(
                        x: -mean.0 / std.0,
                        y: -mean.1 / std.1,
                        z: -mean.2 / std.2,
                        w: 0
                    ),
                ]
            )
        }

        static func asMLXArray(_ image: CIImage) -> MLXArray {
            let width = Int(image.extent.width.rounded())
            let height = Int(image.extent.height.rounded())
            let components = 4
            let bytesPerComponent = MemoryLayout<Float32>.size
            let bytesPerRow = width * components * bytesPerComponent
            var data = Data(count: height * bytesPerRow)
            let context = CIContext(options: [.cacheIntermediates: false])
            data.withUnsafeMutableBytes { bytes in
                guard let address = bytes.baseAddress else { return }
                context.render(
                    image,
                    toBitmap: address,
                    rowBytes: bytesPerRow,
                    bounds: image.extent,
                    format: .RGBAf,
                    colorSpace: nil
                )
            }
            var array = MLXArray(data, [height, width, components], type: Float32.self)
            array = array[0..., 0..., ..<3]
            return array.reshaped(1, height, width, 3).transposed(0, 3, 1, 2)
        }
    }

    private enum OpenMedCompassLoader {
        private static let defaultPrompt =
            "Describe the image accurately and concisely."

        static func loadContainer(from directory: URL) async throws -> ModelContainer {
            let configurationData = try Data(
                contentsOf: directory.appending(path: "config.json")
            )
            let baseConfiguration = try JSONDecoder().decode(
                BaseConfiguration.self,
                from: configurationData
            )
            let compassConfiguration = try JSONDecoder().decode(
                OpenMedCompassConfiguration.self,
                from: configurationData
            )
            try compassConfiguration.validate()

            let model = OpenMedCompassModel(compassConfiguration)
            try loadWeights(
                modelDirectory: directory,
                model: model,
                perLayerQuantization: baseConfiguration.perLayerQuantization
            )

            let tokenizer = try await OpenMedTokenizerLoader().load(from: directory)
            let processorURL = processorConfigurationURL(in: directory)
            let processorConfiguration = try JSONDecoder().decode(
                OpenMedCompassProcessorConfiguration.self,
                from: Data(contentsOf: processorURL)
            )
            guard processorConfiguration.processorClass == "CohereCompassProcessor" else {
                throw ModelFactoryError.invalidConfiguration(
                    "unsupported processor \(processorConfiguration.processorClass)"
                )
            }
            let processor = OpenMedCompassProcessor(
                processorConfiguration,
                tokenizer: tokenizer
            )

            var endTokenIDs = Set(baseConfiguration.eosTokenIds?.values ?? [])
            var stopStrings = Set<String>()
            let generationURL = directory.appending(path: "generation_config.json")
            if let data = try? Data(contentsOf: generationURL),
                let generation = try? JSONDecoder().decode(
                    GenerationConfigFile.self,
                    from: data
                )
            {
                if let values = generation.eosTokenIds?.values {
                    endTokenIDs = Set(values)
                }
                stopStrings.formUnion(generation.stopStrings)
            }
            let modelConfiguration = ModelConfiguration(
                directory: directory,
                defaultPrompt: defaultPrompt,
                stopStrings: stopStrings,
                eosTokenIds: endTokenIDs
            )
            return ModelContainer(
                context: ModelContext(
                    configuration: modelConfiguration,
                    model: model,
                    processor: processor,
                    tokenizer: tokenizer
                )
            )
        }

        private static func processorConfigurationURL(in directory: URL) -> URL {
            let preferred = directory.appending(path: "preprocessor_config.json")
            if FileManager.default.fileExists(atPath: preferred.path) {
                return preferred
            }
            return directory.appending(path: "processor_config.json")
        }
    }

    private struct OpenMedHubDownloader: Downloader, @unchecked Sendable {
        private let hub = HubApi.shared

        func download(
            id: String,
            revision: String?,
            matching patterns: [String],
            useLatest: Bool,
            progressHandler: @Sendable @escaping (Progress) -> Void
        ) async throws -> URL {
            _ = useLatest
            return try await hub.snapshot(
                from: id,
                revision: revision ?? "main",
                matching: patterns,
                progressHandler: progressHandler
            )
        }
    }

    private struct OpenMedTokenizerLoader: TokenizerLoader, Sendable {
        func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
            let tokenizer = try await Tokenizers.AutoTokenizer.from(
                modelFolder: directory
            )
            return OpenMedTokenizerBridge(tokenizer)
        }
    }

    private struct OpenMedTokenizerBridge: MLXLMCommon.Tokenizer, @unchecked Sendable {
        private let tokenizer: any Tokenizers.Tokenizer

        init(_ tokenizer: any Tokenizers.Tokenizer) {
            self.tokenizer = tokenizer
        }

        func encode(text: String, addSpecialTokens: Bool) -> [Int] {
            tokenizer.encode(text: text, addSpecialTokens: addSpecialTokens)
        }

        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
            tokenizer.decode(
                tokens: tokenIds,
                skipSpecialTokens: skipSpecialTokens
            )
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
            let upstreamMessages = messages.map { message in
                message.reduce(into: [String: Any]()) { result, entry in
                    result[entry.key] = entry.value
                }
            }
            let upstreamTools = tools?.map { tool in
                tool.reduce(into: [String: Any]()) { result, entry in
                    result[entry.key] = entry.value
                }
            }
            let upstreamContext = additionalContext?.reduce(
                into: [String: Any](),
                { result, entry in result[entry.key] = entry.value }
            )
            do {
                return try tokenizer.applyChatTemplate(
                    messages: upstreamMessages,
                    chatTemplate: nil,
                    addGenerationPrompt: true,
                    truncation: false,
                    maxLength: nil,
                    tools: upstreamTools,
                    additionalContext: upstreamContext
                )
            } catch Tokenizers.TokenizerError.missingChatTemplate {
                throw MLXLMCommon.TokenizerError.missingChatTemplate
            }
        }
    }

#endif
