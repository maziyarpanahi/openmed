#if canImport(MLX) && !os(watchOS) && !os(visionOS)
    import Foundation
    import MLX

    /// Errors raised before an OpenMedKit MLX runtime touches MLX.
    ///
    /// Every MLX entry point (token classification, zero-shot, Privacy Filter,
    /// Maple and Compass) throws ``unsupportedPlatform`` on the iOS Simulator
    /// and on Intel Macs instead of reaching mlx-swift, which aborts the
    /// process there. The error carries no path, model identifier or input.
    public enum OpenMedMLXRuntimeError: LocalizedError, Equatable, Sendable {
        /// MLX cannot run in this process: use a physical iPhone or iPad, an
        /// Apple silicon Mac, or the CoreML backend.
        case unsupportedPlatform

        public var errorDescription: String? {
            switch self {
            case .unsupportedPlatform:
                return "Swift MLX inference requires Apple Silicon macOS or a real iPhone/iPad device."
            }
        }
    }

    /// The single gate every MLX entry point checks before it creates an
    /// array, a model or a container.
    ///
    /// mlx-swift aborts the process on the iOS Simulator while it searches
    /// app bundles for its Metal library, so a Simulator build must refuse
    /// with ``OpenMedMLXRuntimeError/unsupportedPlatform`` instead.
    enum OpenMedMLXRuntimeSupport {
        /// Test seam: makes the current task behave like an iOS Simulator
        /// build. It can only withdraw MLX support, never grant it.
        @TaskLocal static var simulatesSimulator = false

        /// Whether MLX kernels can run in the current task.
        static var isSupported: Bool {
            supportsMLX(
                isSimulator: isSimulatorBuild || simulatesSimulator,
                isAppleSilicon: isAppleSiliconBuild
            )
        }

        /// MLX needs Apple silicon and cannot run on any simulator.
        static func supportsMLX(isSimulator: Bool, isAppleSilicon: Bool) -> Bool {
            isAppleSilicon && !isSimulator
        }

        /// Throws the typed unsupported error when MLX cannot run.
        static func requireSupported() throws {
            guard isSupported else {
                throw OpenMedMLXRuntimeError.unsupportedPlatform
            }
        }

        private static var isSimulatorBuild: Bool {
            #if targetEnvironment(simulator)
                true
            #else
                false
            #endif
        }

        private static var isAppleSiliconBuild: Bool {
            #if arch(arm64) || arch(arm64e)
                true
            #else
                false
            #endif
        }
    }

    final class MLXTokenClassificationPipeline {
        private let artifact: OpenMedMLXArtifact
        private let model: OpenMedBertForTokenClassification
        private let maxSeqLength: Int

        init(modelDirectoryURL: URL, maxSeqLength: Int = 512) throws {
            guard Self.isRuntimeSupported else {
                throw OpenMedMLXRuntimeError.unsupportedPlatform
            }

            let artifact = try OpenMedMLXArtifact(modelDirectoryURL: modelDirectoryURL)
            self.artifact = artifact
            self.model = try OpenMedMLXModelLoader.loadTokenClassifier(from: artifact)
            self.maxSeqLength = min(maxSeqLength, artifact.manifest.maxSequenceLength ?? maxSeqLength)
        }

        var tokenizerDirectoryURL: URL? {
            artifact.tokenizerDirectoryURL
        }

        var tokenizerName: String? {
            artifact.tokenizerName
        }

        var resolvedMaxSequenceLength: Int {
            maxSeqLength
        }

        func predict(
            inputIDs: [Int],
            attentionMask: [Int],
            tokenTypeIDs: [Int],
            offsets: [(Int, Int)],
            text: String,
            strategy: PostProcessing.AggregationStrategy = .average
        ) throws -> [EntityPrediction] {
            let sequenceLength = inputIDs.count

            let inputArray = MLXArray(inputIDs, [1, sequenceLength])
            let attentionArray = MLXArray(attentionMask, [1, sequenceLength]).asType(.float32)
            let tokenTypeArray: MLXArray? =
                artifact.configuration.typeVocabularySize > 0
                ? MLXArray(tokenTypeIDs, [1, sequenceLength])
                : nil

            let logits = model(
                inputArray,
                tokenTypeIDs: tokenTypeArray,
                attentionMask: attentionArray
            )
            eval(logits)

            let probabilities = softmax(logits[0], axis: -1)
            let predictions = probabilities.argMax(axis: -1)
            eval(probabilities, predictions)

            let flatProbabilities = probabilities.asArray(Float.self)
            let predictedLabelIDs = predictions.asArray(Int32.self).map(Int.init)
            let numLabels = artifact.configuration.numLabels

            var tokenPredictions = [PostProcessing.TokenPrediction]()
            tokenPredictions.reserveCapacity(sequenceLength)

            for tokenIndex in 0..<sequenceLength {
                let offset = offsets[tokenIndex]
                if offset == (0, 0) {
                    continue
                }

                let labelID = predictedLabelIDs[tokenIndex]
                let label = artifact.id2label[labelID] ?? "O"
                let score = flatProbabilities[tokenIndex * numLabels + labelID]

                tokenPredictions.append(
                    PostProcessing.TokenPrediction(
                        labelId: labelID,
                        label: label,
                        score: score,
                        startOffset: offset.0,
                        endOffset: offset.1
                    )
                )
            }

            return PostProcessing.decodeEntities(
                tokens: tokenPredictions,
                text: text,
                strategy: strategy
            )
        }

        static var isRuntimeSupported: Bool {
            OpenMedMLXRuntimeSupport.isSupported
        }
    }
#endif
