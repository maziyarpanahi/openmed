import Foundation
import MLX
import MLXNN

private func openMedFloatScalar(_ value: Float, like array: MLXArray) -> MLXArray {
    MLXArray(value).asType(array.dtype)
}

struct OpenMedGLiNERSpanOutput {
    let logits: MLXArray
    let spanIndex: MLXArray
    let spanMask: MLXArray
    let promptMask: MLXArray
    let wordMask: MLXArray
}

struct OpenMedGLiClassOutput {
    let logits: MLXArray
    let classesMask: MLXArray
}

struct OpenMedGLiNERRelexEntityOutput {
    let entityScores: MLXArray
    let entityPromptMask: MLXArray
    let wordMask: MLXArray
    let encoded: OpenMedGLiNERRelexEncoded
}

struct OpenMedGLiNERRelexRelationOutput {
    let pairScores: MLXArray
    let pairIndex: MLXArray
    let pairMask: MLXArray
    let relationPromptMask: MLXArray
}

struct OpenMedGLiNERRelexEncoded {
    let entityPrompts: MLXArray
    let entityPromptMask: MLXArray
    let relationPrompts: MLXArray
    let relationPromptMask: MLXArray
    let wordsEmbedding: MLXArray
    let wordMask: MLXArray
}

final class OpenMedProjectionMLP: Module {
    @ModuleInfo(key: "linear1") var linear1: Linear
    @ModuleInfo(key: "linear2") var linear2: Linear

    init(inputSize: Int, outputSize: Int? = nil) {
        let outputSize = outputSize ?? inputSize
        _linear1.wrappedValue = Linear(inputSize, outputSize * 4)
        _linear2.wrappedValue = Linear(outputSize * 4, outputSize)
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        linear2(relu(linear1(input)))
    }
}

final class OpenMedBidirectionalLSTM: Module {
    @ModuleInfo(key: "forward_lstm") var forwardLSTM: LSTM
    @ModuleInfo(key: "backward_lstm") var backwardLSTM: LSTM

    init(hiddenSize: Int) {
        _forwardLSTM.wrappedValue = LSTM(inputSize: hiddenSize, hiddenSize: hiddenSize / 2)
        _backwardLSTM.wrappedValue = LSTM(inputSize: hiddenSize, hiddenSize: hiddenSize / 2)
    }

    private static func reversePadded(_ input: MLXArray, lengths: MLXArray) -> MLXArray {
        let steps = MLXArray.arange(input.dim(1), dtype: .int32).expandedDimensions(axis: 0)
        let expandedLengths = lengths.expandedDimensions(axis: 1)
        let gatherIndex = `where`(
            steps .< expandedLengths,
            expandedLengths - 1 - steps,
            steps
        )
        return takeAlong(input, gatherIndex.expandedDimensions(axis: 2), axis: 1)
    }

    func callAsFunction(_ input: MLXArray, mask: MLXArray) -> MLXArray {
        let lengths = sum(mask.asType(.int32), axis: 1)
        let (forwardOutput, _) = forwardLSTM(input)
        let reversedInput = Self.reversePadded(input, lengths: lengths)
        let (backwardReversed, _) = backwardLSTM(reversedInput)
        let backwardOutput = Self.reversePadded(backwardReversed, lengths: lengths)
        return concatenated([forwardOutput, backwardOutput], axis: -1)
            * mask.asType(input.dtype).expandedDimensions(axis: -1)
    }
}

private func openMedPadEmbeddings(_ rows: [MLXArray], width: Int, embedDim: Int) -> MLXArray {
    guard !rows.isEmpty else {
        return MLXArray.zeros([0, width, embedDim], type: Float.self)
    }

    let paddedRows = rows.map { row -> MLXArray in
        let padLength = width - row.dim(0)
        guard padLength > 0 else {
            return row
        }
        let padding = MLXArray.zeros([padLength, row.dim(-1)], type: Float.self).asType(row.dtype)
        return concatenated([row, padding], axis: 0)
    }
    return stacked(paddedRows, axis: 0)
}

private func openMedPadMaskRows(_ rows: [[Int]], width: Int) -> MLXArray {
    let padded = rows.map { row in
        row + Array(repeating: 0, count: max(0, width - row.count))
    }
    return MLXArray(padded.flatMap { $0 }, [rows.count, width]).asType(.bool)
}

private func openMedExtractMarkerEmbeddings(
    tokenEmbeddings: MLXArray,
    inputIDs: MLXArray,
    markerTokenID: Int,
    includeMarkerToken: Bool
) -> (MLXArray, MLXArray) {
    let batchSize = tokenEmbeddings.dim(0)
    let seqLen = tokenEmbeddings.dim(1)
    let embedDim = tokenEmbeddings.dim(2)
    var rows = [MLXArray]()
    var maskRows = [[Int]]()
    var maxItems = 0

    for batchIndex in 0..<batchSize {
        let rowIDs = inputIDs[batchIndex].asArray(Int32.self).map(Int.init)
        let rawPositions = rowIDs.enumerated().compactMap { index, tokenID in
            tokenID == markerTokenID ? index : nil
        }

        guard !rawPositions.isEmpty else {
            rows.append(MLXArray.zeros([0, embedDim], type: Float.self).asType(tokenEmbeddings.dtype))
            maskRows.append([])
            continue
        }

        let positions = rawPositions.map {
            includeMarkerToken ? $0 : min($0 + 1, seqLen - 1)
        }
        let positionArray = MLXArray(positions.map(Int32.init), [positions.count])
        let row = tokenEmbeddings[batchIndex].take(positionArray, axis: 0)
        rows.append(row)
        maxItems = max(maxItems, positions.count)
        maskRows.append(Array(repeating: 1, count: positions.count))
    }

    maxItems = max(maxItems, 1)
    return (
        openMedPadEmbeddings(rows, width: maxItems, embedDim: embedDim),
        openMedPadMaskRows(maskRows, width: maxItems)
    )
}

private func openMedExtractWordEmbeddings(
    tokenEmbeddings: MLXArray,
    wordsMask: MLXArray
) -> (MLXArray, MLXArray) {
    let batchSize = tokenEmbeddings.dim(0)
    let embedDim = tokenEmbeddings.dim(2)
    var rows = [MLXArray]()
    var maskRows = [[Int]]()
    var maxWords = 0

    for batchIndex in 0..<batchSize {
        let wordMask = wordsMask[batchIndex].asArray(Int32.self).map(Int.init)
        let tokenPositions = wordMask.enumerated().compactMap { index, wordIndex in
            wordIndex > 0 ? index : nil
        }

        guard !tokenPositions.isEmpty else {
            rows.append(MLXArray.zeros([0, embedDim], type: Float.self).asType(tokenEmbeddings.dtype))
            maskRows.append([])
            continue
        }

        let positionArray = MLXArray(tokenPositions.map(Int32.init), [tokenPositions.count])
        let row = tokenEmbeddings[batchIndex].take(positionArray, axis: 0)
        rows.append(row)
        maxWords = max(maxWords, tokenPositions.count)
        maskRows.append(Array(repeating: 1, count: tokenPositions.count))
    }

    maxWords = max(maxWords, 1)
    return (
        openMedPadEmbeddings(rows, width: maxWords, embedDim: embedDim),
        openMedPadMaskRows(maskRows, width: maxWords)
    )
}

private func openMedGatherSpanEndpoints(
    startHiddenStates: MLXArray,
    endHiddenStates: MLXArray,
    spanIndex: MLXArray
) -> (MLXArray, MLXArray) {
    let hiddenSize = startHiddenStates.dim(-1)
    let startIndex = broadcast(
        spanIndex[0..., 0..., 0].expandedDimensions(axis: -1),
        to: [spanIndex.dim(0), spanIndex.dim(1), hiddenSize]
    )
    let endIndex = broadcast(
        spanIndex[0..., 0..., 1].expandedDimensions(axis: -1),
        to: [spanIndex.dim(0), spanIndex.dim(1), hiddenSize]
    )
    return (
        takeAlong(startHiddenStates, startIndex, axis: 1),
        takeAlong(endHiddenStates, endIndex, axis: 1)
    )
}

private func openMedBuildAllEntityPairs(
    spanRep: MLXArray,
    spanMask: MLXArray
) -> (MLXArray, MLXArray, MLXArray, MLXArray) {
    let batchSize = spanRep.dim(0)
    let embedDim = spanRep.dim(2)
    var pairRows = [[[Int]]]()
    var pairMaskRows = [[Int]]()
    var headRows = [MLXArray]()
    var tailRows = [MLXArray]()
    var maxPairs = 0

    for batchIndex in 0..<batchSize {
        let spanFlags = spanMask[batchIndex].asArray(Bool.self)
        let entityCount = spanFlags.prefix { $0 }.count
        let pairs = (0..<entityCount).flatMap { head in
            (0..<entityCount).compactMap { tail in
                head == tail ? nil : [head, tail]
            }
        }
        pairRows.append(pairs)
        pairMaskRows.append(Array(repeating: 1, count: pairs.count))
        maxPairs = max(maxPairs, pairs.count)

        guard !pairs.isEmpty else {
            headRows.append(MLXArray.zeros([0, embedDim], type: Float.self).asType(spanRep.dtype))
            tailRows.append(MLXArray.zeros([0, embedDim], type: Float.self).asType(spanRep.dtype))
            continue
        }

        let headIndex = MLXArray(pairs.map { Int32($0[0]) }, [pairs.count])
        let tailIndex = MLXArray(pairs.map { Int32($0[1]) }, [pairs.count])
        headRows.append(spanRep[batchIndex].take(headIndex, axis: 0))
        tailRows.append(spanRep[batchIndex].take(tailIndex, axis: 0))
    }

    maxPairs = max(maxPairs, 1)
    let paddedPairs = pairRows.map { row in
        row + Array(repeating: [0, 0], count: max(0, maxPairs - row.count))
    }
    let flatPairs = paddedPairs.flatMap { $0 }.flatMap { $0 }.map(Int32.init)

    return (
        MLXArray(flatPairs, [batchSize, maxPairs, 2]),
        openMedPadMaskRows(pairMaskRows, width: maxPairs),
        openMedPadEmbeddings(headRows, width: maxPairs, embedDim: embedDim),
        openMedPadEmbeddings(tailRows, width: maxPairs, embedDim: embedDim)
    )
}

final class OpenMedSpanMarkerV0: Module {
    @ModuleInfo(key: "project_start") var projectStart: OpenMedProjectionMLP
    @ModuleInfo(key: "project_end") var projectEnd: OpenMedProjectionMLP
    @ModuleInfo(key: "out_project") var outputProjection: OpenMedProjectionMLP

    init(hiddenSize: Int) {
        _projectStart.wrappedValue = OpenMedProjectionMLP(inputSize: hiddenSize)
        _projectEnd.wrappedValue = OpenMedProjectionMLP(inputSize: hiddenSize)
        _outputProjection.wrappedValue = OpenMedProjectionMLP(
            inputSize: hiddenSize * 2,
            outputSize: hiddenSize
        )
    }

    func callAsFunction(_ hiddenStates: MLXArray, spanIndex: MLXArray) -> MLXArray {
        let startRep = projectStart(hiddenStates)
        let endRep = projectEnd(hiddenStates)
        let (startSpanRep, endSpanRep) = openMedGatherSpanEndpoints(
            startHiddenStates: startRep,
            endHiddenStates: endRep,
            spanIndex: spanIndex
        )
        return outputProjection(relu(concatenated([startSpanRep, endSpanRep], axis: -1)))
    }
}

final class OpenMedGLiNERSpanModel: Module {
    private let configuration: OpenMedMLXBertConfiguration

    @ModuleInfo(key: "deberta") var deberta: OpenMedDebertaV2Model
    @ModuleInfo(key: "token_projection") var tokenProjection: Linear
    @ModuleInfo(key: "rnn") var rnn: OpenMedBidirectionalLSTM?
    @ModuleInfo(key: "span_rep_layer") var spanRepLayer: OpenMedSpanMarkerV0
    @ModuleInfo(key: "prompt_rep_layer") var promptRepLayer: OpenMedProjectionMLP

    init(_ configuration: OpenMedMLXBertConfiguration) {
        self.configuration = configuration
        _deberta.wrappedValue = OpenMedDebertaV2Model(configuration)
        _tokenProjection.wrappedValue = Linear(
            configuration.encoderHiddenSize,
            configuration.hiddenSize
        )
        if configuration.numRNNLayers > 0 {
            _rnn.wrappedValue = OpenMedBidirectionalLSTM(hiddenSize: configuration.hiddenSize)
        }
        _spanRepLayer.wrappedValue = OpenMedSpanMarkerV0(hiddenSize: configuration.hiddenSize)
        _promptRepLayer.wrappedValue = OpenMedProjectionMLP(inputSize: configuration.hiddenSize)
    }

    private func encode(
        inputIDs: MLXArray,
        attentionMask: MLXArray,
        wordsMask: MLXArray
    ) -> (MLXArray, MLXArray, MLXArray, MLXArray) {
        let hiddenStates = deberta(inputIDs: inputIDs, attentionMask: attentionMask)
        let markerTokenID = configuration.classTokenIndex ?? 0
        let (promptEmbeddings, promptMask) = openMedExtractMarkerEmbeddings(
            tokenEmbeddings: hiddenStates,
            inputIDs: inputIDs,
            markerTokenID: markerTokenID,
            includeMarkerToken: configuration.embedEntityToken
        )
        let (wordEmbeddings, wordMask) = openMedExtractWordEmbeddings(
            tokenEmbeddings: hiddenStates,
            wordsMask: wordsMask
        )

        var projectedWords = tokenProjection(wordEmbeddings)
        let projectedPrompts = tokenProjection(promptEmbeddings)
        if let rnn {
            projectedWords = rnn(projectedWords, mask: wordMask)
        }
        return (projectedPrompts, promptMask, projectedWords, wordMask)
    }

    func callAsFunction(
        inputIDs: MLXArray,
        attentionMask: MLXArray,
        wordsMask: MLXArray,
        spanIndex: MLXArray,
        spanMask: MLXArray
    ) -> OpenMedGLiNERSpanOutput {
        let maskedSpanIndex = spanIndex * spanMask.asType(spanIndex.dtype).expandedDimensions(axis: -1)
        let (promptEmbeddings, promptMask, wordEmbeddings, wordMask) = encode(
            inputIDs: inputIDs,
            attentionMask: attentionMask,
            wordsMask: wordsMask
        )
        let spanRep = spanRepLayer(wordEmbeddings, spanIndex: maskedSpanIndex)
        let promptRep = promptRepLayer(promptEmbeddings)
        var logits = einsum("bsd,bcd->bsc", spanRep, promptRep)
        logits = `where`(
            spanMask.expandedDimensions(axis: -1),
            logits,
            openMedFloatScalar(-1.0e9, like: logits)
        )
        return OpenMedGLiNERSpanOutput(
            logits: logits,
            spanIndex: maskedSpanIndex,
            spanMask: spanMask,
            promptMask: promptMask,
            wordMask: wordMask
        )
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        weights.filter { key, _ in
            key != "deberta.embeddings.position_ids" && !key.hasPrefix("_")
        }
    }
}

final class OpenMedGLiClassFeaturesProjector: Module {
    @ModuleInfo(key: "linear1") var linear1: Linear
    @ModuleInfo(key: "linear2") var linear2: Linear

    init(encoderHiddenSize: Int, hiddenSize: Int) {
        _linear1.wrappedValue = Linear(encoderHiddenSize, hiddenSize)
        _linear2.wrappedValue = Linear(hiddenSize, encoderHiddenSize)
    }

    func callAsFunction(_ features: MLXArray) -> MLXArray {
        linear2(gelu(linear1(features)))
    }
}

final class OpenMedGLiClassMLPScorer: Module {
    @ModuleInfo(key: "linear1") var linear1: Linear
    @ModuleInfo(key: "linear2") var linear2: Linear
    @ModuleInfo(key: "linear3") var linear3: Linear

    init(hiddenSize: Int) {
        _linear1.wrappedValue = Linear(hiddenSize * 2, 256)
        _linear2.wrappedValue = Linear(256, 128)
        _linear3.wrappedValue = Linear(128, 1)
    }

    func callAsFunction(textRep: MLXArray, labelRep: MLXArray) -> MLXArray {
        let batchSize = labelRep.dim(0)
        let numLabels = labelRep.dim(1)
        let dim = labelRep.dim(2)
        let expandedText = broadcast(
            textRep.expandedDimensions(axis: 1),
            to: [batchSize, numLabels, dim]
        )
        let combined = concatenated([expandedText, labelRep], axis: -1)
        return linear3(relu(linear2(relu(linear1(combined))))).squeezed(axis: -1)
    }
}

final class OpenMedGLiClassUniEncoderModel: Module {
    private let configuration: OpenMedMLXBertConfiguration

    @ModuleInfo(key: "deberta") var deberta: OpenMedDebertaV2Model
    @ModuleInfo(key: "classes_projector") var classesProjector: OpenMedGLiClassFeaturesProjector
    @ModuleInfo(key: "text_projector") var textProjector: OpenMedGLiClassFeaturesProjector
    @ModuleInfo(key: "segment_embeddings") var segmentEmbeddings: Embedding
    @ModuleInfo(key: "scorer") var scorer: OpenMedGLiClassMLPScorer
    @ParameterInfo(key: "logit_scale") var logitScale: MLXArray

    init(_ configuration: OpenMedMLXBertConfiguration) {
        self.configuration = configuration
        _deberta.wrappedValue = OpenMedDebertaV2Model(configuration)
        _classesProjector.wrappedValue = OpenMedGLiClassFeaturesProjector(
            encoderHiddenSize: configuration.encoderHiddenSize,
            hiddenSize: configuration.hiddenSize
        )
        _textProjector.wrappedValue = OpenMedGLiClassFeaturesProjector(
            encoderHiddenSize: configuration.encoderHiddenSize,
            hiddenSize: configuration.hiddenSize
        )
        _segmentEmbeddings.wrappedValue = Embedding(
            embeddingCount: 3,
            dimensions: configuration.encoderHiddenSize
        )
        _scorer.wrappedValue = OpenMedGLiClassMLPScorer(
            hiddenSize: configuration.encoderHiddenSize
        )
        _logitScale.wrappedValue = MLXArray(Float(configuration.logitScaleInitValue))
    }

    private func createSegmentIDs(_ inputIDs: MLXArray) -> MLXArray {
        let batchSize = inputIDs.dim(0)
        let seqLength = inputIDs.dim(1)
        let textTokenID = configuration.textTokenIndex ?? -1
        let exampleTokenID = configuration.exampleTokenIndex ?? -1
        var rows = [[Int32]]()

        for batchIndex in 0..<batchSize {
            let tokens = inputIDs[batchIndex].asArray(Int32.self).map(Int.init)
            var row = Array(repeating: Int32(0), count: seqLength)
            if let textStart = tokens.firstIndex(of: textTokenID) {
                if let exampleStart = tokens.firstIndex(of: exampleTokenID) {
                    for index in textStart..<exampleStart {
                        row[index] = 1
                    }
                    for index in exampleStart..<seqLength {
                        row[index] = 2
                    }
                } else {
                    for index in textStart..<seqLength {
                        row[index] = 1
                    }
                }
            }
            rows.append(row)
        }

        return MLXArray(rows.flatMap { $0 }, [batchSize, seqLength])
    }

    private func extractClassFeatures(
        encoderLayer: MLXArray,
        inputIDs: MLXArray,
        attentionMask: MLXArray
    ) -> (MLXArray, MLXArray, MLXArray, MLXArray) {
        let batchSize = encoderLayer.dim(0)
        let seqLength = encoderLayer.dim(1)
        let embedDim = encoderLayer.dim(2)
        let classTokenID = configuration.classTokenIndex ?? -1
        let textTokenID = configuration.textTokenIndex ?? -1
        var classRows = [MLXArray]()
        var maskRows = [[Int]]()
        var maxClasses = 0

        for batchIndex in 0..<batchSize {
            let tokenIDs = inputIDs[batchIndex].asArray(Int32.self).map(Int.init)
            let classPositions = tokenIDs.enumerated().compactMap { index, tokenID in
                tokenID == classTokenID ? index : nil
            }
            let textStart = tokenIDs.firstIndex(of: textTokenID) ?? seqLength
            var rowEmbeddings = [MLXArray]()

            for (classIndex, classPosition) in classPositions.enumerated() {
                let startPosition = configuration.embedClassToken
                    ? classPosition
                    : min(classPosition + 1, seqLength - 1)
                let endPosition =
                    classIndex + 1 < classPositions.count
                    ? classPositions[classIndex + 1]
                    : textStart
                if startPosition >= endPosition {
                    rowEmbeddings.append(encoderLayer[batchIndex, startPosition])
                    continue
                }

                let classTokens = encoderLayer[batchIndex, startPosition..<endPosition, 0...]
                let classAttention = attentionMask[batchIndex, startPosition..<endPosition]
                    .asType(classTokens.dtype)
                let denom = sum(classAttention)
                let pooled = `where`(
                    denom .> 0,
                    sum(classTokens * classAttention.expandedDimensions(axis: -1), axis: 0) / denom,
                    mean(classTokens, axis: 0)
                )
                rowEmbeddings.append(pooled)
            }

            let row: MLXArray
            if rowEmbeddings.isEmpty {
                row = MLXArray.zeros([0, embedDim], type: Float.self).asType(encoderLayer.dtype)
            } else {
                row = stacked(rowEmbeddings, axis: 0)
            }
            classRows.append(row)
            maxClasses = max(maxClasses, row.dim(0))
            maskRows.append(Array(repeating: 1, count: row.dim(0)))
        }

        maxClasses = max(maxClasses, 1)
        return (
            openMedPadEmbeddings(classRows, width: maxClasses, embedDim: embedDim),
            openMedPadMaskRows(maskRows, width: maxClasses),
            encoderLayer,
            attentionMask
        )
    }

    private func poolText(_ textEmbeddings: MLXArray, textMask: MLXArray) -> MLXArray {
        switch configuration.poolingStrategy {
        case "mean":
            let mask = textMask.asType(textEmbeddings.dtype)
            let denominator = maximum(
                sum(mask, axis: 1, keepDims: true),
                openMedFloatScalar(1.0, like: mask)
            )
            return sum(textEmbeddings * mask.expandedDimensions(axis: -1), axis: 1) / denominator
        default:
            return textEmbeddings[0..., 0, 0...]
        }
    }

    func callAsFunction(inputIDs: MLXArray, attentionMask: MLXArray) -> OpenMedGLiClassOutput {
        var embedded = deberta.embeddings(inputIDs: inputIDs, attentionMask: attentionMask)
        if configuration.useSegmentEmbeddings {
            embedded = embedded + segmentEmbeddings(createSegmentIDs(inputIDs))
        }

        let hiddenStates = deberta.encoder(embedded, attentionMask: attentionMask)
        let (classesEmbedding, classesMask, textEmbeddings, textMask) = extractClassFeatures(
            encoderLayer: hiddenStates,
            inputIDs: inputIDs,
            attentionMask: attentionMask
        )
        var pooledOutput = textProjector(poolText(textEmbeddings, textMask: textMask))
        var projectedClasses = classesProjector(classesEmbedding)

        if configuration.normalizeFeatures {
            let pooledNorm = maximum(
                sqrt(sum(pooledOutput * pooledOutput, axis: -1, keepDims: true)),
                openMedFloatScalar(1.0e-8, like: pooledOutput)
            )
            let classNorm = maximum(
                sqrt(sum(projectedClasses * projectedClasses, axis: -1, keepDims: true)),
                openMedFloatScalar(1.0e-8, like: projectedClasses)
            )
            pooledOutput = pooledOutput / pooledNorm
            projectedClasses = projectedClasses / classNorm
        }

        var logits = scorer(textRep: pooledOutput, labelRep: projectedClasses)
        if configuration.normalizeFeatures {
            logits = logits * logitScale
        }
        return OpenMedGLiClassOutput(logits: logits, classesMask: classesMask)
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        weights.filter { key, _ in
            key != "deberta.embeddings.position_ids" && !key.hasPrefix("_")
        }
    }
}

final class OpenMedGLiNERTokenScorer: Module {
    @ModuleInfo(key: "proj_token") var tokenProjection: Linear
    @ModuleInfo(key: "proj_label") var labelProjection: Linear
    @ModuleInfo(key: "out_linear1") var outputLinear1: Linear
    @ModuleInfo(key: "out_linear2") var outputLinear2: Linear

    init(hiddenSize: Int) {
        _tokenProjection.wrappedValue = Linear(hiddenSize, hiddenSize * 2)
        _labelProjection.wrappedValue = Linear(hiddenSize, hiddenSize * 2)
        _outputLinear1.wrappedValue = Linear(hiddenSize * 3, hiddenSize * 4)
        _outputLinear2.wrappedValue = Linear(hiddenSize * 4, 3)
    }

    func callAsFunction(tokenRep: MLXArray, labelRep: MLXArray) -> MLXArray {
        let batchSize = tokenRep.dim(0)
        let seqLen = tokenRep.dim(1)
        let hiddenSize = tokenRep.dim(2)
        let numClasses = labelRep.dim(1)
        let tokenProjected = tokenProjection(tokenRep)
            .reshaped(batchSize, seqLen, 1, 2, hiddenSize)
        let labelProjected = labelProjection(labelRep)
            .reshaped(batchSize, 1, numClasses, 2, hiddenSize)

        let tokenLeft = broadcast(
            tokenProjected[0..., 0..., 0..., 0, 0...],
            to: [batchSize, seqLen, numClasses, hiddenSize]
        )
        let tokenRight = broadcast(
            tokenProjected[0..., 0..., 0..., 1, 0...],
            to: [batchSize, seqLen, numClasses, hiddenSize]
        )
        let labelLeft = broadcast(
            labelProjected[0..., 0..., 0..., 0, 0...],
            to: [batchSize, seqLen, numClasses, hiddenSize]
        )
        let labelRight = broadcast(
            labelProjected[0..., 0..., 0..., 1, 0...],
            to: [batchSize, seqLen, numClasses, hiddenSize]
        )
        let combined = concatenated([tokenLeft, labelLeft, tokenRight * labelRight], axis: -1)
        return outputLinear2(relu(outputLinear1(combined)))
    }
}

final class OpenMedGLiNERRelexModel: Module {
    private let configuration: OpenMedMLXBertConfiguration

    @ModuleInfo(key: "deberta") var deberta: OpenMedDebertaV2Model
    @ModuleInfo(key: "token_projection") var tokenProjection: Linear?
    @ModuleInfo(key: "rnn") var rnn: OpenMedBidirectionalLSTM?
    @ModuleInfo(key: "scorer") var scorer: OpenMedGLiNERTokenScorer
    @ModuleInfo(key: "span_rep_layer") var spanRepLayer: OpenMedSpanMarkerV0
    @ModuleInfo(key: "prompt_rep_layer") var promptRepLayer: OpenMedProjectionMLP
    @ModuleInfo(key: "pair_rep_layer") var pairRepLayer: OpenMedProjectionMLP

    init(_ configuration: OpenMedMLXBertConfiguration) {
        self.configuration = configuration
        _deberta.wrappedValue = OpenMedDebertaV2Model(configuration)
        if configuration.encoderHiddenSize != configuration.hiddenSize {
            _tokenProjection.wrappedValue = Linear(
                configuration.encoderHiddenSize,
                configuration.hiddenSize
            )
        }
        if configuration.numRNNLayers > 0 {
            _rnn.wrappedValue = OpenMedBidirectionalLSTM(hiddenSize: configuration.hiddenSize)
        }
        _scorer.wrappedValue = OpenMedGLiNERTokenScorer(hiddenSize: configuration.hiddenSize)
        _spanRepLayer.wrappedValue = OpenMedSpanMarkerV0(hiddenSize: configuration.hiddenSize)
        _promptRepLayer.wrappedValue = OpenMedProjectionMLP(inputSize: configuration.hiddenSize)
        _pairRepLayer.wrappedValue = OpenMedProjectionMLP(
            inputSize: configuration.hiddenSize * 2,
            outputSize: configuration.hiddenSize
        )
    }

    func encode(
        inputIDs: MLXArray,
        attentionMask: MLXArray,
        wordsMask: MLXArray
    ) -> OpenMedGLiNERRelexEncoded {
        let hiddenStates = deberta(inputIDs: inputIDs, attentionMask: attentionMask)
        let (entityPrompts, entityPromptMask) = openMedExtractMarkerEmbeddings(
            tokenEmbeddings: hiddenStates,
            inputIDs: inputIDs,
            markerTokenID: configuration.classTokenIndex ?? 0,
            includeMarkerToken: configuration.embedEntityToken
        )
        let (relationPrompts, relationPromptMask) = openMedExtractMarkerEmbeddings(
            tokenEmbeddings: hiddenStates,
            inputIDs: inputIDs,
            markerTokenID: configuration.relTokenIndex ?? 0,
            includeMarkerToken: configuration.embedRelationToken ?? configuration.embedEntityToken
        )
        let (wordEmbeddings, wordMask) = openMedExtractWordEmbeddings(
            tokenEmbeddings: hiddenStates,
            wordsMask: wordsMask
        )

        var projectedEntities = entityPrompts
        var projectedRelations = relationPrompts
        var projectedWords = wordEmbeddings
        if let tokenProjection {
            projectedEntities = tokenProjection(projectedEntities)
            projectedRelations = tokenProjection(projectedRelations)
            projectedWords = tokenProjection(projectedWords)
        }
        if let rnn {
            projectedWords = rnn(projectedWords, mask: wordMask)
        }

        return OpenMedGLiNERRelexEncoded(
            entityPrompts: projectedEntities,
            entityPromptMask: entityPromptMask,
            relationPrompts: projectedRelations,
            relationPromptMask: relationPromptMask,
            wordsEmbedding: projectedWords,
            wordMask: wordMask
        )
    }

    func entityScores(encoded: OpenMedGLiNERRelexEncoded) -> MLXArray {
        scorer(
            tokenRep: encoded.wordsEmbedding,
            labelRep: promptRepLayer(encoded.entityPrompts)
        )
    }

    func relationScores(
        encoded: OpenMedGLiNERRelexEncoded,
        spanIndex: MLXArray,
        spanMask: MLXArray
    ) -> OpenMedGLiNERRelexRelationOutput {
        let maskedSpanIndex = spanIndex * spanMask.asType(spanIndex.dtype).expandedDimensions(axis: -1)
        let spanRep = spanRepLayer(encoded.wordsEmbedding, spanIndex: maskedSpanIndex)
        let (pairIndex, pairMask, headRep, tailRep) = openMedBuildAllEntityPairs(
            spanRep: spanRep,
            spanMask: spanMask
        )
        let pairRep = pairRepLayer(concatenated([headRep, tailRep], axis: -1))
        let pairScores = einsum("bnd,bcd->bnc", pairRep, encoded.relationPrompts)
        return OpenMedGLiNERRelexRelationOutput(
            pairScores: pairScores,
            pairIndex: pairIndex,
            pairMask: pairMask,
            relationPromptMask: encoded.relationPromptMask
        )
    }

    func callAsFunction(
        inputIDs: MLXArray,
        attentionMask: MLXArray,
        wordsMask: MLXArray
    ) -> OpenMedGLiNERRelexEntityOutput {
        let encoded = encode(
            inputIDs: inputIDs,
            attentionMask: attentionMask,
            wordsMask: wordsMask
        )
        return OpenMedGLiNERRelexEntityOutput(
            entityScores: entityScores(encoded: encoded),
            entityPromptMask: encoded.entityPromptMask,
            wordMask: encoded.wordMask,
            encoded: encoded
        )
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        weights.filter { key, _ in
            key != "deberta.embeddings.position_ids" && !key.hasPrefix("_")
        }
    }
}
