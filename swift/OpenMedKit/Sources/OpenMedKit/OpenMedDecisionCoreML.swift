#if canImport(CoreML)
    import CoreML
    import Foundation

    /// Anything that runs a Core ML prediction; `MLModel` conforms.
    public protocol OpenMedDecisionFeaturePredicting {
        func prediction(from input: MLFeatureProvider) throws -> MLFeatureProvider
    }

    extension MLModel: OpenMedDecisionFeaturePredicting {}

    /// Runs a Core ML export of a GLiClass-style uni-encoder.
    ///
    /// Inputs are `[1, tokens]` multi-arrays named `input_ids` and
    /// `attention_mask`; the output is a `[1, labels]` `logits` multi-array in
    /// any numeric type. Tensors are built and read in memory only; nothing is
    /// written to disk or logged.
    public final class OpenMedCoreMLDecisionEncoder: OpenMedDecisionEncoder {
        private let model: any OpenMedDecisionFeaturePredicting
        private let inputIDsName: String
        private let attentionMaskName: String
        private let logitsName: String
        private let inputDataType: MLMultiArrayDataType

        public init(
            model: any OpenMedDecisionFeaturePredicting,
            inputIDsName: String = "input_ids",
            attentionMaskName: String = "attention_mask",
            logitsName: String = "logits",
            inputDataType: MLMultiArrayDataType = .int32
        ) {
            self.model = model
            self.inputIDsName = inputIDsName
            self.attentionMaskName = attentionMaskName
            self.logitsName = logitsName
            self.inputDataType = inputDataType
        }

        public func logits(inputIDs: [Int], attentionMask: [Int]) throws -> [Double] {
            guard !inputIDs.isEmpty, inputIDs.count == attentionMask.count,
                inputIDs.allSatisfy({ $0 >= 0 && $0 <= Int(Int32.max) })
            else { throw OpenMedDecisionEncoderError.invalidInput }
            let shape = [1, NSNumber(value: inputIDs.count)]
            let ids = try MLMultiArray(shape: shape, dataType: inputDataType)
            let mask = try MLMultiArray(shape: shape, dataType: inputDataType)
            for index in inputIDs.indices {
                ids[[0, NSNumber(value: index)]] = NSNumber(value: inputIDs[index])
                mask[[0, NSNumber(value: index)]] = NSNumber(value: attentionMask[index])
            }
            let output = try model.prediction(
                from: MLDictionaryFeatureProvider(dictionary: [
                    inputIDsName: MLFeatureValue(multiArray: ids),
                    attentionMaskName: MLFeatureValue(multiArray: mask),
                ]))
            guard let logits = output.featureValue(for: logitsName)?.multiArrayValue else {
                throw OpenMedDecisionEncoderError.missingOutput
            }
            let dimensions = logits.shape.map(\.intValue)
            guard dimensions.count == 2, dimensions[0] >= 1, dimensions[1] >= 1 else {
                throw OpenMedDecisionEncoderError.invalidOutput
            }
            return (0..<dimensions[1]).map { logits[[0, NSNumber(value: $0)]].doubleValue }
        }
    }
#endif
