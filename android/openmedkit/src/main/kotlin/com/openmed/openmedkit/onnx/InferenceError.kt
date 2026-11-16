package com.openmed.openmedkit.onnx

import com.openmed.openmedkit.OpenMedCodedError
import com.openmed.openmedkit.OpenMedErrorCode

public sealed class InferenceError(
    message: String,
    cause: Throwable? = null,
) : Exception(message, cause), OpenMedCodedError {
    /**
     * `input_error` for [InvalidInput], `model_load_error` for
     * [SessionCreation], and `inference_error` for a missing or invalid output.
     */
    override val errorCode: OpenMedErrorCode
        get() = when (this) {
            is InvalidInput -> OpenMedErrorCode.INPUT_ERROR
            is SessionCreation -> OpenMedErrorCode.MODEL_LOAD_ERROR
            else -> OpenMedErrorCode.INFERENCE_ERROR
        }

    public class MissingOutput(public val outputName: String) :
        InferenceError("ONNX model output '$outputName' not found")

    public class InvalidInput(message: String) : InferenceError(message)

    public class InvalidOutput(message: String, cause: Throwable? = null) :
        InferenceError(message, cause)

    public class SessionCreation(cause: Throwable? = null) :
        InferenceError("Unable to create an ONNX Runtime CPU session", cause)
}
