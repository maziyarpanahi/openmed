package com.openmed.openmedkit.onnx

public sealed class InferenceError(
    message: String,
    cause: Throwable? = null,
) : Exception(message, cause) {
    public class MissingOutput(public val outputName: String) :
        InferenceError("ONNX model output '$outputName' not found")

    public class InvalidInput(message: String) : InferenceError(message)

    public class InvalidOutput(message: String, cause: Throwable? = null) :
        InferenceError(message, cause)

    public class SessionCreation(cause: Throwable? = null) :
        InferenceError("Unable to create an ONNX Runtime CPU session", cause)
}
