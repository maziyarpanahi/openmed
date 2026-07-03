package com.openmed.openmedkit.onnx

public data class TokenOffset(
    val startOffset: Int,
    val endOffset: Int,
) {
    internal val isSpecialToken: Boolean
        get() = startOffset == 0 && endOffset == 0
}

public data class TokenPrediction(
    val labelId: Int,
    val label: String,
    val score: Float,
    val startOffset: Int,
    val endOffset: Int,
)
