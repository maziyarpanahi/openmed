package com.openmed.openmedkit.ocr

import android.content.Context
import com.google.mlkit.common.MlKit

internal fun initializeMlKitForTests(context: Context) {
    try {
        MlKit.initialize(context)
    } catch (error: IllegalStateException) {
        if (error.message?.contains("already initialized") != true) {
            throw error
        }
    }
}
