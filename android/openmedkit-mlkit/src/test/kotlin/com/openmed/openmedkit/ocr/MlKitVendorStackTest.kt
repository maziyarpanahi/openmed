package com.openmed.openmedkit.ocr

import org.junit.Assert.assertEquals
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

/**
 * Negative control for the core `CoreVendorBoundaryTest`: the same sentinel
 * classes must resolve once the optional ML Kit module is on the classpath,
 * which proves the core test is checking real vendor class names.
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [26])
class MlKitVendorStackTest {
    @Test
    fun optionalModuleCarriesTheVendorStackCoreExcludes() {
        val missing = listOf(
            "com.google.mlkit.vision.common.InputImage",
            "com.google.mlkit.common.internal.MlKitInitProvider",
            "com.google.android.gms.tasks.Task",
            "com.google.android.datatransport.TransportFactory",
            "com.google.android.datatransport.cct.CctBackendFactory",
            "com.google.mlkit.common.internal.MlKitComponentDiscoveryService",
            "com.google.firebase.components.ComponentRegistrar",
        ).filter { className ->
            runCatching { Class.forName(className, false, javaClass.classLoader) }.isFailure
        }

        assertEquals(emptyList<String>(), missing)
    }
}
