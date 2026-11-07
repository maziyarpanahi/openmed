package com.openmed.openmedkit

import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

/**
 * Guards the core artifact against the ML Kit, Play services, and datatransport
 * stack that now lives only in the optional `openmedkit-mlkit` module.
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [26])
class CoreVendorBoundaryTest {
    @Test
    fun vendorOcrAndTelemetryClassesAreAbsentFromTheCoreClasspath() {
        val present = VENDOR_SENTINEL_CLASSES.filter { className ->
            runCatching { Class.forName(className, false, javaClass.classLoader) }.isSuccess
        }

        assertEquals(emptyList<String>(), present)
    }

    @Test
    fun productionSourcesDoNotImportVendorPackages() {
        val sourceRoot = File(moduleDirectory(), "src/main/kotlin")
        check(sourceRoot.isDirectory) { "cannot locate OpenMedKit Android Kotlin sources" }

        val violations = sourceRoot.walkTopDown()
            .filter { it.isFile && it.extension == "kt" }
            .flatMap { file ->
                file.readLines().asSequence().mapIndexedNotNull { index, line ->
                    val vendorPackage = VENDOR_PACKAGES.firstOrNull { line.contains(it) }
                    vendorPackage?.let { "${file.relativeTo(sourceRoot)}:${index + 1}" }
                }
            }
            .toList()

        assertEquals(emptyList<String>(), violations)
    }

    @Test
    fun coreBuildDeclaresNoVendorDependencyAndRunsTheBoundaryCheck() {
        val build = File(moduleDirectory(), "build.gradle.kts").readText()

        assertFalse(build.contains("mlkit.text.recognition"))
        VENDOR_PACKAGES.forEach { assertFalse(build.contains("\"$it")) }
        assertTrue(build.contains("extra[\"openmedkitDependencyBoundary\"] = \"core\""))
        assertTrue(build.contains("dependsOn(\"verifyDependencyBoundary\")"))
    }

    private fun moduleDirectory(): File =
        File(
            requireNotNull(System.getProperty("openmedkit.moduleDirectory")) {
                "openmedkit.moduleDirectory test property is missing"
            },
        )

    private companion object {
        val VENDOR_PACKAGES = listOf(
            "com.google.mlkit",
            "com.google.android.gms",
            "com.google.android.datatransport",
            "com.google.firebase",
        )

        val VENDOR_SENTINEL_CLASSES = listOf(
            "com.google.mlkit.vision.common.InputImage",
            "com.google.mlkit.common.internal.MlKitInitProvider",
            "com.google.android.gms.tasks.Task",
            "com.google.android.datatransport.TransportFactory",
            "com.google.android.datatransport.cct.CctBackendFactory",
            "com.google.mlkit.common.internal.MlKitComponentDiscoveryService",
            "com.google.firebase.components.ComponentRegistrar",
        )
    }
}
