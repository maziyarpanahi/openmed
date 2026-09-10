package com.openmed.openmedkit.util

import com.openmed.openmedkit.OnnxTokenClassifier
import com.openmed.openmedkit.OpenMedKit
import com.openmed.openmedkit.TokenClassificationPrediction
import java.io.File
import java.security.Permission
import java.util.concurrent.CopyOnWriteArrayList
import kotlin.test.Test
import kotlin.test.assertFalse
import kotlin.test.assertTrue
import kotlinx.coroutines.test.runTest
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@Suppress("DEPRECATION")
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [33])
class NoNetworkInferenceTest {
    @Test
    fun analyzeExtractAndDeidentifyOpenNoSockets() = runTest {
        val source = "Patient Jane called 555-1212"
        val networkAttempts = CopyOnWriteArrayList<String>()
        val previousManager = System.getSecurityManager()
        val denyNetwork = DenyNetworkSecurityManager(networkAttempts)
        val kit = OpenMedKit(
            classifier = StaticClassifier(
                TokenClassificationPrediction("PERSON", "Jane", 0.99f, 8, 12),
                TokenClassificationPrediction("PHONE", "555-1212", 0.99f, 20, 28),
            ),
        )

        System.setSecurityManager(denyNetwork)
        try {
            assertTrue(kit.analyzeText(source).isNotEmpty())
            assertTrue(kit.extractPii(source).isNotEmpty())
            assertFalse(kit.deidentify(source).redactedText.contains("Jane"))
        } finally {
            kit.close()
            System.setSecurityManager(previousManager)
        }

        assertTrue(networkAttempts.isEmpty(), "inference attempted network I/O")
    }

    @Test
    fun libraryManifestDoesNotRequestInternetPermission() {
        val manifest = File(
            requireNotNull(System.getProperty("openmedkit.moduleDirectory")) {
                "openmedkit.moduleDirectory test property is missing"
            },
            "src/main/AndroidManifest.xml",
        )

        assertTrue(manifest.isFile, "cannot locate OpenMedKit Android manifest")
        assertFalse(manifest.readText().contains("android.permission.INTERNET"))
    }

    private class DenyNetworkSecurityManager(
        private val attempts: MutableList<String>,
    ) : SecurityManager() {
        override fun checkPermission(permission: Permission?) = Unit

        override fun checkPermission(permission: Permission?, context: Any?) = Unit

        override fun checkConnect(host: String?, port: Int) {
            attempts += "$host:$port"
            throw AssertionError("network access is forbidden during inference")
        }

        override fun checkConnect(host: String?, port: Int, context: Any?) {
            checkConnect(host, port)
        }
    }

    private class StaticClassifier(
        vararg predictions: TokenClassificationPrediction,
    ) : OnnxTokenClassifier {
        private val predictions = predictions.toList()

        override suspend fun predict(text: String): List<TokenClassificationPrediction> =
            predictions
    }
}
