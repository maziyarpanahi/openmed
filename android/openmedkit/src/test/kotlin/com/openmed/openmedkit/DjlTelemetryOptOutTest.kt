package com.openmed.openmedkit

import ai.djl.util.Ec2Utils
import ai.djl.util.Utils
import java.security.Permission
import java.util.concurrent.CopyOnWriteArrayList
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import org.junit.Assume.assumeTrue

/**
 * DJL runs usage telemetry inside every `HuggingFaceTokenizer.newInstance`
 * call. These tests pin that OpenMedKit's opt-out stops it from connecting,
 * and that the probe really does connect without the opt-out.
 */
@Suppress("DEPRECATION")
class DjlTelemetryOptOutTest {
    @Test
    fun optOutSetsTheSwitchDjlReads() {
        DjlTelemetry.optOut()

        assertEquals("true", Utils.getEnvOrSystemProperty(DjlTelemetry.OPT_OUT_PROPERTY))
    }

    @Test
    fun tokenizerTelemetryOpensNoConnectionAfterOptOut() {
        DjlTelemetry.optOut()

        val attempts = runTelemetryProbe()

        assertTrue(attempts.isEmpty(), "DJL telemetry attempted a network connection")
    }

    @Test
    fun tokenizerTelemetryConnectsWithoutOptOut() {
        assumeTrue(System.getenv(DjlTelemetry.OPT_OUT_PROPERTY) == null)
        assumeTrue(System.getenv("DJL_OFFLINE") == null)
        val previous = System.clearProperty(DjlTelemetry.OPT_OUT_PROPERTY)
        try {
            val attempts = runTelemetryProbe()

            assertTrue(attempts.isNotEmpty(), "negative control: DJL telemetry did not probe")
        } finally {
            if (previous != null) {
                System.setProperty(DjlTelemetry.OPT_OUT_PROPERTY, previous)
            }
        }
    }

    /** Run DJL's telemetry step under a security manager that refuses connects. */
    private fun runTelemetryProbe(): List<String> {
        // DJL reports at most once a day per process; clear that memo so each
        // test exercises the probe itself.
        Ec2Utils::class.java.getDeclaredField("lastCheckIn").apply {
            isAccessible = true
            setLong(null, 0L)
        }
        val attempts = CopyOnWriteArrayList<String>()
        val previousManager = System.getSecurityManager()
        System.setSecurityManager(RefuseConnectSecurityManager(attempts))
        try {
            Ec2Utils.callHome("Huggingface")
        } catch (refused: SecurityException) {
            // The refusal propagates out of DJL's probe; the attempt is recorded.
        } finally {
            System.setSecurityManager(previousManager)
        }
        return attempts.toList()
    }

    private class RefuseConnectSecurityManager(
        private val attempts: MutableList<String>,
    ) : SecurityManager() {
        override fun checkPermission(permission: Permission?) = Unit

        override fun checkPermission(permission: Permission?, context: Any?) = Unit

        override fun checkConnect(host: String?, port: Int) {
            attempts += "connect"
            throw SecurityException("network connection refused by test")
        }

        override fun checkConnect(host: String?, port: Int, context: Any?) {
            checkConnect(host, port)
        }
    }
}
