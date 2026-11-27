package com.openmed.openmedkit.emulator

import android.Manifest
import android.content.pm.PackageManager
import android.system.ErrnoException
import android.system.Os
import android.system.OsConstants
import androidx.test.platform.app.InstrumentationRegistry
import java.net.DatagramSocket
import java.net.ServerSocket
import java.net.Socket
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test

/**
 * Negative controls for the lane's offline guarantee.
 *
 * They prove that the checks used by [NoNetworkSocketRule] can fail: the
 * kernel refuses IPv4 and IPv6 sockets for the test process, and every
 * `java.net` socket construction is recorded.
 */
class OfflineGuardTest {
    @Test
    fun testProcessHoldsNoInternetPermission() {
        val context = InstrumentationRegistry.getInstrumentation().context
        assertEquals(
            "internet_permission_granted",
            PackageManager.PERMISSION_DENIED,
            context.checkSelfPermission(Manifest.permission.INTERNET),
        )
    }

    @Test
    fun kernelRefusesInetSocketsForTheTestProcess() {
        for (family in listOf(OsConstants.AF_INET, OsConstants.AF_INET6)) {
            val errno = try {
                val socket = Os.socket(family, OsConstants.SOCK_STREAM, 0)
                Os.close(socket)
                fail("inet_socket_not_refused:$family")
                0
            } catch (error: ErrnoException) {
                error.errno
            }
            // The eBPF INTERNET check returns EPERM; older paranoid-network
            // kernels return EACCES.
            assertTrue(
                "inet_socket_refused_for_unexpected_reason:$family",
                errno == OsConstants.EPERM || errno == OsConstants.EACCES,
            )
        }
        assertEquals("inet_socket_open", 0, InetSocketInventory.openInetSocketCount())
    }

    @Test
    fun javaSocketConstructionIsRecordedAndRefused() {
        NetworkSocketGuard.install()
        NetworkSocketGuard.reset()
        try {
            assertRefused("stream") { Socket().close() }
            assertRefused("server") { ServerSocket().close() }
            assertRefused("datagram") { DatagramSocket().close() }
            assertEquals("network_socket_attempts_not_recorded", 3, NetworkSocketGuard.attempts())
        } finally {
            NetworkSocketGuard.reset()
        }
    }

    private fun assertRefused(kind: String, open: () -> Unit) {
        try {
            open()
            fail("network_socket_not_refused:$kind")
        } catch (error: SecurityException) {
            assertEquals("network_socket_refused:$kind", error.message)
        }
    }
}
