package com.openmed.openmedkit.emulator

import android.Manifest
import android.content.pm.PackageManager
import android.os.ParcelFileDescriptor
import android.system.ErrnoException
import android.system.Os
import androidx.test.platform.app.InstrumentationRegistry
import java.io.File
import java.io.IOException
import java.net.DatagramSocket
import java.net.DatagramSocketImplFactory
import java.net.InetSocketAddress
import java.net.ServerSocket
import java.net.Socket
import java.net.SocketImplFactory
import java.util.concurrent.atomic.AtomicInteger
import org.junit.Assert.assertEquals
import org.junit.rules.TestRule
import org.junit.runner.Description
import org.junit.runners.model.Statement

/**
 * Process-wide recorder that refuses every `java.net` socket.
 *
 * Installing the guard replaces the JVM socket factories for the rest of the
 * test process: every `Socket`, `ServerSocket` and `DatagramSocket`
 * construction (including those made by `HttpURLConnection` and TLS sockets)
 * is counted and then refused with a [SecurityException]. Sockets created in
 * native code bypass these factories; the kernel refuses those because the
 * test APK holds no `INTERNET` permission (see [OfflineGuardTest]).
 */
internal object NetworkSocketGuard {
    private val attempts = AtomicInteger()

    @Volatile
    private var installed = false

    /** Install the refusing factories once per process. */
    @Synchronized
    @Suppress("DEPRECATION")
    fun install() {
        if (installed) {
            return
        }
        Socket.setSocketImplFactory(SocketImplFactory { refuse("stream") })
        ServerSocket.setSocketFactory(SocketImplFactory { refuse("server") })
        DatagramSocket.setDatagramSocketImplFactory(
            DatagramSocketImplFactory { refuse("datagram") },
        )
        installed = true
    }

    /** Number of socket constructions refused since the last [reset]. */
    fun attempts(): Int = attempts.get()

    /** Forget earlier attempts, for example after a negative control. */
    fun reset() {
        attempts.set(0)
    }

    private fun refuse(kind: String): Nothing {
        attempts.incrementAndGet()
        throw SecurityException("network_socket_refused:$kind")
    }
}

/**
 * Counts open IPv4 and IPv6 sockets of the test process.
 *
 * Every descriptor under `/proc/self/fd` that is a socket is duplicated and
 * asked for its local address, so sockets opened from native code are counted
 * too. Only IPv4 and IPv6 sockets report an `InetSocketAddress`; Unix-domain
 * sockets, such as the logging socket, are ignored.
 */
internal object InetSocketInventory {
    fun openInetSocketCount(): Int {
        val descriptors = File("/proc/self/fd").listFiles()
            ?: throw AssertionError("fd_inventory_unavailable")
        return descriptors.count { descriptor ->
            val target = try {
                Os.readlink(descriptor.path)
            } catch (error: ErrnoException) {
                return@count false
            }
            val fd = descriptor.name.toIntOrNull()
            if (!target.startsWith("socket:") || fd == null) {
                return@count false
            }
            val duplicate = try {
                ParcelFileDescriptor.fromFd(fd)
            } catch (error: IOException) {
                return@count false
            }
            duplicate.use { socket ->
                try {
                    Os.getsockname(socket.fileDescriptor) is InetSocketAddress
                } catch (error: ErrnoException) {
                    false
                } catch (error: IllegalArgumentException) {
                    // Address families libcore cannot represent (for example
                    // netlink) are not network sockets.
                    false
                }
            }
        }
    }
}

/**
 * Fails a test that opens, or tries to open, a network socket.
 *
 * Before the test it checks that the process holds no `INTERNET` permission
 * and has no IPv4/IPv6 socket open; afterwards it checks that no `java.net`
 * socket was attempted and that still no IPv4/IPv6 socket is open.
 */
class NoNetworkSocketRule : TestRule {
    override fun apply(base: Statement, description: Description): Statement =
        object : Statement() {
            override fun evaluate() {
                val context = InstrumentationRegistry.getInstrumentation().context
                assertEquals(
                    "internet_permission_granted",
                    PackageManager.PERMISSION_DENIED,
                    context.checkSelfPermission(Manifest.permission.INTERNET),
                )
                NetworkSocketGuard.install()
                NetworkSocketGuard.reset()
                assertEquals(
                    "inet_socket_open_before_test",
                    0,
                    InetSocketInventory.openInetSocketCount(),
                )

                base.evaluate()

                assertEquals("network_socket_attempted", 0, NetworkSocketGuard.attempts())
                assertEquals(
                    "inet_socket_open_after_test",
                    0,
                    InetSocketInventory.openInetSocketCount(),
                )
            }
        }
}
