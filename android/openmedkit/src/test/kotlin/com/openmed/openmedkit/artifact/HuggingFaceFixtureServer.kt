package com.openmed.openmedkit.artifact

import com.openmed.openmedkit.download.HttpHuggingFaceModelClient
import com.openmed.openmedkit.download.ModelIntegrity
import java.io.BufferedInputStream
import java.io.ByteArrayOutputStream
import java.io.Closeable
import java.io.IOException
import java.io.InputStream
import java.io.OutputStream
import java.net.InetAddress
import java.net.ServerSocket
import java.net.Socket
import java.security.MessageDigest
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicReference

/**
 * Loopback-only, Hugging Face shaped HTTP/1.1 fixture built on [ServerSocket]
 * so tests need no extra dependency: a `blobs=true` model-info API, a
 * `resolve` endpoint that redirects to a signed CDN URL, and a CDN endpoint
 * with HTTP Range support plus fault injection. All content is synthetic.
 */
internal class HuggingFaceFixtureServer(
    val repoId: String,
    files: Map<String, ByteArray>,
) : Closeable {
    val files: MutableMap<String, ByteArray> = ConcurrentHashMap(files)

    /** SHA-256 advertised in the manifest; defaults to the served bytes. */
    val advertisedSha256: MutableMap<String, String> = ConcurrentHashMap()

    /** Files published as LFS objects (with size and SHA-256). */
    val lfsPaths: MutableSet<String> = ConcurrentHashMap.newKeySet<String>().apply {
        files.keys.filter { it.endsWith(".onnx") || it.endsWith(".tflite") }.forEach(::add)
    }

    val requests = CopyOnWriteArrayList<RecordedRequest>()

    /** File that one-shot [dropAfterBytes] and [pauseAfterBytes] faults apply to. */
    @Volatile
    var faultPath: String = "model.onnx"
    val dropAfterBytes = AtomicReference<Int?>(null)
    val pauseAfterBytes = AtomicReference<Int?>(null)
    val paused = CountDownLatch(1)
    val release = CountDownLatch(1)

    @Volatile
    var cdnStatusOverride: Int? = null

    private val serverSocket = ServerSocket(0, 64, InetAddress.getLoopbackAddress())
    private val executor = Executors.newCachedThreadPool()

    init {
        executor.execute {
            while (!serverSocket.isClosed) {
                val socket = try {
                    serverSocket.accept()
                } catch (_: IOException) {
                    break
                }
                executor.execute { serve(socket) }
            }
        }
    }

    val baseUrl: String
        get() = "http://127.0.0.1:${serverSocket.localPort}"

    fun client(): HttpHuggingFaceModelClient = HttpHuggingFaceModelClient(
        modelApiBaseUrl = "$baseUrl/api/models",
        resolveBaseUrl = baseUrl,
        connectTimeoutMillis = 10_000,
        readTimeoutMillis = 10_000,
    )

    fun digest(): String = ModelIntegrity.reproducibilityHash(
        repoId = repoId,
        revisionSha = REVISION_SHA,
        released = RELEASED,
        siblings = files.keys.toList(),
    )

    fun cdnRequests(path: String): List<RecordedRequest> =
        requests.filter { it.path == "/cdn/$path" }

    fun manifestRequests(): List<RecordedRequest> =
        requests.filter { it.path.startsWith("/api/models/") }

    override fun close() {
        release.countDown()
        runCatching { serverSocket.close() }
        executor.shutdownNow()
    }

    private fun serve(socket: Socket) {
        socket.use {
            try {
                val input = BufferedInputStream(socket.getInputStream())
                val requestLine = readLine(input) ?: return
                val headers = generateSequence { readLine(input)?.takeIf { it.isNotEmpty() } }
                    .associate { line ->
                        line.substringBefore(':').trim().lowercase() to
                            line.substringAfter(':').trim()
                    }
                val path = requestLine.split(' ').getOrElse(1) { "/" }.substringBefore('?')
                requests += RecordedRequest(path, headers["range"])
                val output = socket.getOutputStream()
                when {
                    path.startsWith("/api/models/") -> respond(output, 200, manifest())
                    path.startsWith("/cdn/") ->
                        serveFile(output, path.removePrefix("/cdn/"), headers["range"])
                    path.contains("/resolve/") -> {
                        val file = path.substringAfter("/resolve/").substringAfter('/')
                        respond(
                            output,
                            302,
                            ByteArray(0),
                            mapOf("Location" to "/cdn/$file?$SIGNED_QUERY_SENTINEL&download=true"),
                        )
                    }
                    else -> respond(output, 404, ByteArray(0))
                }
            } catch (_: IOException) {
                // The client disconnected mid-response.
            }
        }
    }

    private fun manifest(): ByteArray {
        val siblings = files.entries.sortedBy { it.key }.joinToString(",") { (path, bytes) ->
            val sha = advertisedSha256[path] ?: sha256(bytes)
            val lfs = if (path in lfsPaths) {
                ""","lfs":{"sha256":"$sha","size":${bytes.size}}"""
            } else {
                ""
            }
            """{"rfilename":"$path","size":${bytes.size}$lfs}"""
        }
        return (
            """{"sha":"$REVISION_SHA","lastModified":"${RELEASED}T00:00:00.000Z",""" +
                """"siblings":[$siblings]}"""
            ).toByteArray()
    }

    private fun serveFile(output: OutputStream, path: String, range: String?) {
        cdnStatusOverride?.let { status ->
            respond(output, status, ERROR_BODY_SENTINEL.toByteArray())
            return
        }
        val content = files[path] ?: run {
            respond(output, 404, ByteArray(0))
            return
        }
        val start = range?.removePrefix("bytes=")?.substringBefore('-')?.toIntOrNull() ?: 0
        if (start > 0 && start >= content.size) {
            respond(output, 416, ByteArray(0))
            return
        }
        val body = content.copyOfRange(start, content.size)
        val status = if (start > 0) 206 else 200
        val extra = if (start > 0) {
            mapOf("Content-Range" to "bytes $start-${content.size - 1}/${content.size}")
        } else {
            emptyMap()
        }
        val faulted = path == faultPath
        val drop = if (faulted) dropAfterBytes.getAndSet(null) else null
        val pause = if (faulted) pauseAfterBytes.getAndSet(null) else null
        writeHead(output, status, body.size, extra)
        when {
            // Close before Content-Length is met: the client sees a truncated body.
            drop != null -> output.write(body, 0, minOf(drop, body.size))
            pause != null -> {
                val first = minOf(pause, body.size)
                output.write(body, 0, first)
                output.flush()
                paused.countDown()
                release.await(30, TimeUnit.SECONDS)
                output.write(body, first, body.size - first)
            }
            else -> output.write(body)
        }
        output.flush()
    }

    private fun respond(
        output: OutputStream,
        status: Int,
        body: ByteArray,
        extra: Map<String, String> = emptyMap(),
    ) {
        writeHead(output, status, body.size, extra)
        output.write(body)
        output.flush()
    }

    private fun writeHead(
        output: OutputStream,
        status: Int,
        contentLength: Int,
        extra: Map<String, String>,
    ) {
        val head = buildString {
            append("HTTP/1.1 $status ${REASONS[status] ?: "Status"}\r\n")
            extra.forEach { (name, value) -> append("$name: $value\r\n") }
            append("Content-Length: $contentLength\r\n")
            append("Connection: close\r\n\r\n")
        }
        output.write(head.toByteArray(Charsets.US_ASCII))
    }

    private fun readLine(input: InputStream): String? {
        val buffer = ByteArrayOutputStream()
        while (true) {
            val next = input.read()
            if (next < 0) {
                return if (buffer.size() == 0) null else buffer.toString("US-ASCII")
            }
            if (next == '\n'.code) {
                return buffer.toString("US-ASCII").trimEnd('\r')
            }
            buffer.write(next)
        }
    }

    data class RecordedRequest(val path: String, val range: String?)

    companion object {
        const val REVISION_SHA = "0123456789abcdef0123456789abcdef01234567"
        const val RELEASED = "2026-01-01"
        const val SIGNED_QUERY_SENTINEL = "X-Amz-Signature=SENTINEL-SIGNED-QUERY-7f3a"
        const val ERROR_BODY_SENTINEL = "SENTINEL-ERROR-BODY-token-hf_live_abc"

        private val REASONS = mapOf(
            200 to "OK",
            206 to "Partial Content",
            302 to "Found",
            404 to "Not Found",
            416 to "Range Not Satisfiable",
            500 to "Internal Server Error",
            503 to "Service Unavailable",
        )

        fun sha256(bytes: ByteArray): String =
            MessageDigest.getInstance("SHA-256").digest(bytes)
                .joinToString("") { "%02x".format(it) }

        /** Synthetic ONNX-shaped bytes: protobuf tag 0x08 first, then filler. */
        fun syntheticOnnx(sizeBytes: Int, marker: String = "synthetic"): ByteArray {
            val bytes = ByteArray(sizeBytes) { index -> ((index * 31 + 7) % 251).toByte() }
            bytes[0] = 0x08
            bytes[1] = 0x08
            marker.toByteArray().copyInto(bytes, destinationOffset = 16)
            return bytes
        }

        /** Synthetic TFLite-shaped bytes: `TFL3` identifier at byte 4. */
        fun syntheticTflite(sizeBytes: Int): ByteArray {
            val bytes = ByteArray(sizeBytes) { index -> ((index * 17 + 3) % 253).toByte() }
            "TFL3".toByteArray().copyInto(bytes, destinationOffset = 4)
            return bytes
        }
    }
}
