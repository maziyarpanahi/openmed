package com.openmed.openmedkit.privacy

import java.io.ByteArrayInputStream
import java.io.File
import java.io.InputStream
import java.io.OutputStream
import java.net.URLEncoder
import java.nio.file.Files
import java.util.Base64
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue
import org.junit.Rule
import org.junit.rules.TemporaryFolder

class PhiSentinelSweepTest {
    @get:Rule
    val temporary = TemporaryFolder()

    private val name = PhiSentinel("person.name", "Vexorin Qualtrane")

    // Positive controls

    @Test
    fun plantedSentinelIsFoundInCacheFileAndLogWithInjectedRootsAndReaders() {
        val cache = directory("cache")
        val noBackup = directory("no_backup")
        File(cache, "draft.txt").writeText("draft: ${name.plantedValue}\n")
        File(noBackup, "state.bin").writeBytes(name.plantedValue.toByteArray(Charsets.UTF_16LE))
        val reader = CapturedLogReader(listOf("started", "lookup ${name.plantedValue} ok"))

        val report = PhiSentinelSweep(
            sentinels = listOf(name),
            roots = listOf(
                SweepRoot(cache, StorageClass.CACHE),
                SweepRoot(noBackup, StorageClass.NO_BACKUP),
            ),
            logReaders = listOf(reader),
        ).run()

        assertFalse(report.clean)
        assertTrue(report.exhaustive, report.toString())
        assertEquals(
            listOf(
                listOf("file_content", "cache", "text", "utf8"),
                listOf("file_content", "no_backup", "other", "utf16le"),
                listOf("log_entry", "captured", "-", "utf8"),
            ),
            report.findings.map {
                listOf(it.location.wireName, it.surface, it.fileKind?.wireName ?: "-", it.encoding.wireName)
            },
        )
        assertTrue(report.findings.all { it.sentinelIdentifier == "person.name" && it.itemCount == 1 })
        assertEquals(2, report.filesScanned)
        assertEquals(listOf(SweepLogStatus.READ), report.logs.map { it.status })
    }

    @Test
    fun everyEncodingIsDetectedAtEveryByteAlignment() {
        val value = "Zoë Quinterra-Ångström 41"
        val sentinel = PhiSentinel("person.full", value)
        val raw = value.toByteArray(Charsets.UTF_8)
        val percent = URLEncoder.encode(value, "UTF-8").replace("+", "%20")
        val plants = listOf(
            SentinelEncoding.UTF8 to raw,
            SentinelEncoding.UTF16LE to value.toByteArray(Charsets.UTF_16LE),
            SentinelEncoding.UTF16BE to value.toByteArray(Charsets.UTF_16BE),
            SentinelEncoding.PERCENT to percent.toByteArray(),
            SentinelEncoding.PERCENT to lowercaseHex(percent).toByteArray(),
            SentinelEncoding.PERCENT to URLEncoder.encode(value, "UTF-8").toByteArray(),
        )
        for ((encoding, payload) in plants) {
            val found = findings(sentinel, "prefix:".toByteArray() + payload + ";".toByteArray())
            assertEquals(listOf(encoding), found.map { it.encoding }, encoding.wireName)
        }
        for (offset in 0 until 3) {
            val standard = Base64.getEncoder()
                .encodeToString(ByteArray(offset) { 0x41 } + raw + "tail".toByteArray())
            assertEquals(
                listOf(SentinelEncoding.BASE64),
                findings(sentinel, standard.toByteArray()).map { it.encoding },
                "offset $offset",
            )
            val urlSafe = Base64.getUrlEncoder()
                .encodeToString(ByteArray(offset) { 0x41 } + raw + "tail".toByteArray())
            val urlFindings = findings(sentinel, urlSafe.toByteArray()).map { it.encoding }
            assertTrue(urlFindings.isNotEmpty(), "offset $offset")
            assertTrue(
                urlFindings.all { it == SentinelEncoding.BASE64 || it == SentinelEncoding.BASE64URL },
            )
        }
    }

    @Test
    fun sentinelInFileOrDirectoryNameIsReportedWithoutTheName() {
        val root = directory("files")
        File(root, name.plantedValue).mkdirs()
        File(File(root, name.plantedValue), "notes.json").writeText("ok")
        val encoded = URLEncoder.encode(name.plantedValue, "UTF-8").replace("+", "%20")
        File(root, "export-$encoded.json").writeText("ok")

        val report = PhiSentinelSweep(listOf(name), listOf(SweepRoot(root, StorageClass.FILES))).run()

        assertEquals(
            listOf(listOf("file_name", "json", "percent"), listOf("file_name", "other", "utf8")),
            report.findings.map {
                listOf(it.location.wireName, it.fileKind?.wireName ?: "-", it.encoding.wireName)
            },
        )
        assertFalse((report.toJson() + report).contains("export-"))
    }

    @Test
    fun matchSpanningReadChunksIsFound() {
        val sentinel = PhiSentinel("record.id", "OMS-CHUNK-BOUNDARY-77")
        val contents = ByteArray(PhiSentinelSweep.CHUNK_SIZE - 7) { '.'.code.toByte() } +
            sentinel.plantedValue.toByteArray() +
            ByteArray(4_096) { '.'.code.toByte() }

        assertEquals(listOf(SentinelEncoding.UTF8), findings(sentinel, contents).map { it.encoding })
    }

    // Negative controls

    @Test
    fun negativeControlsStayCleanAndExhaustive() {
        val other = PhiSentinel("person.other", "Morvessa Tillcrane")
        val root = directory("cache")
        File(root, "prefix.txt").writeText("Vexorin Qualtran")
        File(root, "case.txt").writeText("vexorin qualtrane")
        File(root, "split.txt").writeText("Vexorin\nQualtrane")
        File(root, "other.txt").writeText(other.plantedValue)
        File(root, "empty.bin").writeBytes(ByteArray(0))

        val report = PhiSentinelSweep(
            sentinels = listOf(name),
            roots = listOf(
                SweepRoot(root, StorageClass.CACHE),
                SweepRoot(File(temporary.root, "missing"), StorageClass.FILES),
            ),
            logReaders = listOf(CapturedLogReader(listOf("redacted [NAME] seen", other.plantedValue))),
        ).run()

        assertTrue(report.clean, report.toString())
        assertTrue(report.exhaustive, report.toString())
        assertEquals(5, report.filesScanned)
        assertEquals(listOf(true, false), report.roots.map { it.present })
    }

    // Bounds

    @Test
    fun perFileAndTotalByteBoundsAreEnforcedAndReported() {
        val sentinel = PhiSentinel("record.id", "OMS-LATE-VALUE-0042")
        val root = directory("cache")
        File(root, "a.log").writeBytes(ByteArray(64) { '.'.code.toByte() } + sentinel.plantedValue.toByteArray())
        File(root, "b.log").writeText(sentinel.plantedValue)
        val roots = listOf(SweepRoot(root, StorageClass.CACHE))

        val perFile = PhiSentinelSweep(listOf(sentinel), roots, bounds = SweepBounds(maxBytesPerFile = 32)).run()
        assertEquals(listOf(SweepBound.MAX_BYTES_PER_FILE), perFile.boundsReached)
        assertEquals(1, perFile.truncatedFiles)
        assertEquals(32L + 19L, perFile.bytesScanned)
        assertEquals(listOf(1), perFile.findings.map { it.itemCount })
        assertFalse(perFile.exhaustive)

        val total = PhiSentinelSweep(listOf(sentinel), roots, bounds = SweepBounds(maxTotalBytes = 40)).run()
        assertEquals(listOf(SweepBound.MAX_TOTAL_BYTES), total.boundsReached)
        assertEquals(40L, total.bytesScanned)
        assertEquals(1, total.filesScanned)
        assertTrue(total.clean)
        assertFalse(total.exhaustive)
    }

    @Test
    fun fileCountAndDepthBoundsAreEnforcedAndReported() {
        val sentinel = PhiSentinel("record.id", "OMS-DEEP-VALUE-0042")
        val root = directory("files")
        listOf("a.txt", "b.txt", "c.txt").forEach { File(root, it).writeText("filler") }
        File(root, "one/two").mkdirs()
        File(root, "one/two/deep.txt").writeText(sentinel.plantedValue)
        val roots = listOf(SweepRoot(root, StorageClass.FILES))

        val files = PhiSentinelSweep(listOf(sentinel), roots, bounds = SweepBounds(maxFiles = 2)).run()
        assertEquals(listOf(SweepBound.MAX_FILES), files.boundsReached)
        assertEquals(2, files.filesScanned)

        val shallow = PhiSentinelSweep(listOf(sentinel), roots, bounds = SweepBounds(maxDepth = 1)).run()
        assertEquals(listOf(SweepBound.MAX_DEPTH), shallow.boundsReached)
        assertTrue(shallow.clean)
        assertEquals(3, shallow.filesScanned)

        val deep = PhiSentinelSweep(listOf(sentinel), roots, bounds = SweepBounds(maxDepth = 2)).run()
        assertEquals(emptyList(), deep.boundsReached)
        assertEquals(listOf(SweepLocation.FILE_CONTENT), deep.findings.map { it.location })
    }

    @Test
    fun timeBoundUsesInjectedClockAndSkipsRemainingReaders() {
        val sentinel = PhiSentinel("record.id", "OMS-SLOW-VALUE-0042")
        val root = directory("cache")
        (0 until 10).forEach { File(root, "f$it.txt").writeText("filler") }
        var ticks = 0L

        val report = PhiSentinelSweep(
            sentinels = listOf(sentinel),
            roots = listOf(SweepRoot(root, StorageClass.CACHE)),
            logReaders = listOf(CapturedLogReader(listOf(sentinel.plantedValue))),
            bounds = SweepBounds(maxDurationMillis = 5),
            clockMillis = { ++ticks },
        ).run()

        assertEquals(listOf(SweepBound.MAX_DURATION), report.boundsReached)
        assertTrue(report.filesScanned < 10)
        assertEquals(listOf(SweepLogStatus.SKIPPED), report.logs.map { it.status })
        assertTrue(report.clean)
        assertFalse(report.exhaustive)
    }

    @Test
    fun logEntryAndByteBoundsAreEnforcedAndReported() {
        val sentinel = PhiSentinel("record.id", "OMS-LOG-VALUE-0042")
        val entries = PhiSentinelSweep(
            sentinels = listOf(sentinel),
            roots = emptyList(),
            logReaders = listOf(CapturedLogReader(listOf("one", "two", sentinel.plantedValue))),
            bounds = SweepBounds(maxLogEntries = 2),
        ).run()
        assertEquals(listOf(SweepBound.MAX_LOG_ENTRIES), entries.boundsReached)
        assertEquals(2, entries.logEntriesScanned)
        assertTrue(entries.clean)

        val bytes = PhiSentinelSweep(
            sentinels = listOf(sentinel),
            roots = emptyList(),
            logReaders = listOf(CapturedLogReader(listOf("0123456", sentinel.plantedValue))),
            bounds = SweepBounds(maxLogBytes = 10),
        ).run()
        assertEquals(listOf(SweepBound.MAX_LOG_BYTES), bytes.boundsReached)
        assertEquals(10L, bytes.logBytesScanned)
        assertTrue(bytes.clean)
        assertFalse(bytes.exhaustive)
    }

    @Test
    fun invalidBoundsAndSentinelsAreRefusedWithValueFreeCodes() {
        val invalidBounds = listOf(
            SweepBounds(maxFiles = 0),
            SweepBounds(maxDepth = -1),
            SweepBounds(maxBytesPerFile = SweepBounds.CEILING.maxBytesPerFile + 1),
            SweepBounds(maxDurationMillis = 0),
            SweepBounds(maxLogEntries = 0),
        )
        for (bounds in invalidBounds) {
            val error = assertFailsWith<PhiSentinelSweepException> {
                PhiSentinelSweep(listOf(name), emptyList(), bounds = bounds)
            }
            assertEquals(SweepRefusal.BOUNDS_INVALID, error.refusal)
        }
        val refusals = listOf(
            SweepRefusal.SENTINEL_IDENTIFIER_INVALID to { PhiSentinel("Person", "OMS-VALID-VALUE") },
            SweepRefusal.SENTINEL_IDENTIFIER_INVALID to { PhiSentinel(".dot", "OMS-VALID-VALUE") },
            SweepRefusal.SENTINEL_LENGTH_OUT_OF_RANGE to { PhiSentinel("short", "Ada Lee") },
            SweepRefusal.SENTINELS_EMPTY to { PhiSentinelSweep(emptyList(), emptyList()) },
            SweepRefusal.SENTINEL_IDENTIFIER_DUPLICATE to { PhiSentinelSweep(listOf(name, name), emptyList()) },
            SweepRefusal.SENTINELS_TOO_MANY to {
                PhiSentinelSweep(
                    (0..PhiSentinel.MAXIMUM_COUNT).map { PhiSentinel.synthetic("s$it") },
                    emptyList(),
                )
            },
        )
        for ((code, action) in refusals) {
            val error = assertFailsWith<PhiSentinelSweepException> { action() }
            assertEquals(code, error.refusal)
            assertEquals(code.code, error.message)
        }
    }

    // Allow-list, links and storage layout

    @Test
    fun allowListSkipsModelArtifactContentsButStillSweepsNames() {
        val noBackup = directory("no_backup")
        val files = directory("files")
        File(noBackup, "weights.onnx").writeText(name.plantedValue)
        File(noBackup, "Model.mlmodelc").mkdirs()
        File(noBackup, "Model.mlmodelc/coremldata.bin").writeText(name.plantedValue)
        File(noBackup, "Model.mlmodelc/${name.plantedValue}.bin").writeText("weights")
        File(noBackup, "openmed-model-cache/repo").mkdirs()
        File(noBackup, "openmed-model-cache/repo/tokenizer.json").writeText(name.plantedValue)
        File(files, "openmed-model-cache/repo").mkdirs()
        File(files, "openmed-model-cache/repo/tokenizer.json").writeText(name.plantedValue)
        val roots = listOf(SweepRoot(noBackup, StorageClass.NO_BACKUP), SweepRoot(files, StorageClass.FILES))

        val report = PhiSentinelSweep(listOf(name), roots, allowList = ArtifactAllowList.MODEL_ARTIFACTS).run()

        assertEquals(4, report.allowListedFiles)
        assertEquals(
            listOf(listOf("file_content", "files", "json"), listOf("file_name", "no_backup", "other")),
            report.findings.map { listOf(it.location.wireName, it.surface, it.fileKind?.wireName ?: "-") },
        )
        val unlisted = PhiSentinelSweep(listOf(name), roots.take(1)).run()
        assertEquals(0, unlisted.allowListedFiles)
        assertEquals(
            3,
            unlisted.findings.filter { it.location == SweepLocation.FILE_CONTENT }.sumOf { it.itemCount },
        )
    }

    @Test
    fun symbolicLinksAreCountedButNotFollowed() {
        val root = directory("cache")
        val outside = directory("outside")
        File(outside, "secret.txt").writeText(name.plantedValue)
        Files.createSymbolicLink(File(root, "link").toPath(), outside.toPath())
        Files.createSymbolicLink(File(root, "file-link.txt").toPath(), File(outside, "secret.txt").toPath())

        val report = PhiSentinelSweep(listOf(name), listOf(SweepRoot(root, StorageClass.CACHE))).run()

        assertTrue(report.clean, report.toString())
        assertEquals(2, report.skippedSymbolicLinks)
        assertEquals(0, report.filesScanned)
    }

    @Test
    fun storageRootsCoverEachStorageClassExactlyOnce() {
        val data = directory("data")
        val external = directory("external/files")
        val externalCache = directory("external/cache")
        val planted = listOf(
            "files/a.txt", "cache/b.json", "code_cache/c.bin", "no_backup/d.db",
            "shared_prefs/org.openmed.xml", "databases/e.db-wal", "app_webview/f.log",
        )
        planted.forEach { relative ->
            File(data, relative).apply { parentFile?.mkdirs() }.writeText(name.plantedValue)
        }
        File(external, "g.txt").writeText(name.plantedValue)
        File(externalCache, "h.txt").writeText(name.plantedValue)
        val roots = AndroidAppStorageRoots.roots(
            dataDirectory = data,
            externalFilesDirectories = listOf(external),
            externalCacheDirectories = listOf(externalCache),
        )

        val report = PhiSentinelSweep(listOf(name), roots).run()

        assertEquals(StorageClass.values().toList(), roots.map { it.storageClass })
        assertEquals(planted.size + 2, report.filesScanned)
        assertEquals(
            listOf(
                "cache/json", "code_cache/other", "data_other/text", "databases/sqlite",
                "external_cache/text", "external_files/text", "files/text", "no_backup/sqlite",
                "shared_prefs/xml",
            ),
            report.findings.map { "${it.surface}/${it.fileKind?.wireName}" },
        )
        assertTrue(report.findings.all { it.itemCount == 1 })
    }

    // Value-free reporting

    @Test
    fun reportsAndDescriptionsContainNoSentinelValuesContentsNamesOrPaths() {
        val sentinels = listOf(name, PhiSentinel("record.id", "MRN 4471-OMS-90"))
        val root = directory("cache")
        sentinels.forEachIndexed { index, sentinel ->
            val value = sentinel.plantedValue
            File(root, "plain-$index.txt").writeText("context-$index $value")
            File(root, "wide-$index.bin").writeBytes(value.toByteArray(Charsets.UTF_16LE))
            File(root, "b64-$index.txt").writeText(Base64.getEncoder().encodeToString(value.toByteArray()))
            File(root, "$value-$index.txt").writeText("x")
        }
        val failing = object : ProcessLogReader {
            override val source = ProcessLogSource.LOGCAT

            override fun readEntries(visit: (String) -> Boolean) {
                throw IllegalStateException(name.plantedValue)
            }
        }

        val report = PhiSentinelSweep(
            sentinels = sentinels,
            roots = listOf(SweepRoot(root, StorageClass.CACHE)),
            logReaders = listOf(CapturedLogReader(sentinels.map { it.plantedValue }), failing),
        ).run()
        val text = (report.toJson() + report.toString()).toByteArray()

        assertFalse(report.clean)
        assertFalse(report.exhaustive)
        assertEquals(listOf(SweepLogStatus.READ, SweepLogStatus.UNREADABLE), report.logs.map { it.status })
        assertEquals(setOf("person.name", "record.id"), report.findings.map { it.sentinelIdentifier }.toSet())
        for (sentinel in sentinels) {
            for (needle in SentinelNeedles.make(sentinel.plantedValue, 0)) {
                assertFalse(text.containsBytes(needle.bytes), needle.encoding.wireName)
            }
            assertFalse(sentinel.toString().contains(sentinel.plantedValue))
        }
        val decoded = String(text)
        for (forbidden in listOf(temporary.root.path, root.path, "context-", "plain-", "wide-", "b64-", "-0.txt")) {
            assertFalse(decoded.contains(forbidden), forbidden)
        }
        assertTrue(report.toJson().startsWith("{\"allow_listed_files\":"))
        assertTrue(report.toJson().contains("\"schema\":\"openmed.phi_sentinel_sweep.v1\""))
    }

    @Test
    fun syntheticSentinelsAreUniqueAndValid() {
        val values = (0 until 32).map { PhiSentinel.synthetic("s$it").plantedValue }
        assertEquals(values.size, values.toSet().size)
        assertTrue(values.all { it.startsWith("OMS") && it.length == 16 })
    }

    // Logcat reader

    @Test
    fun logcatReaderRunsAPidScopedDumpAndFeedsEveryLine() {
        val launched = mutableListOf<List<String>>()
        val process = FakeProcess("--------- beginning of main\nI/OpenMed: lookup ${name.plantedValue}\nD/Other: ok\n")
        val reader = LogcatProcessLogReader(pid = 4242) { command ->
            launched += command
            process
        }

        val report = PhiSentinelSweep(listOf(name), emptyList(), logReaders = listOf(reader)).run()

        assertEquals(listOf(listOf("logcat", "-d", "-v", "tag", "--pid=4242")), launched)
        assertEquals(3, report.logEntriesScanned)
        assertEquals(listOf(listOf("log_entry", "logcat")), report.findings.map { listOf(it.location.wireName, it.surface) })
        assertTrue(report.exhaustive)
        assertTrue(process.destroyed)
    }

    @Test
    fun logcatReaderStopsAtTheEntryBoundAndReportsFailedDumpsValueFree() {
        val process = FakeProcess("a\nb\nc\nd\n")
        val bounded = PhiSentinelSweep(
            sentinels = listOf(name),
            roots = emptyList(),
            logReaders = listOf(LogcatProcessLogReader(1) { process }),
            bounds = SweepBounds(maxLogEntries = 2),
        ).run()
        assertEquals(2, bounded.logEntriesScanned)
        assertEquals(listOf(SweepBound.MAX_LOG_ENTRIES), bounded.boundsReached)
        assertTrue(process.destroyed)

        val failed = PhiSentinelSweep(
            sentinels = listOf(name),
            roots = emptyList(),
            logReaders = listOf(LogcatProcessLogReader(1) { FakeProcess("denied ${name.plantedValue}\n", exitCode = 1) }),
        ).run()
        assertEquals(listOf(SweepLogStatus.UNREADABLE), failed.logs.map { it.status })
        assertFalse(failed.exhaustive)
        assertFalse(failed.toJson().contains(name.plantedValue))
        assertEquals("logcat_unavailable", LogcatUnavailableException().message)
    }

    // Helpers

    private class FakeProcess(output: String, private val exitCode: Int = 0) : Process() {
        private val stdout = ByteArrayInputStream(output.toByteArray())
        var destroyed = false
            private set

        override fun getOutputStream(): OutputStream = OutputStream.nullOutputStream()

        override fun getInputStream(): InputStream = stdout

        override fun getErrorStream(): InputStream = InputStream.nullInputStream()

        override fun waitFor(): Int = exitCode

        override fun exitValue(): Int = exitCode

        override fun destroy() {
            destroyed = true
        }
    }

    private fun lowercaseHex(encoded: String): String {
        val output = StringBuilder()
        var pending = 0
        for (char in encoded) {
            when {
                char == '%' -> {
                    pending = 2
                    output.append(char)
                }
                pending > 0 -> {
                    pending -= 1
                    output.append(char.lowercaseChar())
                }
                else -> output.append(char)
            }
        }
        return output.toString()
    }

    private fun findings(sentinel: PhiSentinel, contents: ByteArray): List<SweepFinding> {
        val root = temporary.newFolder()
        File(root, "payload.bin").writeBytes(contents)
        return PhiSentinelSweep(listOf(sentinel), listOf(SweepRoot(root, StorageClass.CACHE))).run().findings
    }

    private fun directory(relative: String): File =
        File(temporary.root, relative).apply { mkdirs() }
}
