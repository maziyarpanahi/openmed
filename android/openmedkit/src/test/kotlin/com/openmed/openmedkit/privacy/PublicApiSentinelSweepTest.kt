package com.openmed.openmedkit.privacy

import android.content.Context
import android.util.Log
import com.openmed.openmedkit.OnnxTokenClassifier
import com.openmed.openmedkit.OpenMedKit
import com.openmed.openmedkit.TokenClassificationPrediction
import com.openmed.openmedkit.policy.PolicyProfiles
import com.openmed.openmedkit.util.SafeLog
import com.openmed.openmedkit.util.SafeLogEvent
import java.io.File
import kotlin.test.AfterTest
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue
import kotlinx.coroutines.test.runTest
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.RuntimeEnvironment
import org.robolectric.annotation.Config
import org.robolectric.shadows.ShadowLog

/**
 * Runs the public OpenMedKit text APIs on sentinel inputs inside a Robolectric app
 * sandbox, then sweeps the sandbox's storage roots, `android.util.Log` output and the
 * internal SafeLog diagnostics a host sink would receive.
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [33])
class PublicApiSentinelSweepTest {
    private lateinit var context: Context

    @BeforeTest
    fun setUp() {
        context = RuntimeEnvironment.getApplication()
        ShadowLog.clear()
    }

    @AfterTest
    fun tearDown() {
        ShadowLog.clear()
    }

    @Test
    fun publicTextApisLeaveNoSentinelInAppStorageOrProcessLog() = runTest {
        val person = PhiSentinel.synthetic("workload.person")
        val record = PhiSentinel.synthetic("workload.record")
        val phone = PhiSentinel("workload.phone", "+1 (555) 01${(10..99).random()}-${(1000..9999).random()}")
        val sentinels = listOf(person, record, phone)
        val source = "Patient ${person.plantedValue} MRN ${record.plantedValue} " +
            "phone ${phone.plantedValue}. Follow-up for ${person.plantedValue} next week."
        val diagnostics = mutableListOf<SafeLogEvent>()
        val previous = SafeLog.installSinkForTesting(diagnostics::add)
        val kit = OpenMedKit(classifier = SentinelClassifier(sentinels))
        try {
            assertTrue(kit.analyzeText(source).isNotEmpty())
            kit.extractPii(source)
            kit.extractPiiChunked(source, chunkTokenLimit = 4, tokenOverlap = 1)
            for (profile in PolicyProfiles.bundledProfileNames) {
                kit.deidentify(source, policy = profile)
            }
        } finally {
            kit.close()
            SafeLog.installSinkForTesting(previous)
        }
        assertTrue(diagnostics.isNotEmpty())

        val report = sweep(sentinels, diagnostics.map(SafeLogEvent::toString))

        assertTrue(report.clean, report.toString())
        assertTrue(report.exhaustive, report.toString())
        assertTrue(report.roots.first { it.storageClass == StorageClass.FILES }.present)
        assertEquals(
            listOf(SweepLogStatus.READ, SweepLogStatus.READ),
            report.logs.map { it.status },
        )
    }

    @Test
    fun sandboxRootsAndShadowLogDetectPlantedLeaks() {
        val leaked = PhiSentinel.synthetic("planted.leak")
        context.getSharedPreferences("planted", Context.MODE_PRIVATE)
            .edit()
            .putString("draft", leaked.plantedValue)
            .commit()
        File(context.cacheDir, "draft.txt").writeText(leaked.plantedValue)
        File(context.noBackupFilesDir, "state.bin")
            .writeBytes(leaked.plantedValue.toByteArray(Charsets.UTF_16BE))
        Log.w("OpenMedHost", "lookup ${leaked.plantedValue}")

        val report = sweep(listOf(leaked), emptyList())

        assertFalse(report.clean)
        assertEquals(
            listOf(
                listOf("file_content", "cache", "text", "utf8"),
                listOf("file_content", "no_backup", "other", "utf16be"),
                listOf("file_content", "shared_prefs", "xml", "utf8"),
                listOf("log_entry", "robolectric_shadow_log", "-", "utf8"),
            ),
            report.findings.map {
                listOf(it.location.wireName, it.surface, it.fileKind?.wireName ?: "-", it.encoding.wireName)
            },
        )
        assertFalse(report.toJson().contains(leaked.plantedValue))
        assertFalse(report.toJson().contains(context.dataDir.path))
    }

    private fun sweep(sentinels: List<PhiSentinel>, captured: List<String>): PhiSentinelSweepReport =
        PhiSentinelSweep(
            sentinels = sentinels,
            roots = AndroidAppStorageRoots.of(context),
            logReaders = listOf(ShadowLogProcessLogReader(), CapturedLogReader(captured)),
            allowList = ArtifactAllowList.MODEL_ARTIFACTS,
        ).run()

    /** Labels every planted sentinel occurrence, with offsets into the text it is given. */
    private class SentinelClassifier(private val sentinels: List<PhiSentinel>) : OnnxTokenClassifier {
        override suspend fun predict(text: String): List<TokenClassificationPrediction> =
            sentinels.flatMap { sentinel ->
                Regex(Regex.escape(sentinel.plantedValue)).findAll(text).map { match ->
                    TokenClassificationPrediction(
                        label = if (sentinel.identifier.endsWith("phone")) "PHONE" else "PERSON",
                        text = match.value,
                        confidence = 0.99f,
                        start = match.range.first,
                        end = match.range.last + 1,
                    )
                }.toList()
            }.sortedBy { it.start }
    }
}

/** Reads Robolectric's captured `android.util.Log` lines, tags and throwables included. */
class ShadowLogProcessLogReader : ProcessLogReader {
    override val source: ProcessLogSource = ProcessLogSource.ROBOLECTRIC_SHADOW_LOG

    override fun readEntries(visit: (String) -> Boolean) {
        for (item in ShadowLog.getLogs()) {
            val line = buildString {
                append(item.tag).append(": ").append(item.msg)
                item.throwable?.let { append(' ').append(it) }
            }
            if (!visit(line)) {
                return
            }
        }
    }
}
