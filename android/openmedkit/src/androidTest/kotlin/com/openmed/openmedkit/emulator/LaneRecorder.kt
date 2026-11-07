package com.openmed.openmedkit.emulator

import android.os.SystemClock
import android.util.Log
import kotlinx.serialization.json.JsonObjectBuilder
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import kotlinx.serialization.json.putJsonArray

/**
 * Writes one value-free JSON record per lane check to logcat.
 *
 * Records carry only the lane name, case identifiers, check names, labels,
 * offsets, outcomes, reason codes and timings: never input text, span surface
 * text, file paths or exception messages. CI collects them with
 * `adb logcat -s OpenMedEmulatorLane:I`.
 */
internal object LaneRecorder {
    const val TAG = "OpenMedEmulatorLane"
    private val reasonCode = Regex("^[a-z][a-z0-9_]*(:[A-Za-z0-9_.-]+)*")

    /** Record the device facts a lane run depends on. */
    fun environment(abi: String, pageSizeKb: Long, sdkInt: Int) {
        write {
            put("record", "environment")
            put("lane", LaneArguments.lane)
            put("abi", abi)
            put("page_size_kb", pageSizeKb)
            put("sdk_int", sdkInt)
        }
    }

    /**
     * Time [block] and record its outcome for [caseId].
     *
     * [block] returns the spans it verified; a failure is recorded with its
     * reason code (the prefix of a lane assertion message, or the exception
     * class) and rethrown.
     */
    inline fun check(caseId: String, check: String, block: () -> List<LaneSpan>) {
        val started = SystemClock.elapsedRealtime()
        val spans = try {
            block()
        } catch (failure: Throwable) {
            recordFailure(caseId, check, failure, SystemClock.elapsedRealtime() - started)
            throw failure
        }
        recordPass(caseId, check, spans, SystemClock.elapsedRealtime() - started)
    }

    @PublishedApi
    internal fun recordPass(caseId: String, check: String, spans: List<LaneSpan>, elapsedMs: Long) {
        write {
            put("record", "check")
            put("lane", LaneArguments.lane)
            put("case", caseId)
            put("check", check)
            put("outcome", "pass")
            put("elapsed_ms", elapsedMs)
            putJsonArray("spans") {
                spans.forEach { span ->
                    add(
                        buildJsonObject {
                            put("label", span.label)
                            put("start", span.start)
                            put("end", span.end)
                        },
                    )
                }
            }
        }
    }

    @PublishedApi
    internal fun recordFailure(caseId: String, check: String, failure: Throwable, elapsedMs: Long) {
        write {
            put("record", "check")
            put("lane", LaneArguments.lane)
            put("case", caseId)
            put("check", check)
            put("outcome", "fail")
            put("reason", reasonFor(failure))
            put("elapsed_ms", elapsedMs)
        }
    }

    /** Record whether one of [NativeLibraries.required] loaded. */
    fun nativeLibrary(library: String, outcome: String) {
        write {
            put("record", "native_library")
            put("lane", LaneArguments.lane)
            put("library", library)
            put("outcome", outcome)
        }
    }

    private fun reasonFor(failure: Throwable): String {
        val message = failure.message.orEmpty()
        return reasonCode.find(message)?.value?.takeIf { failure is AssertionError }
            ?: failure.javaClass.simpleName.ifEmpty { "Throwable" }
    }

    private fun write(build: JsonObjectBuilder.() -> Unit) {
        Log.i(TAG, buildJsonObject(build).toString())
    }
}
