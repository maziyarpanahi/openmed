package com.openmed.openmedkit.emulator

import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assume.assumeTrue

/** Native libraries OpenMedKit needs at run time, in load order. */
internal object NativeLibraries {
    const val ONNX_RUNTIME = "onnxruntime"
    const val ONNX_RUNTIME_JNI = "onnxruntime4j_jni"
    const val TOKENIZER = "djl_tokenizer"

    val required: List<String> = listOf(ONNX_RUNTIME, ONNX_RUNTIME_JNI, TOKENIZER)
}

/**
 * Instrumentation arguments that describe the lane a run belongs to.
 *
 * CI passes them as `-Pandroid.testInstrumentationRunnerArguments.<name>=<value>`.
 * A local run without arguments checks everything and expects no page size.
 *
 * - `openmedLane`: lane name used in the value-free result records.
 * - `openmedExpectedPageSizeKb`: page size the device must report (4 or 16).
 * - `openmedKnownUnloadableNativeLibraries`: comma-separated libraries from
 *   [NativeLibraries.required] that are known not to load on this lane. The
 *   packaging test then requires the load to fail, so the lane notices when
 *   the blocker is fixed, and tests that need the library are skipped.
 */
internal object LaneArguments {
    private const val LANE = "openmedLane"
    private const val EXPECTED_PAGE_SIZE_KB = "openmedExpectedPageSizeKb"
    private const val KNOWN_UNLOADABLE = "openmedKnownUnloadableNativeLibraries"
    private val laneName = Regex("[a-z0-9][a-z0-9_-]{0,47}")

    val lane: String
        get() = argument(LANE)?.takeIf(laneName::matches) ?: "local"

    val expectedPageSizeKb: Int?
        get() = argument(EXPECTED_PAGE_SIZE_KB)?.let { value ->
            val kilobytes = value.toIntOrNull()
            require(kilobytes == 4 || kilobytes == 16) { "lane_argument_page_size" }
            kilobytes
        }

    val knownUnloadableLibraries: Set<String>
        get() {
            val libraries = argument(KNOWN_UNLOADABLE).orEmpty()
                .split(',')
                .map(String::trim)
                .filter(String::isNotEmpty)
                .toSet()
            require(NativeLibraries.required.containsAll(libraries)) {
                "lane_argument_known_unloadable"
            }
            return libraries
        }

    /** Skip the calling test when this lane cannot load [library]. */
    fun assumeLoadable(library: String) {
        assumeTrue(
            "known_unloadable_native_library:$library",
            library !in knownUnloadableLibraries,
        )
    }

    private fun argument(name: String): String? =
        InstrumentationRegistry.getArguments().getString(name)?.trim()?.ifEmpty { null }
}
