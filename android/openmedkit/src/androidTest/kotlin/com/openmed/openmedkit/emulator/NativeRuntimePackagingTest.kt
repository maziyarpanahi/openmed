package com.openmed.openmedkit.emulator

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtProvider
import android.os.Build
import android.system.Os
import android.system.OsConstants
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test

/**
 * Native packaging checks: the device matches its lane and every native
 * library OpenMedKit needs loads from the instrumented APK.
 *
 * On a 16 KB page-size device the dynamic linker refuses libraries whose
 * LOAD segments are aligned to less than 16 KB, so this test also catches a
 * misaligned library. The emulator lane's negative control strips the
 * libraries from the APK and expects `native_library_unloadable:<name>`.
 */
class NativeRuntimePackagingTest {
    @get:Rule
    val noNetwork = NoNetworkSocketRule()

    @Test
    fun deviceMatchesTheRequestedLane() {
        val pageSizeKb = Os.sysconf(OsConstants._SC_PAGESIZE) / 1024
        val abi = Build.SUPPORTED_ABIS.firstOrNull().orEmpty()
        LaneRecorder.environment(abi = abi, pageSizeKb = pageSizeKb, sdkInt = Build.VERSION.SDK_INT)

        LaneArguments.expectedPageSizeKb?.let { expected ->
            assertEquals("page_size_mismatch", expected.toLong(), pageSizeKb)
        }
    }

    @Test
    fun onnxRuntimeLibrariesLoad() {
        assertLoadsUnlessKnownBlocked(NativeLibraries.ONNX_RUNTIME)
        assertLoadsUnlessKnownBlocked(NativeLibraries.ONNX_RUNTIME_JNI)
        LaneArguments.assumeLoadable(NativeLibraries.ONNX_RUNTIME)
        LaneArguments.assumeLoadable(NativeLibraries.ONNX_RUNTIME_JNI)

        val providers = OrtEnvironment.getAvailableProviders()
        assertTrue("onnxruntime_cpu_provider_missing", OrtProvider.CPU in providers)
    }

    @Test
    fun tokenizerLibraryLoads() {
        assertLoadsUnlessKnownBlocked(NativeLibraries.TOKENIZER)
    }

    private fun assertLoadsUnlessKnownBlocked(library: String) {
        val loaded = try {
            System.loadLibrary(library)
            true
        } catch (error: UnsatisfiedLinkError) {
            false
        }
        if (library in LaneArguments.knownUnloadableLibraries) {
            // A known blocker must still reproduce; once it is fixed the lane
            // argument has to be removed so the dependent tests run again.
            val outcome = if (loaded) "known_blocker_resolved" else "known_unloadable"
            LaneRecorder.nativeLibrary(library, outcome)
            assertFalse("known_blocker_resolved:$library", loaded)
        } else {
            LaneRecorder.nativeLibrary(library, if (loaded) "loaded" else "unloadable")
            assertTrue("native_library_unloadable:$library", loaded)
        }
    }
}
