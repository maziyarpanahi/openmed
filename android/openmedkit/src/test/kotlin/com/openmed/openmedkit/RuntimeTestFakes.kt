@file:OptIn(ExperimentalCoroutinesApi::class)

package com.openmed.openmedkit

import java.io.Closeable
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.Deferred
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.async
import kotlinx.coroutines.delay
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.TestScope
import kotlinx.coroutines.withContext

/** Synthetic note; every identifier in it is fabricated. */
internal const val SYNTHETIC_NOTE = "Patient Jane Roe called 555-0100 about asthma."

/** Values that must never appear on any diagnostic or failure surface. */
internal val RUNTIME_SENTINELS = listOf(
    "Jane Roe",
    "555-0100",
    "/data/user/0/org.example.app/files/jane-roe",
    "sk-sentinel-secret",
    "reversible:Jane Roe",
)

/**
 * Token classifier that behaves like a native ONNX session.
 *
 * The native window run cannot observe coroutine cancellation (it runs under
 * `NonCancellable`), and the fake records any close that happens while a run
 * holds it, and any use after close.
 */
internal class FakeNativeClassifier(
    val sessionId: Int,
    private val nativeMillis: Long,
    private val hooks: FakeHooks,
) : OnnxTokenClassifier {
    var runs = 0
        private set
    var activeRuns = 0
        private set
    var closeCount = 0
        private set
    var closedWhileRunning = false
        private set
    var usedAfterClose = false
        private set

    override suspend fun predict(text: String): List<TokenClassificationPrediction> {
        if (closeCount > 0) {
            usedAfterClose = true
        }
        runs += 1
        activeRuns += 1
        try {
            withContext(NonCancellable) {
                hooks.nativeGate?.await() ?: delay(nativeMillis)
            }
        } finally {
            activeRuns -= 1
        }
        if (closeCount > 0) {
            usedAfterClose = true
        }
        hooks.afterNative?.invoke()
        return syntheticPredictions(text)
    }

    override fun tokenOffsets(text: String): List<IntRange> {
        hooks.onTokenize?.invoke()
        return defaultTokenOffsets(text)
    }

    override fun close() {
        if (activeRuns > 0) {
            closedWhileRunning = true
        }
        closeCount += 1
    }
}

/** Mutable stage hooks shared by every classifier a factory opens. */
internal class FakeHooks {
    var nativeGate: CompletableDeferred<Unit>? = null
    var afterNative: (() -> Unit)? = null
    var onTokenize: (() -> Unit)? = null
}

/** Session factory with virtual-time loading and optional failures. */
internal class FakeSessionFactory(
    private val loadMillis: Long = 10L,
    private val nativeMillis: Long = 100L,
    private val sessionIdBase: Int = 0,
) : OpenMedSessionFactory {
    val hooks = FakeHooks()
    val opened = mutableListOf<FakeNativeClassifier>()
    var openCalls = 0
        private set
    var failure: (() -> Throwable)? = null

    override suspend fun open(): OpenMedKit {
        openCalls += 1
        delay(loadMillis)
        failure?.let { throw it() }
        val classifier = FakeNativeClassifier(
            sessionId = sessionIdBase + opened.size + 1,
            nativeMillis = nativeMillis,
            hooks = hooks,
        )
        opened += classifier
        return OpenMedKit(classifier = classifier)
    }

    fun liveSessions(): Int = opened.count { it.closeCount == 0 }

    fun assertNoUnsafeClose() {
        opened.forEach { classifier ->
            check(!classifier.closedWhileRunning) { "session closed while running" }
            check(!classifier.usedAfterClose) { "session used after close" }
            check(classifier.closeCount <= 1) { "session closed more than once" }
        }
    }
}

/** Deterministic predictions over [SYNTHETIC_NOTE]-style text. */
internal fun syntheticPredictions(text: String): List<TokenClassificationPrediction> =
    listOf("Jane Roe" to "PERSON", "555-0100" to "PHONE").mapNotNull { (needle, label) ->
        val start = text.indexOf(needle)
        if (start < 0) {
            null
        } else {
            TokenClassificationPrediction(label, needle, 0.97f, start, start + needle.length)
        }
    }

/** Value-free outcome of one runtime call. */
internal sealed class CallOutcome {
    data class Completed(val value: Any?) : CallOutcome()

    data class Refused(val reason: OpenMedRuntimeFailureReason) : CallOutcome()
}

/**
 * Start a runtime call whose typed refusal is captured instead of failing the
 * test scope. Cancellation still cancels the returned deferred.
 */
internal fun TestScope.call(block: suspend () -> Any?): Deferred<CallOutcome> = async {
    try {
        CallOutcome.Completed(block())
    } catch (error: OpenMedRuntimeException) {
        CallOutcome.Refused(error.reason)
    }
}

/** Runtime on the test scheduler with a virtual-time clock. */
internal fun TestScope.testRuntime(
    factory: OpenMedSessionFactory,
    policy: OpenMedRuntimePolicy = OpenMedRuntimePolicy(),
): OpenMedRuntime = OpenMedRuntime(
    factory = factory,
    policy = policy,
    dispatcher = StandardTestDispatcher(testScheduler),
    clock = { testScheduler.currentTime },
)

/** Lifecycle source that a test drives by hand. */
internal class FakeLifecycleSource : OpenMedLifecycleSource {
    val observers = mutableListOf<OpenMedLifecycleObserver>()

    override fun subscribe(observer: OpenMedLifecycleObserver): Closeable {
        observers += observer
        return Closeable { observers.remove(observer) }
    }

    fun background() = observers.toList().forEach { it.onHostBackground() }

    fun foreground() = observers.toList().forEach { it.onHostForeground() }

    fun pressure(level: OpenMedMemoryPressure) =
        observers.toList().forEach { it.onMemoryPressure(level) }
}

/** Unwrap a deferred that is expected to have been cancelled. */
internal suspend fun Deferred<*>.awaitCancelled(): CancellationException {
    try {
        await()
    } catch (error: CancellationException) {
        return error
    }
    error("call completed instead of being cancelled")
}
