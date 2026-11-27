package com.openmed.openmedkit

import java.io.Closeable
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

/**
 * Coroutine scope and [OpenMedRuntime] owned by one bridge host.
 *
 * A bridge (for example the React Native module) launches every operation in
 * [scope] and calls [invalidate] when its host is torn down. Invalidation
 * cancels every owned coroutine, closes the runtime so native sessions are
 * released after their final lease, and unsubscribes lifecycle sources. The
 * host scope keeps no reference to caller input.
 *
 * @param policy Policy for the runtime this host creates.
 * @param lifecycleSources Sources bound to the runtime, such as [OpenMedTrimMemorySource].
 * @param dispatcher Dispatcher for owned coroutines and runtime work.
 * @param clock Monotonic clock for memory-pressure cool-downs.
 */
class OpenMedHostScope(
    private val policy: OpenMedRuntimePolicy = OpenMedRuntimePolicy(),
    private val lifecycleSources: List<OpenMedLifecycleSource> = emptyList(),
    private val dispatcher: CoroutineDispatcher = Dispatchers.Default,
    private val clock: OpenMedRuntimeClock = OpenMedRuntimeClock.MONOTONIC,
) : Closeable {
    private val job = SupervisorJob()
    private val lock = Any()
    private val loadMutex = Mutex()
    private var runtime: OpenMedRuntime? = null
    private var loadedKey: String? = null
    private var invalidated = false

    /** Scope that owns every bridge coroutine; [invalidate] cancels it. */
    val scope: CoroutineScope = CoroutineScope(job + dispatcher)

    /** True once [invalidate] has run. */
    val isInvalidated: Boolean
        get() = synchronized(lock) { invalidated }

    /**
     * Launch bridge work in [scope].
     *
     * @param block The work; it is cancelled when the host is invalidated.
     * @return The launched job, already cancelled when the host is invalidated.
     */
    fun launch(block: suspend CoroutineScope.() -> Unit): Job = scope.launch(block = block)

    /**
     * Load a model, or replace the loaded one when [key] differs.
     *
     * Replacement uses [OpenMedRuntime.replaceModel], so operations still
     * running on the previous model report
     * [OpenMedRuntimeFailureReason.STALE_GENERATION]. The session is opened
     * eagerly so load failures surface here.
     *
     * @param key Stable cache key for the model source.
     * @param factory Factory that opens the verified model.
     * @return True when a model was loaded; false when [key] was already loaded.
     * @throws OpenMedRuntimeException When invalidated or the load failed.
     */
    suspend fun load(key: String, factory: OpenMedSessionFactory): Boolean {
        return loadMutex.withLock {
            val (target, created) = synchronized(lock) {
                if (invalidated) {
                    throw OpenMedRuntimeException(OpenMedRuntimeFailureReason.RUNTIME_CLOSED)
                }
                val existing = runtime
                if (existing != null && loadedKey == key) {
                    return false
                }
                loadedKey = null
                if (existing != null) {
                    existing.replaceModel(factory)
                    existing to false
                } else {
                    OpenMedRuntime(factory, policy, dispatcher, clock).also {
                        runtime = it
                    } to true
                }
            }
            if (created) {
                lifecycleSources.forEach { target.bindLifecycle(it) }
            }
            target.warmUp()
            synchronized(lock) {
                if (runtime === target && !invalidated) {
                    loadedKey = key
                }
            }
            true
        }
    }

    /**
     * The loaded runtime.
     *
     * @throws OpenMedRuntimeException When invalidated or no model is loaded.
     */
    fun requireRuntime(): OpenMedRuntime = synchronized(lock) {
        if (invalidated) {
            throw OpenMedRuntimeException(OpenMedRuntimeFailureReason.RUNTIME_CLOSED)
        }
        runtime ?: throw OpenMedRuntimeException(OpenMedRuntimeFailureReason.MODEL_NOT_LOADED)
    }

    /** The loaded runtime, or null. */
    fun runtimeOrNull(): OpenMedRuntime? = synchronized(lock) { runtime }

    /**
     * Tear down the host idempotently.
     *
     * Cancels every coroutine in [scope], closes the runtime (cancelling its
     * active operations), and drops every reference to it.
     */
    fun invalidate() {
        val closing = synchronized(lock) {
            if (invalidated) {
                return
            }
            invalidated = true
            loadedKey = null
            runtime.also { runtime = null }
        }
        job.cancel()
        closing?.close()
    }

    /** Same as [invalidate]. */
    override fun close() {
        invalidate()
    }
}
