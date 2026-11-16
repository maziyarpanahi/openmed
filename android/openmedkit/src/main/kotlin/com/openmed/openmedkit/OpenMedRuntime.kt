package com.openmed.openmedkit

import com.openmed.openmedkit.cache.ModelCache
import com.openmed.openmedkit.catalog.ModelCatalogEntry
import com.openmed.openmedkit.policy.PolicyProfiles
import java.io.Closeable
import java.io.File
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.CoroutineStart
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.async
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.isActive
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext

/** Lifecycle states of an [OpenMedRuntime]. */
enum class OpenMedRuntimeState {
    /** No session is open; the next admitted operation opens one. */
    IDLE,

    /** One session is being opened; concurrent callers wait for the same load. */
    LOADING,

    /** A session is open and no operation holds a lease. */
    READY,

    /** One or more operations hold leases; see [OpenMedRuntimeStatus.activeLeases]. */
    BUSY,

    /** Close was requested; admission is closed and active leases are finishing. */
    DRAINING,

    /** Terminal: every session is closed and no lease remains. */
    RELEASED,
}

/** Whether an [OpenMedRuntime] currently admits new operations. */
enum class OpenMedAdmission {
    OPEN,
    PAUSED_BACKGROUND,
    PAUSED_MEMORY_PRESSURE,
    CLOSED,
}

/**
 * Value-free snapshot of an [OpenMedRuntime].
 *
 * @property state Current lifecycle state.
 * @property admission Whether new work is admitted.
 * @property activeLeases Operations currently holding a session lease.
 * @property generation Model generation; [OpenMedRuntime.replaceModel] increments it.
 * @property liveSessions Sessions opened and not yet closed, including draining ones.
 * @property sessionsOpened Sessions opened over the runtime's lifetime.
 */
data class OpenMedRuntimeStatus(
    val state: OpenMedRuntimeState,
    val admission: OpenMedAdmission,
    val activeLeases: Int,
    val generation: Long,
    val liveSessions: Int,
    val sessionsOpened: Long,
)

/**
 * Admission and resource policy for an [OpenMedRuntime].
 *
 * Defaults: [OpenMedMemoryPressure.LOW] releases idle sessions,
 * [OpenMedMemoryPressure.CRITICAL] also stops admission for
 * [pressureCooldownMillis] (or until a foreground return or an explicit
 * [OpenMedMemoryPressure.NONE] signal), background hosts refuse new work and
 * release idle sessions, and three consecutive load failures stop reloading.
 * Active work is never cancelled by memory pressure or backgrounding.
 *
 * @property releaseIdleAt Lowest pressure that releases idle sessions, or `null` to never release.
 * @property refuseAdmissionAt Lowest pressure that stops admission, or `null` to never stop.
 * @property pressureCooldownMillis How long a memory-pressure pause lasts after the last signal.
 * @property admitInBackground Whether new work is admitted while the host is in the background.
 * @property releaseIdleInBackground Whether a background transition releases idle sessions.
 * @property maxConsecutiveLoadFailures Load failures tolerated before reloading stops.
 */
data class OpenMedRuntimePolicy(
    val releaseIdleAt: OpenMedMemoryPressure? = OpenMedMemoryPressure.LOW,
    val refuseAdmissionAt: OpenMedMemoryPressure? = OpenMedMemoryPressure.CRITICAL,
    val pressureCooldownMillis: Long = DEFAULT_PRESSURE_COOLDOWN_MILLIS,
    val admitInBackground: Boolean = false,
    val releaseIdleInBackground: Boolean = true,
    val maxConsecutiveLoadFailures: Int = DEFAULT_MAX_CONSECUTIVE_LOAD_FAILURES,
) {
    init {
        require(releaseIdleAt != OpenMedMemoryPressure.NONE) {
            "releaseIdleAt must be a pressure level or null"
        }
        require(refuseAdmissionAt != OpenMedMemoryPressure.NONE) {
            "refuseAdmissionAt must be a pressure level or null"
        }
        require(pressureCooldownMillis >= 0) { "pressureCooldownMillis must be non-negative" }
        require(maxConsecutiveLoadFailures > 0) {
            "maxConsecutiveLoadFailures must be greater than zero"
        }
    }

    companion object {
        /** Default memory-pressure admission pause. */
        const val DEFAULT_PRESSURE_COOLDOWN_MILLIS: Long = 30_000L

        /** Default bound on consecutive session load failures. */
        const val DEFAULT_MAX_CONSECUTIVE_LOAD_FAILURES: Int = 3
    }
}

/** Monotonic clock used for memory-pressure cool-downs. */
fun interface OpenMedRuntimeClock {
    /** Milliseconds from an arbitrary fixed origin; must never decrease. */
    fun elapsedMillis(): Long

    companion object {
        /** `System.nanoTime` based monotonic clock. */
        @JvmField
        val MONOTONIC: OpenMedRuntimeClock = OpenMedRuntimeClock { System.nanoTime() / 1_000_000L }
    }
}

/**
 * Opens one verified OpenMedKit session.
 *
 * The runtime calls [open] for the first operation and again after a
 * policy-driven release, so every reacquisition re-runs the factory's bundle,
 * execution-plan, and policy verification before new work starts. A factory
 * must open the same model, backend, and local-only posture each time; only
 * [OpenMedRuntime.replaceModel] changes them.
 */
fun interface OpenMedSessionFactory {
    /** Open a new session; the runtime owns and closes the returned facade. */
    suspend fun open(): OpenMedKit

    companion object {
        /**
         * Open an exported ONNX directory with [OpenMedKit.fromDirectory].
         *
         * @param modelDirectory Local model directory.
         * @param variant `int8`, `fp32`, or `fp16`.
         * @return A factory that opens the directory on each acquisition.
         */
        @JvmStatic
        fun fromDirectory(
            modelDirectory: File,
            variant: String = "int8",
        ): OpenMedSessionFactory {
            OpenMedKit.modelFileNameForVariant(variant)
            return OpenMedSessionFactory { OpenMedKit.fromDirectory(modelDirectory, variant) }
        }

        /**
         * Open a verified [ModelCache] entry and hold a cache lease while open.
         *
         * Each acquisition re-checks the entry's integrity metadata. The lease
         * protects the directory from [ModelCache.evict] and budget trimming
         * until the session is closed.
         *
         * @param cache The app-private model cache.
         * @param entry The catalog entry whose integrity hash must match.
         * @param variant `int8`, `fp32`, or `fp16`.
         * @return A factory that opens the cached model on each acquisition.
         */
        @JvmStatic
        fun fromCache(
            cache: ModelCache,
            entry: ModelCatalogEntry,
            variant: String = "int8",
        ): OpenMedSessionFactory {
            val modelFileName = OpenMedKit.modelFileNameForVariant(variant)
            return fromCache(cache, entry) { directory ->
                BackendOnnxTokenClassifier(
                    OpenMedBackend(
                        modelDirectory = directory,
                        modelFile = File(directory, modelFileName),
                    ),
                )
            }
        }

        internal fun fromCache(
            cache: ModelCache,
            entry: ModelCatalogEntry,
            classifierFactory: (File) -> OnnxTokenClassifier,
        ): OpenMedSessionFactory = OpenMedSessionFactory {
            val lease = cache.acquireLease(entry) ?: throw CachedModelUnavailableException()
            try {
                OpenMedKit(
                    classifier = LeaseReleasingClassifier(
                        classifierFactory(lease.directory),
                        lease,
                    ),
                )
            } catch (error: Throwable) {
                lease.close()
                throw error
            }
        }
    }
}

/** Raised by a cache-backed factory when no verified entry can be leased. */
internal class CachedModelUnavailableException :
    IllegalStateException("verified cached model is not available")

private class LeaseReleasingClassifier(
    private val delegate: OnnxTokenClassifier,
    private val lease: Closeable,
) : OnnxTokenClassifier by delegate {
    override fun close() {
        try {
            delegate.close()
        } finally {
            lease.close()
        }
    }
}

/**
 * Lifecycle-owned OpenMedKit runtime for Android hosts.
 *
 * The runtime opens sessions lazily through an [OpenMedSessionFactory] and
 * moves through [OpenMedRuntimeState]. Every operation holds a lease on the
 * session it started on; a session is closed only after its final lease is
 * released, so `close`, [replaceModel], memory pressure, and backgrounding can
 * never free a native session while a run uses it.
 *
 * Cancellation: cancelling the calling coroutine cancels the operation. The
 * pipeline checks for cancellation before tokenization, after native
 * inference, and after decoding, merging, and policy application, so the
 * cancellation bound is one in-flight native window run (native execution
 * itself cannot be interrupted). A cancelled caller receives a
 * `CancellationException` and never a late result.
 *
 * Outcomes stay distinguishable: completed operations return their result;
 * cancelled callers get `CancellationException`; refusals and revocations
 * raise [OpenMedRuntimeException] with a stable reason. An operation running
 * when [close] or [replaceModel] is called is cancelled and reports
 * [OpenMedRuntimeFailureReason.RUNTIME_CLOSED] or
 * [OpenMedRuntimeFailureReason.STALE_GENERATION]; its result is discarded.
 *
 * The runtime never changes backend, model, policy, or local-only posture on
 * its own and never persists caller input. It launches no coroutines of its
 * own: operations run as children of the caller.
 *
 * @param factory Opens verified sessions for the current model generation.
 * @param policy Admission and resource policy.
 * @param dispatcher Dispatcher for loading sessions and running operations.
 * @param clock Monotonic clock for memory-pressure cool-downs.
 */
class OpenMedRuntime(
    factory: OpenMedSessionFactory,
    private val policy: OpenMedRuntimePolicy = OpenMedRuntimePolicy(),
    private val dispatcher: CoroutineDispatcher = Dispatchers.Default,
    private val clock: OpenMedRuntimeClock = OpenMedRuntimeClock.MONOTONIC,
) : Closeable, OpenMedLifecycleObserver {
    private val lock = Any()
    private val loadMutex = Mutex()

    private var factory: OpenMedSessionFactory = factory
    private var generation = 1L
    private var current: SessionHolder? = null
    private val retired = mutableListOf<SessionHolder>()
    private val leases = LinkedHashSet<SessionLease>()
    private val subscriptions = mutableListOf<Closeable>()
    private var loading = false
    private var pendingReleaseWhenIdle = false
    private var backgrounded = false
    private var pressurePausedAt: Long? = null
    private var consecutiveLoadFailures = 0
    private var closeRequested = false
    private var cancelActiveOnClose = false
    private var released = false
    private var sessionsOpened = 0L
    private var sessionsClosed = 0L
    private val history = ArrayDeque<OpenMedRuntimeState>()
    private val mutableStatus = MutableStateFlow(snapshotLocked())

    init {
        history.addLast(OpenMedRuntimeState.IDLE)
    }

    /** Value-free status, updated on every transition. */
    val status: StateFlow<OpenMedRuntimeStatus> = mutableStatus.asStateFlow()

    /**
     * Run token classification and return entities above the threshold.
     *
     * @param text Source text; it is never retained after the call.
     * @param confidenceThreshold Minimum confidence in `0.0..1.0`.
     * @return Entities sorted by offset.
     * @throws OpenMedRuntimeException When refused, revoked, or the load failed.
     */
    suspend fun analyzeText(
        text: String,
        confidenceThreshold: Float = 0.5f,
    ): List<EntityPrediction> = execute { it.analyzeText(text, confidenceThreshold) }

    /**
     * Run PII detection with span repair and optional smart merging.
     *
     * @param text Source text; it is never retained after the call.
     * @param confidenceThreshold Minimum confidence in `0.0..1.0`.
     * @param useSmartMerging Whether to merge overlapping spans.
     * @return Entities sorted by offset.
     * @throws OpenMedRuntimeException When refused, revoked, or the load failed.
     */
    suspend fun extractPii(
        text: String,
        confidenceThreshold: Float = 0.5f,
        useSmartMerging: Boolean = true,
    ): List<EntityPrediction> = execute {
        it.extractPii(text, confidenceThreshold, useSmartMerging)
    }

    /**
     * Run PII detection over overlapping token windows.
     *
     * @param text Source text; it is never retained after the call.
     * @param confidenceThreshold Minimum confidence in `0.0..1.0`.
     * @param chunkTokenLimit Tokens per window.
     * @param tokenOverlap Tokens shared by adjacent windows.
     * @param useSmartMerging Whether to merge overlapping spans.
     * @return Entities whose offsets reference the full input.
     * @throws OpenMedRuntimeException When refused, revoked, or the load failed.
     */
    suspend fun extractPiiChunked(
        text: String,
        confidenceThreshold: Float = 0.5f,
        chunkTokenLimit: Int = 256,
        tokenOverlap: Int = 32,
        useSmartMerging: Boolean = true,
    ): List<EntityPrediction> = execute {
        it.extractPiiChunked(
            text = text,
            confidenceThreshold = confidenceThreshold,
            chunkTokenLimit = chunkTokenLimit,
            tokenOverlap = tokenOverlap,
            useSmartMerging = useSmartMerging,
        )
    }

    /**
     * De-identify text under a bundled policy profile.
     *
     * @param text Source text; it is never retained after the call.
     * @param policy Bundled policy profile name.
     * @param confidenceThreshold Minimum confidence in `0.0..1.0`.
     * @param useSmartMerging Whether to merge overlapping spans.
     * @return The policy result.
     * @throws OpenMedRuntimeException When refused, revoked, or the load failed.
     */
    suspend fun deidentify(
        text: String,
        policy: String = PolicyProfiles.DEFAULT_PROFILE,
        confidenceThreshold: Float = 0.5f,
        useSmartMerging: Boolean = true,
    ): PolicyDeidentificationResult = execute {
        it.deidentify(text, policy, confidenceThreshold, useSmartMerging)
    }

    /**
     * Open the current generation's session now instead of on first use.
     *
     * @throws OpenMedRuntimeException When refused or the load failed.
     */
    suspend fun warmUp() {
        execute { }
    }

    /**
     * Switch to a new model generation.
     *
     * New operations open sessions through [factory]. Operations still running
     * on the previous generation are cancelled and report
     * [OpenMedRuntimeFailureReason.STALE_GENERATION]; the previous session is
     * closed once its final lease is released. The load-failure bound resets.
     *
     * @param factory Factory for the new generation.
     * @throws OpenMedRuntimeException When the runtime is closed.
     */
    fun replaceModel(factory: OpenMedSessionFactory) {
        val stale = synchronized(lock) {
            if (closeRequested || released) {
                throw OpenMedRuntimeException(OpenMedRuntimeFailureReason.RUNTIME_CLOSED)
            }
            this.factory = factory
            generation += 1
            consecutiveLoadFailures = 0
            pendingReleaseWhenIdle = false
            current?.let { previous ->
                current = null
                if (previous.leases == 0) {
                    closeHolderLocked(previous)
                } else {
                    previous.retired = true
                    retired += previous
                }
            }
            publishLocked()
            leases.filter { it.holder.generation < generation }
        }
        stale.forEach { it.revoke(OpenMedRuntimeFailureReason.STALE_GENERATION) }
    }

    override fun onHostForeground() {
        synchronized(lock) {
            if (closeRequested || released) {
                return
            }
            backgrounded = false
            pressurePausedAt = null
            pendingReleaseWhenIdle = false
            current?.releaseWhenIdle = false
            publishLocked()
        }
    }

    override fun onHostBackground() {
        synchronized(lock) {
            if (closeRequested || released) {
                return
            }
            backgrounded = true
            if (policy.releaseIdleInBackground) {
                releaseIdleLocked()
            }
            publishLocked()
        }
    }

    override fun onMemoryPressure(pressure: OpenMedMemoryPressure) {
        synchronized(lock) {
            if (closeRequested || released) {
                return
            }
            if (pressure == OpenMedMemoryPressure.NONE) {
                pressurePausedAt = null
                pendingReleaseWhenIdle = false
                current?.releaseWhenIdle = false
            }
            val refuseAt = policy.refuseAdmissionAt
            if (refuseAt != null && pressure >= refuseAt) {
                pressurePausedAt = clock.elapsedMillis()
            }
            val releaseAt = policy.releaseIdleAt
            if (releaseAt != null && pressure >= releaseAt) {
                releaseIdleLocked()
            }
            publishLocked()
        }
    }

    /**
     * Apply a raw Android `onTrimMemory` level.
     *
     * @param level The `ComponentCallbacks2` trim level.
     */
    fun onTrimMemory(level: Int) {
        onMemoryPressure(OpenMedMemoryPressure.fromTrimLevel(level))
    }

    /**
     * Subscribe this runtime to a lifecycle source until either is closed.
     *
     * @param source Process-lifecycle or trim-memory source.
     * @return A handle that unsubscribes; [close] unsubscribes automatically.
     * @throws OpenMedRuntimeException When the runtime is closed.
     */
    fun bindLifecycle(source: OpenMedLifecycleSource): Closeable {
        synchronized(lock) {
            if (closeRequested || released) {
                throw OpenMedRuntimeException(OpenMedRuntimeFailureReason.RUNTIME_CLOSED)
            }
        }
        val subscription = source.subscribe(this)
        val handle = SubscriptionHandle(subscription) { handle ->
            synchronized(lock) { subscriptions.remove(handle) }
        }
        val closeNow = synchronized(lock) {
            if (closeRequested || released) {
                true
            } else {
                subscriptions += handle
                false
            }
        }
        if (closeNow) {
            handle.close()
        }
        return handle
    }

    /**
     * Close idempotently, cancelling active operations.
     *
     * Admission stops immediately and lifecycle subscriptions are dropped.
     * Active operations are cancelled and report
     * [OpenMedRuntimeFailureReason.RUNTIME_CLOSED]; each native session is
     * closed when its final lease is released, after which the state is
     * [OpenMedRuntimeState.RELEASED]. Use [awaitReleased] to wait for that.
     */
    override fun close() {
        requestClose(cancelActive = true)
    }

    /**
     * Stop admission, let active operations finish normally, then release.
     *
     * Active callers receive their results. A later [close] still cancels
     * whatever is running.
     */
    suspend fun drainAndClose() {
        requestClose(cancelActive = false)
        awaitReleased()
    }

    /** Suspend until the runtime reaches [OpenMedRuntimeState.RELEASED]. */
    suspend fun awaitReleased() {
        status.first { it.state == OpenMedRuntimeState.RELEASED }
    }

    /** Distinct states visited, oldest first; bounded value-free evidence. */
    internal fun stateHistory(): List<OpenMedRuntimeState> = synchronized(lock) {
        history.toList()
    }

    private suspend fun <T> execute(block: suspend (OpenMedKit) -> T): T {
        currentCoroutineContext().ensureActive()
        val lease = acquire()
        try {
            currentCoroutineContext().ensureActive()
            val result = try {
                coroutineScope {
                    val work = async(dispatcher, start = CoroutineStart.LAZY) {
                        block(lease.holder.kit)
                    }
                    lease.attach(work)
                    work.await()
                }
            } catch (error: Throwable) {
                val revoked = lease.revocation
                if (revoked != null && currentCoroutineContext().isActive) {
                    throw OpenMedRuntimeException(revoked)
                }
                throw error
            }
            // Late-result suppression: a caller cancelled, or a lease revoked,
            // after the native run finished never receives the result.
            currentCoroutineContext().ensureActive()
            lease.revocation?.let { throw OpenMedRuntimeException(it) }
            return result
        } finally {
            release(lease)
        }
    }

    private suspend fun acquire(): SessionLease {
        synchronized(lock) {
            checkAdmissionLocked()
            current?.let { return leaseLocked(it) }
        }
        return loadMutex.withLock { loadAndLease() }
    }

    private suspend fun loadAndLease(): SessionLease {
        while (true) {
            currentCoroutineContext().ensureActive()
            val ticket = synchronized(lock) {
                checkAdmissionLocked()
                current?.let { return leaseLocked(it) }
                if (consecutiveLoadFailures >= policy.maxConsecutiveLoadFailures) {
                    throw OpenMedRuntimeException(
                        OpenMedRuntimeFailureReason.SESSION_RELOAD_LIMIT,
                    )
                }
                loading = true
                pendingReleaseWhenIdle = false
                publishLocked()
                LoadTicket(generation, factory)
            }
            val outcome = openSession(ticket.factory)
            val lease = synchronized(lock) { installLocked(ticket, outcome) }
            if (lease != null) {
                return lease
            }
        }
    }

    /**
     * Run the factory to completion and return its outcome.
     *
     * A started load always completes, so its session is either installed or
     * closed and never abandoned half-open. `withContext` discards its return
     * value when the caller was cancelled meanwhile (prompt cancellation), so
     * the outcome is captured inside the block instead; the caller's
     * cancellation is observed after installation.
     */
    private suspend fun openSession(factory: OpenMedSessionFactory): LoadOutcome {
        var outcome: LoadOutcome? = null
        try {
            withContext(dispatcher + NonCancellable) {
                outcome = try {
                    LoadOutcome.Opened(factory.open())
                } catch (error: Exception) {
                    LoadOutcome.Failed(error.javaClass.simpleName)
                } catch (error: LinkageError) {
                    LoadOutcome.Failed(error.javaClass.simpleName)
                }
            }
        } catch (_: CancellationException) {
            // The block has completed; its outcome is handled below.
        }
        return outcome ?: LoadOutcome.Failed(CancellationException::class.java.simpleName)
    }

    /** Returns a lease, or null when the generation changed and the load must retry. */
    private fun installLocked(ticket: LoadTicket, outcome: LoadOutcome): SessionLease? {
        loading = false
        val superseded = ticket.generation != generation
        when (outcome) {
            is LoadOutcome.Failed -> {
                if (!superseded) {
                    consecutiveLoadFailures += 1
                }
                finalizeIfDrainedLocked()
                publishLocked()
                if (closeRequested) {
                    throw OpenMedRuntimeException(OpenMedRuntimeFailureReason.RUNTIME_CLOSED)
                }
                if (superseded) {
                    return null
                }
                throw OpenMedRuntimeException(
                    reason = OpenMedRuntimeFailureReason.SESSION_LOAD_FAILED,
                    causeKind = outcome.causeKind,
                )
            }
            is LoadOutcome.Opened -> {
                sessionsOpened += 1
                val holder = SessionHolder(ticket.generation, outcome.kit)
                if (closeRequested || superseded) {
                    closeHolderLocked(holder)
                    finalizeIfDrainedLocked()
                    publishLocked()
                    if (closeRequested) {
                        throw OpenMedRuntimeException(
                            OpenMedRuntimeFailureReason.RUNTIME_CLOSED,
                        )
                    }
                    return null
                }
                consecutiveLoadFailures = 0
                holder.releaseWhenIdle = pendingReleaseWhenIdle
                pendingReleaseWhenIdle = false
                current = holder
                return leaseLocked(holder)
            }
        }
    }

    private fun leaseLocked(holder: SessionHolder): SessionLease {
        val lease = SessionLease(holder)
        holder.leases += 1
        leases += lease
        publishLocked()
        return lease
    }

    private fun release(lease: SessionLease) {
        synchronized(lock) {
            if (!leases.remove(lease)) {
                return
            }
            val holder = lease.holder
            holder.leases -= 1
            if (holder.leases == 0) {
                if (holder.retired) {
                    retired.remove(holder)
                    closeHolderLocked(holder)
                } else if (
                    holder === current && (holder.releaseWhenIdle || closeRequested)
                ) {
                    current = null
                    closeHolderLocked(holder)
                }
            }
            finalizeIfDrainedLocked()
            publishLocked()
        }
    }

    private fun requestClose(cancelActive: Boolean) {
        val (toRevoke, toUnsubscribe) = synchronized(lock) {
            if (released) {
                return
            }
            val revoke = if (cancelActive && !cancelActiveOnClose) {
                cancelActiveOnClose = true
                leases.toList()
            } else {
                emptyList()
            }
            closeRequested = true
            val unsubscribe = subscriptions.toList()
            subscriptions.clear()
            current?.let { holder ->
                if (holder.leases == 0) {
                    current = null
                    closeHolderLocked(holder)
                }
            }
            finalizeIfDrainedLocked()
            publishLocked()
            revoke to unsubscribe
        }
        toUnsubscribe.forEach { subscription ->
            try {
                subscription.close()
            } catch (_: Exception) {
                // Unsubscribing is best effort and has no value to report.
            }
        }
        toRevoke.forEach { it.revoke(OpenMedRuntimeFailureReason.RUNTIME_CLOSED) }
    }

    private fun releaseIdleLocked() {
        if (loading) {
            pendingReleaseWhenIdle = true
        }
        val holder = current ?: return
        if (holder.leases == 0) {
            current = null
            closeHolderLocked(holder)
        } else {
            holder.releaseWhenIdle = true
        }
    }

    private fun finalizeIfDrainedLocked() {
        if (!closeRequested || released || loading || leases.isNotEmpty()) {
            return
        }
        current?.let(::closeHolderLocked)
        current = null
        retired.forEach(::closeHolderLocked)
        retired.clear()
        released = true
    }

    private fun closeHolderLocked(holder: SessionHolder) {
        sessionsClosed += 1
        try {
            holder.kit.close()
        } catch (_: Exception) {
            // Close failures carry no actionable, value-free detail.
        }
    }

    private fun checkAdmissionLocked() {
        val pausedAt = pressurePausedAt
        if (pausedAt != null &&
            clock.elapsedMillis() - pausedAt >= policy.pressureCooldownMillis
        ) {
            pressurePausedAt = null
            publishLocked()
        }
        when (admissionLocked()) {
            OpenMedAdmission.OPEN -> Unit
            OpenMedAdmission.CLOSED -> throw OpenMedRuntimeException(
                OpenMedRuntimeFailureReason.RUNTIME_CLOSED,
            )
            OpenMedAdmission.PAUSED_BACKGROUND -> throw OpenMedRuntimeException(
                OpenMedRuntimeFailureReason.ADMISSION_PAUSED_BACKGROUND,
            )
            OpenMedAdmission.PAUSED_MEMORY_PRESSURE -> {
                val elapsed = clock.elapsedMillis() - (pressurePausedAt ?: 0L)
                throw OpenMedRuntimeException(
                    reason = OpenMedRuntimeFailureReason.ADMISSION_PAUSED_MEMORY_PRESSURE,
                    retryAfterMillis = (policy.pressureCooldownMillis - elapsed)
                        .coerceAtLeast(0L),
                )
            }
        }
    }

    private fun admissionLocked(): OpenMedAdmission = when {
        closeRequested || released -> OpenMedAdmission.CLOSED
        pressurePausedAt != null -> OpenMedAdmission.PAUSED_MEMORY_PRESSURE
        backgrounded && !policy.admitInBackground -> OpenMedAdmission.PAUSED_BACKGROUND
        else -> OpenMedAdmission.OPEN
    }

    private fun stateLocked(): OpenMedRuntimeState = when {
        released -> OpenMedRuntimeState.RELEASED
        closeRequested -> OpenMedRuntimeState.DRAINING
        loading -> OpenMedRuntimeState.LOADING
        leases.isNotEmpty() -> OpenMedRuntimeState.BUSY
        current != null -> OpenMedRuntimeState.READY
        else -> OpenMedRuntimeState.IDLE
    }

    private fun snapshotLocked(): OpenMedRuntimeStatus = OpenMedRuntimeStatus(
        state = stateLocked(),
        admission = admissionLocked(),
        activeLeases = leases.size,
        generation = generation,
        liveSessions = (sessionsOpened - sessionsClosed).toInt(),
        sessionsOpened = sessionsOpened,
    )

    private fun publishLocked() {
        val snapshot = snapshotLocked()
        if (history.lastOrNull() != snapshot.state) {
            history.addLast(snapshot.state)
            while (history.size > MAX_HISTORY) {
                history.removeFirst()
            }
        }
        mutableStatus.value = snapshot
    }

    private class SessionHolder(
        val generation: Long,
        val kit: OpenMedKit,
    ) {
        var leases = 0
        var releaseWhenIdle = false
        var retired = false
    }

    private class SessionLease(val holder: SessionHolder) {
        private var work: Job? = null

        @Volatile
        var revocation: OpenMedRuntimeFailureReason? = null
            private set

        fun attach(job: Job) {
            val reason = synchronized(this) {
                work = job
                revocation
            }
            if (reason != null) {
                job.cancel(revokedCancellation(reason))
            }
        }

        fun revoke(reason: OpenMedRuntimeFailureReason) {
            val job = synchronized(this) {
                if (revocation != null) {
                    return
                }
                revocation = reason
                work
            }
            job?.cancel(revokedCancellation(reason))
        }
    }

    private class SubscriptionHandle(
        private val subscription: Closeable,
        private val onClosed: (SubscriptionHandle) -> Unit,
    ) : Closeable {
        private var closed = false

        override fun close() {
            synchronized(this) {
                if (closed) {
                    return
                }
                closed = true
            }
            try {
                subscription.close()
            } finally {
                onClosed(this)
            }
        }
    }

    private class LoadTicket(
        val generation: Long,
        val factory: OpenMedSessionFactory,
    )

    private sealed class LoadOutcome {
        class Opened(val kit: OpenMedKit) : LoadOutcome()

        class Failed(val causeKind: String) : LoadOutcome()
    }

    private companion object {
        const val MAX_HISTORY = 256

        fun revokedCancellation(reason: OpenMedRuntimeFailureReason): CancellationException =
            CancellationException("OpenMedKit runtime: ${reason.code}")
    }
}
