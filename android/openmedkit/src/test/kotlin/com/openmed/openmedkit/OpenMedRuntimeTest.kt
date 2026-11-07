@file:OptIn(ExperimentalCoroutinesApi::class)

package com.openmed.openmedkit

import android.content.ComponentCallbacks2
import com.openmed.openmedkit.cache.ModelCache
import com.openmed.openmedkit.catalog.ModelCatalogEntry
import com.openmed.openmedkit.util.SafeLog
import com.openmed.openmedkit.util.SafeLogOperation
import java.io.File
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertIs
import kotlin.test.assertNull
import kotlin.test.assertTrue
import kotlinx.coroutines.Deferred
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.cancel
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.test.advanceTimeBy
import kotlinx.coroutines.test.advanceUntilIdle
import kotlinx.coroutines.test.runCurrent
import kotlinx.coroutines.test.runTest
import org.junit.Rule
import org.junit.rules.TemporaryFolder
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.RuntimeEnvironment
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [33])
class OpenMedRuntimeTest {
    @get:Rule
    val temporaryFolder = TemporaryFolder()

    @Test
    fun concurrentFirstUseOpensExactlyOneSessionAndRecordsTransitions() = runTest {
        val factory = FakeSessionFactory()
        val runtime = testRuntime(factory)

        val calls = List(5) { call { runtime.analyzeText(SYNTHETIC_NOTE) } }
        runCurrent()
        assertEquals(OpenMedRuntimeState.LOADING, runtime.status.value.state)
        val outcomes = calls.awaitAll()

        assertEquals(1, factory.openCalls)
        assertTrue(outcomes.all { it == outcomes.first() })
        assertIs<CallOutcome.Completed>(outcomes.first())
        assertEquals(OpenMedRuntimeState.READY, runtime.status.value.state)
        assertEquals(0, runtime.status.value.activeLeases)
        assertEquals(1, runtime.status.value.liveSessions)

        runtime.close()
        runtime.close()

        assertEquals(OpenMedRuntimeState.RELEASED, runtime.status.value.state)
        assertEquals(0, runtime.status.value.liveSessions)
        assertEquals(
            listOf(
                OpenMedRuntimeState.IDLE,
                OpenMedRuntimeState.LOADING,
                OpenMedRuntimeState.BUSY,
                OpenMedRuntimeState.READY,
                OpenMedRuntimeState.RELEASED,
            ),
            runtime.stateHistory(),
        )
        assertEquals(1, factory.opened.single().closeCount)
        factory.assertNoUnsafeClose()
    }

    @Test
    fun cancellationBeforeAdmissionTakesNoLeaseAndOpensNoSession() = runTest {
        val factory = FakeSessionFactory()
        val runtime = testRuntime(factory)

        val pending = call {
            currentCoroutineContext().cancel()
            runtime.analyzeText(SYNTHETIC_NOTE)
        }
        advanceUntilIdle()

        pending.awaitCancelled()
        assertEquals(0, factory.openCalls)
        assertEquals(OpenMedRuntimeState.IDLE, runtime.status.value.state)
        assertEquals(0, runtime.status.value.activeLeases)
    }

    @Test
    fun cancellationDuringLoadKeepsTheVerifiedSessionForTheNextCaller() = runTest {
        val factory = FakeSessionFactory(loadMillis = 100L)
        val runtime = testRuntime(factory)

        val first = call { runtime.analyzeText(SYNTHETIC_NOTE) }
        advanceTimeBy(50L)
        runCurrent()
        assertEquals(OpenMedRuntimeState.LOADING, runtime.status.value.state)
        first.cancel()
        advanceUntilIdle()

        first.awaitCancelled()
        assertEquals(OpenMedRuntimeState.READY, runtime.status.value.state)
        assertEquals(1, runtime.status.value.liveSessions)
        assertEquals(0, factory.opened.single().runs)

        val second = call { runtime.analyzeText(SYNTHETIC_NOTE) }
        advanceUntilIdle()

        assertIs<CallOutcome.Completed>(second.await())
        assertEquals(1, factory.openCalls)
    }

    @Test
    fun cancellationDuringNativeRunResolvesAtTheWindowBoundaryWithoutAResult() = runTest {
        val factory = FakeSessionFactory(nativeMillis = 100L)
        val runtime = testRuntime(factory)
        runtime.warmUp()
        val startedAt = testScheduler.currentTime

        val pending = call { runtime.analyzeText(SYNTHETIC_NOTE) }
        var completedAt = -1L
        pending.invokeOnCompletion { completedAt = testScheduler.currentTime }
        advanceTimeBy(50L)
        runCurrent()
        assertEquals(1, factory.opened.single().activeRuns)

        pending.cancel()
        runCurrent()
        // The native window cannot be interrupted, so the lease stays held.
        assertFalse(pending.isCompleted)
        assertEquals(1, runtime.status.value.activeLeases)

        advanceUntilIdle()

        pending.awaitCancelled()
        assertEquals(startedAt + 100L, completedAt)
        assertEquals(OpenMedRuntimeState.READY, runtime.status.value.state)
        assertEquals(0, runtime.status.value.activeLeases)
        assertEquals(0, factory.opened.single().closeCount)
    }

    @Test
    fun cancellationImmediatelyAfterNativeInferenceSuppressesTheResult() = runTest {
        val events = mutableListOf<SafeLogOperation>()
        val previous = SafeLog.installSinkForTesting { events += it.operation }
        try {
            val factory = FakeSessionFactory()
            val runtime = testRuntime(factory)
            runtime.warmUp()
            lateinit var pending: Deferred<CallOutcome>
            factory.hooks.afterNative = { pending.cancel() }

            pending = call { runtime.analyzeText(SYNTHETIC_NOTE) }
            advanceUntilIdle()

            pending.awaitCancelled()
            assertEquals(1, factory.opened.single().runs)
            assertTrue(events.isEmpty(), "decoded output must not be published")
            assertEquals(OpenMedRuntimeState.READY, runtime.status.value.state)
        } finally {
            SafeLog.installSinkForTesting(previous)
        }
    }

    @Test
    fun cancellationDuringTokenizationAndBetweenWindowsStopsTheNextNativeRun() = runTest {
        val longNote = List(6) { SYNTHETIC_NOTE }.joinToString(" ")
        val factory = FakeSessionFactory()
        val runtime = testRuntime(factory)
        runtime.warmUp()
        val classifier = factory.opened.single()

        lateinit var tokenizing: Deferred<CallOutcome>
        factory.hooks.onTokenize = { tokenizing.cancel() }
        tokenizing = call { runtime.extractPiiChunked(longNote, 0.5f, 4, 1) }
        advanceUntilIdle()
        tokenizing.awaitCancelled()
        assertEquals(0, classifier.runs)

        factory.hooks.onTokenize = null
        lateinit var windowed: Deferred<CallOutcome>
        factory.hooks.afterNative = { if (classifier.runs == 1) windowed.cancel() }
        windowed = call { runtime.extractPiiChunked(longNote, 0.5f, 4, 1) }
        advanceUntilIdle()
        windowed.awaitCancelled()
        assertEquals(1, classifier.runs, "no window may start after cancellation")
        assertEquals(0, runtime.status.value.activeLeases)
    }

    @Test
    fun cancellationDuringMergingAndPolicyApplicationPublishesNothingLater() = runTest {
        val factory = FakeSessionFactory()
        val runtime = testRuntime(factory)
        runtime.warmUp()

        suspend fun cancelAt(
            trigger: SafeLogOperation,
            operation: suspend () -> Any?,
        ): List<SafeLogOperation> {
            val events = mutableListOf<SafeLogOperation>()
            lateinit var pending: Deferred<CallOutcome>
            val previous = SafeLog.installSinkForTesting { event ->
                events += event.operation
                if (event.operation == trigger) {
                    pending.cancel()
                }
            }
            try {
                pending = call { operation() }
                advanceUntilIdle()
                pending.awaitCancelled()
            } finally {
                SafeLog.installSinkForTesting(previous)
            }
            return events
        }

        // Cancelled after decoding, before span repair and merging finish.
        assertEquals(
            listOf(SafeLogOperation.ANALYZE_TEXT),
            cancelAt(SafeLogOperation.ANALYZE_TEXT) { runtime.extractPii(SYNTHETIC_NOTE) },
        )
        // Cancelled after merging, before policy application finishes.
        assertEquals(
            listOf(SafeLogOperation.ANALYZE_TEXT, SafeLogOperation.EXTRACT_PII),
            cancelAt(SafeLogOperation.EXTRACT_PII) { runtime.deidentify(SYNTHETIC_NOTE) },
        )
        assertEquals(0, runtime.status.value.activeLeases)
    }

    @Test
    fun closeWhileBusyDrainsBeforeReleasingTheNativeSession() = runTest {
        val factory = FakeSessionFactory()
        val runtime = testRuntime(factory)
        runtime.warmUp()
        val classifier = factory.opened.single()

        val pending = call { runtime.analyzeText(SYNTHETIC_NOTE) }
        advanceTimeBy(50L)
        runCurrent()
        runtime.close()

        assertEquals(OpenMedRuntimeState.DRAINING, runtime.status.value.state)
        assertEquals(OpenMedAdmission.CLOSED, runtime.status.value.admission)
        assertEquals(1, runtime.status.value.activeLeases)
        // The runtime itself keeps the leased session open, independently of
        // the facade's own close guard.
        assertEquals(1, runtime.status.value.liveSessions)
        assertEquals(0, classifier.closeCount)
        val refused = assertFailsWith<OpenMedRuntimeException> {
            runtime.analyzeText(SYNTHETIC_NOTE)
        }
        assertEquals(OpenMedRuntimeFailureReason.RUNTIME_CLOSED, refused.reason)
        assertEquals(OpenMedRetryGuidance.RECREATE_RUNTIME, refused.retryGuidance)

        advanceUntilIdle()

        assertEquals(
            CallOutcome.Refused(OpenMedRuntimeFailureReason.RUNTIME_CLOSED),
            pending.await(),
        )
        assertEquals(OpenMedRuntimeState.RELEASED, runtime.status.value.state)
        assertEquals(0, runtime.status.value.liveSessions)
        assertEquals(1, classifier.closeCount)
        factory.assertNoUnsafeClose()
        runtime.close()
        assertEquals(1, classifier.closeCount)
    }

    @Test
    fun drainAndCloseLetsActiveWorkFinishWithItsResult() = runTest {
        val factory = FakeSessionFactory()
        val runtime = testRuntime(factory)
        val expected = runtime.analyzeText(SYNTHETIC_NOTE)

        val pending = call { runtime.analyzeText(SYNTHETIC_NOTE) }
        advanceTimeBy(50L)
        runCurrent()
        val closing = async { runtime.drainAndClose() }
        runCurrent()

        assertEquals(OpenMedRuntimeState.DRAINING, runtime.status.value.state)
        assertEquals(
            OpenMedRuntimeFailureReason.RUNTIME_CLOSED,
            assertFailsWith<OpenMedRuntimeException> {
                runtime.analyzeText(SYNTHETIC_NOTE)
            }.reason,
        )
        advanceUntilIdle()

        assertEquals(CallOutcome.Completed(expected), pending.await())
        assertTrue(closing.isCompleted)
        assertEquals(OpenMedRuntimeState.RELEASED, runtime.status.value.state)
        assertEquals(1, factory.opened.single().closeCount)
        factory.assertNoUnsafeClose()
    }

    @Test
    fun replacingTheModelMidRunMarksTheOldResultStaleAndNeverDuplicatesSessions() =
        runTest {
            val first = FakeSessionFactory()
            val second = FakeSessionFactory(sessionIdBase = 100)
            val runtime = testRuntime(first)
            runtime.warmUp()

            val stale = call { runtime.analyzeText(SYNTHETIC_NOTE) }
            advanceTimeBy(50L)
            runCurrent()
            runtime.replaceModel(second)

            assertEquals(2L, runtime.status.value.generation)
            assertEquals(1, runtime.status.value.liveSessions)
            assertEquals(0, first.opened.single().closeCount)
            val fresh = call { runtime.analyzeText(SYNTHETIC_NOTE) }
            advanceUntilIdle()

            assertEquals(
                CallOutcome.Refused(OpenMedRuntimeFailureReason.STALE_GENERATION),
                stale.await(),
            )
            assertIs<CallOutcome.Completed>(fresh.await())
            assertEquals(1, first.opened.single().closeCount)
            assertEquals(1, second.openCalls)
            assertEquals(1, runtime.status.value.liveSessions)
            first.assertNoUnsafeClose()
            second.assertNoUnsafeClose()
        }

    @Test
    fun replacingTheModelDuringLoadDiscardsTheSupersededSession() = runTest {
        val first = FakeSessionFactory(loadMillis = 100L)
        val second = FakeSessionFactory(sessionIdBase = 100)
        val runtime = testRuntime(first)

        val pending = call { runtime.analyzeText(SYNTHETIC_NOTE) }
        advanceTimeBy(50L)
        runCurrent()
        runtime.replaceModel(second)
        advanceUntilIdle()

        assertIs<CallOutcome.Completed>(pending.await())
        assertEquals(1, first.openCalls)
        assertEquals(1, first.opened.single().closeCount)
        assertEquals(0, first.opened.single().runs)
        assertEquals(1, second.opened.single().runs)
        assertEquals(1, runtime.status.value.liveSessions)
        assertEquals(2L, runtime.status.value.sessionsOpened)
    }

    @Test
    fun lowMemoryReleasesOnlyIdleSessionsAndReacquiresThroughTheFactory() = runTest {
        val factory = FakeSessionFactory()
        val runtime = testRuntime(factory)
        val expected = runtime.analyzeText(SYNTHETIC_NOTE)

        runtime.onMemoryPressure(OpenMedMemoryPressure.MODERATE)
        assertEquals(OpenMedRuntimeState.READY, runtime.status.value.state)

        runtime.onMemoryPressure(OpenMedMemoryPressure.LOW)
        assertEquals(OpenMedRuntimeState.IDLE, runtime.status.value.state)
        assertEquals(OpenMedAdmission.OPEN, runtime.status.value.admission)
        assertEquals(1, factory.opened[0].closeCount)

        val reacquired = call { runtime.analyzeText(SYNTHETIC_NOTE) }
        advanceTimeBy(50L)
        runCurrent()
        runtime.onTrimMemory(ComponentCallbacks2.TRIM_MEMORY_BACKGROUND)
        assertEquals(OpenMedRuntimeState.BUSY, runtime.status.value.state)
        assertEquals(1, runtime.status.value.liveSessions)
        assertEquals(0, factory.opened[1].closeCount)
        advanceUntilIdle()

        assertEquals(CallOutcome.Completed(expected), reacquired.await())
        assertEquals(2, factory.openCalls)
        assertEquals(OpenMedRuntimeState.IDLE, runtime.status.value.state)
        assertEquals(1, factory.opened[1].closeCount)
        factory.assertNoUnsafeClose()
    }

    @Test
    fun criticalPressureStopsAdmissionPreservesActiveWorkAndReopensAfterCooldown() =
        runTest {
            val factory = FakeSessionFactory()
            val runtime = testRuntime(
                factory,
                OpenMedRuntimePolicy(pressureCooldownMillis = 1_000L),
            )
            val expected = runtime.analyzeText(SYNTHETIC_NOTE)

            val active = call { runtime.analyzeText(SYNTHETIC_NOTE) }
            advanceTimeBy(50L)
            runCurrent()
            runtime.onTrimMemory(ComponentCallbacks2.TRIM_MEMORY_COMPLETE)

            val status = runtime.status.value
            assertEquals(OpenMedAdmission.PAUSED_MEMORY_PRESSURE, status.admission)
            assertEquals(OpenMedRuntimeState.BUSY, status.state)
            val refused = assertFailsWith<OpenMedRuntimeException> {
                runtime.analyzeText(SYNTHETIC_NOTE)
            }
            assertEquals(
                OpenMedRuntimeFailureReason.ADMISSION_PAUSED_MEMORY_PRESSURE,
                refused.reason,
            )
            assertEquals(OpenMedRetryGuidance.RETRY_AFTER_BACKOFF, refused.retryGuidance)
            assertEquals(1_000L, refused.retryAfterMillis)

            advanceUntilIdle()
            assertEquals(CallOutcome.Completed(expected), active.await())
            assertEquals(OpenMedRuntimeState.IDLE, runtime.status.value.state)
            assertEquals(0, runtime.status.value.liveSessions)

            advanceTimeBy(1_000L)
            val reopened = call { runtime.analyzeText(SYNTHETIC_NOTE) }
            advanceUntilIdle()
            assertEquals(CallOutcome.Completed(expected), reopened.await())
            assertEquals(2, factory.openCalls)
            factory.assertNoUnsafeClose()
        }

    @Test
    fun clearedPressureOrForegroundReturnLiftsTheAdmissionPause() = runTest {
        val runtime = testRuntime(FakeSessionFactory())

        runtime.onMemoryPressure(OpenMedMemoryPressure.CRITICAL)
        assertEquals(OpenMedAdmission.PAUSED_MEMORY_PRESSURE, runtime.status.value.admission)
        runtime.onMemoryPressure(OpenMedMemoryPressure.NONE)
        assertEquals(OpenMedAdmission.OPEN, runtime.status.value.admission)

        runtime.onMemoryPressure(OpenMedMemoryPressure.CRITICAL)
        runtime.onHostForeground()
        assertEquals(OpenMedAdmission.OPEN, runtime.status.value.admission)
        assertEquals(2, runtime.analyzeText(SYNTHETIC_NOTE).size)
    }

    @Test
    fun backgroundRefusesNewWorkAndForegroundReacquiresAVerifiedSession() = runTest {
        val factory = FakeSessionFactory()
        val runtime = testRuntime(factory)
        val source = FakeLifecycleSource()
        runtime.bindLifecycle(source)
        runtime.warmUp()

        source.background()

        assertEquals(OpenMedAdmission.PAUSED_BACKGROUND, runtime.status.value.admission)
        assertEquals(OpenMedRuntimeState.IDLE, runtime.status.value.state)
        val refused = assertFailsWith<OpenMedRuntimeException> {
            runtime.analyzeText(SYNTHETIC_NOTE)
        }
        assertEquals(OpenMedRuntimeFailureReason.ADMISSION_PAUSED_BACKGROUND, refused.reason)
        assertEquals(OpenMedRetryGuidance.RETRY_AFTER_FOREGROUND, refused.retryGuidance)

        source.foreground()
        assertEquals(OpenMedAdmission.OPEN, runtime.status.value.admission)
        runtime.analyzeText(SYNTHETIC_NOTE)
        assertEquals(2, factory.openCalls)

        runtime.close()
        assertTrue(source.observers.isEmpty())
    }

    @Test
    fun backgroundPolicyCanKeepAdmittingWork() = runTest {
        val factory = FakeSessionFactory()
        val runtime = testRuntime(
            factory,
            OpenMedRuntimePolicy(admitInBackground = true, releaseIdleInBackground = false),
        )
        runtime.warmUp()

        runtime.onHostBackground()
        runtime.analyzeText(SYNTHETIC_NOTE)

        assertEquals(OpenMedAdmission.OPEN, runtime.status.value.admission)
        assertEquals(1, factory.openCalls)
    }

    @Test
    fun hostRecreationDropsThePreviousHostWithoutChangingRuntimeState() = runTest {
        val runtime = testRuntime(FakeSessionFactory())
        val previousHost = FakeLifecycleSource()
        val subscription = runtime.bindLifecycle(previousHost)
        runtime.warmUp()

        subscription.close()
        subscription.close()
        previousHost.background()
        val nextHost = FakeLifecycleSource()
        runtime.bindLifecycle(nextHost)

        assertTrue(previousHost.observers.isEmpty())
        assertEquals(1, nextHost.observers.size)
        assertEquals(OpenMedRuntimeState.READY, runtime.status.value.state)
        assertEquals(OpenMedAdmission.OPEN, runtime.status.value.admission)

        runtime.close()
        assertTrue(nextHost.observers.isEmpty())
        assertEquals(
            OpenMedRuntimeFailureReason.RUNTIME_CLOSED,
            assertFailsWith<OpenMedRuntimeException> {
                runtime.bindLifecycle(FakeLifecycleSource())
            }.reason,
        )
    }

    @Test
    fun trimMemorySourceRegistersOnTheApplicationAndUnregistersOnClose() = runTest {
        val application = RuntimeEnvironment.getApplication()
        val runtime = testRuntime(FakeSessionFactory())
        val subscription = runtime.bindLifecycle(OpenMedTrimMemorySource(application))
        runtime.warmUp()

        application.onTrimMemory(ComponentCallbacks2.TRIM_MEMORY_COMPLETE)
        assertEquals(OpenMedAdmission.PAUSED_MEMORY_PRESSURE, runtime.status.value.admission)
        assertEquals(OpenMedRuntimeState.IDLE, runtime.status.value.state)

        runtime.onHostForeground()
        subscription.close()
        application.onTrimMemory(ComponentCallbacks2.TRIM_MEMORY_COMPLETE)
        assertEquals(OpenMedAdmission.OPEN, runtime.status.value.admission)
    }

    @Test
    fun trimLevelsMapToDocumentedPressure() {
        val expected = mapOf(
            ComponentCallbacks2.TRIM_MEMORY_RUNNING_MODERATE to OpenMedMemoryPressure.MODERATE,
            ComponentCallbacks2.TRIM_MEMORY_RUNNING_LOW to OpenMedMemoryPressure.LOW,
            ComponentCallbacks2.TRIM_MEMORY_RUNNING_CRITICAL to OpenMedMemoryPressure.CRITICAL,
            ComponentCallbacks2.TRIM_MEMORY_UI_HIDDEN to OpenMedMemoryPressure.MODERATE,
            ComponentCallbacks2.TRIM_MEMORY_BACKGROUND to OpenMedMemoryPressure.LOW,
            ComponentCallbacks2.TRIM_MEMORY_MODERATE to OpenMedMemoryPressure.LOW,
            ComponentCallbacks2.TRIM_MEMORY_COMPLETE to OpenMedMemoryPressure.CRITICAL,
            0 to OpenMedMemoryPressure.MODERATE,
        )
        expected.forEach { (level, pressure) ->
            assertEquals(pressure, OpenMedMemoryPressure.fromTrimLevel(level), "level $level")
        }

        val received = mutableListOf<OpenMedMemoryPressure>()
        val callbacks = OpenMedComponentCallbacks(
            object : OpenMedLifecycleObserver {
                override fun onHostForeground() = Unit

                override fun onHostBackground() = Unit

                override fun onMemoryPressure(pressure: OpenMedMemoryPressure) {
                    received += pressure
                }
            },
        )
        callbacks.onLowMemory()
        callbacks.onTrimMemory(ComponentCallbacks2.TRIM_MEMORY_RUNNING_LOW)
        assertEquals(
            listOf(OpenMedMemoryPressure.CRITICAL, OpenMedMemoryPressure.LOW),
            received,
        )
    }

    @Test
    fun loadFailuresAreValueFreeAndReacquisitionIsBounded() = runTest {
        val factory = FakeSessionFactory().apply {
            failure = {
                IllegalStateException(
                    "cannot open ${RUNTIME_SENTINELS[2]}/model.onnx for " +
                        "${RUNTIME_SENTINELS[0]} ${RUNTIME_SENTINELS[1]} " +
                        RUNTIME_SENTINELS[3],
                )
            }
        }
        val runtime = testRuntime(
            factory,
            OpenMedRuntimePolicy(maxConsecutiveLoadFailures = 2),
        )

        repeat(2) {
            val error = assertFailsWith<OpenMedRuntimeException> {
                runtime.analyzeText(SYNTHETIC_NOTE)
            }
            assertEquals(OpenMedRuntimeFailureReason.SESSION_LOAD_FAILED, error.reason)
            assertEquals("IllegalStateException", error.causeKind)
            assertNull(error.cause)
            assertValueFree(error)
        }
        val limited = assertFailsWith<OpenMedRuntimeException> {
            runtime.analyzeText(SYNTHETIC_NOTE)
        }
        assertEquals(OpenMedRuntimeFailureReason.SESSION_RELOAD_LIMIT, limited.reason)
        assertEquals(2, factory.openCalls)
        assertEquals(OpenMedRuntimeState.IDLE, runtime.status.value.state)

        runtime.replaceModel(FakeSessionFactory(sessionIdBase = 100))
        assertEquals(2, runtime.analyzeText(SYNTHETIC_NOTE).size)
    }

    @Test
    fun everyFailureAndDiagnosticSurfaceIsValueFree() = runTest {
        OpenMedRuntimeFailureReason.values().forEach { reason ->
            val error = OpenMedRuntimeException(reason, 5L, "IllegalStateException")
            assertTrue(reason.code.matches(Regex("[a-z_]+")), reason.code)
            assertEquals("OpenMedKit runtime: ${reason.code}", error.message)
            assertValueFree(error)
        }

        val factory = FakeSessionFactory()
        val runtime = testRuntime(factory)
        runtime.warmUp()
        val surfaces = mutableListOf<String>()
        val stale = call { runtime.deidentify(SYNTHETIC_NOTE) }
        advanceTimeBy(50L)
        runCurrent()
        runtime.replaceModel(FakeSessionFactory(sessionIdBase = 100))
        val closed = call { runtime.deidentify(SYNTHETIC_NOTE) }
        advanceTimeBy(20L)
        runCurrent()
        runtime.close()
        advanceUntilIdle()

        surfaces += stale.await().toString()
        surfaces += closed.await().toString()
        surfaces += runtime.status.value.toString()
        surfaces += runtime.stateHistory().toString()
        surfaces += OpenMedRuntimePolicy().toString()
        surfaces.forEach { surface ->
            RUNTIME_SENTINELS.forEach { sentinel ->
                assertFalse(surface.contains(sentinel), "surface leaked a sentinel")
            }
        }
    }

    @Test
    fun facadeCloseNeverFreesTheSessionUnderAnInFlightOperation() = runTest {
        val classifier = FakeNativeClassifier(1, 100L, FakeHooks())
        val kit = OpenMedKit(classifier = classifier)

        val pending = call { kit.analyzeText(SYNTHETIC_NOTE) }
        advanceTimeBy(50L)
        runCurrent()
        kit.close()

        assertEquals(0, classifier.closeCount)
        assertEquals(
            OpenMedRuntimeFailureReason.RUNTIME_CLOSED,
            assertFailsWith<OpenMedRuntimeException> {
                kit.analyzeText(SYNTHETIC_NOTE)
            }.reason,
        )
        advanceUntilIdle()

        assertIs<CallOutcome.Completed>(pending.await())
        assertEquals(1, classifier.closeCount)
        assertFalse(classifier.closedWhileRunning)
        kit.close()
        assertEquals(1, classifier.closeCount)
    }

    @Test
    fun cacheLeaseProtectsTheActiveModelFromEvictionUntilTheSessionCloses() = runTest {
        val cache = ModelCache(temporaryFolder.newFolder("cache"))
        val entry = catalogEntry()
        storeModel(cache, entry)
        val opened = mutableListOf<FakeNativeClassifier>()
        val factory = OpenMedSessionFactory.fromCache(cache, entry) {
            FakeNativeClassifier(opened.size + 1, 100L, FakeHooks()).also { opened += it }
        }
        val runtime = testRuntime(factory)
        runtime.warmUp()
        assertTrue(cache.isLeased(entry.repoId))

        val active = call { runtime.analyzeText(SYNTHETIC_NOTE) }
        advanceTimeBy(50L)
        runCurrent()
        assertFalse(cache.evict(entry.repoId))
        assertTrue(cache.isAvailable(entry))
        advanceUntilIdle()
        assertIs<CallOutcome.Completed>(active.await())

        runtime.onMemoryPressure(OpenMedMemoryPressure.LOW)
        assertFalse(cache.isLeased(entry.repoId))
        assertEquals(1, opened.single().closeCount)
        assertTrue(cache.evict(entry.repoId))

        val missing = assertFailsWith<OpenMedRuntimeException> {
            runtime.analyzeText(SYNTHETIC_NOTE)
        }
        assertEquals(OpenMedRuntimeFailureReason.SESSION_LOAD_FAILED, missing.reason)
        assertEquals("CachedModelUnavailableException", missing.causeKind)
        assertEquals(1, opened.size)
        runtime.close()
        assertFalse(cache.isLeased(entry.repoId))
    }

    @Test
    fun stressedLifecycleAndCancellationKeepResourcesBoundedAndResultBytesStable() =
        runTest {
            val baseline = OpenMedKit(
                classifier = FakeNativeClassifier(0, 0L, FakeHooks()),
            ).deidentify(SYNTHETIC_NOTE)
            val factories = mutableListOf(FakeSessionFactory())
            val runtime = testRuntime(
                factories.first(),
                OpenMedRuntimePolicy(pressureCooldownMillis = 200L),
            )
            val random = Random(3647)
            val calls = mutableListOf<Deferred<CallOutcome>>()
            var maxLiveSessions = 0

            repeat(400) {
                when (random.nextInt(12)) {
                    in 0..4 -> calls += call { runtime.deidentify(SYNTHETIC_NOTE) }
                    5 -> calls.filter { it.isActive }.randomOrNull(random)?.cancel()
                    6 -> runtime.onMemoryPressure(
                        OpenMedMemoryPressure.values()[random.nextInt(4)],
                    )
                    7 -> if (random.nextBoolean()) {
                        runtime.onHostBackground()
                    } else {
                        runtime.onHostForeground()
                    }
                    8 -> {
                        val next = FakeSessionFactory(sessionIdBase = factories.size * 1_000)
                        factories += next
                        runtime.replaceModel(next)
                    }
                    else -> advanceTimeBy(random.nextLong(1L, 120L))
                }
                runCurrent()
                maxLiveSessions = maxOf(maxLiveSessions, runtime.status.value.liveSessions)
            }
            runtime.onHostForeground()
            runtime.onMemoryPressure(OpenMedMemoryPressure.NONE)
            advanceUntilIdle()

            val completed = mutableListOf<PolicyDeidentificationResult>()
            val refused = mutableListOf<OpenMedRuntimeFailureReason>()
            var cancelled = 0
            calls.forEach { pending ->
                if (pending.isCancelled) {
                    cancelled += 1
                } else {
                    when (val outcome = pending.await()) {
                        is CallOutcome.Completed ->
                            completed += outcome.value as PolicyDeidentificationResult
                        is CallOutcome.Refused -> refused += outcome.reason
                    }
                }
            }
            assertTrue(completed.isNotEmpty())
            assertTrue(cancelled > 0)
            assertTrue(refused.contains(OpenMedRuntimeFailureReason.STALE_GENERATION))
            completed.forEach { result ->
                assertEquals(baseline.redactedText, result.redactedText)
                assertEquals(baseline.actions, result.actions)
            }
            assertTrue(
                refused.all {
                    it in setOf(
                        OpenMedRuntimeFailureReason.STALE_GENERATION,
                        OpenMedRuntimeFailureReason.ADMISSION_PAUSED_BACKGROUND,
                        OpenMedRuntimeFailureReason.ADMISSION_PAUSED_MEMORY_PRESSURE,
                    )
                },
            )

            val idle = runtime.status.value
            assertEquals(0, idle.activeLeases)
            assertTrue(idle.liveSessions <= 1)
            assertEquals(factories.sumOf { it.liveSessions() }, idle.liveSessions)
            // At most the current session plus one draining superseded session.
            assertTrue(maxLiveSessions <= 2, "live sessions peaked at $maxLiveSessions")

            runtime.close()
            assertEquals(OpenMedRuntimeState.RELEASED, runtime.status.value.state)
            assertEquals(0, factories.sumOf { it.liveSessions() })
            factories.forEach { it.assertNoUnsafeClose() }
        }

    private fun assertValueFree(error: OpenMedRuntimeException) {
        val surfaces = listOf(
            error.message.orEmpty(),
            error.toString(),
            error.code,
            error.causeKind.orEmpty(),
            error.stackTraceToString(),
        )
        surfaces.forEach { surface ->
            RUNTIME_SENTINELS.forEach { sentinel ->
                assertFalse(surface.contains(sentinel), "exception leaked a sentinel")
            }
        }
    }

    private fun catalogEntry(): ModelCatalogEntry = ModelCatalogEntry(
        repoId = "OpenMed/synthetic-runtime-onnx",
        formats = listOf("onnx"),
        tier = "tiny",
        paramCount = 1,
        languages = listOf("en"),
        license = "apache-2.0",
        reproducibilityHash =
            "sha256:2222222222222222222222222222222222222222222222222222222222222222",
    )

    private fun storeModel(cache: ModelCache, entry: ModelCatalogEntry) {
        val staged = temporaryFolder.newFolder("staged")
        File(staged, "model_int8.onnx").writeText("synthetic-weights")
        cache.storeReadyModel(
            repoId = entry.repoId,
            sourceDirectory = staged,
            expectedIntegrityHash = entry.reproducibilityHash,
            actualIntegrityHash = entry.reproducibilityHash,
            revisionSha = "synthetic",
        )
    }
}
