@file:OptIn(ExperimentalCoroutinesApi::class)

package com.openmed.openmedkit

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertSame
import kotlin.test.assertTrue
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.Job
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.TestScope
import kotlinx.coroutines.test.advanceTimeBy
import kotlinx.coroutines.test.advanceUntilIdle
import kotlinx.coroutines.test.runCurrent
import kotlinx.coroutines.test.runTest
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [33])
class OpenMedHostScopeTest {
    @Test
    fun invalidateCancelsOwnedWorkAndReleasesSessionsAfterTheNativeWindow() = runTest {
        val source = FakeLifecycleSource()
        val factory = FakeSessionFactory()
        val host = testHost(listOf(source))
        assertTrue(host.load("model-a", factory))
        assertEquals(1, source.observers.size)
        val runtime = host.requireRuntime()

        var published: Any? = null
        val job = host.launch { published = host.requireRuntime().deidentify(SYNTHETIC_NOTE) }
        advanceTimeBy(50L)
        runCurrent()
        assertEquals(OpenMedRuntimeState.BUSY, runtime.status.value.state)

        host.invalidate()

        assertTrue(job.isCancelled)
        assertTrue(host.isInvalidated)
        assertTrue(source.observers.isEmpty())
        assertNull(host.runtimeOrNull())
        assertEquals(OpenMedRuntimeState.DRAINING, runtime.status.value.state)
        assertEquals(1, runtime.status.value.liveSessions)
        assertEquals(0, factory.opened.single().closeCount)

        advanceUntilIdle()

        assertNull(published, "an invalidated host must not publish a late result")
        assertEquals(OpenMedRuntimeState.RELEASED, runtime.status.value.state)
        assertEquals(1, factory.opened.single().closeCount)
        factory.assertNoUnsafeClose()
        assertTrue(host.scope.coroutineContext[Job]!!.children.none())
        assertEquals(
            OpenMedRuntimeFailureReason.RUNTIME_CLOSED,
            assertFailsWith<OpenMedRuntimeException> { host.requireRuntime() }.reason,
        )
        assertEquals(
            OpenMedRuntimeFailureReason.RUNTIME_CLOSED,
            assertFailsWith<OpenMedRuntimeException> {
                host.load("model-a", factory)
            }.reason,
        )
        assertTrue(host.launch { }.isCancelled)
        host.invalidate()
        host.close()
    }

    @Test
    fun loadingTheSameKeyIsANoOpAndANewKeyReplacesTheModel() = runTest {
        val first = FakeSessionFactory()
        val second = FakeSessionFactory(sessionIdBase = 100)
        val host = testHost()

        assertTrue(host.load("model-a", first))
        assertFalse(host.load("model-a", first))
        assertEquals(1, first.openCalls)
        val runtime = host.requireRuntime()

        val stale = call { runtime.analyzeText(SYNTHETIC_NOTE) }
        advanceTimeBy(50L)
        runCurrent()
        assertTrue(host.load("model-b", second))
        assertSame(runtime, host.requireRuntime())
        advanceUntilIdle()

        assertEquals(
            CallOutcome.Refused(OpenMedRuntimeFailureReason.STALE_GENERATION),
            stale.await(),
        )
        assertEquals(1, first.opened.single().closeCount)
        assertEquals(1, second.openCalls)
        assertEquals(2L, runtime.status.value.generation)
        assertEquals(1, runtime.status.value.liveSessions)

        host.invalidate()
        assertEquals(OpenMedRuntimeState.RELEASED, runtime.status.value.state)
        first.assertNoUnsafeClose()
        second.assertNoUnsafeClose()
    }

    @Test
    fun hostFailuresAreTypedAndAFailedLoadIsRetried() = runTest {
        val host = testHost()
        val missing = assertFailsWith<OpenMedRuntimeException> { host.requireRuntime() }
        assertEquals(OpenMedRuntimeFailureReason.MODEL_NOT_LOADED, missing.reason)
        assertEquals(OpenMedRetryGuidance.LOAD_MODEL, missing.retryGuidance)

        val factory = FakeSessionFactory().apply {
            failure = { IllegalStateException("cannot read ${RUNTIME_SENTINELS[2]}") }
        }
        val failed = assertFailsWith<OpenMedRuntimeException> {
            host.load("model-a", factory)
        }
        assertEquals(OpenMedRuntimeFailureReason.SESSION_LOAD_FAILED, failed.reason)
        assertFalse(failed.message.orEmpty().contains(RUNTIME_SENTINELS[2]))

        factory.failure = null
        assertTrue(host.load("model-a", factory))
        assertEquals(2, host.requireRuntime().analyzeText(SYNTHETIC_NOTE).size)
        host.invalidate()
    }

    private fun TestScope.testHost(
        sources: List<OpenMedLifecycleSource> = emptyList(),
    ): OpenMedHostScope = OpenMedHostScope(
        lifecycleSources = sources,
        dispatcher = StandardTestDispatcher(testScheduler),
        clock = { testScheduler.currentTime },
    )
}
