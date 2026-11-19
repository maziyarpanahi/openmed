package com.openmed.openmedkit.util

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class LeaseGateTest {
    @Test
    fun closeWithoutLeasesReleasesOnceAndPropagatesFailures() {
        var releases = 0
        val gate = LeaseGate {
            releases += 1
            throw IllegalStateException("release failed")
        }

        assertFailsWith<IllegalStateException> { gate.close() }
        gate.close()

        assertEquals(1, releases)
        assertTrue(gate.isReleased)
        assertFalse(gate.tryAcquire())
    }

    @Test
    fun closeWhileLeasedDefersReleaseUntilTheFinalLeaseAndSuppressesItsFailure() {
        var releases = 0
        val gate = LeaseGate {
            releases += 1
            throw IllegalStateException("deferred release failed")
        }
        assertTrue(gate.tryAcquire())
        assertTrue(gate.tryAcquire())

        gate.close()
        assertTrue(gate.isClosed)
        assertFalse(gate.tryAcquire())
        gate.release()
        assertEquals(0, releases)
        assertEquals(1, gate.activeLeases)

        gate.release()
        assertEquals(1, releases)
        assertTrue(gate.isReleased)
        assertFailsWith<IllegalStateException> { gate.release() }
    }

    @Test
    fun withLeaseReleasesOnFailureAndRefusesAfterClose() {
        val gate = LeaseGate {}

        assertFailsWith<IllegalArgumentException> {
            gate.withLease(onClosed = { error("closed") }) {
                throw IllegalArgumentException("work failed")
            }
        }
        assertEquals(0, gate.activeLeases)

        gate.close()
        val refused = assertFailsWith<IllegalStateException> {
            gate.withLease(onClosed = { error("closed") }) { Unit }
        }
        assertEquals("closed", refused.message)
    }
}
