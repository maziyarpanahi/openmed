package com.openmed.openmedkit.util

/**
 * Reference-counted close guard for a native resource.
 *
 * Every use of the resource runs between [tryAcquire] and [release]. [close]
 * stops new acquisitions immediately but defers [onRelease] until the final
 * active lease is released, so a native session is never freed while a run
 * still holds it. [onRelease] runs at most once.
 *
 * A failure thrown by an immediate release propagates to the [close] caller.
 * A deferred release runs on the thread that drops the final lease; its
 * failure has no caller to report to and is suppressed so it can never replace
 * the result or failure of the run that triggered it.
 */
internal class LeaseGate(
    private val onRelease: () -> Unit,
) {
    private var active = 0
    private var closeRequested = false
    private var released = false

    /** Number of uses currently holding the resource. */
    val activeLeases: Int
        @Synchronized get() = active

    /** True once [close] has been requested. */
    val isClosed: Boolean
        @Synchronized get() = closeRequested

    /** True once [onRelease] has run. */
    val isReleased: Boolean
        @Synchronized get() = released

    /** Take a lease, or return false when the resource is closing or closed. */
    @Synchronized
    fun tryAcquire(): Boolean {
        if (closeRequested) {
            return false
        }
        active += 1
        return true
    }

    /** Drop a lease taken with [tryAcquire]. */
    fun release() {
        val releaseNow = synchronized(this) {
            check(active > 0) { "lease released more times than acquired" }
            active -= 1
            claimReleaseLocked()
        }
        if (releaseNow) {
            try {
                onRelease()
            } catch (_: Exception) {
                // Deferred release has no caller; see the class documentation.
            }
        }
    }

    /** Stop new leases and release the resource once no lease remains. */
    fun close() {
        val releaseNow = synchronized(this) {
            if (closeRequested) {
                return
            }
            closeRequested = true
            claimReleaseLocked()
        }
        if (releaseNow) {
            onRelease()
        }
    }

    private fun claimReleaseLocked(): Boolean {
        if (!closeRequested || active != 0 || released) {
            return false
        }
        released = true
        return true
    }
}

/** Run [block] under a lease, throwing [onClosed] when the gate is closed. */
internal inline fun <T> LeaseGate.withLease(
    onClosed: () -> Nothing,
    block: () -> T,
): T {
    if (!tryAcquire()) {
        onClosed()
    }
    try {
        return block()
    } finally {
        release()
    }
}
