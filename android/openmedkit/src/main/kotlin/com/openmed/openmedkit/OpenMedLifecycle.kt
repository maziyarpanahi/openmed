package com.openmed.openmedkit

import android.content.ComponentCallbacks2
import android.content.Context
import android.content.res.Configuration
import java.io.Closeable
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Value-free memory-pressure levels understood by [OpenMedRuntime].
 *
 * Levels are ordered by severity. [OpenMedRuntimePolicy] decides which level
 * releases idle sessions and which level stops admission.
 */
enum class OpenMedMemoryPressure {
    /** Pressure has cleared; a memory-pressure admission pause is lifted. */
    NONE,

    /** Informational; the default policy takes no action. */
    MODERATE,

    /** Release idle sessions; admission stays open. */
    LOW,

    /** Stop admission and release sessions once they are idle. */
    CRITICAL,
    ;

    companion object {
        /**
         * Map an Android `ComponentCallbacks2.onTrimMemory` level.
         *
         * | Android level | Pressure |
         * | --- | --- |
         * | `TRIM_MEMORY_RUNNING_MODERATE` | [MODERATE] |
         * | `TRIM_MEMORY_RUNNING_LOW` | [LOW] |
         * | `TRIM_MEMORY_RUNNING_CRITICAL` | [CRITICAL] |
         * | `TRIM_MEMORY_UI_HIDDEN` | [MODERATE] |
         * | `TRIM_MEMORY_BACKGROUND`, `TRIM_MEMORY_MODERATE` | [LOW] |
         * | `TRIM_MEMORY_COMPLETE` and above | [CRITICAL] |
         *
         * Unknown levels below `TRIM_MEMORY_RUNNING_MODERATE` map to
         * [MODERATE] so an unrecognised signal never lifts a pause.
         *
         * @param level The raw Android trim level.
         * @return The value-free pressure level.
         */
        @JvmStatic
        fun fromTrimLevel(level: Int): OpenMedMemoryPressure = when {
            level >= ComponentCallbacks2.TRIM_MEMORY_COMPLETE -> CRITICAL
            level >= ComponentCallbacks2.TRIM_MEMORY_BACKGROUND -> LOW
            level >= ComponentCallbacks2.TRIM_MEMORY_UI_HIDDEN -> MODERATE
            level >= ComponentCallbacks2.TRIM_MEMORY_RUNNING_CRITICAL -> CRITICAL
            level >= ComponentCallbacks2.TRIM_MEMORY_RUNNING_LOW -> LOW
            else -> MODERATE
        }
    }
}

/**
 * Host lifecycle signals an [OpenMedRuntime] reacts to.
 *
 * Signals carry no host object, so a runtime never retains an Activity,
 * a bridge context, or caller input through this interface.
 */
interface OpenMedLifecycleObserver {
    /** The host process moved to the foreground. */
    fun onHostForeground()

    /** The host process moved to the background. */
    fun onHostBackground()

    /**
     * The platform reported memory pressure.
     *
     * @param pressure The value-free pressure level.
     */
    fun onMemoryPressure(pressure: OpenMedMemoryPressure)
}

/**
 * A source of host lifecycle signals, such as `ProcessLifecycleOwner` or trim-memory callbacks.
 *
 * OpenMedKit does not depend on `androidx.lifecycle`; hosts adapt their own
 * lifecycle owner with a few lines (see `docs/android-integration.md`).
 */
fun interface OpenMedLifecycleSource {
    /**
     * Start delivering signals to [observer].
     *
     * @param observer The runtime that receives signals.
     * @return A handle that stops delivery and drops the observer when closed.
     */
    fun subscribe(observer: OpenMedLifecycleObserver): Closeable
}

/**
 * `ComponentCallbacks2` adapter that forwards trim-memory signals.
 *
 * @param observer The runtime that receives value-free pressure levels.
 */
class OpenMedComponentCallbacks(
    private val observer: OpenMedLifecycleObserver,
) : ComponentCallbacks2 {
    override fun onTrimMemory(level: Int) {
        observer.onMemoryPressure(OpenMedMemoryPressure.fromTrimLevel(level))
    }

    override fun onLowMemory() {
        observer.onMemoryPressure(OpenMedMemoryPressure.CRITICAL)
    }

    override fun onConfigurationChanged(newConfig: Configuration) = Unit
}

/**
 * Trim-memory source registered on the application context.
 *
 * Only the application context is retained, so binding from an Activity or a
 * bridge module never keeps that host alive after it is recreated.
 *
 * @param context Any context; its application context is used.
 */
class OpenMedTrimMemorySource(context: Context) : OpenMedLifecycleSource {
    private val applicationContext: Context = context.applicationContext ?: context

    override fun subscribe(observer: OpenMedLifecycleObserver): Closeable {
        val callbacks = OpenMedComponentCallbacks(observer)
        applicationContext.registerComponentCallbacks(callbacks)
        val open = AtomicBoolean(true)
        return Closeable {
            if (open.compareAndSet(true, false)) {
                applicationContext.unregisterComponentCallbacks(callbacks)
            }
        }
    }
}
