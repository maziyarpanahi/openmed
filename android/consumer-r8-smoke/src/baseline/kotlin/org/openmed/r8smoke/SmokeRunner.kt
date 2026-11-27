package org.openmed.r8smoke

import android.content.Context

/**
 * Baseline flavor: the same app shell without OpenMedKit.
 *
 * Its release APKs are the reference that per-ABI size deltas are measured
 * against, so it must stay free of any OpenMedKit dependency.
 */
object SmokeRunner {
    fun run(@Suppress("UNUSED_PARAMETER") context: Context): SmokeReport = SmokeReport(
        flavor = "baseline",
        checks = listOf(
            SmokeCheck("openmedkit_absent", SmokeStatus.SKIPPED, reason = "baseline"),
        ),
    )
}
