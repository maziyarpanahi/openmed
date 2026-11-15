package org.openmed.r8smoke

import android.app.Activity
import android.os.Bundle
import android.util.Log
import android.widget.TextView

/**
 * Launcher activity of the minified consumer smoke app.
 *
 * It runs [SmokeRunner] once on a worker thread and writes the value-free
 * report to logcat under [SmokeReport.LOG_TAG]. An emulator lane launches it
 * with `am start -W` and waits for the `result=` line.
 */
class SmokeActivity : Activity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val status = TextView(this)
        status.text = "running"
        setContentView(status)
        val appContext = applicationContext
        Thread(
            {
                val report = SmokeRunner.run(appContext)
                report.lines().forEach { Log.i(SmokeReport.LOG_TAG, it) }
                runOnUiThread { status.text = report.resultLine() }
            },
            "openmed-r8-smoke",
        ).start()
    }
}
