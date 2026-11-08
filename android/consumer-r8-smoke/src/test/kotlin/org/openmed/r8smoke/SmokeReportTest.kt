package org.openmed.r8smoke

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

class SmokeReportTest {
    @Test
    fun formatsChecksAsValueFreeKeyValueLines() {
        val report = SmokeReport(
            flavor = "openmedkit",
            checks = listOf(
                SmokeCheck("catalog", SmokeStatus.PASS, count = 3),
                SmokeCheck(
                    "extract_pii",
                    SmokeStatus.PASS,
                    count = 1,
                    spans = listOf(SmokeSpan("NAME", 8, 19)),
                ),
            ),
        )

        assertEquals(
            listOf(
                "openmed_r8_smoke check=catalog status=PASS count=3",
                "openmed_r8_smoke check=extract_pii status=PASS count=1 spans=NAME:8-19",
                "openmed_r8_smoke flavor=openmedkit result=PASS checks=2 failed=0",
            ),
            report.lines(),
        )
        assertTrue(report.passed)
    }

    @Test
    fun anyFailedCheckFailsTheRun() {
        val report = SmokeReport(
            flavor = "openmedkit",
            checks = listOf(
                SmokeCheck("catalog", SmokeStatus.PASS),
                SmokeCheck("session", SmokeStatus.FAIL, reason = "spans_mismatch"),
            ),
        )

        assertFalse(report.passed)
        assertEquals(
            "openmed_r8_smoke flavor=openmedkit result=FAIL checks=2 failed=1",
            report.resultLine(),
        )
    }

    @Test
    fun anEmptyRunDoesNotPass() {
        assertFalse(SmokeReport("openmedkit", emptyList()).passed)
    }

    @Test
    fun failureKeepsTheErrorClassAndDropsItsMessage() {
        val sentinel = "Patient Avery Quill /data/user/0/secret"
        val check = SmokeCheck.failure("session", UnsatisfiedLinkError(sentinel))

        val line = check.format()
        assertEquals(
            "openmed_r8_smoke check=session status=FAIL " +
                "reason=error:java.lang.UnsatisfiedLinkError",
            line,
        )
        assertFalse(line.contains("Avery"))
        assertFalse(line.contains("/data"))
    }

    @Test
    fun freeTextCannotEnterTheReport() {
        assertThrows(IllegalArgumentException::class.java) {
            SmokeSpan("Avery Quill", 8, 19)
        }
        assertThrows(IllegalArgumentException::class.java) {
            SmokeCheck("catalog", SmokeStatus.FAIL, reason = "contains spaces")
        }
        assertThrows(IllegalArgumentException::class.java) {
            SmokeCheck("Patient note", SmokeStatus.PASS)
        }
        assertThrows(IllegalArgumentException::class.java) {
            SmokeSpan("NAME", 19, 8)
        }
    }
}
