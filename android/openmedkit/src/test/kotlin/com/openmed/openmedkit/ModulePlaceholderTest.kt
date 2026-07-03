package com.openmed.openmedkit

import org.junit.Assert.assertEquals
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [26])
class ModulePlaceholderTest {
    @Test
    fun exposesPlaceholderVersion() {
        assertEquals("0.0.0-dev", OpenMedKit.VERSION)
    }
}
