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
        assertEquals("2.2.0", OpenMedKit.VERSION)
    }
}
