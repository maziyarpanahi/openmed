package com.openmed.openmedkit.deid

/**
 * Supported text rewrite strategies for Android de-identification.
 */
enum class DeidentifyMethod {
    MASK,
    REMOVE,
    REPLACE,
    HASH,
}
