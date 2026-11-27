package com.openmed.openmedkit

/**
 * An OpenMedKit failure that carries a portable [OpenMedErrorCode] from the
 * shared error-code registry next to its own reason code. Branch on
 * [errorCode] the way Python callers branch on `OpenMedError.code`; the
 * kit-specific reason codes stay unchanged.
 */
interface OpenMedCodedError {
    /** The portable registry code of this failure. */
    val errorCode: OpenMedErrorCode
}

/**
 * An invalid argument or configuration with a portable [errorCode]. It is still
 * an [IllegalArgumentException], so existing handlers keep catching it, and its
 * message carries no input text.
 */
class OpenMedArgumentException internal constructor(
    override val errorCode: OpenMedErrorCode,
    message: String,
) : IllegalArgumentException(message), OpenMedCodedError
