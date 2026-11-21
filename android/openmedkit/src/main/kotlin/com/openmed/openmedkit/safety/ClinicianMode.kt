package com.openmed.openmedkit.safety

/** Explicit purposes for restricted review. None authorizes an autonomous action. */
enum class ClinicianPurpose(val code: String) {
    REVIEW("review"), RESEARCH("research"), QUALITY_ASSURANCE("quality_assurance")
}

class ClinicianModeRequired : IllegalStateException("clinician_mode_required")
class NoticeAcknowledgementRequired : IllegalArgumentException("acknowledgement_required")
class InvalidNoticeAcknowledgement : IllegalArgumentException("invalid_acknowledgement")

/** Value-free receipt. Possession of this record does not re-enable the mode. */
data class ClinicianModeReceipt(val purpose: String, val noticeId: String, val noticeDigest: String)

/** Host-owned in-memory mode. Each new instance starts disabled. */
class ClinicianMode {
    @Volatile private var active: ClinicianModeReceipt? = null

    fun enable(purpose: ClinicianPurpose, acknowledgement: OpenMedNoticeAcknowledgement?): ClinicianModeReceipt {
        if (acknowledgement == null) throw NoticeAcknowledgementRequired()
        val notice = OpenMedDisclaimer.byId(acknowledgement.noticeId)
        if (notice == null || notice.tier != "restricted" || !notice.permitsResults(acknowledgement)) {
            throw InvalidNoticeAcknowledgement()
        }
        return ClinicianModeReceipt(purpose.code, notice.id, notice.versionDigest).also { active = it }
    }

    fun disable() { active = null }

    /** Check the registered purpose or tier, applying restricted safeguards to unknown codes. */
    fun requireCapability(capability: String): ClinicianModeReceipt? {
        val tier = OpenMedDisclaimer.forCapability(capability)?.tier
        return requireTier(tier)
    }

    fun requireTier(tier: String?): ClinicianModeReceipt? {
        if (OpenMedDisclaimer.forTier(tier).tier != "restricted") return null
        return active ?: throw ClinicianModeRequired()
    }
}
