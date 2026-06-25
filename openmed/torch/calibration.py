"""Synthetic calibration text for PyTorch export recipes."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

SYNTHETIC_CLINICAL_AWQ_CALIBRATION_TEXTS: tuple[str, ...] = (
    "Discharge note: Sample Patient A returned for blood pressure follow-up "
    "after lisinopril adjustment. Home readings averaged 128/76.",
    "Emergency triage: Synthetic adult with fever, cough, and oxygen saturation "
    "of 96 percent on room air. Chest exam clear.",
    "Medication review: Placeholder patient reports taking metformin 500 mg "
    "twice daily and atorvastatin 20 mg nightly without side effects.",
    "Clinic message: Care team asked the patient to repeat a basic metabolic "
    "panel before the next telehealth visit.",
    "Radiology summary: CT abdomen showed no acute process. Incidental simple "
    "renal cyst was discussed with the ordering clinician.",
    "Procedure note: Synthetic case received local anesthetic before a skin "
    "biopsy. Hemostasis achieved with pressure dressing.",
    "Nursing handoff: Patient resting comfortably, pain score 2 of 10, tolerating "
    "oral fluids, ambulating with standby assistance.",
    "Care plan: Continue inhaled corticosteroid, review spacer technique, and "
    "schedule pulmonary function testing in six weeks.",
    "Lab interpretation: Hemoglobin A1c improved from 8.1 percent to 7.2 percent "
    "after nutrition counseling and medication adherence support.",
    "Consult note: Neurology recommended migraine trigger diary, hydration, and "
    "trial of magnesium supplementation.",
    "Behavioral health note: Synthetic patient describes interrupted sleep and "
    "situational anxiety related to work stressors.",
    "Pediatric visit: Guardian reports mild sore throat and decreased appetite; "
    "rapid strep test was negative.",
)


def load_awq_calibration_texts(limit: int | None = None) -> list[str]:
    """Return deterministic synthetic clinical notes for AWQ calibration.

    Args:
        limit: Optional positive maximum number of samples to return. ``None``
            returns the full committed set.

    Raises:
        ValueError: If ``limit`` is zero or negative.
    """

    if limit is not None and limit <= 0:
        raise ValueError("limit must be a positive integer")
    return list(SYNTHETIC_CLINICAL_AWQ_CALIBRATION_TEXTS[:limit])


def calibration_texts_sha256(texts: Sequence[str]) -> str:
    """Return a stable digest for a calibration text sequence."""

    digest = hashlib.sha256()
    for text in texts:
        digest.update(text.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


__all__ = [
    "SYNTHETIC_CLINICAL_AWQ_CALIBRATION_TEXTS",
    "calibration_texts_sha256",
    "load_awq_calibration_texts",
]
