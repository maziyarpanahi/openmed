"""Notice identity, independent version pins and value-free receipts."""

from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone

import pytest

from openmed.clinical import OPENMED_CLINICAL_DISCLAIMER
from openmed.core.disclaimer import OpenMedDisclaimer, notice_catalog
from openmed.core.intended_use import RegulatoryTier
from openmed.mlx.maple import MAPLE_MEDICAL_DISCLAIMER
from openmed.multimodal.notices import NOTICE_CATALOG, NoticeKind


def test_public_notices_keep_their_identity():
    assert (
        OpenMedDisclaimer.for_tier("assistive_review").text
        is OPENMED_CLINICAL_DISCLAIMER
    )
    assert (
        OpenMedDisclaimer.for_capability("openmed.MapleClinicalAssistant.chat").text
        is MAPLE_MEDICAL_DISCLAIMER
    )
    assert (
        OpenMedDisclaimer.for_capability("generate_text").text
        is OPENMED_CLINICAL_DISCLAIMER
    )
    notice = OpenMedDisclaimer.for_capability("ecg")
    assert notice.notice_id == NOTICE_CATALOG[NoticeKind.MEASUREMENT].identifier
    assert notice.text is NOTICE_CATALOG[NoticeKind.MEASUREMENT].text
    assert (
        OpenMedDisclaimer.for_capability("visual_qa").notice_id
        == NOTICE_CATALOG[NoticeKind.VISUAL_DESCRIPTION].identifier
    )


@pytest.mark.parametrize("tier", list(RegulatoryTier))
def test_receipt_contains_only_versioned_notice_and_utc_instant(tier):
    notice = OpenMedDisclaimer.for_tier(tier)
    instant = datetime(2024, 5, 1, 11, tzinfo=timezone(timedelta(hours=1)))
    receipt = notice.acknowledge(at=instant)
    if tier in (RegulatoryTier.RESTRICTED, RegulatoryTier.ASSISTIVE_REVIEW):
        assert asdict(receipt) == {
            "notice_id": notice.notice_id,
            "version_digest": notice.version_digest,
            "acknowledged_at": "2024-05-01T10:00:00+00:00",
        }
        with pytest.raises(ValueError, match="invalid_acknowledgement_time"):
            notice.acknowledge(at=instant.replace(tzinfo=None))
    else:
        assert receipt is None


def test_catalog_matches_published_notice_versions():
    for notice in notice_catalog().values():
        pinned = PUBLISHED[(notice.notice_id, notice.version, notice.locale)]
        assert notice.version_digest == pinned
        assert replace(notice, text=notice.text + "!").version_digest != pinned
    assert set(notice_catalog()) == {key[0] for key in PUBLISHED}


# Keep historical pins when publishing a new version.
PUBLISHED = {
    (
        "openmed.non_clinical",
        "1",
        "en",
    ): "sha256:29d49fd604e07a994751c55fda4167c63c9e53422a0670c4c0206024392f6bad",
    (
        "openmed.informational",
        "1",
        "en",
    ): "sha256:6f92c5f262c6b301c0596a4bda0391361a6f792ec3febd3f863fcab7cd164bc9",
    (
        "openmed.clinical",
        "1",
        "en",
    ): "sha256:81a2c40b9c512914075b36c4e88bbe9be9018b712fc8279c55868059f25ecc87",
    (
        "openmed.restricted",
        "1",
        "en",
    ): "sha256:fa1788f696f18cdd80866d9c9132fa2e799bd7ffe353493c52034396b954cdad",
    (
        "openmed.maple",
        "1",
        "en",
    ): "sha256:02d5c3716042f489748c9da1603c71f842f5ef84e6cc2abe2a456a9233dd2713",
    (
        "openmed.multimodal.measurement_for_review.v1",
        "1",
        "en",
    ): "sha256:ff1bf73b4a86a84faf560a986c8988947820318aee52f03ea8588149425ec177",
    (
        "openmed.multimodal.visual_description.v1",
        "1",
        "en",
    ): "sha256:9aeced47b086b1116174ef9b9d7a98082d3a3d56fb409abbbdcf10ce78666d79",
    (
        "openmed.multimodal.draft_for_review.v1",
        "1",
        "en",
    ): "sha256:e3669fa76dfcaff464c4d0fbbfc579481b151eef6b4f3498b4e08d6e6855bb7a",
}
