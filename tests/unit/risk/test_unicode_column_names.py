"""End-to-end release-policy tests for real-world schema labels."""

import pytest

from openmed.risk import AnonymityPolicy, anonymize_release, assess_release


def test_release_policy_accepts_unicode_spaces_and_commas() -> None:
    rows = [
        {
            "患者 ID": "patient-a",
            "Patient Age": 40,
            "cohort, stratum": "A",
            "diagnóstico": "condition-a",
        },
        {
            "患者 ID": "patient-b",
            "Patient Age": 40,
            "cohort, stratum": "B",
            "diagnóstico": "condition-b",
        },
    ]
    policy = AnonymityPolicy(
        quasi_identifiers=("Patient Age",),
        sensitive_attributes=("diagnóstico",),
        non_sensitive_attributes=("cohort, stratum",),
        privacy_unit="患者 ID",
        target_k=2,
    )

    assessment = assess_release(rows, policy)
    result = anonymize_release(rows, policy)

    assert assessment.meets_policy is True
    assert result.after.meets_policy is True
    assert all("患者 ID" not in row for row in result.records)
    assert result.policy.to_dict()["quasi_identifiers"] == ["Patient Age"]


@pytest.mark.parametrize(
    "column",
    (
        " leading",
        "trailing ",
        "line\nbreak",
        "zero\u200bwidth",
        "line\u2028separator",
        "paragraph\u2029separator",
    ),
)
def test_release_policy_rejects_ambiguous_or_controlled_names(column: str) -> None:
    with pytest.raises(ValueError, match="column names"):
        AnonymityPolicy(quasi_identifiers=(column,), target_k=1)
