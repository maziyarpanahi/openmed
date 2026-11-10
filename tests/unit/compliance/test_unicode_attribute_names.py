"""Schema-name compatibility tests for expert-review evidence."""

import pytest

from openmed.compliance import AttributeRoleReview, TransformationAggregate
from openmed.compliance import expert_review as expert_review_module


def test_evidence_accepts_real_world_attribute_names(subtests) -> None:
    for attribute in ("Patient Age", "âge", "患者年龄", "cohort, stratum"):
        with subtests.test(attribute=attribute):
            review = AttributeRoleReview(
                attribute=attribute, roles=("quasi_identifier",)
            )
            transformation = TransformationAggregate(
                attribute=attribute,
                method="generalize",
                affected_privacy_unit_count=1,
                hierarchy_level_before=0,
                hierarchy_level_after=1,
            )

            assert review.to_dict()["attribute"] == attribute
            assert transformation.to_dict()["attribute"] == attribute


def test_evidence_rejects_ambiguous_or_controlled_attribute_names(subtests) -> None:
    for attribute in (
        " leading",
        "trailing ",
        "line\nbreak",
        "\x00",
        "line\u2028separator",
        "paragraph\u2029separator",
    ):
        with subtests.test(attribute=attribute):
            with pytest.raises(ValueError, match="source column name"):
                AttributeRoleReview(attribute=attribute, roles=("quasi_identifier",))


def test_attribute_names_render_as_table_safe_markdown_code() -> None:
    rendered = expert_review_module._markdown_code("measure|`tick`")

    assert r"\|" in rendered
    assert rendered.startswith("``")
    assert rendered.endswith("``")
