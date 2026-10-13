"""Synthetic source/translation/rights gates and restricted custody controls."""

import json
from dataclasses import replace

import pytest

from openmed.training.languages.corpus_license_gate import (
    ParallelCorpusShard,
    corpus_use_report,
    render_corpus_use_report,
    require_corpus_permission,
)


def digest(code):
    return "sha256:" + code * 64


def shard(**changes):
    args = dict(
        shard_digest=digest("b"),
        source_digest=digest("a"),
        license_id="MIT",
        translation_method="human",
        consent_category="synthetic",
        permitted_uses=("training", "evaluation", "export"),
        derivative_policy="redistributable",
        lineage_digests=(digest("a"), digest("b")),
        translation_evidence_digest=digest("c"),
    )
    return ParallelCorpusShard(**(args | changes))


def test_complete_permissive_synthetic_shard_allows_explicit_declared_uses():
    for use in ("training", "evaluation", "export"):
        assert corpus_use_report(shard(), use)["outcome"] == "allow"
        assert require_corpus_permission(shard(), use) is None


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"source_digest": None}, "source_missing"),
        ({"lineage_digests": ()}, "lineage_missing"),
        ({"lineage_digests": (digest("c"), digest("b"))}, "lineage_source_unbound"),
        ({"lineage_digests": (digest("a"), digest("c"))}, "lineage_shard_unbound"),
        ({"translation_method": "unknown"}, "translation_unknown"),
        ({"translation_evidence_digest": None}, "translation_evidence_missing"),
        ({"consent_category": "unknown"}, "consent_unknown"),
        ({"consent_category": "denied"}, "consent_denied"),
        ({"consent_category": "explicit"}, "consent_evidence_missing"),
        ({"permitted_uses": ("evaluation",)}, "use_not_permitted"),
        ({"derivative_policy": "forbidden"}, "derivative_forbidden"),
        ({"derivative_policy": "unknown"}, "derivative_unknown"),
        ({"license_id": None}, "license_missing"),
    ],
)
def test_each_missing_or_forbidden_dimension_blocks_training(changes, reason):
    result = corpus_use_report(shard(**changes), "training")
    assert result["outcome"] == "deny"
    assert reason in result["reasons"]
    with pytest.raises(ValueError, match="corpus_permission_denied"):
        require_corpus_permission(shard(**changes), "training")


def test_explicit_consent_requires_separate_digest_and_does_not_override_other_policy():
    good = shard(consent_category="explicit", consent_evidence_digest=digest("d"))
    assert corpus_use_report(good, "training")["outcome"] == "allow"
    assert (
        corpus_use_report(replace(good, derivative_policy="forbidden"), "training")[
            "outcome"
        ]
        == "deny"
    )


def test_restricted_reviewed_corpus_stays_user_supplied_and_never_exportable():
    private = shard(
        license_id="LicenseRef-PrivateCorpus",
        license_review_digest=digest("d"),
        user_supplied=True,
        derivative_policy="user_supplied_only",
    )
    assert corpus_use_report(private, "training")["outcome"] == "allow"
    assert corpus_use_report(private, "evaluation")["outcome"] == "allow"
    result = corpus_use_report(private, "export")
    assert result["outcome"] == "deny"
    assert "restricted_export_prohibited" in result["reasons"]
    assert "user_supplied_export_prohibited" in result["reasons"]
    assert "LicenseRef-PrivateCorpus" not in render_corpus_use_report(
        private, "training"
    )


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"license_review_digest": None}, "restricted_license_unreviewed"),
        ({"user_supplied": False}, "restricted_input_not_user_supplied"),
        (
            {"derivative_policy": "redistributable"},
            "restricted_derivative_not_user_supplied_only",
        ),
    ],
)
def test_restricted_custody_review_and_derivative_controls_are_independent(
    changes, reason
):
    private = shard(
        license_id="LicenseRef-PrivateCorpus",
        license_review_digest=digest("d"),
        user_supplied=True,
        derivative_policy="user_supplied_only",
    )
    result = corpus_use_report(replace(private, **changes), "training")
    assert result["outcome"] == "deny"
    assert reason in result["reasons"]


def test_gate_stops_caller_action_before_incomplete_lineage_is_used():
    operations = []

    def entrypoint(meta, use):
        require_corpus_permission(meta, use)
        operations.append(use)

    with pytest.raises(ValueError):
        entrypoint(shard(lineage_digests=()), "training")
    with pytest.raises(ValueError):
        entrypoint(
            shard(derivative_policy="user_supplied_only", user_supplied=True), "export"
        )
    assert operations == []
    entrypoint(shard(), "training")
    assert operations == ["training"]


def test_unknown_license_never_becomes_permissive_from_spelling():
    result = corpus_use_report(shard(license_id="UnknownLicense"), "training")
    assert result["license_class"] == "restricted_or_unknown"
    assert result["outcome"] == "deny"


def test_reports_are_stable_and_exclude_source_translation_consent_payload_metadata():
    result = json.loads(render_corpus_use_report(shard(), "training"))
    assert set(result) == {
        "schema_version",
        "shard_digest",
        "use",
        "license_class",
        "outcome",
        "reasons",
    }
    assert "source_digest" not in result
    assert "license_id" not in result
    assert render_corpus_use_report(shard(), "training") == render_corpus_use_report(
        shard(permitted_uses=("export", "training", "evaluation")), "training"
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"source_digest": "synthetic/private"},
        {"license_id": "synthetic/private"},
        {"consent_category": "raw name"},
        {"translation_method": "cloud"},
        {"user_supplied": "yes"},
        {"lineage_digests": (digest("a"), digest("a"))},
        {"permitted_uses": ("training", "training")},
        {"lineage_digests": (None,)},
    ],
)
def test_malformed_metadata_rejects_without_echo(changes):
    with pytest.raises(ValueError) as exc:
        shard(**changes)
    assert "synthetic/private" not in str(exc.value)
    assert "raw name" not in str(exc.value)


def test_lineage_iterables_have_a_hard_bound():
    consumed = []

    def infinite():
        while True:
            consumed.append(1)
            yield digest("a")

    with pytest.raises(ValueError, match="limit"):
        shard(lineage_digests=infinite())
    assert len(consumed) == 33
