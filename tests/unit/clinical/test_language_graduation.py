"""Low-resource language cohort records and non-compensable graduation gates."""

from __future__ import annotations

import json
import re
from dataclasses import FrozenInstanceError, replace

import pytest

from openmed.clinical.languages.graduation import (
    CRITICAL_IDENTIFIER_LABELS,
    EXPERIMENTAL,
    GRADUATION_TASKS,
    INITIAL_LOW_RESOURCE_COHORT,
    REVIEW_ROLES,
    STATUSES,
    SUPPORTED,
    TASKS,
    UNSUPPORTED,
    CohortLocale,
    CrossLingualCell,
    GraduationContract,
    GraduationEvidence,
    IdentifierRecall,
    LanguageCohort,
    LanguageGraduationError,
    LicenseLineage,
    SyntheticFixturePlan,
    cohort_inventory_findings,
    evaluate_graduation,
    graduate,
    repository_language_inventory,
    wilson_lower_bound,
)

COHORT = INITIAL_LOW_RESOURCE_COHORT
REFERENCES = ("en", "hi")
_CODE = re.compile(r"[A-Za-z0-9_./:-]*")


def _active(language: str = "sw") -> CohortLocale:
    return replace(COHORT.get(language), reviewer_commitment="active")


def _recalls(**overrides: tuple[int, int]) -> tuple[IdentifierRecall, ...]:
    counts = {label: (200, 0) for label in CRITICAL_IDENTIFIER_LABELS}
    counts.update(overrides)
    return tuple(
        IdentifierRecall(label, detected, missed)
        for label, (detected, missed) in counts.items()
    )


def _cells(references=REFERENCES, **overrides: tuple[int, int, int]):
    cells = {
        (language, label): (150, 148, 148)
        for language in references
        for label in CRITICAL_IDENTIFIER_LABELS
    }
    for key, value in overrides.items():
        language, label = key.split("__")
        cells[(language, label)] = value
    return tuple(
        CrossLingualCell(language, label, *counts)
        for (language, label), counts in cells.items()
    )


def _evidence(**overrides) -> GraduationEvidence:
    args = dict(
        language="sw",
        task="deidentification",
        documents=250,
        identifier_recall=_recalls(),
        terminology_terms=250,
        terminology_covered=240,
        abstention_probes=60,
        abstention_failures=0,
        cross_lingual=_cells(),
        fixture_authoring="native_authored_synthetic",
        review_role="native_clinical_reviewer",
        review_status="approved",
        model_tiers=("small",),
        package_bytes=500_000_000,
        offline_verified=True,
        maintainer_role="language_pack_maintainer",
        review_interval_days=90,
        lexicon_digest="a" * 64,
        fixture_digest="b" * 64,
        evaluation_digest="c" * 64,
        native_review_digest="d" * 64,
        package_digest="e" * 64,
    )
    args.update(overrides)
    return GraduationEvidence(**args)


def _failures(entry=None, evidence=None, **kwargs) -> tuple[str, ...]:
    report = evaluate_graduation(
        entry or _active(),
        evidence or _evidence(),
        reference_languages=kwargs.pop("reference_languages", REFERENCES),
        **kwargs,
    )
    return report.failures


def _assert_codes_only(value) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            assert _CODE.fullmatch(key)
            _assert_codes_only(item)
    elif isinstance(value, list):
        for item in value:
            _assert_codes_only(item)
    elif isinstance(value, str):
        assert _CODE.fullmatch(value), value


# --- Cohort records -------------------------------------------------------


def test_every_cohort_locale_records_lineage_role_plan_and_status():
    assert COHORT.languages == ("am", "ne", "or", "pa", "sw", "ur", "yo", "zu")
    for entry in COHORT.entries:
        assert isinstance(entry.license_lineage, LicenseLineage)
        assert entry.license_lineage.bundles_restricted_data is False
        assert entry.review_role in REVIEW_ROLES
        assert isinstance(entry.fixture_plan, SyntheticFixturePlan)
        assert entry.fixture_plan.target_documents >= 200
        assert entry.fixture_plan.authoring != "machine_translation"
        assert entry.status in STATUSES
        assert entry.candidate_tasks == GRADUATION_TASKS
        # The repository holds no native-review or evaluation evidence yet.
        assert entry.status == EXPERIMENTAL
        assert entry.supported_tasks == ()
        assert entry.reviewer_commitment == "unfilled"


def test_cohort_selection_matches_repository_evidence():
    inventory = repository_language_inventory(COHORT.languages)
    assert cohort_inventory_findings(COHORT, inventory) == ()
    assert inventory["ne"]["route"] == "user_supplied_model"
    assert inventory["yo"]["route"] == "national_id_only"
    assert COHORT.get("ne").fixture_plan.golden_records == 0
    assert "PERSON" in COHORT.get("yo").fixture_plan.missing_labels
    assert "tokenizer_script_unclaimed" in COHORT.get("pa").limitations


def test_inventory_findings_flag_overstated_or_contradicted_records():
    inventory = repository_language_inventory(COHORT.languages)
    drifted = {code: dict(row) for code, row in inventory.items()}
    drifted["sw"]["route"] = "dedicated_checkpoint"
    drifted["am"]["golden_records"] = 1
    drifted["or"]["repository_evidence"] = ("language_pack",)
    drifted["zu"]["surrogate_origin"] = "faker_approximation"
    drifted["ur"]["tokenizer_unclaimed"] = True
    del drifted["yo"]
    assert cohort_inventory_findings(COHORT, drifted) == (
        ("am", "golden_records_overstated"),
        ("or", "evidence_unbacked"),
        ("sw", "route_mismatch"),
        ("ur", "tokenizer_limitation_missing"),
        ("yo", "inventory_missing"),
        ("zu", "surrogate_origin_overstated"),
    )


def test_inventory_findings_tolerate_evidence_added_later():
    inventory = repository_language_inventory(COHORT.languages)
    grown = {code: dict(row) for code, row in inventory.items()}
    grown["ne"]["golden_records"] = 2
    grown["ne"]["golden_labels"] = ("DATE", "ID_NUM", "PHONE")
    grown["ne"]["repository_evidence"] += ("golden_i18n_fixture",)
    grown["yo"]["golden_records"] = 5
    grown["ur"]["surrogate_origin"] = "faker_native"
    assert cohort_inventory_findings(COHORT, grown) == ()


def test_repository_inventory_reads_only_synthetic_fixture_metadata(tmp_path):
    record = {
        "id": "golden-i18n-sw-synthetic",
        "language": "sw",
        "text": "Synthetic Mgonjwa Juma",
        "gold_spans": [{"start": 17, "end": 21, "label": "PERSON"}],
        "metadata": {"synthetic": True},
    }
    (tmp_path / "sw.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    row = repository_language_inventory(["sw"], fixture_root=tmp_path)["sw"]
    assert row["golden_records"] == 1
    assert row["golden_labels"] == ("PERSON",)
    assert "Juma" not in json.dumps(row)

    record["metadata"] = {"synthetic": False}
    (tmp_path / "sw.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError) as exc:
        repository_language_inventory(["sw"], fixture_root=tmp_path)
    assert "Juma" not in str(exc.value)


# --- Fail-closed task gate ------------------------------------------------


@pytest.mark.parametrize("language", COHORT.languages)
def test_experimental_locales_fail_closed_for_every_task(language):
    with pytest.raises(LanguageGraduationError, match="task_not_supported"):
        COHORT.require_task(language, "deidentification")
    assert (
        COHORT.require_task(language, "deidentification", allow_experimental=True)
        == EXPERIMENTAL
    )
    for task in set(TASKS) - set(GRADUATION_TASKS):
        assert COHORT.task_status(language, task) == UNSUPPORTED
        with pytest.raises(LanguageGraduationError, match="task_not_supported"):
            COHORT.require_task(language, task, allow_experimental=True)


@pytest.mark.parametrize(
    "language,task,code",
    [
        ("en", "deidentification", "language_not_in_cohort"),
        ("sw-TZ", "deidentification", "locale_not_in_cohort"),
        ("/private/raw note", "deidentification", "invalid_language"),
        ("sw", "raw clinical note", "unknown_task"),
    ],
)
def test_lookup_refusals_use_controlled_codes(language, task, code):
    with pytest.raises(LanguageGraduationError) as exc:
        COHORT.require_task(language, task, allow_experimental=True)
    assert exc.value.code == code
    assert str(exc.value) == code
    assert COHORT.task_status(language, task) == UNSUPPORTED


def test_matching_region_and_aliases_resolve_to_the_recorded_locale():
    assert COHORT.get("AM_et").language == "am"
    assert COHORT.get("zu-ZA").language == "zu"


# --- Graduation -----------------------------------------------------------


def test_complete_evidence_graduates_only_the_evaluated_task():
    promoted, report = graduate(_active(), _evidence(), reference_languages=REFERENCES)
    assert report.passed and report.status == SUPPORTED
    assert report.failures == ()
    assert promoted.status == SUPPORTED
    assert promoted.graduations == (("deidentification", report.digest),)

    cohort = COHORT.with_entry(promoted)
    assert cohort.require_task("sw", "deidentification") == SUPPORTED
    with pytest.raises(LanguageGraduationError, match="task_not_supported"):
        cohort.require_task("sw", "summarization", allow_experimental=True)
    with pytest.raises(LanguageGraduationError, match="task_not_supported"):
        cohort.require_task("am", "deidentification")
    assert COHORT.get("sw").status == EXPERIMENTAL


def test_critical_recall_is_non_compensable_across_labels():
    evidence = _evidence(
        identifier_recall=_recalls(PHONE=(194, 6), PERSON=(5000, 0), DATE=(5000, 0))
    )
    report = evaluate_graduation(_active(), evidence, reference_languages=REFERENCES)
    assert report.micro_recall > 0.99
    assert not report.passed
    assert "critical_recall_below_floor:PHONE" in report.failures
    rows = {row["label"]: row for row in report.critical_identifiers}
    assert rows["PHONE"]["passed"] is False
    assert rows["PERSON"]["passed"] is True


def test_missing_or_thin_critical_labels_cannot_be_skipped():
    recalls = tuple(r for r in _recalls() if r.label != "PERSON")
    assert "critical_label_missing:PERSON" in _failures(
        evidence=_evidence(identifier_recall=recalls)
    )
    assert "critical_label_support_low:ZIPCODE" in _failures(
        evidence=_evidence(identifier_recall=_recalls(ZIPCODE=(50, 0)))
    )


def test_point_recall_alone_cannot_pass_an_uncertain_slice():
    failures = _failures(evidence=_evidence(identifier_recall=_recalls(ID_NUM=(99, 1))))
    assert "critical_recall_uncertain:ID_NUM" in failures
    assert "critical_recall_below_floor:ID_NUM" not in failures


def test_cross_lingual_regression_is_bounded_per_cell_without_offsets():
    cells = _cells(en__PERSON=(150, 148, 147), hi__PERSON=(150, 140, 150))
    report = evaluate_graduation(
        _active(), _evidence(cross_lingual=cells), reference_languages=REFERENCES
    )
    assert report.failures == ("cross_lingual_regression:en:PERSON",)


def test_cross_lingual_cells_must_match_the_reference_set():
    failures = _failures(reference_languages=("bn", "en", "hi"))
    assert "cross_lingual_cell_missing:bn:PERSON" in failures
    extra = _cells() + (CrossLingualCell("fr", "DATE", 150, 140, 140),)
    assert "cross_lingual_cell_unexpected:fr:DATE" in _failures(
        evidence=_evidence(cross_lingual=extra)
    )
    thin = _cells(en__DATE=(10, 10, 10))
    assert "cross_lingual_slice_too_small:en:DATE" in _failures(
        evidence=_evidence(cross_lingual=thin)
    )
    for references in ((), ("sw", "en")):
        with pytest.raises(ValueError):
            _failures(reference_languages=references)


@pytest.mark.parametrize(
    "overrides,code",
    [
        ({"language": "am"}, "language_mismatch"),
        ({"review_status": "pending"}, "native_review_not_approved"),
        ({"review_role": "privacy_reviewer"}, "review_role_mismatch"),
        ({"fixture_authoring": "machine_translation"}, "translation_only_evidence"),
        ({"documents": 120}, "slice_too_small"),
        ({"terminology_covered": 150}, "terminology_coverage_below_floor"),
        (
            {"terminology_terms": 120, "terminology_covered": 120},
            "terminology_slice_too_small",
        ),
        ({"abstention_failures": 1}, "abstention_failures"),
        ({"abstention_probes": 10}, "abstention_probes_insufficient"),
        ({"model_tiers": ()}, "device_tiers_missing"),
        ({"offline_verified": False}, "package_not_offline"),
        ({"package_bytes": 1024**3 + 1}, "package_over_budget"),
        ({"review_interval_days": 400}, "maintenance_interval_too_long"),
    ],
)
def test_each_graduation_gate_fails_independently(overrides, code):
    assert _failures(evidence=_evidence(**overrides)) == (code,)


def test_unfilled_reviewer_role_blocks_graduation():
    failures = _failures(entry=COHORT.get("sw"))
    assert failures == ("reviewer_commitment_not_active",)
    with pytest.raises(LanguageGraduationError, match="graduation_failed"):
        graduate(COHORT.get("sw"), _evidence(), reference_languages=REFERENCES)


@pytest.mark.parametrize("language", ["ne", "yo"])
def test_routes_without_bundled_weights_cannot_graduate(language):
    evidence = _evidence(language=language)
    assert "route_without_bundled_model" in _failures(
        entry=_active(language), evidence=evidence
    )
    with pytest.raises(LanguageGraduationError, match="graduation_failed"):
        graduate(_active(language), evidence, reference_languages=REFERENCES)


# --- Contract and record validation ---------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"critical_recall_floor": 0.95},
        {"critical_recall_lower_bound": 0.90},
        {"critical_recall_floor": 0.995, "critical_recall_lower_bound": 0.999},
        {"min_label_support": 10},
        {"min_documents": 50},
        {"max_cross_lingual_regression": 0.05},
        {"critical_labels": ("DATE", "ID_NUM")},
        {"min_terminology_coverage": 0.5},
        {"min_abstention_probes": 1},
        {"min_cross_lingual_trials": 5},
        {"max_package_bytes": 3 * 1024**3},
        {"max_review_interval_days": 1000},
        {"critical_recall_floor": float("nan")},
        {"min_documents": True},
    ],
)
def test_contract_cannot_loosen_policy_floors(kwargs):
    with pytest.raises(ValueError):
        GraduationContract(**kwargs)


def test_contract_can_tighten_and_extend_critical_labels():
    contract = GraduationContract(
        critical_recall_floor=1.0,
        critical_recall_lower_bound=0.97,
        critical_labels=CRITICAL_IDENTIFIER_LABELS + ("AGE",),
    )
    failures = _failures(contract=contract)
    assert "critical_label_missing:AGE" in failures
    assert "cross_lingual_cell_missing:en:AGE" in failures


def test_supported_status_requires_graduation_and_bundled_weights():
    sw = COHORT.get("sw")
    with pytest.raises(ValueError):
        replace(sw, status=SUPPORTED)
    graduated = (("deidentification", "f" * 64),)
    with pytest.raises(ValueError):
        replace(sw, status=SUPPORTED, graduations=graduated)
    assert replace(
        _active(), status=SUPPORTED, graduations=graduated
    ).supported_tasks == ("deidentification",)
    with pytest.raises(ValueError):
        replace(_active("ne"), status=SUPPORTED, graduations=graduated)
    with pytest.raises(ValueError):
        replace(_active(), graduations=graduated)


@pytest.mark.parametrize(
    "build",
    [
        lambda sw: replace(sw.license_lineage, bundles_restricted_data=True),
        lambda sw: replace(sw.license_lineage, model_license="GPL-3.0-only"),
        lambda sw: replace(sw.license_lineage, model_license=None),
        lambda sw: replace(sw.fixture_plan, authoring="machine_translation"),
        lambda sw: replace(sw.fixture_plan, target_documents=20),
        lambda sw: replace(sw.fixture_plan, stressors=()),
        lambda sw: replace(sw, review_role="Dr Synthetic Reviewer"),
        lambda sw: replace(sw, limitations=("no_bundled_model",)),
        lambda sw: replace(sw, route="national_id_only"),
        lambda sw: replace(sw, candidate_tasks=("summarization",)),
        lambda sw: LanguageCohort((sw, sw)),
    ],
)
def test_records_reject_restricted_unreviewed_or_contradictory_values(build):
    with pytest.raises(ValueError) as exc:
        build(COHORT.get("sw"))
    assert "Synthetic Reviewer" not in str(exc.value)


def test_records_are_frozen():
    with pytest.raises(FrozenInstanceError):
        COHORT.get("sw").status = SUPPORTED


# --- Determinism, privacy and compatibility -------------------------------


def test_reports_are_order_invariant_and_content_free():
    ordered = _evidence()
    shuffled = _evidence(
        identifier_recall=tuple(reversed(_recalls())),
        cross_lingual=tuple(reversed(_cells())),
    )
    assert ordered == shuffled
    first = evaluate_graduation(_active(), ordered, reference_languages=REFERENCES)
    second = evaluate_graduation(_active(), shuffled, reference_languages=("hi", "en"))
    assert first.digest == second.digest
    _assert_codes_only(first.to_dict())
    report = json.loads(COHORT.to_json())
    _assert_codes_only(report)
    assert COHORT.to_json() == LanguageCohort(tuple(reversed(COHORT.entries))).to_json()
    assert report["contract_digest"] == GraduationContract().digest


def test_evidence_projects_onto_readiness_evidence_fields():
    rows = _evidence(model_tiers=("nano", "small")).to_readiness_evidence()
    assert {row["kind"] for row in rows} == {
        "device",
        "evaluation",
        "fixture",
        "lexicon",
        "native_review",
    }
    for row in rows:
        assert set(row) == {"kind", "task", "digest", "model_tier"}
        assert re.fullmatch(r"[0-9a-f]{64}", row["digest"])
        assert (row["model_tier"] is not None) == (row["kind"] == "device")
    assert sorted(r["model_tier"] for r in rows if r["kind"] == "device") == [
        "nano",
        "small",
    ]


def test_evidence_collections_are_consumed_with_a_hard_bound():
    consumed = []

    def endless():
        while True:
            consumed.append(1)
            yield IdentifierRecall("DATE", 1, 0)

    with pytest.raises(ValueError):
        _evidence(identifier_recall=endless())
    assert len(consumed) <= 4097


@pytest.mark.parametrize(
    "overrides",
    [
        {"lexicon_digest": "raw note"},
        {"identifier_recall": _recalls() + (IdentifierRecall("DATE", 1, 0),)},
        {"terminology_covered": 300},
        {"offline_verified": "yes"},
        {"review_status": "Dr Synthetic Reviewer"},
        {"model_tiers": ("Raw Tier",)},
    ],
)
def test_invalid_evidence_is_rejected_without_echoing_values(overrides):
    with pytest.raises(ValueError) as exc:
        _evidence(**overrides)
    assert "raw note" not in str(exc.value)
    assert "Synthetic Reviewer" not in str(exc.value)


def test_wilson_lower_bound_is_conservative():
    assert wilson_lower_bound(0, 0) == 0.0
    assert wilson_lower_bound(100, 100) == pytest.approx(0.963, abs=1e-3)
    assert wilson_lower_bound(99, 100) < 0.95 < wilson_lower_bound(198, 200)
    with pytest.raises(ValueError):
        wilson_lower_bound(5, 4)
