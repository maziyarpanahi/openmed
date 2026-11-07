from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable, Mapping
from pathlib import Path

import pytest

from openmed.core.anonymizer import Anonymizer
from openmed.core.anonymizer.registry import LANGUAGE_PACK_GENERATORS
from openmed.core.language_pack import (
    LANGUAGE_PACK_REGISTRY,
    LanguagePack,
    LanguagePackRegistry,
)
from openmed.core.language_pack_catalog import NATIONAL_ID_ONLY_CAPABILITIES
from openmed.eval import release_gates
from openmed.eval import surrogate_quality_live as live
from openmed.eval.release_gates import (
    LIVE_SURROGATE_QUALITY_GATE,
    QUARANTINED,
    RELEASABLE,
    evaluate_live_surrogate_quality_gate,
)

Override = Callable[[str, int], str]

_ARABIC_GIVEN = ("سلمى", "نادر", "ريم", "باسل", "هالة", "كريم")
_ARABIC_FAMILY = ("حداد", "نجار", "عطار", "خياط", "صباغ", "دباغ")
_ARABIC_STREETS = ("شارع الزيتون", "شارع الياسمين", "طريق النخيل", "شارع الرمان")
_ARABIC_CITIES = ("الواحة", "الندى", "السنابل", "الربوة")
_LATIN_NAMES = ("Avery Morgan", "Jordan Blake", "Casey Quinn", "Riley Parker")


def _pick(values: tuple[str, ...], original: str, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}|{original}".encode("utf-8")).digest()
    return values[digest[0] % len(values)]


def _arabic_name(original: str, seed: int) -> str:
    given = _pick(_ARABIC_GIVEN, original, seed)
    family = _pick(_ARABIC_FAMILY, original, seed + 1)
    return f"{given} {family}"


def _arabic_street(original: str, seed: int) -> str:
    number = 10 + hashlib.sha256(original.encode("utf-8")).digest()[1] % 80
    return f"{_pick(_ARABIC_STREETS, original, seed)} {number}"


def _arabic_city(original: str, seed: int) -> str:
    return _pick(_ARABIC_CITIES, original, seed)


def _latin_name(original: str, seed: int) -> str:
    return _pick(_LATIN_NAMES, original, seed)


_ARABIC_TEXT: Mapping[str, Override] = {
    "PERSON": _arabic_name,
    "STREET_ADDRESS": _arabic_street,
    "CITY": _arabic_city,
}


class _OverrideSource:
    """Production anonymizer with selected source labels replaced."""

    def __init__(
        self,
        language: str,
        seed: int,
        overrides: Mapping[str, Override],
    ) -> None:
        self._base = Anonymizer(lang=language, consistent=True, seed=seed)
        self._seed = seed
        self._overrides = overrides

    def surrogate(self, original: str, label: str, *, locale: str | None = None) -> str:
        override = self._overrides.get(label)
        if override is not None:
            return override(original, self._seed)
        return self._base.surrogate(original, label, locale=locale)


def _factory(overrides: Mapping[str, Override]) -> live.LiveSurrogateFactory:
    def build(language: str, seed: int) -> _OverrideSource:
        return _OverrideSource(language, seed, overrides)

    return build


def _registry(
    *codes: str, extra: tuple[LanguagePack, ...] = ()
) -> LanguagePackRegistry:
    registry = LanguagePackRegistry()
    for code in codes:
        registry.register(LANGUAGE_PACK_REGISTRY.get(code))
    for pack in extra:
        registry.register(pack)
    return registry


def _evaluate(
    *codes: str,
    overrides: Mapping[str, Override] | None = None,
    **kwargs,
) -> live.LiveSurrogateQualityReport:
    return live.evaluate_live_surrogate_quality(
        registry=_registry(*codes),
        national_id_only={},
        surrogate_factory=_factory(overrides or {}),
        **kwargs,
    )


@pytest.fixture(scope="module")
def default_report() -> live.LiveSurrogateQualityReport:
    return live.evaluate_live_surrogate_quality()


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------


def test_live_targets_cover_every_registered_pack_and_national_id_only_language():
    targets = {target.language: target for target in live.live_surrogate_targets()}

    expected = set(LANGUAGE_PACK_REGISTRY.iter_codes()) | set(
        NATIONAL_ID_ONLY_CAPABILITIES
    )
    assert set(targets) == expected
    assert targets["fa"].expected_scripts == ("Arabic",)
    assert targets["fa"].script_source == live.SCRIPT_SOURCE_DECLARED
    assert targets["ja"].expected_scripts == ("Han", "Hiragana/Katakana")
    assert targets["id"].script_source == live.SCRIPT_SOURCE_FALLBACK
    assert targets["ha"].source == live.TARGET_NATIONAL_ID_ONLY
    assert targets["ha"].national_id_method == "nigeria_nin"
    assert targets["te"].approximation is not None


@pytest.mark.slow
def test_default_live_report_lists_every_pack_and_label(default_report):
    expected = set(LANGUAGE_PACK_REGISTRY.iter_codes()) | set(
        NATIONAL_ID_ONLY_CAPABILITIES
    )

    assert default_report.missing_languages == ()
    assert default_report.missing_labels == ()
    assert set(default_report.languages) == expected
    for language, row in default_report.matrix().items():
        assert tuple(row) == live.LIVE_SURROGATE_LABELS, language
        assert set(row.values()) <= set(live.CELL_STATUSES)
    for cell in default_report.cells():
        if cell.status == live.CELL_UNSCORED:
            assert cell.unscored_reason
        else:
            assert cell.samples == live.DEFAULT_LIVE_SURROGATE_SAMPLES
            assert set(cell.dimension_rates) == set(
                live.LIVE_SURROGATE_LABEL_DIMENSIONS[cell.label]
            )

    gate = evaluate_live_surrogate_quality_gate(default_report)
    assert gate.gate == LIVE_SURROGATE_QUALITY_GATE
    assert gate.passed is default_report.passed
    assert tuple(gate.details["failing_cells"]) == default_report.failing_cells
    assert tuple(gate.details["unscored_cells"]) == default_report.unscored_cells
    if not gate.passed:
        assert gate.reason != "ok"
        assert gate.details["failing_languages"]


def test_pack_registered_later_is_scored_without_code_changes():
    registry = _registry("fa")
    before = {
        target.language
        for target in live.live_surrogate_targets(
            registry=registry, national_id_only={}
        )
    }
    registry.register(
        LanguagePack(
            code="qa",
            scripts=("Arabic",),
            default_model="local/unit-model",
            segmenter_id="pysbd",
            recognizers=("builtin-patterns",),
            surrogate_locale="fa_IR",
            national_id_providers={"iran_national_id": "fa_IR"},
        )
    )

    report = live.evaluate_live_surrogate_quality(
        registry=registry,
        national_id_only={},
        surrogate_factory=_factory(_ARABIC_TEXT),
    )

    assert before == {"fa"}
    assert report.expected_languages == ("fa", "qa")
    assert report.locale_reports["qa"].target.expected_scripts == ("Arabic",)
    assert report.matrix()["qa"]["name"] == live.CELL_PASS


def test_pack_without_a_scorer_is_reported_unscored_and_blocks_the_gate():
    registry = LanguagePackRegistry()
    registry.register(
        LanguagePack(
            code="qq",
            scripts=("Unrouted",),
            default_model="local/unit-model",
            segmenter_id="pysbd",
            recognizers=("builtin-patterns",),
            surrogate_locale="en_US",
        )
    )

    report = live.evaluate_live_surrogate_quality(
        registry=registry,
        national_id_only={},
    )
    cells = report.locale_reports["qq"].cells
    gate = evaluate_live_surrogate_quality_gate(report)

    assert cells["name"].unscored_reason == "no_script_expectation"
    assert cells["address"].unscored_reason == "no_script_expectation"
    assert cells["city"].unscored_reason == "no_script_expectation"
    assert cells["id"].unscored_reason == "no_national_id_provider"
    assert cells["date"].unscored_reason == "no_locale_date_order"
    assert cells["phone"].status == live.CELL_PASS
    assert report.unscored_cells == (
        "qq/name",
        "qq/address",
        "qq/city",
        "qq/id",
        "qq/date",
    )
    assert gate.passed is False
    assert gate.reason == "live surrogate quality has unscored cells"
    assert gate.details["unscored_cells"]["qq/id"] == "no_national_id_provider"


def test_subset_report_cannot_pass_the_gate():
    report = live.evaluate_live_surrogate_quality(
        registry=_registry("fa", "en"),
        national_id_only={},
        languages=("fa",),
        surrogate_factory=_factory(_ARABIC_TEXT),
    )
    gate = evaluate_live_surrogate_quality_gate(report)

    assert report.missing_languages == ("en",)
    assert gate.passed is False
    assert gate.details["missing_languages"] == ["en"]


# ---------------------------------------------------------------------------
# Negative and positive controls
# ---------------------------------------------------------------------------


def test_latin_names_for_an_arabic_script_pack_fail_the_gate():
    passing = _evaluate("fa", overrides=_ARABIC_TEXT)
    failing = _evaluate("fa", overrides={**_ARABIC_TEXT, "PERSON": _latin_name})

    passing_gate = evaluate_live_surrogate_quality_gate(passing)
    failing_gate = evaluate_live_surrogate_quality_gate(failing)

    assert passing_gate.passed is True, passing.gate_details()["failing_cells"]
    assert failing_gate.passed is False
    assert failing.failing_cells == ("fa/name",)
    assert (
        failing_gate.reason == "live surrogate quality below per-cell pass-rate floor"
    )
    name_cell = failing_gate.details["failing_cells"]["fa/name"]
    assert name_cell["pass_rate"] == 0.0
    assert name_cell["reason_counts"]["script_ratio_below_floor"] == (
        live.DEFAULT_LIVE_SURROGATE_SAMPLES
    )


def test_live_run_reads_providers_registered_for_the_arabic_pack(monkeypatch):
    def latin_generator(faker, original, *, locale):
        return f"{faker.random.choice(('Avery', 'Jordan'))} Morgan"

    def arabic_generator(faker, original, *, locale):
        given = faker.random.choice(_ARABIC_GIVEN)
        return f"{given} {faker.random.choice(_ARABIC_FAMILY)}"

    key = ("ar", "Arabic", "PERSON")
    monkeypatch.setitem(LANGUAGE_PACK_GENERATORS, key, latin_generator)
    latin = live.evaluate_live_surrogate_quality(languages=("ar",), labels=("name",))
    monkeypatch.setitem(LANGUAGE_PACK_GENERATORS, key, arabic_generator)
    arabic = live.evaluate_live_surrogate_quality(languages=("ar",), labels=("name",))

    latin_cell = latin.locale_reports["ar"].cells["name"]
    arabic_cell = arabic.locale_reports["ar"].cells["name"]
    assert latin_cell.status == live.CELL_FAIL
    assert latin_cell.dimension_rates["script"] == 0.0
    assert arabic_cell.status == live.CELL_PASS
    assert arabic_cell.dimension_rates == {
        "script": 1.0,
        "consistency": 1.0,
        "naturalness": 1.0,
    }


@pytest.mark.parametrize(
    ("language", "label", "value", "dimension", "reason"),
    [
        (
            "sw",
            "STREET_ADDRESS",
            "8453 Shida Knolls Apt. 803",
            "format",
            "us_street_template",
        ),
        ("sw", "STREET_ADDRESS", "Barabara 12 Suite 4", "format", "us_secondary_unit"),
        ("sw", "CITY", "Nakiview", "format", "us_city_template"),
        ("sw", "CITY", "Lake Hodari", "format", "us_city_template"),
        ("fa", "PHONE", "(415) 555-0134", "format", "us_phone_shape"),
        ("fa", "PHONE", "+1-415-555-0134", "format", "us_phone_shape"),
        ("fa", "PHONE", "021 555 0134 x42", "format", "us_phone_shape"),
        ("fa", "national_id", "0000000001", "checksum", "national_id_checksum_failed"),
        ("de", "DATE_OF_BIRTH", "1987-04-23", "format", "locale_date_order_mismatch"),
        ("vi", "PERSON", "Bà Xuân Lê", "naturalness", "inserted_honorific"),
        ("hi", "PERSON", "हभऊडधऋएऐ", "naturalness", "implausible_letter_sequence"),
        ("hi", "PERSON", "हभऊडधऋएऐ", "naturalness", "name_shape_mismatch"),
        ("fa", "PERSON", "[PERSON]", "naturalness", "fallback_placeholder"),
    ],
)
def test_injected_defects_are_scored_on_the_matching_dimension(
    language, label, value, dimension, reason
):
    live_label = next(
        key
        for key, source in live.LIVE_SURROGATE_SOURCE_LABELS.items()
        if source == label
    )
    report = _evaluate(
        language,
        overrides={label: lambda original, seed: value},
        labels=(live_label,),
    )
    cell = report.locale_reports[language].cells[live_label]

    assert cell.status == live.CELL_FAIL
    assert cell.dimension_rates[dimension] == 0.0
    assert cell.reason_counts[reason] == live.DEFAULT_LIVE_SURROGATE_SAMPLES


def test_us_shapes_are_native_for_us_locales():
    report = _evaluate(
        "en",
        overrides={
            "STREET_ADDRESS": lambda original, seed: "8453 Isaiah Fields Apt. 366",
            "PHONE": lambda original, seed: "(415) 555-0134",
        },
        labels=("address", "phone"),
    )

    assert report.locale_reports["en"].cells["address"].dimension_rates["format"] == 1.0
    assert report.locale_reports["en"].cells["phone"].status == live.CELL_PASS


def test_unstable_and_identity_surrogates_fail_consistency():
    counter = iter(range(10_000))

    def unstable(original: str, seed: int) -> str:
        return f"{_ARABIC_GIVEN[next(counter) % 6]} {_ARABIC_FAMILY[0]}"

    unstable_report = _evaluate("fa", overrides={"PERSON": unstable}, labels=("name",))
    identity_report = _evaluate(
        "fa", overrides={"CITY": lambda original, seed: original}, labels=("city",)
    )

    assert unstable_report.locale_reports["fa"].cells["name"].reason_counts == {
        "nondeterministic": live.DEFAULT_LIVE_SURROGATE_SAMPLES
    }
    assert identity_report.locale_reports["fa"].cells["city"].reason_counts == {
        "surrogate_equals_original": live.DEFAULT_LIVE_SURROGATE_SAMPLES
    }


def test_provider_errors_are_scored_not_raised():
    def broken(original: str, seed: int) -> str:
        raise RuntimeError("provider failure")

    report = _evaluate("fa", overrides={"PERSON": broken}, labels=("name",))

    assert report.locale_reports["fa"].cells["name"].reason_counts == {
        "generation_error": live.DEFAULT_LIVE_SURROGATE_SAMPLES
    }


# ---------------------------------------------------------------------------
# Exemptions
# ---------------------------------------------------------------------------


def test_documented_approximation_exempts_a_cell_with_a_recorded_reason():
    reason = "Telugu generic fields are generated by the en_IN backend"
    report = _evaluate(
        "te",
        overrides={"STREET_ADDRESS": lambda original, seed: "H.No. 12, Gandhi Road"},
        labels=("address",),
        exemptions=[{"language": "te", "labels": ["address"], "reason": reason}],
    )
    cell = report.locale_reports["te"].cells["address"]

    assert cell.status == live.CELL_EXEMPT
    assert cell.scored_status == live.CELL_FAIL
    assert cell.exemption == {
        "reason": reason,
        "approximation": LANGUAGE_PACK_REGISTRY.get(
            "te"
        ).surrogate_locale_approximation,
    }
    assert report.exempt_cells == ("te/address",)
    assert report.failing_cells == ()
    details = report.gate_details()
    assert details["exempt_cells"]["te/address"]["exemption"]["reason"] == reason


def test_exemption_without_a_documented_approximation_is_rejected():
    with pytest.raises(ValueError, match="documented surrogate-locale approximation"):
        _evaluate(
            "fa",
            exemptions=[{"language": "fa", "labels": ["name"], "reason": "unit"}],
        )


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"language": "te", "labels": ["address"], "reason": " "}, "recorded reason"),
        ({"language": "te", "labels": ["street"], "reason": "unit"}, "unknown"),
        ({"language": "te", "labels": [], "reason": "unit"}, "at least one label"),
        ({"language": "", "labels": ["address"], "reason": "unit"}, "language"),
    ],
)
def test_exemption_declarations_are_validated(payload, message):
    with pytest.raises(ValueError, match=message):
        live.LiveSurrogateExemption.from_mapping(payload)


def test_duplicate_exemptions_are_rejected():
    exemption = {"language": "te", "labels": ["address"], "reason": "unit"}
    with pytest.raises(ValueError, match="duplicate"):
        _evaluate("te", exemptions=[exemption, exemption], labels=("address",))


# ---------------------------------------------------------------------------
# Privacy and determinism
# ---------------------------------------------------------------------------


def test_report_and_gate_hold_digests_not_values(caplog):
    seen: list[str] = []

    class Recording:
        def __init__(self, language: str, seed: int) -> None:
            self._base = live.default_live_surrogate_factory(language, seed)

        def surrogate(
            self, original: str, label: str, *, locale: str | None = None
        ) -> str:
            value = self._base.surrogate(original, label, locale=locale)
            seen.extend((original, value))
            return value

    caplog.set_level(logging.DEBUG)
    report = live.evaluate_live_surrogate_quality(
        registry=_registry("en", "hi", "zh", "ar"),
        national_id_only={"ha": NATIONAL_ID_ONLY_CAPABILITIES["ha"]},
        surrogate_factory=Recording,
    )
    gate = evaluate_live_surrogate_quality_gate(report)
    payload = json.dumps(report.to_dict(), ensure_ascii=False) + json.dumps(
        gate.to_dict(), ensure_ascii=False
    )

    assert seen
    for value in {item.strip() for item in seen if len(item.strip()) >= 4}:
        assert value not in payload
        assert value not in caplog.text
    for cell in report.cells():
        assert all(len(digest) == 16 for digest in cell.value_digests)
        int("".join(cell.value_digests) or "0", 16)


def test_live_run_leaves_one_time_locale_warnings_untouched(monkeypatch):
    from openmed.core.anonymizer import locales

    monkeypatch.setattr(locales, "_warned", set())
    live.evaluate_live_surrogate_quality(languages=("te",), labels=("address",))

    assert locales._warned == set()
    with pytest.warns(UserWarning, match="'te'"):
        Anonymizer(lang="te").surrogate("12 Gandhi Road", "STREET_ADDRESS")


def test_live_report_is_deterministic():
    first = _evaluate("fa", "de", overrides=_ARABIC_TEXT).to_dict()
    second = _evaluate("fa", "de", overrides=_ARABIC_TEXT).to_dict()

    assert first == second
    assert first["report_digest"].startswith("sha256:")


def test_invalid_arguments_are_rejected():
    with pytest.raises(ValueError, match="samples"):
        _evaluate("fa", samples=0)
    with pytest.raises(ValueError, match="min_pass_rate"):
        _evaluate("fa", min_pass_rate=1.5)
    with pytest.raises(ValueError, match="labels"):
        _evaluate("fa", labels=("street",))
    with pytest.raises(ValueError, match="not registered"):
        _evaluate("fa", languages=("de",))
    with pytest.raises(TypeError):
        evaluate_live_surrogate_quality_gate({"records": []})


# ---------------------------------------------------------------------------
# Gate configuration, release wiring and CLI
# ---------------------------------------------------------------------------


@pytest.fixture()
def failing_report() -> live.LiveSurrogateQualityReport:
    return _evaluate("fa", overrides={**_ARABIC_TEXT, "PERSON": _latin_name})


@pytest.fixture()
def passing_report() -> live.LiveSurrogateQualityReport:
    return _evaluate("fa", overrides=_ARABIC_TEXT)


def test_gate_min_pass_rate_regrades_a_report():
    def half_latin(original: str, seed: int) -> str:
        if hashlib.sha256(original.encode("utf-8")).digest()[0] % 2:
            return _latin_name(original, seed)
        return _arabic_name(original, seed)

    report = _evaluate("fa", overrides={**_ARABIC_TEXT, "PERSON": half_latin})
    rate = report.locale_reports["fa"].cells["name"].pass_rate

    assert 0.0 < rate < live.DEFAULT_LIVE_SURROGATE_PASS_RATE
    assert evaluate_live_surrogate_quality_gate(report).passed is False
    assert (
        evaluate_live_surrogate_quality_gate(report, min_pass_rate=rate).passed is True
    )


def test_release_check_is_not_applicable_without_evidence():
    check = release_gates._live_surrogate_quality_release_check({}, {})

    assert check.gate == LIVE_SURROGATE_QUALITY_GATE
    assert check.passed is True
    assert check.reason == "not applicable"


def test_release_check_requires_evidence_when_required():
    check = release_gates._live_surrogate_quality_release_check(
        {}, {"surrogate_quality_live_required": True}
    )

    assert check.passed is False
    assert check.reason == "live surrogate-quality evidence is required"


def test_release_check_rejects_unknown_evidence_keys():
    check = release_gates._live_surrogate_quality_release_check(
        {"surrogate_quality_live": {"seed": 7}}, {}
    )

    assert check.passed is False
    assert check.reason == "live surrogate-quality evidence is malformed"
    assert check.details["unknown_keys"] == ["seed"]


def test_release_check_rejects_undocumented_exemptions():
    check = release_gates._live_surrogate_quality_release_check(
        {
            "surrogate_quality_live": {
                "exemptions": [{"language": "fa", "labels": ["name"], "reason": "unit"}]
            }
        },
        {},
    )

    assert check.passed is False
    assert check.reason == "live surrogate-quality evidence is invalid"


def test_release_check_cannot_loosen_floor_or_samples(monkeypatch, passing_report):
    captured: dict[str, object] = {}

    def fake_evaluate(**kwargs):
        captured.update(kwargs)
        return passing_report

    monkeypatch.setattr(live, "evaluate_live_surrogate_quality", fake_evaluate)
    check = release_gates._live_surrogate_quality_release_check(
        {"surrogate_quality_live": {"min_pass_rate": 0.5, "samples": 2}}, {}
    )

    assert check.passed is True
    assert captured["min_pass_rate"] == live.DEFAULT_LIVE_SURROGATE_PASS_RATE
    assert captured["samples"] == live.DEFAULT_LIVE_SURROGATE_SAMPLES
    assert captured["seed"] == live.DEFAULT_LIVE_SURROGATE_SEED
    assert check.details["requested_min_pass_rate"] == 0.5


def test_release_gate_quarantines_failing_live_evidence(
    monkeypatch, tmp_path: Path, failing_report
):
    monkeypatch.setattr(
        live, "evaluate_live_surrogate_quality", lambda **kwargs: failing_report
    )
    candidate = {
        "metadata": {
            "repo_id": "OpenMed/unit-model",
            "family": "PII",
            "tier": "Tiny",
            "param_count": 44_000_000,
            "format": "mlx-fp",
            "eval_set_hash": "sha256:eval",
            "leakage_fixture_hash": "sha256:leakage",
            "surrogate_quality_live_required": True,
        },
        "metrics": {"surrogate_quality_live": True},
    }

    report = release_gates.preview(
        candidate,
        baseline_path=tmp_path / "missing-baseline.json",
    )
    check = next(
        item for item in report.gate_results if item.gate == LIVE_SURROGATE_QUALITY_GATE
    )

    assert report.decision == QUARANTINED
    assert check.passed is False
    assert check.details["required"] is True
    assert list(check.details["failing_cells"]) == ["fa/name"]


@pytest.mark.parametrize(
    ("fixture_name", "exit_code", "decision"),
    [("failing_report", 1, QUARANTINED), ("passing_report", 0, RELEASABLE)],
)
def test_cli_writes_hash_only_live_gate_payload(
    monkeypatch, tmp_path: Path, request, capsys, fixture_name, exit_code, decision
):
    quality_report = request.getfixturevalue(fixture_name)
    monkeypatch.setattr(
        live, "evaluate_live_surrogate_quality", lambda **kwargs: quality_report
    )
    output = tmp_path / "live-surrogate-gate.json"

    code = release_gates.main(["--live-surrogate-quality", "--output", str(output)])
    payload = json.loads(output.read_text(encoding="utf-8"))
    summary = json.loads(capsys.readouterr().out)

    assert code == exit_code
    assert payload["decision"] == decision
    assert payload["gate_result"]["gate"] == LIVE_SURROGATE_QUALITY_GATE
    assert (
        payload["report"]["report_digest"] == quality_report.to_dict()["report_digest"]
    )
    assert summary["failing_cells"] == list(quality_report.failing_cells)
