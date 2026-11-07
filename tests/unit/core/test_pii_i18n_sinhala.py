"""Sinhala user-supplied-model route with Sri Lanka NIC detection.

Sinhala ships no bundled weights. These checks pin the route itself (script
routing, catalog projections, no bundled-model claim) and the deterministic
safety net that runs with an empty model detector: ``+94`` phone numbers,
context-gated National Identity Card (NIC) numbers whose embedded year and
day-of-year fields are validated, numeric dates, street addresses and postal
codes. They also pin the ``si_LK`` surrogate locale and the structural NIC
surrogate provider. Every value is synthetic.
"""

from __future__ import annotations

import json
import random
import re
import warnings
from pathlib import Path

import pytest
from faker import Faker

import openmed
from openmed.core.anonymizer import Anonymizer
from openmed.core.anonymizer import locales as anonymizer_locales
from openmed.core.anonymizer.providers.clinical_ids import (
    SriLankaNICProvider,
    generate_sri_lanka_nic,
    register_clinical_providers,
)
from openmed.core.anonymizer.providers.registry_ids import get_national_id
from openmed.core.errors import ModelLoadError
from openmed.core.language_pack import (
    LanguagePack,
    LanguagePackRegistry,
    get_language_pack,
)
from openmed.core.language_pack_catalog import (
    DEFAULT_PII_MODELS as BUILTIN_DEFAULT_PII_MODELS,
)
from openmed.core.language_pack_catalog import (
    LANG_TO_LOCALE,
    LANGUAGE_PACK_ADAPTERS,
    NATIONAL_ID_PROVIDERS,
    USER_SUPPLIED_MODEL_ROUTE_PACKS,
    LanguagePackAdapters,
)
from openmed.core.language_pack_coherence import pack_coherence_report
from openmed.core.language_router import LanguageRouter
from openmed.core.locale_formats import LOCALE_DATE_ORDER
from openmed.core.model_registry import get_default_pii_model
from openmed.core.pii import deidentify
from openmed.core.pii_i18n import (
    DEFAULT_PII_MODELS,
    INDIAN_MULTI_ID_PII_PATTERNS,
    LANGUAGE_NAMES,
    LANGUAGE_PII_PATTERNS,
    LOCALE_FAKE_DATA,
    SUPPORTED_LANGUAGES,
    USER_SUPPLIED_MODEL_LANGUAGES,
    USER_SUPPLIED_PII_MODEL,
    get_patterns_for_language,
    validate_aadhaar,
    validate_sri_lanka_nic,
)
from openmed.core.pipeline import Pipeline
from openmed.core.safety_sweep import safety_sweep
from openmed.core.script_detect import (
    candidate_languages_for_script,
    detect_script,
)
from openmed.processing.outputs import PredictionResult

FIXTURE_PATH = (
    Path(__file__).resolve().parents[3] / "openmed/eval/golden/fixtures/i18n/si.jsonl"
)
_MODEL = "unit-test-model"
_FIXTURE_LABEL_TO_ENTITY_TYPE = {
    "DATE": "date",
    "PHONE": "phone_number",
    "ID_NUM": "national_id",
    "STREET_ADDRESS": "street_address",
    "ZIPCODE": "postcode",
}
_SINHALA_NOTE = "රෝගියා ස්ථාවර තත්ත්වයේ සිටී."
_TAMIL_SRI_LANKAN_NOTE = "நோயாளி கொழும்பில் உள்ளார். தொலைபேசி +94 77 123 4567."
_NIC_CONTEXT = "ජාතික හැඳුනුම්පත් අංකය"
_SINHALA_SCRIPT = re.compile(r"[\u0D80-\u0DFF]")


def _empty_prediction(text, **kwargs):
    return PredictionResult(
        text=text, entities=[], model_name="synthetic", timestamp="2026-01-01"
    )


def _redacted_outputs(text: str, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    monkeypatch.setattr(openmed, "analyze_text", _empty_prediction)
    direct = deidentify(text, lang="si", model_name=_MODEL).deidentified_text
    staged = (
        Pipeline(lang="si", model_name=_MODEL, model_detector=_empty_prediction)
        .run(text)
        .redacted_text
    )
    automatic = (
        Pipeline(lang="auto", model_name=_MODEL, model_detector=_empty_prediction)
        .run(text)
        .redacted_text
    )
    return [direct, staged, automatic]


def _fixture_rows() -> list[dict]:
    return [
        json.loads(line)
        for line in FIXTURE_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sweep(text: str, lang: str = "si") -> list[tuple[str, str]]:
    return [(entity.label, entity.text) for entity in safety_sweep(text, [], lang=lang)]


# ---------------------------------------------------------------------------
# Route registration: user-supplied model, no bundled-model claim
# ---------------------------------------------------------------------------


def test_sinhala_is_a_user_supplied_model_route_without_bundled_weights() -> None:
    assert "si" in USER_SUPPLIED_MODEL_LANGUAGES
    assert "si" not in SUPPORTED_LANGUAGES
    assert "si" not in BUILTIN_DEFAULT_PII_MODELS
    assert DEFAULT_PII_MODELS["si"] == USER_SUPPLIED_PII_MODEL
    assert get_default_pii_model("si") is None
    # The route pack is never registered, so it cannot claim a default model.
    assert get_language_pack("si") is None


def test_sinhala_route_pack_feeds_locale_provider_and_script_hints() -> None:
    (pack,) = (p for p in USER_SUPPLIED_MODEL_ROUTE_PACKS if p.code == "si")

    assert {p.code for p in USER_SUPPLIED_MODEL_ROUTE_PACKS} <= (
        USER_SUPPLIED_MODEL_LANGUAGES
    )
    assert pack.scripts == ("Sinhala",)
    assert pack.default_model == USER_SUPPLIED_PII_MODEL
    assert pack.surrogate_locale_approximation
    assert LANGUAGE_NAMES["si"] == "Sinhala"
    assert LANG_TO_LOCALE["si"] == "si_LK"
    assert NATIONAL_ID_PROVIDERS["si"] == ("si_LK", "sri_lanka_nic")
    assert LOCALE_DATE_ORDER["si"] == "dmy"
    assert candidate_languages_for_script("Sinhala") == ("si",)
    assert pack in LANGUAGE_PACK_ADAPTERS.routing_packs()


def test_sinhala_route_pack_passes_the_language_pack_coherence_check() -> None:
    registry = LanguagePackRegistry()
    for pack in USER_SUPPLIED_MODEL_ROUTE_PACKS:
        registry.register(pack)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        (row,) = pack_coherence_report(registry=registry)

    assert row["language"] == "si"
    assert row["coherent"] is True, row


def test_a_registered_pack_replaces_the_user_supplied_route() -> None:
    registry = LanguagePackRegistry()
    adapters = LanguagePackAdapters(
        registry,
        user_supplied_model_languages=("si",),
        user_supplied_route_packs=USER_SUPPLIED_MODEL_ROUTE_PACKS,
    )

    assert adapters.supported_languages == set()
    assert adapters.default_pii_models == {}
    assert adapters.lang_to_locale == {"si": "si_LK"}
    assert adapters.national_id_providers == {"si": ("si_LK", "sri_lanka_nic")}
    assert adapters.script_language_hints == {"Sinhala": ("si",)}
    assert adapters.routing_packs() == USER_SUPPLIED_MODEL_ROUTE_PACKS

    bundled = LanguagePack(
        code="si",
        scripts=("Sinhala",),
        default_model="OpenMed/synthetic-sinhala-pii",
        segmenter_id="unicode-sentence",
        recognizers=("builtin-patterns", "model"),
        surrogate_locale="en_US",
    )
    registry.register(bundled)

    assert adapters.routing_packs() == (bundled,)
    assert adapters.default_pii_models == {"si": "OpenMed/synthetic-sinhala-pii"}
    assert adapters.lang_to_locale == {"si": "en_US"}
    assert adapters.national_id_providers == {}


# ---------------------------------------------------------------------------
# Script routing
# ---------------------------------------------------------------------------


def test_sinhala_text_routes_to_si_and_not_to_english() -> None:
    router = LanguageRouter(use_optional_lid=False)
    decision = router.route(_SINHALA_NOTE)

    assert detect_script(_SINHALA_NOTE) == "Sinhala"
    assert decision.language == "si"
    assert decision.dominant_script == "Sinhala"
    assert [(run.script, run.language, run.reason) for run in decision.runs] == [
        ("Sinhala", "si", None)
    ]
    assert decision.runs[0].source == "stdlib:script"


def test_auto_pipeline_routes_sinhala_with_a_user_supplied_model() -> None:
    route = Pipeline(lang="auto", model_name=_MODEL).stage2_language_script(
        _SINHALA_NOTE
    )

    assert route.lang == "si"
    assert route.script == "Sinhala"
    assert route.model_name == _MODEL
    # The placeholder is reported as "no default model", as for lang="si".
    assert route.metadata["available_default_model"] is None
    assert (
        Pipeline(lang="si", model_name=_MODEL)
        .stage2_language_script(_SINHALA_NOTE)
        .metadata["available_default_model"]
        is None
    )


def test_auto_pipeline_requires_a_model_for_sinhala_without_echoing_text() -> None:
    with pytest.raises(ModelLoadError, match="pass an explicit model_name") as exc:
        Pipeline(lang="auto").stage2_language_script(_SINHALA_NOTE)

    message = str(exc.value)
    assert "'si'" in message
    assert _SINHALA_NOTE not in message
    assert not _SINHALA_SCRIPT.search(message)


def test_tamil_script_sri_lankan_text_still_routes_to_tamil() -> None:
    router = LanguageRouter(use_optional_lid=False)

    assert router.route(_TAMIL_SRI_LANKAN_NOTE).language == "ta"
    assert (
        Pipeline(lang="auto", model_name=_MODEL)
        .stage2_language_script(_TAMIL_SRI_LANKAN_NOTE)
        .lang
        == "ta"
    )
    # The Sinhala pack stays off the Tamil route.
    sinhala = LANGUAGE_PII_PATTERNS["si"]
    tamil = get_patterns_for_language("ta")
    assert not any(any(p is q for q in sinhala) for p in tamil)


def test_mixed_sinhala_and_english_keeps_exact_run_languages() -> None:
    text = "Patient stable. රෝගියා ස්ථාවරයි."
    runs = LanguageRouter(use_optional_lid=False).route_runs(text)

    assert [(run.script, run.language) for run in runs] == [
        ("Latin", "en"),
        ("Sinhala", "si"),
    ]
    assert "".join(text[run.start : run.end] for run in runs) == text


def test_explicit_registry_only_router_still_rejects_sinhala() -> None:
    router = LanguageRouter(
        packs=tuple(LANGUAGE_PACK_ADAPTERS.registry.iter_packs()),
        use_optional_lid=False,
    )

    assert router.route(_SINHALA_NOTE).runs[0].reason == "unsupported_script"


# ---------------------------------------------------------------------------
# NIC validator: embedded year and day-of-year rules
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        "853400937V",
        "853400937X",
        "853400937v",
        "850010937V",  # day 001
        "853660937V",  # day 366
        "855010937V",  # day 501, the +500 range
        "858660937X",  # day 866
        "198534000937",
        "190000100001",  # birth year 1900, day 001
        "200286600017",  # day 866
        " 198534000937 ",
    ],
)
def test_structurally_valid_nics_are_accepted(value: str) -> None:
    assert validate_sri_lanka_nic(value)


@pytest.mark.parametrize(
    "value",
    [
        "850000937V",  # day 000
        "853670937V",  # day 367
        "855000937V",  # day 500
        "858670937V",  # day 867
        "859990937X",  # day 999
        "853400937Z",  # unknown letter
        "85340093V",  # eight digits
        "8534009370V",  # ten digits
        "189934000937",  # birth year before 1900
        "198500000937",  # day 000
        "198536700937",  # day 367
        "198550000937",  # day 500
        "198586700937",  # day 867
        "19853400093",  # eleven digits
        "1985340009377",  # thirteen digits
        "1985-340-00937",
        "",
    ],
)
def test_nics_with_out_of_range_fields_or_shapes_are_rejected(value: str) -> None:
    assert not validate_sri_lanka_nic(value)


def test_nic_birth_year_bound_uses_an_injectable_reference_year() -> None:
    assert validate_sri_lanka_nic("203034000937", reference_year=2030)
    assert not validate_sri_lanka_nic("203034000937", reference_year=2029)
    # The nine-digit layout has a two-digit year, so it has no future bound.
    assert validate_sri_lanka_nic("993400937V", reference_year=1950)


def test_nic_validator_rejects_non_strings() -> None:
    assert not validate_sri_lanka_nic(None)  # type: ignore[arg-type]
    assert not validate_sri_lanka_nic(853400937)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Deterministic patterns with an empty model detector
# ---------------------------------------------------------------------------


def test_sinhala_route_has_a_dedicated_pattern_list() -> None:
    sinhala = LANGUAGE_PII_PATTERNS["si"]
    combined = get_patterns_for_language("si")

    assert {pattern.entity_type for pattern in sinhala} == {
        "phone_number",
        "national_id",
        "street_address",
        "postcode",
    }
    assert all(any(p is q for q in combined) for p in sinhala)
    # Sri Lanka is not India: the India-only multi-ID pack stays off the route.
    assert not any(any(p is q for q in INDIAN_MULTI_ID_PII_PATTERNS) for p in sinhala)
    nic_patterns = [p for p in sinhala if p.entity_type == "national_id"]
    assert len(nic_patterns) == 2
    assert all(p.validator is validate_sri_lanka_nic for p in nic_patterns)
    assert all(p.requires_context for p in nic_patterns)
    assert all(p.reject_on_validation_failure for p in nic_patterns)


@pytest.mark.parametrize(
    ("text", "value", "label"),
    [
        ("ජංගම දුරකථන +94 71 234 5678 ය.", "+94 71 234 5678", "phone_number"),
        ("දුරකථන +94-77-123-4567 ය.", "+94-77-123-4567", "phone_number"),
        ("දුරකථන +94771234567 ය.", "+94771234567", "phone_number"),
        ("දුරකථන +94 (0) 77 123 4567 ය.", "+94 (0) 77 123 4567", "phone_number"),
        ("දුරකථන 0094 81 223 4567 ය.", "0094 81 223 4567", "phone_number"),
        ("ජංගම 077 345 6789 ය.", "077 345 6789", "phone_number"),
        ("ජංගම 077-3456789 ය.", "077-3456789", "phone_number"),
        ("ස්ථාවර දුරකථන 011 245 6789 ය.", "011 245 6789", "phone_number"),
        (f"{_NIC_CONTEXT} 853400937V ය.", "853400937V", "national_id"),
        (f"{_NIC_CONTEXT} 855010937x ය.", "855010937x", "national_id"),
        (f"{_NIC_CONTEXT} 198534000937 ය.", "198534000937", "national_id"),
        ("ජා.හැ.අ. 200286600017 ය.", "200286600017", "national_id"),
        (
            "රෝගියාගේ ජාතික හැඳුනුම්පත (NIC) 853400937V ය.",
            "853400937V",
            "national_id",
        ),
        ("උපන් දිනය 14/03/1985 ය.", "14/03/1985", "date"),
        ("ඇතුළත් කළ දිනය 2024-11-02 ය.", "2024-11-02", "date"),
        ("දිනය 05.08.2025 ය.", "05.08.2025", "date"),
        ("ලිපිනය 45, ගාලු පාර, කොළඹ.", "45, ගාලු පාර", "street_address"),
        ("ලිපිනය 27/3, මල් වීදිය, මහනුවර.", "27/3, මල් වීදිය", "street_address"),
        (
            "ලිපිනය අංක 12/A, ශ්\u200dරී ධර්මපාල මාවත, ගාල්ල.",
            "12/A, ශ්\u200dරී ධර්මපාල මාවත",
            "street_address",
        ),
        ("තැපැල් කේතය 00300 ය.", "00300", "postcode"),
    ],
)
def test_sinhala_values_are_swept_whole_and_masked_with_an_empty_detector(
    monkeypatch: pytest.MonkeyPatch,
    text: str,
    value: str,
    label: str,
) -> None:
    start = text.index(value)
    entities = safety_sweep(text, [], lang="si")

    assert any(
        (e.start, e.end, e.text, e.label) == (start, start + len(value), value, label)
        for e in entities
    ), [(e.label, e.text) for e in entities]
    # No competing rule keeps a fragment of the value under another label.
    assert not any(
        e.start < start + len(value) and e.end > start and e.label != label
        for e in entities
    )
    for output in _redacted_outputs(text, monkeypatch):
        assert value not in output
        assert output.startswith(text[:start])
        if label == "phone_number" and "94" in value:
            # The whole number is masked, not only the digits after +94.
            assert "+94" not in output and "0094" not in output


def test_english_nic_context_works_on_an_explicit_sinhala_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = "Patient NIC: 853400937V, N.I.C 198534000937."

    assert _sweep(text) == [
        ("national_id", "853400937V"),
        ("national_id", "198534000937"),
    ]
    monkeypatch.setattr(openmed, "analyze_text", _empty_prediction)
    output = deidentify(text, lang="si", model_name=_MODEL).deidentified_text
    assert "853400937V" not in output and "198534000937" not in output


@pytest.mark.parametrize(
    "text",
    [
        # Out-of-range day fields are rejected even right after NIC context.
        f"{_NIC_CONTEXT} 853670937V ය.",
        f"{_NIC_CONTEXT} 850000937V ය.",
        f"{_NIC_CONTEXT} 855000937V ය.",
        f"{_NIC_CONTEXT} 858670937V ය.",
        f"{_NIC_CONTEXT} 198536700937 ය.",
        f"{_NIC_CONTEXT} 198550000937 ය.",
        # Birth years outside 1900 and the current year are rejected.
        f"{_NIC_CONTEXT} 189934000937 ය.",
        f"{_NIC_CONTEXT} 299934000937 ය.",
    ],
)
def test_nics_with_out_of_range_embedded_fields_are_not_identifiers(
    text: str,
) -> None:
    assert not any(label == "national_id" for label, _ in _sweep(text))


@pytest.mark.parametrize(
    "text",
    [
        # Valid NIC shapes without NIC context: a clinic or lab number.
        "සායන අංකය 853400937V ය.",
        "රසායනාගාර අංකය 198534000937 ය.",
        # "nic" inside "clinic" is not NIC context.
        "Clinic No 853400937V, clinic number 198534000937.",
        # Lab values and doses are not identifiers or phones.
        "හිමොග්ලොබින් 12.5 g/dL, පට්ටිකා 250000 /µL.",
        "ඖෂධ 500 mg දිනකට 2 වරක්.",
        "රුධිර පීඩනය 120/80, නාඩිය 72.",
        # A landline shape without phone context.
        "වාට්ටු අංකය 0112345678 ය.",
        # A road and a five-digit count without address or postal context.
        "කාමර 45, ගාලු පාර දෙසට.",
        "සාම්පලය 80000 ක්.",
    ],
)
def test_numbers_without_sinhala_identifier_context_are_not_flagged(
    text: str,
) -> None:
    assert _sweep(text) == []


# ---------------------------------------------------------------------------
# NIC surrogates and the si_LK surrogate locale
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "source",
    ["853400937V", "855010937x", "198534000937", None, "not-a-nic"],
)
def test_nic_surrogates_pass_the_structural_validator(source: str | None) -> None:
    for seed in range(200):
        surrogate = generate_sri_lanka_nic(source, rng=random.Random(seed))
        assert validate_sri_lanka_nic(surrogate), (seed, surrogate)
        assert surrogate != source
        if source is not None and validate_sri_lanka_nic(source):
            # Only the layout and the letter case of the source survive.
            assert len(surrogate) == len(source)
            assert surrogate[-1].islower() == source[-1].islower()
        else:
            assert re.fullmatch(r"[0-9]{12}", surrogate)


def test_nic_surrogates_draw_both_day_ranges_independently_of_the_source() -> None:
    days = {
        int(generate_sri_lanka_nic("853400937V", rng=random.Random(seed))[2:5])
        for seed in range(200)
    }

    assert any(day <= 366 for day in days)
    assert any(day >= 501 for day in days)


def test_corrupted_nic_surrogates_fail_the_validator_negative_control() -> None:
    for seed in range(50):
        old = generate_sri_lanka_nic("853400937V", rng=random.Random(seed))
        new = generate_sri_lanka_nic("198534000937", rng=random.Random(seed))

        assert not validate_sri_lanka_nic(old[:2] + "367" + old[5:])
        assert not validate_sri_lanka_nic(old[:2] + "500" + old[5:])
        assert not validate_sri_lanka_nic(old[:-1] + "Q")
        assert not validate_sri_lanka_nic("1899" + new[4:])
        assert not validate_sri_lanka_nic(new[:4] + "867" + new[7:])


def test_nic_provider_is_registered_for_the_sinhala_route() -> None:
    for alias in ("lk", "si", "si_LK"):
        spec = get_national_id(alias, "sri_lanka_nic")
        assert spec is not None
        assert spec.validate is validate_sri_lanka_nic
        assert spec.faker_provider is SriLankaNICProvider

    faker = Faker("en_IN")
    register_clinical_providers(faker)
    faker.seed_instance(3890)
    assert validate_sri_lanka_nic(faker.sri_lanka_nic())
    assert validate_sri_lanka_nic(faker.sri_lanka_nic("853400937V"))


@pytest.mark.parametrize("source", ["853400937V", "853400937v", "198534000937"])
def test_anonymizer_replaces_nics_with_validated_surrogates(source: str) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        anonymizer = Anonymizer(lang="si", consistent=True, seed=3890)
        first = anonymizer.surrogate(source, "national_id")
        again = anonymizer.surrogate(source, "national_id")

    assert first == again
    assert first != source
    assert len(first) == len(source)
    assert validate_sri_lanka_nic(first)


def test_twelve_digit_nic_that_also_passes_aadhaar_keeps_the_nic_layout() -> None:
    # Birth year 2001, day 123: some serials also satisfy the Aadhaar check.
    source = next(
        candidate
        for serial in range(100_000)
        if validate_aadhaar(candidate := f"2001123{serial:05d}")
    )
    assert validate_sri_lanka_nic(source)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        surrogate = Anonymizer(lang="si", consistent=True, seed=7).surrogate(
            source, "national_id"
        )

    assert validate_sri_lanka_nic(surrogate)
    assert surrogate != source


def test_si_lk_is_a_documented_approximation_with_synthetic_sinhala_names() -> None:
    assert "si" in anonymizer_locales._APPROXIMATE_LOCALES
    assert anonymizer_locales.FAKER_BACKEND_LOCALE["si_LK"] == "en_IN"
    curated = LOCALE_FAKE_DATA["si_LK"]

    anonymizer_locales._warned.discard("si")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        anonymizer = Anonymizer(lang="si", consistent=True, seed=3890)
        person = anonymizer.surrogate("නිමල් පෙරේරා", "PERSON")
        first = anonymizer.surrogate("කසුන්", "FIRST_NAME")
        last = anonymizer.surrogate("ජයසූරිය", "LAST_NAME")
        place = anonymizer.surrogate("කොළඹ", "LOCATION")
        street = anonymizer.surrogate("45, ගාලු පාර", "STREET_ADDRESS")

    approximation_warnings = [
        w for w in caught if issubclass(w.category, UserWarning) and "'si'" in str(w)
    ]
    assert len(approximation_warnings) == 1
    assert person in curated["NAME"] and person != "නිමල් පෙරේරා"
    assert first in curated["FIRST_NAME"] and first != "කසුන්"
    assert last in curated["LAST_NAME"] and last != "ජයසූරිය"
    assert place in curated["LOCATION"] and place != "කොළඹ"
    assert street in curated["STREET_ADDRESS"] and street != "45, ගාලු පාර"
    for value in (person, first, last, place):
        assert _SINHALA_SCRIPT.search(value)
        assert not re.search(r"[A-Za-z]", value)
    # Curated street addresses are themselves recognized on the Sinhala route.
    for address in curated["STREET_ADDRESS"]:
        text = f"ලිපිනය {address}, කොළඹ."
        assert ("street_address", address) in _sweep(text)


# ---------------------------------------------------------------------------
# Synthetic golden fixture
# ---------------------------------------------------------------------------


def test_sinhala_fixture_spans_are_swept_exactly() -> None:
    rows = _fixture_rows()

    assert 3 <= len(rows) <= 5
    for row in rows:
        assert row["language"] == "si"
        assert row["metadata"]["synthetic"] is True
        text = row["text"]
        swept = {(e.label, e.start, e.end) for e in safety_sweep(text, [], lang="si")}
        for span in row["gold_spans"]:
            assert text[span["start"] : span["end"]] == span["text"]
            key = (
                _FIXTURE_LABEL_TO_ENTITY_TYPE[span["label"]],
                span["start"],
                span["end"],
            )
            assert key in swept, (row["id"], span["label"])


def test_sinhala_fixture_nics_cover_both_layouts_and_pass_the_validator() -> None:
    nics = [
        span["text"]
        for row in _fixture_rows()
        for span in row["gold_spans"]
        if span["label"] == "ID_NUM"
    ]

    assert all(validate_sri_lanka_nic(value) for value in nics)
    assert any(len(value) == 10 for value in nics)
    assert any(len(value) == 12 for value in nics)


def test_sinhala_fixture_is_leak_free_with_an_empty_detector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels: set[str] = set()
    for row in _fixture_rows():
        text = row["text"]
        outputs = _redacted_outputs(text, monkeypatch)
        for span in row["gold_spans"]:
            labels.add(span["label"])
            for output in outputs:
                assert span["text"] not in output, (row["id"], span["label"])

    assert {"ID_NUM", "PHONE", "DATE"} <= labels
