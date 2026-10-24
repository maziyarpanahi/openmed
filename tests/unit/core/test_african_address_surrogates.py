"""Local-convention address and town surrogates for African language packs."""

from __future__ import annotations

import hashlib
import re
import string
import unicodedata
import warnings
from pathlib import Path

import pytest
from faker.providers.address.en_US import Provider as USAddressProvider
from hypothesis import given, settings
from hypothesis import strategies as st

from openmed.core import labels as L
from openmed.core.anonymizer import Anonymizer
from openmed.core.anonymizer.locales import (
    CONCEPTUAL_LOCALE_LANGUAGES,
    locale_coherence_report,
)
from openmed.core.anonymizer.providers.african_places import (
    ADDRESS_CONVENTIONS,
    AFRICAN_ADDRESS_APPROXIMATIONS,
    AFRICAN_ADDRESS_LOCALES,
    AfricanPlaceDispatcher,
    _distinctive_tokens,
    african_address_convention,
    generate_african_location,
    generate_african_street_address,
)
from openmed.core.anonymizer.registry import (
    LABEL_GENERATORS,
    _gen_india_street_address,
    _gen_location,
)
from openmed.core.pii import _build_deidentification_result
from openmed.processing.outputs import EntityPrediction, PredictionResult

ROOT = Path(__file__).resolve().parents[3]
SEED = 3891

# (language, explicit locale) pairs that must use the curated conventions.
AFRICAN_CASES = (
    ("sw", None),
    ("sw", "sw_TZ"),
    ("ha", None),
    ("ig", None),
    ("yo", None),
    ("rw", None),
)

_US_STREET_SUFFIXES = frozenset(
    suffix.casefold() for suffix in USAddressProvider.street_suffixes
)
_US_CITY_PREFIXES = frozenset(
    prefix.casefold() for prefix in USAddressProvider.city_prefixes
)
_US_CITY_SUFFIXES = tuple(USAddressProvider.city_suffixes)
_US_UNIT_DESIGNATORS = frozenset(
    fmt.split()[0].rstrip(".").casefold()
    for fmt in USAddressProvider.secondary_address_formats
) | {"unit"}

_ADDRESS_SOURCES = (
    *((f"{n} Synthetic Example Road", "STREET_ADDRESS") for n in range(1, 41)),
    *((f"Plot {n}, Sample Area", "ADDRESS") for n in range(1, 21)),
    *((f"P.O. Box {n}", "STREET_ADDRESS") for n in range(100, 120)),
)
_LOCATION_SOURCES = (
    *((f"Sampletown {n}", "LOCATION") for n in range(1, 41)),
    *((f"Exampleville {n}", "CITY") for n in range(1, 21)),
)
_SAMPLE = _ADDRESS_SOURCES + _LOCATION_SOURCES

_BOX_SHAPE_RE = re.compile(
    r"^(?:S\.L\.P\.|Sanduku la Posta|P\.M\.B\.|Akwatin Gidan Waya|"
    r"Àpótí Ìfìwéránṣẹ́|P\.O\. Box|B\.P\.|Agasanduku k'iposita) \d{1,4}"
)


def _fold(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    return "".join(
        character for character in decomposed if not unicodedata.combining(character)
    ).casefold()


def _letter_tokens(value: str) -> set[str]:
    return set(re.findall(r"[^\W\d_]+", _fold(value)))


def _us_template_markers(value: str) -> set[str]:
    tokens = _letter_tokens(value)
    markers = tokens & (_US_STREET_SUFFIXES | _US_CITY_PREFIXES | _US_UNIT_DESIGNATORS)
    markers |= {
        token
        for token in tokens
        if len(token) > 4 and token.endswith(_US_CITY_SUFFIXES)
    }
    if re.search(r"\d{5,}", value):
        markers.add("five-digit-house-number")
    return markers


def _surrogates(lang: str, locale: str | None, *, seed: int = SEED) -> list[str]:
    anonymizer = Anonymizer(lang=lang, locale=locale, consistent=True, seed=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return [anonymizer.surrogate(source, label) for source, label in _SAMPLE]


def _digest(values: list[str]) -> str:
    return hashlib.sha256("\x1f".join(values).encode("utf-8")).hexdigest()[:16]


@pytest.mark.parametrize(("lang", "locale"), AFRICAN_CASES)
def test_african_address_and_city_surrogates_have_no_us_template_markers(
    lang: str,
    locale: str | None,
) -> None:
    surrogates = _surrogates(lang, locale)

    flagged = {value: _us_template_markers(value) for value in surrogates}
    assert {value: found for value, found in flagged.items() if found} == {}
    # The sample is large enough to exercise several templates and towns.
    address_count = len(_ADDRESS_SOURCES)
    assert len(set(surrogates[:address_count])) >= 20
    assert len(set(surrogates[address_count:])) >= 4


@pytest.mark.parametrize("locale", sorted(AFRICAN_ADDRESS_LOCALES))
def test_marker_check_flags_the_generic_faker_fallback(locale: str) -> None:
    """Negative control: the bypassed fallback still emits US templates."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        faker = Anonymizer(lang="en", locale=locale)._get_faker(locale)
    flagged = 0
    for index in range(20):
        faker.seed_instance(SEED + index)
        street = LABEL_GENERATORS[L.STREET_ADDRESS].fallback(
            faker,
            f"{index} Synthetic Example Road",
            locale=locale,
        )
        town = LABEL_GENERATORS[L.LOCATION].fallback(
            faker,
            f"Sampletown {index}",
            locale=locale,
        )
        flagged += bool(_us_template_markers(street))
        flagged += bool(_us_template_markers(town))

    assert flagged >= 20


@pytest.mark.parametrize(("lang", "locale"), AFRICAN_CASES)
def test_african_address_surrogates_are_deterministic_per_seed(
    lang: str,
    locale: str | None,
) -> None:
    first = _surrogates(lang, locale)

    assert _surrogates(lang, locale) == first
    assert _surrogates(lang, locale, seed=SEED + 1) != first


@pytest.mark.parametrize("lang", sorted(ADDRESS_CONVENTIONS))
def test_each_language_draws_from_its_own_convention(lang: str) -> None:
    convention = ADDRESS_CONVENTIONS[lang]
    other_names = {
        name
        for other, other_convention in ADDRESS_CONVENTIONS.items()
        if other != lang
        for name in (*other_convention.areas, *other_convention.towns)
    }
    anonymizer = Anonymizer(lang=lang, consistent=True, seed=SEED)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        addresses = [
            anonymizer.surrogate(f"{n} Synthetic Example Road", "STREET_ADDRESS")
            for n in range(1, 21)
        ]
        towns = [anonymizer.surrogate(f"Sampletown {n}", "CITY") for n in range(20)]

    own_names = (*convention.areas, *convention.towns)
    assert all(any(name in value for name in own_names) for value in addresses)
    assert not any(name in value for value in addresses for name in other_names)
    assert set(towns) <= set(convention.towns)


@pytest.mark.parametrize(("lang", "locale"), AFRICAN_CASES)
def test_postal_box_sources_keep_a_box_shape(lang: str, locale: str | None) -> None:
    anonymizer = Anonymizer(lang=lang, locale=locale, consistent=True, seed=SEED)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        boxes = [
            anonymizer.surrogate(source, "STREET_ADDRESS")
            for source in (
                "P.O. Box 400",
                "S.L.P. 7781 Sampletown",
                "P.M.B. 1024",
                "B.P. 330 Sampletown",
            )
        ]
        streets = [
            anonymizer.surrogate(f"{n} Synthetic Example Road", "STREET_ADDRESS")
            for n in range(1, 21)
        ]

    assert all(_BOX_SHAPE_RE.match(value) for value in boxes), boxes
    assert not any(_BOX_SHAPE_RE.match(value) for value in streets), streets


@st.composite
def _curated_source(draw):
    """Build a worst-case source from the curated vocabulary itself."""

    language = draw(st.sampled_from(sorted(ADDRESS_CONVENTIONS)))
    convention = ADDRESS_CONVENTIONS[language]
    template = draw(
        st.sampled_from(convention.street_templates + convention.box_templates)
    )
    area, cell, sector = draw(
        st.lists(
            st.sampled_from(convention.areas),
            min_size=3,
            max_size=3,
            unique=True,
        )
    )
    slots = {
        "street": (
            draw(st.sampled_from(convention.street_names))
            if convention.street_names
            else ""
        ),
        "area": area,
        "cell": cell,
        "sector": sector,
        "town": draw(st.sampled_from(convention.towns)),
        "number": draw(st.integers(min_value=1, max_value=199)),
        "box": draw(st.integers(min_value=1, max_value=9999)),
    }
    fields = {name for _, name, _, _ in string.Formatter().parse(template) if name}
    used_names = [str(slots[name]) for name in fields if name not in {"number", "box"}]
    return language, template.format(**slots), used_names


@settings(max_examples=150, deadline=None)
@given(_curated_source())
def test_curated_sources_are_never_reproduced(case) -> None:
    language, source, used_names = case
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        anonymizer = Anonymizer(lang=language, consistent=True, seed=SEED)
        surrogate = anonymizer.surrogate(source, "STREET_ADDRESS")
        town = anonymizer.surrogate(source, "LOCATION")

    assert " ".join(_fold(surrogate).split()) != " ".join(_fold(source).split())
    for name in used_names:
        assert _distinctive_tokens(name).isdisjoint(_letter_tokens(surrogate))
        assert _distinctive_tokens(name).isdisjoint(_letter_tokens(town))
    assert set(re.findall(r"\d+", surrogate)).isdisjoint(re.findall(r"\d+", source))


@pytest.mark.parametrize("lang", sorted(ADDRESS_CONVENTIONS))
def test_location_surrogate_never_repeats_a_curated_town(lang: str) -> None:
    anonymizer = Anonymizer(lang=lang, consistent=True, seed=SEED)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for town in ADDRESS_CONVENTIONS[lang].towns:
            surrogate = anonymizer.surrogate(town, "CITY")
            assert surrogate != town
            assert surrogate in ADDRESS_CONVENTIONS[lang].towns


@pytest.mark.parametrize("lang", sorted(ADDRESS_CONVENTIONS))
def test_replace_pipeline_swaps_address_spans_without_us_templates(lang: str) -> None:
    street = "17 Synthetic Example Road"
    city = "Sampletown"
    prefix = "Synthetic patient lives at "
    text = f"{prefix}{street} in {city}."
    entities = [
        EntityPrediction(
            text=value,
            label=label,
            confidence=1.0,
            start=text.index(value),
            end=text.index(value) + len(value),
        )
        for value, label in ((street, "STREET_ADDRESS"), (city, "CITY"))
    ]
    model_result = PredictionResult(
        text=text,
        entities=entities,
        model_name="synthetic-address-fixture",
        timestamp="2026-10-24T00:00:00Z",
        metadata={},
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = _build_deidentification_result(
            text,
            model_result,
            effective_method="replace",
            keep_year=False,
            date_shift_days=None,
            keep_mapping=False,
            lang=lang,
            consistent=True,
            seed=SEED,
            locale=None,
            use_safety_sweep=False,
        )

    output = result.deidentified_text
    assert output.startswith(prefix)
    assert street not in output
    assert city not in output
    assert not _us_template_markers(output.removeprefix(prefix))
    convention = ADDRESS_CONVENTIONS[lang]
    assert any(town in output for town in convention.towns)


def test_curated_values_are_nfc_and_templates_fill_every_slot() -> None:
    slots = {
        "street": "S",
        "area": "A",
        "cell": "C",
        "sector": "R",
        "town": "T",
        "number": 1,
        "box": 2,
    }
    for convention in ADDRESS_CONVENTIONS.values():
        values = (
            *convention.street_names,
            *convention.areas,
            *convention.towns,
            *convention.street_templates,
            *convention.box_templates,
        )
        assert all(unicodedata.normalize("NFC", value) == value for value in values)
        assert len(set(convention.areas)) >= 6
        assert len(set(convention.towns)) >= 6
        for template in convention.street_templates + convention.box_templates:
            filled = template.format(**slots)
            assert "{" not in filled and "}" not in filled
        for value in (*convention.street_names, *convention.areas, *convention.towns):
            assert _distinctive_tokens(value), value
            assert not _us_template_markers(value), value


def test_locale_table_covers_every_curated_language() -> None:
    assert set(AFRICAN_ADDRESS_LOCALES.values()) == set(ADDRESS_CONVENTIONS)
    assert african_address_convention("ha-NG") is ADDRESS_CONVENTIONS["ha"]
    assert african_address_convention("SW_tz") is ADDRESS_CONVENTIONS["sw"]
    for locale in ("en_KE", "en_NG", "en_TZ", "zu_ZA", "xh_ZA", "am_ET", "fr_SN"):
        assert african_address_convention(locale) is None


def test_curated_generators_reject_non_african_locales() -> None:
    faker = Anonymizer(lang="en")._get_faker("en_US")
    with pytest.raises(ValueError, match="no African address convention"):
        generate_african_street_address(faker, "1 Example Road", locale="en_US")
    with pytest.raises(ValueError, match="no African address convention"):
        generate_african_location(faker, "Sampletown", locale="en_US")


def test_dispatchers_wrap_the_previous_generators() -> None:
    street = LABEL_GENERATORS[L.STREET_ADDRESS]
    location = LABEL_GENERATORS[L.LOCATION]

    assert isinstance(street, AfricanPlaceDispatcher)
    assert isinstance(location, AfricanPlaceDispatcher)
    assert street.fallback is _gen_india_street_address
    assert location.fallback is _gen_location


@pytest.mark.parametrize(
    "locale",
    ("it_IT", "en_US", "en_KE", "en_NG", "zu_ZA", "xh_ZA", "am_ET", "fr_SN"),
)
@pytest.mark.parametrize("label", (L.STREET_ADDRESS, L.LOCATION))
def test_non_african_locales_reach_the_fallback_with_identical_faker_state(
    locale: str,
    label: str,
) -> None:
    dispatcher = LABEL_GENERATORS[label]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        faker = Anonymizer(lang="en", locale=locale)._get_faker(locale)
    for source in ("12 Example Road", "Sampletown", "P.O. Box 400"):
        faker.seed_instance(SEED)
        routed = dispatcher(faker, source, locale=locale)
        faker.seed_instance(SEED)
        direct = dispatcher.fallback(faker, source, locale=locale)
        assert routed == direct


# Digests of five address/location surrogates per locale, captured on master
# before the African providers existed. ``it_IT`` is omitted because Faker's
# ``it_IT`` city draw depends on PYTHONHASHSEED; the fallback-state test above
# covers it in-process instead.
_NON_AFRICAN_DIGEST_PROBES = (
    ("12 Example Road", "STREET_ADDRESS"),
    ("Plot 7 Sample Lane", "ADDRESS"),
    ("P.O. Box 400", "STREET_ADDRESS"),
    ("Sampletown", "LOCATION"),
    ("Exampleville", "CITY"),
)
_NON_AFRICAN_BASELINE_DIGESTS = {
    ("af", "af_ZA"): "f077fddd0a83a936",
    ("am", "am_ET"): "a2d8ba6b183d6f83",
    ("ar", "ar_EG"): "20ea8d556e095b83",
    ("as", "as_IN"): "e62b09c18104b56f",
    ("bg", "bg_BG"): "56e8a6317c9b164b",
    ("bn", "bn_BD"): "63402e8eb357dce9",
    ("cs", "cs_CZ"): "0bd0de210c5b0156",
    ("da", "da_DK"): "b4c1976383893ece",
    ("de", "de_DE"): "31704c3e0ab57624",
    ("el", "el_GR"): "b34d386e661c1e9f",
    ("en", "en_ET"): "20ea8d556e095b83",
    ("en", "en_GH"): "812850a9f193d3e2",
    ("en", "en_KE"): "a2d8ba6b183d6f83",
    ("en", "en_NG"): "7fe42e649b6c4f64",
    ("en", "en_TZ"): "20ea8d556e095b83",
    ("en", "en_UG"): "20ea8d556e095b83",
    ("en", "en_US"): "20ea8d556e095b83",
    ("en", "en_ZA"): "042ca27915fe26c4",
    ("es", "es_ES"): "72a7210e1fb7a0be",
    ("et", "et_EE"): "12a5c29875b7e377",
    ("fa", "fa_IR"): "501736820c8c2635",
    ("fi", "fi_FI"): "daf5552b85332bc6",
    ("fr", "fr_CI"): "80041d9efa180239",
    ("fr", "fr_CM"): "435d5f4892bd12b8",
    ("fr", "fr_FR"): "6cef9709b760a078",
    ("fr", "fr_SN"): "89e7b561e345e5ad",
    ("gu", "gu_IN"): "ff39314aefadaaa8",
    ("he", "he_IL"): "6bdab1f295d7d4dc",
    ("hi", "hi_IN"): "7c6d76d28871f02b",
    ("hr", "hr_HR"): "178cf7cf4aec549d",
    ("hu", "hu_HU"): "583a0057e4becd21",
    ("id", "id_ID"): "0e7190e7e37c5fad",
    ("ja", "ja_JP"): "74e7c3c2fc5ef1e6",
    ("kn", "kn_IN"): "f6b3c2265d480911",
    ("ko", "ko_KR"): "beff56371ec15d43",
    ("lv", "lv_LV"): "26d91741f6ca0452",
    ("ml", "ml_IN"): "f6b3c2265d480911",
    ("mr", "mr_IN"): "7c6d76d28871f02b",
    ("ms", "ms_MY"): "0e7190e7e37c5fad",
    ("ne", "ne_NP"): "23eeb1c8469a8e89",
    ("nl", "nl_NL"): "f077fddd0a83a936",
    ("no", "no_NO"): "816e0c0fa55d1aa9",
    ("or", "or_IN"): "ff0dc16b247b39a7",
    ("pa", "pa_IN"): "f6b3c2265d480911",
    ("pl", "pl_PL"): "3eb8d7341fa9e649",
    ("pt", "pt_AO"): "738294acbc0b4b07",
    ("pt", "pt_MZ"): "a5f1916761f21818",
    ("pt", "pt_PT"): "c1eae9333d1c892d",
    ("ro", "ro_RO"): "9819cec4daf60fc8",
    ("ru", "ru_RU"): "6fedb05e4767092c",
    ("sk", "sk_SK"): "9f1dd2f9dcc64070",
    ("sr", "sr_RS"): "178cf7cf4aec549d",
    ("sv", "sv_SE"): "1d5a0c7411ea968d",
    ("ta", "ta_IN"): "6f48b722318e001c",
    ("te", "en_IN"): "f6b3c2265d480911",
    ("th", "th_TH"): "a640243056e8649d",
    ("tl", "fil_PH"): "b2cbca748037b5b0",
    ("tr", "tr_TR"): "8f5f752cd17818f1",
    ("uk", "uk_UA"): "eddab24de0b9af02",
    ("ur", "ur_IN"): "f6b3c2265d480911",
    ("vi", "vi_VN"): "2fb54276e38704a6",
    ("xh", "xh_ZA"): "042ca27915fe26c4",
    ("zh", "zh_CN"): "108dd425a2c1985c",
    ("zu", "zu_ZA"): "042ca27915fe26c4",
}


@pytest.mark.parametrize(
    ("lang", "locale"),
    sorted(_NON_AFRICAN_BASELINE_DIGESTS),
)
def test_non_african_address_outputs_match_pre_change_digests(
    lang: str,
    locale: str,
) -> None:
    anonymizer = Anonymizer(lang=lang, locale=locale, consistent=True, seed=SEED)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        values = [
            anonymizer.surrogate(source, label)
            for source, label in _NON_AFRICAN_DIGEST_PROBES
        ]

    assert _digest(values) == _NON_AFRICAN_BASELINE_DIGESTS[(lang, locale)]


def test_africa_guide_address_example_matches_runtime_output() -> None:
    guide = (ROOT / "docs" / "africa-onboarding.md").read_text(encoding="utf-8")
    anonymizer = Anonymizer(lang="sw", consistent=True, seed=42)

    for source, label in (
        ("12 Synthetic Example Road", "STREET_ADDRESS"),
        ("P.O. Box 400", "STREET_ADDRESS"),
        ("Sampletown", "CITY"),
    ):
        assert f"# {anonymizer.surrogate(source, label)}\n" in guide


def test_locale_coherence_report_records_address_approximations() -> None:
    rows = locale_coherence_report()
    default_rows = {
        row["language"]: row
        for row in rows
        if row["locale"] not in CONCEPTUAL_LOCALE_LANGUAGES
    }

    documented = {
        lang for lang, row in default_rows.items() if row["address_approximation"]
    }
    assert documented == {"sw", "ha", "ig", "yo", "rw", "xh"}
    for lang in ADDRESS_CONVENTIONS:
        assert (
            default_rows[lang]["address_approximation"]
            == (AFRICAN_ADDRESS_APPROXIMATIONS[lang])
        )
        assert "synthetic" in default_rows[lang]["address_approximation"]
    assert "zu_ZA" in default_rows["xh"]["address_approximation"]
    assert default_rows["zu"]["address_approximation"] is None
    assert default_rows["en"]["address_approximation"] is None
    assert all(
        row["address_approximation"] is None
        for row in rows
        if row["locale"] in CONCEPTUAL_LOCALE_LANGUAGES
    )
