"""Tests for shaping generated PERSON surrogates (#3887).

All name samples are synthetic: common given and family names composed into
fictional full names, not real individuals. Faker runs offline.
"""

from __future__ import annotations

import pytest
from faker import Faker
from faker.providers.person.hi_IN import Provider as HindiProvider
from faker.providers.person.id_ID import Provider as IndonesianProvider
from faker.providers.person.th_TH import Provider as ThaiProvider
from faker.providers.person.vi_VN import Provider as VietnameseProvider

from openmed.core.anonymizer.locales import URDU_FAMILY_NAMES, URDU_GIVEN_NAMES
from openmed.core.anonymizer.person_surrogates import (
    AMHARIC_FEMININE_GIVEN_NAMES,
    AMHARIC_MASCULINE_GIVEN_NAMES,
    URDU_FEMININE_GIVEN_NAMES,
    URDU_MASCULINE_GIVEN_NAMES,
    shape_person_surrogate,
    strip_inserted_honorifics,
)
from openmed.core.script_detect import detect_script


def _faker(locale: str, seed: int = 3887) -> Faker:
    faker = Faker(locale)
    faker.seed_instance(seed)
    return faker


class TestStripInsertedHonorifics:
    @pytest.mark.parametrize(
        ("locale", "lang", "generated", "expected"),
        [
            ("de_DE", "de", "Univ.Prof. Dorothea Gotthard", "Dorothea Gotthard"),
            ("de_DE", "de", "Uwe Hänel B.Sc.", "Uwe Hänel"),
            ("de_DE", "de", "Prof. Artur Hein MBA.", "Artur Hein"),
            ("hu_HU", "hu", "Prof. Dr. Szabó Mária", "Szabó Mária"),
            ("tr_TR", "tr", "Arş. Gör. Atalay Bilgin", "Atalay Bilgin"),
            ("vi_VN", "vi", "Quý cô Duyên Dương", "Duyên Dương"),
            ("id_ID", "id", "R.A. Fathonah Iswahyudi, S.I.Kom", "Fathonah Iswahyudi"),
            ("id_ID", "id", "Fitriani Wijaya, M.Farm", "Fitriani Wijaya"),
            ("sw", "sw", "Bi. Nigesa Owuor", "Nigesa Owuor"),
            ("yo_NG", "yo", "Kabiyesi Oluwamuyiwa Bakare", "Oluwamuyiwa Bakare"),
            ("bg_BG", "bg", "Г-н Авел Муев", "Авел Муев"),
            ("fa_IR", "fa", "جناب آقای دکتر رضا حسینی", "رضا حسینی"),
            ("or_IN", "or", "ଶ୍ରୀମତୀ ଶିବାନୀ ରଞ୍ଜନ", "ଶିବାନୀ ରଞ୍ଜନ"),
            ("en_US", "en", "Dr. Jane Roe", "Jane Roe"),
            ("en_US", "en", "John Roe Jr.", "John Roe"),
        ],
    )
    def test_faker_titles_are_removed(self, locale, lang, generated, expected):
        stripped = strip_inserted_honorifics(
            generated,
            reference="Synthetic Source",
            lang=lang,
            faker=_faker(locale),
        )

        assert stripped == expected

    def test_title_glued_to_a_known_thai_given_name(self):
        given = ThaiProvider.first_names_male[0]
        generated = f"นาย{given} ใจดี"

        stripped = strip_inserted_honorifics(
            generated,
            reference="สมหญิง ใจดี",
            lang="th",
            faker=_faker("th_TH"),
        )

        assert stripped == f"{given} ใจดี"

    def test_suffix_glued_to_a_known_hindi_surname(self):
        given = HindiProvider.first_names_female[0]
        family = HindiProvider.last_names[0]

        stripped = strip_inserted_honorifics(
            f"{given} {family}जी",
            reference="अनिता शर्मा",
            lang="hi",
            faker=_faker("hi_IN"),
        )

        assert stripped == f"{given} {family}"

    @pytest.mark.parametrize(
        ("locale", "lang", "generated"),
        [
            # Surname particles in Faker's generic prefix lists are not titles.
            ("fr_FR", "fr", "Louis du Guérin"),
            ("es_ES", "es", "Fidel Sandoval de Tejada"),
            # Hungarian married-name forms and initials are part of the name.
            ("hu_HU", "hu", "Kissné Németh Sára"),
            ("hu_HU", "hu", "T. Horváth Zsolt"),
            # Thai ``ณ <place>`` suffixes are noble surname particles.
            ("th_TH", "th", "สมชาย ณ ระนอง"),
            # A lone token is never emptied.
            ("en_US", "en", "Dr."),
        ],
    )
    def test_name_parts_are_kept(self, locale, lang, generated):
        stripped = strip_inserted_honorifics(
            generated,
            reference="Synthetic Source",
            lang=lang,
            faker=_faker(locale),
        )

        assert stripped is generated

    def test_title_also_present_in_the_source_is_kept(self):
        generated = "Dr. Jane Roe"

        stripped = strip_inserted_honorifics(
            generated,
            reference="Dr. John Doe",
            lang="en",
            faker=_faker("en_US"),
        )

        assert stripped is generated

    def test_registered_honorifics_are_removed_without_faker(self):
        assert (
            strip_inserted_honorifics(
                "Mama Amina Hassan",
                reference="Baraka Mushi",
                lang="sw",
            )
            == "Amina Hassan"
        )

    def test_unregistered_language_without_faker_is_unchanged(self):
        generated = "Dr. Jane Roe"

        assert (
            strip_inserted_honorifics(generated, reference="Jane Doe", lang="en")
            is generated
        )


class TestFamilyFirstOrder:
    @pytest.mark.parametrize(
        ("generated", "expected"),
        [
            ("Lan Nguyễn", "Nguyễn Lan"),
            ("Khoa Xuân Trần", "Trần Xuân Khoa"),
            ("Chị Ánh Trần", "Trần Ánh"),
            # Already family-first: unchanged.
            ("Nguyễn Văn Nam", "Nguyễn Văn Nam"),
        ],
    )
    def test_vietnamese_names_lead_with_the_family_name(self, generated, expected):
        shaped = shape_person_surrogate(
            generated,
            source_name="Lê Thu Hà",
            lang="vi",
            gender=None,
            faker=_faker("vi_VN"),
        )

        assert shaped == expected

    def test_given_first_languages_are_not_reordered(self):
        generated = "Amina Hassan"

        shaped = shape_person_surrogate(
            generated,
            source_name="Baraka Mushi",
            lang="sw",
            gender=None,
            faker=_faker("sw"),
        )

        assert shaped is generated


class TestGenderCue:
    def test_given_name_of_the_other_gender_is_replaced(self):
        male_given = ThaiProvider.first_names_male[0]
        female_names = set(ThaiProvider.first_names_female)

        shaped = shape_person_surrogate(
            f"{male_given} ใจดี",
            source_name="สมหญิง ใจดี",
            lang="th",
            gender="female",
            faker=_faker("th_TH"),
        )

        given, family = shaped.split()
        assert given in female_names
        assert family == "ใจดี"

    def test_given_name_of_the_cued_gender_is_kept(self):
        generated = f"{ThaiProvider.first_names_female[0]} ใจดี"

        shaped = shape_person_surrogate(
            generated,
            source_name="สมหญิง ใจดี",
            lang="th",
            gender="female",
            faker=_faker("th_TH"),
        )

        assert shaped is generated

    def test_replacement_never_reuses_a_source_token(self):
        female_names = tuple(VietnameseProvider.first_names_female)
        source = f"Lê Thị {female_names[0]}"

        for seed in range(40):
            shaped = shape_person_surrogate(
                "Nam Trần",
                source_name=source,
                lang="vi",
                gender="female",
                faker=_faker("vi_VN", seed),
            )
            family, given = shaped.split()
            assert family == "Trần"
            assert given in female_names
            assert given not in source.split()

    def test_replacement_is_deterministic_for_a_seeded_faker(self):
        def shape() -> str:
            return shape_person_surrogate(
                "Nam Trần",
                source_name="Lê Thu Hà",
                lang="vi",
                gender="female",
                faker=_faker("vi_VN", 11),
            )

        assert shape() == shape()

    def test_vietnamese_middle_marker_follows_the_cue(self):
        shaped = shape_person_surrogate(
            f"Trần Văn {VietnameseProvider.first_names_female[0]}",
            source_name="Lê Thị Hà",
            lang="vi",
            gender="female",
            faker=_faker("vi_VN"),
        )

        assert shaped.split()[1] == "Thị"

    def test_indonesian_family_name_follows_the_cue(self):
        male_given = IndonesianProvider.first_names_male[0]
        female_family = IndonesianProvider.last_names_female[0]

        shaped = shape_person_surrogate(
            f"{male_given} {female_family}",
            source_name="Agus Pratama",
            lang="id",
            gender="male",
            faker=_faker("id_ID"),
        )

        given, family = shaped.split()
        assert given == male_given
        assert family in IndonesianProvider.last_names_male

    def test_urdu_given_name_uses_the_bundled_gender_split(self):
        shaped = shape_person_surrogate(
            "عارف خان",
            source_name="فاطمہ قریشی",
            lang="ur",
            gender="female",
            faker=_faker("en_IN"),
            preserve_unknown_gender=True,
        )

        given, family = shaped.split()
        assert given in URDU_FEMININE_GIVEN_NAMES
        assert given != "فاطمہ"
        assert family == "خان"

    def test_urdu_feminine_family_token_is_replaced_for_a_male_cue(self):
        shaped = shape_person_surrogate(
            "سلمان بیگم",
            source_name="عارف قریشی",
            lang="ur",
            gender="male",
            faker=_faker("en_IN"),
            preserve_unknown_gender=True,
        )

        given, family = shaped.split()
        assert given == "سلمان"
        assert family in URDU_FAMILY_NAMES
        assert family not in {"بیگم", "قریشی"}

    def test_amharic_given_name_uses_the_bundled_gender_split(self):
        shaped = shape_person_surrogate(
            "ዳዊት ከበደ",
            source_name="ሜሮን አለሙ",
            lang="am",
            gender="female",
            faker=_faker("en_KE"),
            preserve_unknown_gender=True,
        )

        given, family = shaped.split()
        assert given in AMHARIC_FEMININE_GIVEN_NAMES
        assert given != "ሜሮን"
        assert family == "ከበደ"

    def test_unknown_synthetic_name_is_kept_for_script_providers(self):
        generated = "कखग घचछ"

        shaped = shape_person_surrogate(
            generated,
            source_name="अनिता शर्मा",
            lang="hi",
            gender="female",
            faker=_faker("hi_IN"),
            preserve_unknown_gender=True,
        )

        assert shaped is generated

    def test_unknown_given_name_is_replaced_for_faker_providers(self):
        shaped = shape_person_surrogate(
            "Zyxw Hassan",
            source_name="Amina Juma",
            lang="sw",
            gender="male",
            faker=_faker("sw"),
        )

        given, family = shaped.split()
        assert given != "Zyxw"
        assert detect_script(given) == "Latin"
        assert family == "Hassan"

    def test_no_in_script_inventory_leaves_the_name(self):
        # No gendered inventory exists in the generated token's script, so no
        # cross-script replacement is made and the name is left as generated.
        generated = "ሰላም ተስፋዬ"

        shaped = shape_person_surrogate(
            generated,
            source_name="ሜሮን አለሙ",
            lang="sw",
            gender="female",
            faker=_faker("sw"),
        )

        assert shaped is generated

    def test_faker_stub_without_person_provider(self):
        class _Random:
            @staticmethod
            def choice(values):
                return values[0]

        class _Stub:
            random = _Random()

        shaped = shape_person_surrogate(
            "Bi. Amina Hassan",
            source_name="Baraka Mushi",
            lang="sw",
            gender="male",
            faker=_Stub(),
        )

        assert shaped == "Amina Hassan"


@pytest.mark.parametrize("lang", ["ja", "ko_KR", "zh-TW"])
def test_cjk_surrogates_are_left_to_name_order(lang):
    generated = "Dr. Jane Roe"

    shaped = shape_person_surrogate(
        generated,
        source_name="Synthetic Source",
        lang=lang,
        gender="male",
        faker=_faker("en_US"),
    )

    assert shaped is generated


class TestBundledInventories:
    def test_urdu_gender_split_covers_the_bundled_given_names(self):
        assert set(URDU_FEMININE_GIVEN_NAMES) | set(URDU_MASCULINE_GIVEN_NAMES) == set(
            URDU_GIVEN_NAMES
        )
        assert not set(URDU_FEMININE_GIVEN_NAMES) & set(URDU_MASCULINE_GIVEN_NAMES)

    def test_amharic_inventories_are_disjoint_and_ethiopic(self):
        assert not set(AMHARIC_FEMININE_GIVEN_NAMES) & set(
            AMHARIC_MASCULINE_GIVEN_NAMES
        )
        for name in (*AMHARIC_FEMININE_GIVEN_NAMES, *AMHARIC_MASCULINE_GIVEN_NAMES):
            assert detect_script(name) == "Ethiopic"
