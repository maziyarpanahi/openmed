"""Tests for the non-CJK honorific and name-order registry (#3887).

All name samples are synthetic: common given and family names composed into
fictional full names, not real individuals.
"""

from __future__ import annotations

import pytest

from openmed.core.name_order import CJK_LANGUAGES
from openmed.core.person_honorifics import (
    PERSON_NAME_CONVENTIONS,
    Honorific,
    fold_name_token,
    is_family_first,
    name_convention_for,
    peel_person_honorifics,
    register_person_honorific,
)

LISTED_LANGUAGES = ("th", "am", "sw", "ar", "fa", "ur", "hi", "vi", "id", "yo", "ha")


@pytest.fixture(autouse=True)
def _restore_conventions():
    """Isolate mutations to the module-global convention registry."""

    snapshot = dict(PERSON_NAME_CONVENTIONS)
    try:
        yield
    finally:
        PERSON_NAME_CONVENTIONS.clear()
        PERSON_NAME_CONVENTIONS.update(snapshot)


class TestRegistry:
    def test_every_listed_language_has_gendered_honorifics(self):
        for lang in LISTED_LANGUAGES:
            convention = name_convention_for(lang)
            assert convention is not None, lang
            genders = {honorific.gender for honorific in convention.honorifics}
            assert {"female", "male"} <= genders, lang

    def test_cjk_languages_stay_with_name_order(self):
        assert not set(PERSON_NAME_CONVENTIONS) & CJK_LANGUAGES

    def test_only_vietnamese_is_family_first(self):
        assert is_family_first("vi")
        assert [lang for lang in LISTED_LANGUAGES if is_family_first(lang)] == ["vi"]
        assert not is_family_first("en")

    def test_region_qualified_code_uses_primary_language(self):
        assert name_convention_for("ar-DZ") is name_convention_for("ar")
        assert name_convention_for("sw_TZ") is name_convention_for("sw")

    def test_unknown_language_has_no_convention(self):
        assert name_convention_for("en") is None
        assert name_convention_for("") is None
        assert name_convention_for(None) is None

    def test_register_new_language_honorific(self):
        register_person_honorific("ms", "Puan", gender="female")

        peeled = peel_person_honorifics("Puan Siti Aminah", "ms")

        assert (peeled.prefix, peeled.name, peeled.gender) == (
            "Puan ",
            "Siti Aminah",
            "female",
        )

    def test_register_suffix_extends_existing_convention(self):
        register_person_honorific("sw", "Mzee", gender="male", position="suffix")

        peeled = peel_person_honorifics("Juma Hamisi Mzee", "sw")

        assert (peeled.name, peeled.suffix, peeled.gender) == (
            "Juma Hamisi",
            " Mzee",
            "male",
        )

    def test_register_is_idempotent(self):
        before = len(name_convention_for("sw").honorifics)
        register_person_honorific("sw", "Bi.", gender="female")
        assert len(name_convention_for("sw").honorifics) == before

    @pytest.mark.parametrize(
        ("lang", "text", "kwargs"),
        [
            ("ja", "どの", {}),
            ("", "Puan", {}),
            ("ms", "  ", {}),
            ("ms", "Puan", {"gender": "unknown"}),
            ("ms", "Puan", {"position": "infix"}),
            ("ms", "Puan", {"min_name_tokens": 0}),
        ],
    )
    def test_register_rejects_invalid_input(self, lang, text, kwargs):
        with pytest.raises(ValueError):
            register_person_honorific(lang, text, **kwargs)

    def test_honorific_dataclass_validates_text(self):
        with pytest.raises(ValueError):
            Honorific("")


class TestPeelPerLanguage:
    @pytest.mark.parametrize(
        ("lang", "span", "prefix", "name", "suffix", "gender"),
        [
            # Thai titles are written attached to the given name.
            ("th", "นางสาวสมหญิง ใจดี", "นางสาว", "สมหญิง ใจดี", "", "female"),
            ("th", "นางมาลี พรหมมา", "นาง", "มาลี พรหมมา", "", "female"),
            ("th", "นายสมชาย ใจดี", "นาย", "สมชาย ใจดี", "", "male"),
            ("th", "นาง สุดา แก้วใส", "นาง ", "สุดา แก้วใส", "", "female"),
            ("am", "ወይዘሮ ሜሮን አለሙ", "ወይዘሮ ", "ሜሮን አለሙ", "", "female"),
            ("am", "አቶ ዳዊት ከበደ", "አቶ ", "ዳዊት ከበደ", "", "male"),
            ("sw", "Bi. Amina Hassan", "Bi. ", "Amina Hassan", "", "female"),
            ("sw", "bwana Baraka Mushi", "bwana ", "Baraka Mushi", "", "male"),
            ("ar", "السيدة مريم محمود", "السيدة ", "مريم محمود", "", "female"),
            ("ar", "الدكتورة ليلى حسن", "الدكتورة ", "ليلى حسن", "", "female"),
            ("ar", "السيد يوسف إبراهيم", "السيد ", "يوسف إبراهيم", "", "male"),
            ("fa", "خانم سارا احمدی", "خانم ", "سارا احمدی", "", "female"),
            ("fa", "جناب آقای رضا حسینی", "جناب آقای ", "رضا حسینی", "", "male"),
            ("fa", "مریم خانم", "", "مریم", " خانم", "female"),
            ("ur", "محترمہ فاطمہ خان", "محترمہ ", "فاطمہ خان", "", "female"),
            ("ur", "جناب عارف قریشی صاحب", "جناب ", "عارف قریشی", " صاحب", "male"),
            ("hi", "श्रीमती अनिता शर्मा", "श्रीमती ", "अनिता शर्मा", "", "female"),
            ("hi", "श्री राजेश कुमार", "श्री ", "राजेश कुमार", "", "male"),
            ("vi", "Chị Lê Thu Hà", "Chị ", "Lê Thu Hà", "", "female"),
            ("vi", "Quý bà Phạm Thị Mai", "Quý bà ", "Phạm Thị Mai", "", "female"),
            ("vi", "Ông Trần Hoàng Nam", "Ông ", "Trần Hoàng Nam", "", "male"),
            ("vi", "Anh Nguyễn Quang Huy", "Anh ", "Nguyễn Quang Huy", "", "male"),
            ("id", "Ibu Dewi Lestari", "Ibu ", "Dewi Lestari", "", "female"),
            ("id", "Ny. Siti Aminah", "Ny. ", "Siti Aminah", "", "female"),
            ("id", "Bapak Agus Pratama", "Bapak ", "Agus Pratama", "", "male"),
            (
                "yo",
                "Arábìnrin Bọ́láńlé Adébáyọ̀",
                "Arábìnrin ",
                "Bọ́láńlé Adébáyọ̀",
                "",
                "female",
            ),
            ("yo", "Ọ̀gbẹ́ni Adéọlá Ọládìpọ̀", "Ọ̀gbẹ́ni ", "Adéọlá Ọládìpọ̀", "", "male"),
            # Tone marks may be omitted in running text.
            ("yo", "Ogbeni Adeola Oladipo", "Ogbeni ", "Adeola Oladipo", "", "male"),
            (
                "ha",
                "Hajiya Bilkisu Balarabe",
                "Hajiya ",
                "Bilkisu Balarabe",
                "",
                "female",
            ),
            ("ha", "Malama Amina Ɗanladi", "Malama ", "Amina Ɗanladi", "", "female"),
            ("ha", "Malam Musa Ɗanjuma", "Malam ", "Musa Ɗanjuma", "", "male"),
        ],
    )
    def test_honorific_and_gender_cue(self, lang, span, prefix, name, suffix, gender):
        peeled = peel_person_honorifics(span, lang)

        assert (peeled.prefix, peeled.name, peeled.suffix) == (prefix, name, suffix)
        assert peeled.gender == gender
        assert peeled.has_honorific
        assert peeled.prefix + peeled.name + peeled.suffix == span

    @pytest.mark.parametrize(
        ("lang", "span", "prefix"),
        [
            ("sw", "Dkt. Amina Hassan", "Dkt. "),
            ("am", "ዶ/ር ሰላም ተስፋዬ", "ዶ/ር "),
            ("th", "ดร.สมชาย ใจดี", "ดร."),
            ("hi", "डॉ. प्रिया वर्मा", "डॉ. "),
        ],
    )
    def test_neutral_honorific_carries_no_gender(self, lang, span, prefix):
        peeled = peel_person_honorifics(span, lang)

        assert peeled.prefix == prefix
        assert peeled.gender is None

    def test_stacked_titles_are_peeled_together(self):
        peeled = peel_person_honorifics("جناب آقای دکتر علی رضایی", "fa")

        assert (peeled.prefix, peeled.name, peeled.gender) == (
            "جناب آقای دکتر ",
            "علی رضایی",
            "male",
        )

    def test_conflicting_titles_give_no_gender_cue(self):
        peeled = peel_person_honorifics("Bi. Bw. Baraka Mushi", "sw")

        assert peeled.name == "Baraka Mushi"
        assert peeled.gender is None

    def test_neutral_suffix(self):
        peeled = peel_person_honorifics("राजेश कुमार जी", "hi")

        assert (peeled.name, peeled.suffix, peeled.gender) == (
            "राजेश कुमार",
            " जी",
            None,
        )

    def test_surrounding_whitespace_is_kept_in_the_honorific_parts(self):
        peeled = peel_person_honorifics("  Bi.  Amina Hassan ", "sw")

        assert peeled.prefix == "  Bi.  "
        assert peeled.name == "Amina Hassan"
        assert peeled.suffix == " "

    @pytest.mark.parametrize(
        ("lang", "span"),
        [("sw", "Bi."), ("th", "นางสาว"), ("ur", "جناب"), ("fa", "جناب آقای")],
    )
    def test_honorific_only_span_has_empty_name(self, lang, span):
        peeled = peel_person_honorifics(span, lang)

        assert peeled.name == ""
        assert peeled.prefix + peeled.suffix == span

    def test_arabic_script_spelling_variants_match(self):
        # Heh instead of teh marbuta, Arabic yeh instead of Persian yeh.
        assert peel_person_honorifics("السيده مريم محمود", "ar").gender == "female"
        assert peel_person_honorifics("آقاي رضا حسینی", "fa").gender == "male"
        assert peel_person_honorifics("محترمه فاطمہ خان", "ur").gender == "female"
        assert fold_name_token("السيدة") == fold_name_token("السيده")


class TestVietnameseMarkers:
    def test_medial_thi_is_a_female_cue_without_honorific(self):
        peeled = peel_person_honorifics("Nguyễn Thị Lan", "vi")

        assert not peeled.has_honorific
        assert peeled.name == "Nguyễn Thị Lan"
        assert peeled.gender == "female"

    def test_medial_van_is_a_male_cue(self):
        assert peel_person_honorifics("Trần Văn Nam", "vi").gender == "male"

    def test_honorific_wins_over_marker(self):
        assert peel_person_honorifics("Bà Trần Văn Lan", "vi").gender == "female"

    def test_marker_at_name_edge_is_not_a_cue(self):
        assert peel_person_honorifics("Thị Lan", "vi").gender is None


class TestNegativeControls:
    """Honorific-like words that belong to the name stay part of the name."""

    @pytest.mark.parametrize(
        ("lang", "span"),
        [
            ("sw", "Amina Hassan"),
            ("th", "สมหญิง ใจดี"),
            # ``คุณ`` + a vowel that cannot start a Thai syllable: a given name.
            ("th", "คุณากร ใจดี"),
            # Urdu lineage and family names that look like titles.
            ("ur", "سید عارف"),
            ("ur", "زہرہ بیگم"),
            # ``السيد`` used as a family name at the end of the span.
            ("ar", "محمد السيد"),
            # ``श्री`` as the start of a given name, not a separate title.
            ("hi", "श्रीकांत वर्मा"),
            # ``Anh`` as a given name, and ``Anh`` before a bare given name.
            ("vi", "Trần Hoàng Anh"),
            ("vi", "Anh Thư"),
            ("id", "Budi Pakpahan"),
            ("sw", "Amina Bibi"),
            ("ha", "Malamiya Bello"),
            # Languages without a convention are passed through untouched.
            ("en", "Dr. Jane Roe"),
            ("de", "Frau Erika Muster"),
            ("ja", "田中さん"),
        ],
    )
    def test_span_is_not_peeled(self, lang, span):
        peeled = peel_person_honorifics(span, lang)

        assert not peeled.has_honorific
        assert peeled.name == span
        assert peeled.prefix + peeled.name + peeled.suffix == span

    def test_ambiguous_honorific_alone_is_treated_as_a_name(self):
        peeled = peel_person_honorifics("Anh", "vi")

        assert peeled.name == "Anh"
        assert not peeled.has_honorific

    @pytest.mark.parametrize("span", ["", "   "])
    def test_empty_span(self, span):
        peeled = peel_person_honorifics(span, "sw")

        assert peeled.name == span
        assert not peeled.has_honorific
