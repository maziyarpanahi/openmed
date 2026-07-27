"""Locale-aware synthetic PHI note generation for training augmentation."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Final, Mapping, Sequence

from openmed.core import labels as L
from openmed.core.anonymizer import Anonymizer
from openmed.core.anonymizer.locales import (
    LANG_TO_LOCALE,
    NATIONAL_ID_PROVIDERS,
    resolve_locale,
)
from openmed.core.anonymizer.providers import clinical_ids
from openmed.core.pii_i18n import (
    INDIC_NER_LANGUAGES,
    SUPPORTED_LANGUAGES,
    validate_aadhaar,
    validate_chinese_resident_id,
    validate_czech_rodne_cislo,
    validate_danish_cpr,
    validate_dutch_bsn,
    validate_ethiopia_fayda,
    validate_french_nir,
    validate_german_steuer_id,
    validate_greek_amka,
    validate_indonesian_nik,
    validate_israeli_teudat_zehut,
    validate_italian_codice_fiscale,
    validate_kenya_maisha_namba,
    validate_korean_rrn,
    validate_marathi_aadhaar,
    validate_norwegian_fodselsnummer,
    validate_portuguese_cpf,
    validate_romanian_cnp,
    validate_russian_snils,
    validate_spanish_nie,
    validate_swedish_personnummer,
    validate_thai_national_id,
    validate_turkish_tckn,
    validate_ukrainian_rnokpp,
    validate_za_id_number,
)

SUPPORTED_LOCALE_PHI_LANGUAGES: Final[tuple[str, ...]] = (
    "as",
    "bn",
    "en",
    "fr",
    "de",
    "it",
    "es",
    "nl",
    "hi",
    "gu",
    "kn",
    "ml",
    "mr",
    "or",
    "pa",
    "ta",
    "te",
    "am",
    "pt",
    "ar",
    "he",
    "ja",
    "tr",
    "id",
    "th",
    "ko",
    "ro",
    "ru",
    "sv",
    "da",
    "no",
    "sw",
    "zu",
    "xh",
    "zh",
    "uk",
    "cs",
    "el",
)

LOCALE_PHI_LABELS: Final[tuple[str, ...]] = (
    L.PERSON,
    L.DATE_OF_BIRTH,
    L.ID_NUM,
    L.PHONE,
    L.STREET_ADDRESS,
    L.DATE,
)

_FIELD_ORDER: Final[tuple[str, ...]] = (
    "person",
    "date_of_birth",
    "identifier",
    "phone",
    "street_address",
    "date",
)
_FIELD_LABELS: Final[Mapping[str, str]] = {
    "person": L.PERSON,
    "date_of_birth": L.DATE_OF_BIRTH,
    "identifier": L.ID_NUM,
    "phone": L.PHONE,
    "street_address": L.STREET_ADDRESS,
    "date": L.DATE,
}

_TEMPLATES: Final[Mapping[str, tuple[str, ...]]] = {
    "as": (
        "ক্লিনিকেল টোকা: ৰোগী ",
        " জন্ম ",
        ". ৰোগী ID ",
        ". ফোন ",
        ". ঠিকনা ",
        ". পৰৱৰ্তী তাৰিখ ",
        ".",
    ),
    "bn": (
        "ক্লিনিক্যাল নোট: রোগী ",
        " জন্ম ",
        ". রোগী ID ",
        ". ফোন ",
        ". ঠিকানা ",
        ". পরবর্তী তারিখ ",
        ".",
    ),
    "en": (
        "Clinical note: patient ",
        " was born on ",
        ". Patient ID ",
        ". Phone ",
        ". Address ",
        ". Follow-up date ",
        ".",
    ),
    "fr": (
        "Note clinique : patient ",
        " ne le ",
        ". Identifiant patient ",
        ". Telephone ",
        ". Adresse ",
        ". Suivi le ",
        ".",
    ),
    "de": (
        "Klinische Notiz: Patient ",
        " geboren am ",
        ". Patienten-ID ",
        ". Telefon ",
        ". Adresse ",
        ". Kontrolle am ",
        ".",
    ),
    "it": (
        "Nota clinica: paziente ",
        " nato il ",
        ". ID paziente ",
        ". Telefono ",
        ". Indirizzo ",
        ". Controllo il ",
        ".",
    ),
    "es": (
        "Nota clinica: paciente ",
        " nacido el ",
        ". ID de paciente ",
        ". Telefono ",
        ". Direccion ",
        ". Seguimiento el ",
        ".",
    ),
    "nl": (
        "Klinische notitie: patient ",
        " geboren op ",
        ". Patient-ID ",
        ". Telefoon ",
        ". Adres ",
        ". Controle op ",
        ".",
    ),
    "hi": (
        "\u0915\u094d\u0932\u093f\u0928\u093f\u0915\u0932 \u0928\u094b"
        "\u091f: \u0930\u094b\u0917\u0940 ",
        " \u0915\u093e \u091c\u0928\u094d\u092e ",
        " \u0915\u094b \u0939\u0941\u0906. \u0930\u094b\u0917\u0940 ID ",
        ". \u092b\u094b\u0928 ",
        ". \u092a\u0924\u093e ",
        ". \u0905\u0917\u0932\u0940 \u0924\u093e\u0930\u0940\u0916 ",
        ".",
    ),
    "gu": (
        "ક્લિનિકલ નોંધ: દર્દી ",
        " જન્મ ",
        ". દર્દી ID ",
        ". ફોન ",
        ". સરનામું ",
        ". અનુવર્તી તારીખ ",
        ".",
    ),
    "kn": (
        "ಕ್ಲಿನಿಕಲ್ ಟಿಪ್ಪಣಿ: ರೋಗಿ ",
        " ಜನನ ",
        ". ರೋಗಿ ID ",
        ". ಫೋನ್ ",
        ". ವಿಳಾಸ ",
        ". ಮುಂದಿನ ದಿನಾಂಕ ",
        ".",
    ),
    "ml": (
        "ക്ലിനിക്കൽ കുറിപ്പ്: രോഗി ",
        " ജനനം ",
        ". രോഗി ID ",
        ". ഫോൺ ",
        ". വിലാസം ",
        ". തുടർ തീയതി ",
        ".",
    ),
    "mr": (
        "क्लिनिकल नोंद: रुग्ण ",
        " जन्म ",
        ". रुग्ण ID ",
        ". फोन ",
        ". पत्ता ",
        ". पुढील तारीख ",
        ".",
    ),
    "or": (
        "କ୍ଲିନିକାଲ ଟିପ୍ପଣୀ: ରୋଗୀ ",
        " ଜନ୍ମ ",
        ". ରୋଗୀ ID ",
        ". ଫୋନ ",
        ". ଠିକଣା ",
        ". ପରବର୍ତ୍ତୀ ତାରିଖ ",
        ".",
    ),
    "pa": (
        "ਕਲੀਨਿਕਲ ਨੋਟ: ਮਰੀਜ਼ ",
        " ਜਨਮ ",
        ". ਮਰੀਜ਼ ID ",
        ". ਫੋਨ ",
        ". ਪਤਾ ",
        ". ਅਗਲੀ ਮਿਤੀ ",
        ".",
    ),
    "ta": (
        "மருத்துவ குறிப்பு: நோயாளி ",
        " பிறந்த தேதி ",
        ". நோயாளி ID ",
        ". தொலைபேசி ",
        ". முகவரி ",
        ". அடுத்த தேதி ",
        ".",
    ),
    "te": (
        "\u0c15\u0c4d\u0c32\u0c3f\u0c28\u0c3f\u0c15\u0c32\u0c4d"
        " \u0c28\u0c4b\u0c1f\u0c4d: \u0c30\u0c4b\u0c17\u0c3f ",
        " \u0c1c\u0c28\u0c28 \u0c24\u0c47\u0c26\u0c40 ",
        ". \u0c30\u0c4b\u0c17\u0c3f ID ",
        ". \u0c2b\u0c4b\u0c28\u0c4d ",
        ". \u0c1a\u0c3f\u0c30\u0c41\u0c28\u0c3e\u0c2e\u0c3e ",
        ". \u0c2b\u0c3e\u0c32\u0c4b-\u0c05\u0c2a\u0c4d \u0c24\u0c47\u0c26\u0c40 ",
        ".",
    ),
    "am": (
        "ክሊኒካዊ ማስታወሻ፡ ታካሚ ",
        " የትውልድ ቀን ",
        "። የታካሚ መለያ ",
        "። ስልክ ",
        "። አድራሻ ",
        "። የቀጠሮ ቀን ",
        "።",
    ),
    "pt": (
        "Nota clinica: paciente ",
        " nascido em ",
        ". ID do paciente ",
        ". Telefone ",
        ". Endereco ",
        ". Seguimento em ",
        ".",
    ),
    "ar": (
        "\u0645\u0644\u0627\u062d\u0638\u0629 \u0633\u0631\u064a"
        "\u0631\u064a\u0629: \u0627\u0644\u0645\u0631\u064a"
        "\u0636 ",
        " \u062a\u0627\u0631\u064a\u062e \u0627\u0644\u0645\u064a\u0644\u0627\u062f ",
        ". \u0645\u0639\u0631\u0641 \u0627\u0644\u0645\u0631\u064a\u0636 ",
        ". \u0647\u0627\u062a\u0641 ",
        ". \u0639\u0646\u0648\u0627\u0646 ",
        ". \u0645\u0648\u0639\u062f \u0627\u0644\u0645\u062a\u0627\u0628\u0639\u0629 ",
        ".",
    ),
    "he": (
        "\u05d4\u05e2\u05e8\u05d4 \u05e7\u05dc\u05d9\u05e0\u05d9\u05ea: \u05de\u05d8\u05d5\u05e4\u05dc ",
        " \u05e0\u05d5\u05dc\u05d3 \u05d1\u05ea\u05d0\u05e8\u05d9\u05da ",
        ". \u05de\u05d6\u05d4\u05d4 \u05de\u05d8\u05d5\u05e4\u05dc ",
        ". \u05d8\u05dc\u05e4\u05d5\u05df ",
        ". \u05db\u05ea\u05d5\u05d1\u05ea ",
        ". \u05ea\u05d0\u05e8\u05d9\u05da \u05de\u05e2\u05e7\u05d1 ",
        ".",
    ),
    "ja": (
        "\u81e8\u5e8a\u30ce\u30fc\u30c8: \u60a3\u8005 ",
        "\u306e\u751f\u5e74\u6708\u65e5 ",
        "\u3002\u60a3\u8005ID ",
        "\u3002\u96fb\u8a71 ",
        "\u3002\u4f4f\u6240 ",
        "\u3002\u518d\u8a3a\u65e5 ",
        "\u3002",
    ),
    "tr": (
        "Klinik not: hasta ",
        " dogum tarihi ",
        ". Hasta kimlik no ",
        ". Telefon ",
        ". Adres ",
        ". Kontrol tarihi ",
        ".",
    ),
    "id": (
        "Catatan klinis: pasien ",
        " lahir pada ",
        ". ID pasien ",
        ". Telepon ",
        ". Alamat ",
        ". Tanggal kontrol ",
        ".",
    ),
    "th": (
        "บันทึกคลินิก: ผู้ป่วย ",
        " วันเกิด ",
        ". ID ผู้ป่วย ",
        ". โทรศัพท์ ",
        ". ที่อยู่ ",
        ". วันที่นัด ",
        ".",
    ),
    "ko": (
        "임상 기록: 환자 ",
        " 생년월일 ",
        ". 환자 등록번호 ",
        ". 전화번호 ",
        ". 주소 ",
        ". 추적 관찰일 ",
        ".",
    ),
    "ro": (
        "Nota clinica: pacient ",
        " nascut la ",
        ". CNP pacient ",
        ". Telefon ",
        ". Adresa ",
        ". Control la ",
        ".",
    ),
    "ru": (
        "Клиническая запись: пациент ",
        " родился ",
        ". СНИЛС пациента ",
        ". Телефон ",
        ". Адрес ",
        ". Контроль ",
        ".",
    ),
    "sv": (
        "Klinisk anteckning: patient ",
        " född ",
        ". Patient-ID ",
        ". Telefon ",
        ". Adress ",
        ". Uppföljning ",
        ".",
    ),
    "da": (
        "Klinisk notat: patient ",
        " født ",
        ". Patient-ID ",
        ". Telefon ",
        ". Adresse ",
        ". Opfølgning ",
        ".",
    ),
    "no": (
        "Klinisk notat: pasient ",
        " født ",
        ". Pasient-ID ",
        ". Telefon ",
        ". Adresse ",
        ". Oppfølging ",
        ".",
    ),
    "zh": (
        "临床记录：患者",
        "出生于",
        "。患者编号",
        "。电话",
        "。地址",
        "。复诊日期",
        "。",
    ),
    "sw": (
        "Dokezo la kliniki: mgonjwa ",
        " alizaliwa tarehe ",
        ". Nambari ya mgonjwa ",
        ". Simu ",
        ". Anwani ",
        ". Tarehe ya ufuatiliaji ",
        ".",
    ),
    "zu": (
        "Inothi lomtholampilo: isiguli ",
        " sazalwa ngo ",
        ". Inombolo kamazisi ",
        ". Ucingo ",
        ". Ikheli ",
        ". Usuku lokulandelela ",
        ".",
    ),
    "xh": (
        "Inqaku lekliniki: isigulane ",
        " sazalwa ngo ",
        ". Inombolo yesazisi ",
        ". Ifowuni ",
        ". Idilesi ",
        ". Umhla wokulandelela ",
        ".",
    ),
    "uk": (
        "Клінічна нотатка: пацієнт ",
        " народився ",
        ". РНОКПП пацієнта ",
        ". Телефон ",
        ". Адреса ",
        ". Дата наступного огляду ",
        ".",
    ),
    "cs": (
        "Klinická poznámka: pacient ",
        " se narodil ",
        ". Rodné číslo pacienta ",
        ". Telefon ",
        ". Adresa ",
        ". Datum kontroly ",
        ".",
    ),
    "el": (
        "Κλινική σημείωση: ασθενής ",
        " γεννήθηκε στις ",
        ". ΑΜΚΑ ασθενούς ",
        ". Τηλέφωνο ",
        ". Διεύθυνση ",
        ". Ημερομηνία επανελέγχου ",
        ".",
    ),
}

_NATIONAL_ID_VALIDATORS: Final[Mapping[str, Callable[[str], bool]]] = {
    "am": validate_ethiopia_fayda,
    "en": clinical_ids.validate_ssn,
    "fr": validate_french_nir,
    "de": validate_german_steuer_id,
    "it": validate_italian_codice_fiscale,
    "es": validate_spanish_nie,
    "nl": validate_dutch_bsn,
    "hi": validate_aadhaar,
    "mr": validate_marathi_aadhaar,
    "te": validate_aadhaar,
    "pt": validate_portuguese_cpf,
    "tr": validate_turkish_tckn,
    "he": validate_israeli_teudat_zehut,
    "id": validate_indonesian_nik,
    "th": validate_thai_national_id,
    "ko": validate_korean_rrn,
    "ro": validate_romanian_cnp,
    "ru": validate_russian_snils,
    "sv": validate_swedish_personnummer,
    "da": validate_danish_cpr,
    "no": validate_norwegian_fodselsnummer,
    "sw": validate_kenya_maisha_namba,
    "zu": validate_za_id_number,
    "xh": validate_za_id_number,
    "zh": validate_chinese_resident_id,
    "uk": validate_ukrainian_rnokpp,
    "cs": validate_czech_rodne_cislo,
    "el": validate_greek_amka,
}

_NATIONAL_ID_VALIDATOR_NAMES: Final[Mapping[str, str]] = {
    "am": "pii_i18n.validate_ethiopia_fayda",
    "en": "clinical_ids.validate_ssn",
    "fr": "pii_i18n.validate_french_nir",
    "de": "pii_i18n.validate_german_steuer_id",
    "it": "pii_i18n.validate_italian_codice_fiscale",
    "es": "pii_i18n.validate_spanish_nie",
    "nl": "pii_i18n.validate_dutch_bsn",
    "hi": "pii_i18n.validate_aadhaar",
    "mr": "pii_i18n.validate_marathi_aadhaar",
    "te": "pii_i18n.validate_aadhaar",
    "pt": "pii_i18n.validate_portuguese_cpf",
    "tr": "pii_i18n.validate_turkish_tckn",
    "he": "pii_i18n.validate_israeli_teudat_zehut",
    "id": "pii_i18n.validate_indonesian_nik",
    "th": "pii_i18n.validate_thai_national_id",
    "ko": "pii_i18n.validate_korean_rrn",
    "ro": "pii_i18n.validate_romanian_cnp",
    "ru": "pii_i18n.validate_russian_snils",
    "sv": "pii_i18n.validate_swedish_personnummer",
    "da": "pii_i18n.validate_danish_cpr",
    "no": "pii_i18n.validate_norwegian_fodselsnummer",
    "sw": "pii_i18n.validate_kenya_maisha_namba",
    "zu": "pii_i18n.validate_za_id_number",
    "xh": "pii_i18n.validate_za_id_number",
    "zh": "pii_i18n.validate_chinese_resident_id",
    "uk": "pii_i18n.validate_ukrainian_rnokpp",
    "cs": "pii_i18n.validate_czech_rodne_cislo",
    "el": "pii_i18n.validate_greek_amka",
}


@dataclass(frozen=True)
class SyntheticPhiSpan:
    """A canonical gold span in a synthetic PHI training example."""

    start: int
    end: int
    label: str
    text: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "end": self.end,
            "label": self.label,
            "metadata": dict(self.metadata),
            "start": self.start,
            "text": self.text,
        }


@dataclass(frozen=True)
class LocalePhiExample:
    """Synthetic locale PHI text plus aligned canonical gold spans."""

    text: str
    gold_spans: tuple[SyntheticPhiSpan, ...]
    language: str
    locale: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_training_item(self) -> dict[str, Any]:
        return {
            "is_synthetic": True,
            "labels": [span.to_dict() for span in self.gold_spans],
            "language": self.language,
            "locale": self.locale,
            "metadata": dict(self.metadata),
            "synthetic_source": "locale_phi",
            "text": self.text,
        }


class LocalePhiGenerator:
    """Generate deterministic locale PHI augmentation examples."""

    def __init__(self, *, seed: int | None = None) -> None:
        self.seed = seed
        self._rng = random.Random(seed)

    def generate(self, language: str) -> LocalePhiExample:
        """Generate one synthetic PHI example for ``language``."""

        if language not in SUPPORTED_LOCALE_PHI_LANGUAGES:
            raise ValueError(
                f"unsupported locale PHI language {language!r}; "
                f"supported={list(SUPPORTED_LOCALE_PHI_LANGUAGES)!r}"
            )
        if language not in SUPPORTED_LANGUAGES | INDIC_NER_LANGUAGES:
            raise ValueError(f"language {language!r} is not wired in OpenMed")

        locale = resolve_locale(language)
        values = self._values_for(language, locale)
        text, spans = self._render(language, values)
        metadata = {
            "augmentation_only": True,
            "contains_real_phi": False,
            "language": language,
            "locale": locale,
            "seed": self.seed,
            "synthetic": True,
            "synthetic_source": "locale_phi",
        }
        return LocalePhiExample(
            text=text,
            gold_spans=spans,
            language=language,
            locale=locale,
            metadata=metadata,
        )

    def generate_all(self) -> tuple[LocalePhiExample, ...]:
        """Generate one example for every wired locale PHI language."""

        return tuple(
            self.generate(language) for language in SUPPORTED_LOCALE_PHI_LANGUAGES
        )

    def _values_for(
        self, language: str, locale: str
    ) -> Mapping[str, tuple[str, Mapping[str, Any]]]:
        identifier, identifier_metadata = self._identifier(language)
        return {
            "person": (
                self._surrogate(language, L.PERSON, "Example Patient", locale),
                {"field": "person"},
            ),
            "date_of_birth": (
                self._surrogate(language, L.DATE_OF_BIRTH, "1970-01-15", locale),
                {"field": "date_of_birth"},
            ),
            "identifier": (identifier, identifier_metadata),
            "phone": (
                self._surrogate(language, L.PHONE, "", locale),
                {"field": "phone"},
            ),
            "street_address": (
                self._surrogate(language, L.STREET_ADDRESS, "", locale),
                {"field": "street_address"},
            ),
            "date": (
                self._surrogate(language, L.DATE, "2024-04-18", locale),
                {"field": "date"},
            ),
        }

    def _identifier(self, language: str) -> tuple[str, Mapping[str, Any]]:
        seed = self._field_seed(language, "identifier")
        if language in NATIONAL_ID_PROVIDERS:
            id_locale, method = NATIONAL_ID_PROVIDERS[language]
            value = self._surrogate(
                language,
                L.ID_NUM,
                "123456789",
                id_locale,
                seed=seed,
            )
            validator = _NATIONAL_ID_VALIDATORS[language]
            if not validator(value):
                raise RuntimeError(
                    f"generated invalid national ID for {language!r}: {value!r}"
                )
            return value, {
                "field": "identifier",
                "id_locale": id_locale,
                "id_method": method,
                "id_subtype": "national_id",
                "validator": _NATIONAL_ID_VALIDATOR_NAMES[language],
            }

        rng = random.Random(seed)
        value = clinical_ids.generate_luhn_identifier(rng=rng)
        if not clinical_ids.validate_luhn(value):
            raise RuntimeError(f"generated invalid Luhn identifier: {value!r}")
        return value, {
            "field": "identifier",
            "id_locale": LANG_TO_LOCALE[language],
            "id_method": "generate_luhn_identifier",
            "id_subtype": "luhn",
            "validator": "clinical_ids.validate_luhn",
        }

    def _surrogate(
        self,
        language: str,
        label: str,
        original: str,
        locale: str,
        *,
        seed: int | None = None,
    ) -> str:
        anonymizer = Anonymizer(
            lang=language,
            consistent=True,
            seed=seed if seed is not None else self._field_seed(language, label),
        )
        value = anonymizer.surrogate(original, label, lang=language, locale=locale)
        return " ".join(str(value).split())

    def _render(
        self, language: str, values: Mapping[str, tuple[str, Mapping[str, Any]]]
    ) -> tuple[str, tuple[SyntheticPhiSpan, ...]]:
        literals = _TEMPLATES[language]
        chunks: list[str] = []
        spans: list[SyntheticPhiSpan] = []
        cursor = 0

        for index, field_name in enumerate(_FIELD_ORDER):
            literal = literals[index]
            chunks.append(literal)
            cursor += len(literal)

            value, metadata = values[field_name]
            start = cursor
            chunks.append(value)
            cursor += len(value)
            spans.append(
                SyntheticPhiSpan(
                    start=start,
                    end=cursor,
                    label=_FIELD_LABELS[field_name],
                    text=value,
                    metadata={"synthetic": True, **dict(metadata)},
                )
            )

        chunks.append(literals[-1])
        return "".join(chunks), tuple(spans)

    def _field_seed(self, language: str, field_name: str) -> int:
        if self.seed is None:
            return self._rng.getrandbits(64)
        material = f"{self.seed}|{language}|{field_name}".encode("utf-8")
        digest = hashlib.blake2b(material, digest_size=8).digest()
        return int.from_bytes(digest, "big", signed=False)


def generate_locale_phi_examples(
    *,
    languages: Sequence[str] = SUPPORTED_LOCALE_PHI_LANGUAGES,
    seed: int | None = None,
) -> tuple[LocalePhiExample, ...]:
    """Generate synthetic PHI examples for ``languages``."""

    generator = LocalePhiGenerator(seed=seed)
    return tuple(generator.generate(language) for language in languages)


__all__ = [
    "LOCALE_PHI_LABELS",
    "SUPPORTED_LOCALE_PHI_LANGUAGES",
    "LocalePhiExample",
    "LocalePhiGenerator",
    "SyntheticPhiSpan",
    "generate_locale_phi_examples",
]
