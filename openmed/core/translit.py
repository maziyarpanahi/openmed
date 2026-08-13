"""Small, dependency-free name romanization helpers.

This module supports regression tests for non-Latin PERSON surrogates.  It is
deliberately not a general transliteration API: Unicode does not expose Han
readings through :mod:`unicodedata`, so common deterministic Faker names use a
small phrase table and unknown ideographs use a stable ASCII code-point token.
Kana, Hangul, Cyrillic, and Greek are transliterated algorithmically.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Mapping

from .script_detect import UNKNOWN_SCRIPT, detect_script, segment_by_script


def _case_variants(mapping: Mapping[str, str]) -> dict[str, str]:
    variants = dict(mapping)
    variants.update(
        {
            source.upper(): target[:1].upper() + target[1:]
            for source, target in mapping.items()
            if source.upper() != source
        }
    )
    return variants


_CYRILLIC = _case_variants(
    {
        "а": "a",
        "б": "b",
        "в": "v",
        "г": "g",
        "д": "d",
        "е": "e",
        "ё": "yo",
        "ж": "zh",
        "з": "z",
        "и": "i",
        "й": "y",
        "к": "k",
        "л": "l",
        "м": "m",
        "н": "n",
        "о": "o",
        "п": "p",
        "р": "r",
        "с": "s",
        "т": "t",
        "у": "u",
        "ф": "f",
        "х": "kh",
        "ц": "ts",
        "ч": "ch",
        "ш": "sh",
        "щ": "shch",
        "ъ": "",
        "ы": "y",
        "ь": "'",
        "э": "e",
        "ю": "yu",
        "я": "ya",
        "і": "i",
        "ї": "yi",
        "є": "ye",
        "ґ": "g",
        "ў": "u",
        "ј": "j",
        "љ": "lj",
        "њ": "nj",
        "ћ": "c",
        "ђ": "dj",
        "џ": "dz",
    }
)

_GREEK = _case_variants(
    {
        "α": "a",
        "β": "v",
        "γ": "g",
        "δ": "d",
        "ε": "e",
        "ζ": "z",
        "η": "i",
        "θ": "th",
        "ι": "i",
        "κ": "k",
        "λ": "l",
        "μ": "m",
        "ν": "n",
        "ξ": "x",
        "ο": "o",
        "π": "p",
        "ρ": "r",
        "σ": "s",
        "ς": "s",
        "τ": "t",
        "υ": "u",
        "φ": "f",
        "χ": "ch",
        "ψ": "ps",
        "ω": "o",
    }
)

_KANA = {
    "あ": "a",
    "い": "i",
    "う": "u",
    "え": "e",
    "お": "o",
    "か": "ka",
    "き": "ki",
    "く": "ku",
    "け": "ke",
    "こ": "ko",
    "が": "ga",
    "ぎ": "gi",
    "ぐ": "gu",
    "げ": "ge",
    "ご": "go",
    "さ": "sa",
    "し": "shi",
    "す": "su",
    "せ": "se",
    "そ": "so",
    "ざ": "za",
    "じ": "ji",
    "ず": "zu",
    "ぜ": "ze",
    "ぞ": "zo",
    "た": "ta",
    "ち": "chi",
    "つ": "tsu",
    "て": "te",
    "と": "to",
    "だ": "da",
    "ぢ": "ji",
    "づ": "zu",
    "で": "de",
    "ど": "do",
    "な": "na",
    "に": "ni",
    "ぬ": "nu",
    "ね": "ne",
    "の": "no",
    "は": "ha",
    "ひ": "hi",
    "ふ": "fu",
    "へ": "he",
    "ほ": "ho",
    "ば": "ba",
    "び": "bi",
    "ぶ": "bu",
    "べ": "be",
    "ぼ": "bo",
    "ぱ": "pa",
    "ぴ": "pi",
    "ぷ": "pu",
    "ぺ": "pe",
    "ぽ": "po",
    "ま": "ma",
    "み": "mi",
    "む": "mu",
    "め": "me",
    "も": "mo",
    "や": "ya",
    "ゆ": "yu",
    "よ": "yo",
    "ら": "ra",
    "り": "ri",
    "る": "ru",
    "れ": "re",
    "ろ": "ro",
    "わ": "wa",
    "を": "o",
    "ん": "n",
    "ゔ": "vu",
    "ぁ": "a",
    "ぃ": "i",
    "ぅ": "u",
    "ぇ": "e",
    "ぉ": "o",
}

_KANA_DIGRAPHS = {
    "きゃ": "kya",
    "きゅ": "kyu",
    "きょ": "kyo",
    "ぎゃ": "gya",
    "ぎゅ": "gyu",
    "ぎょ": "gyo",
    "しゃ": "sha",
    "しゅ": "shu",
    "しょ": "sho",
    "じゃ": "ja",
    "じゅ": "ju",
    "じょ": "jo",
    "ちゃ": "cha",
    "ちゅ": "chu",
    "ちょ": "cho",
    "にゃ": "nya",
    "にゅ": "nyu",
    "にょ": "nyo",
    "ひゃ": "hya",
    "ひゅ": "hyu",
    "ひょ": "hyo",
    "びゃ": "bya",
    "びゅ": "byu",
    "びょ": "byo",
    "ぴゃ": "pya",
    "ぴゅ": "pyu",
    "ぴょ": "pyo",
    "みゃ": "mya",
    "みゅ": "myu",
    "みょ": "myo",
    "りゃ": "rya",
    "りゅ": "ryu",
    "りょ": "ryo",
}

_HANGUL_INITIALS = (
    "g",
    "kk",
    "n",
    "d",
    "tt",
    "r",
    "m",
    "b",
    "pp",
    "s",
    "ss",
    "",
    "j",
    "jj",
    "ch",
    "k",
    "t",
    "p",
    "h",
)
_HANGUL_VOWELS = (
    "a",
    "ae",
    "ya",
    "yae",
    "eo",
    "e",
    "yeo",
    "ye",
    "o",
    "wa",
    "wae",
    "oe",
    "yo",
    "u",
    "wo",
    "we",
    "wi",
    "yu",
    "eu",
    "ui",
    "i",
)
_HANGUL_FINALS = (
    "",
    "k",
    "k",
    "ks",
    "n",
    "nj",
    "nh",
    "t",
    "l",
    "lk",
    "lm",
    "lb",
    "ls",
    "lt",
    "lp",
    "lh",
    "m",
    "p",
    "ps",
    "t",
    "t",
    "ng",
    "t",
    "t",
    "k",
    "t",
    "p",
    "h",
)

_PHRASES: dict[str, dict[str, str]] = {
    "ja": {
        "佐藤 花子": "Sato Hanako",
        "田中 太郎": "Tanaka Taro",
        "鈴木 美咲": "Suzuki Misaki",
        "高橋 健": "Takahashi Ken",
        "鈴木 健一": "Suzuki Kenichi",
        "佐藤 翔太": "Sato Shota",
        "渡辺 春香": "Watanabe Haruka",
        "松本 陽一": "Matsumoto Yoichi",
        "小林 治": "Kobayashi Osamu",
        "佐藤 里佳": "Sato Rika",
        "佐藤 洋介": "Sato Yosuke",
        "井上 直子": "Inoue Naoko",
        "三浦 舞": "Miura Mai",
        "藤井 さゆり": "Fujii Sayuri",
        "前田 あすか": "Maeda Asuka",
        "中島 学": "Nakajima Manabu",
    },
    "zh": {
        "李博": "Li Bo",
        "王娜": "Wang Na",
        "刘兰英": "Liu Lanying",
        "郭博": "Guo Bo",
        "黄浩": "Huang Hao",
        "王成": "Wang Cheng",
        "顾秀华": "Gu Xiuhua",
        "刘霞": "Liu Xia",
        "杨彬": "Yang Bin",
        "谭鹏": "Tan Peng",
        "钟伟": "Zhong Wei",
        "谢春梅": "Xie Chunmei",
        "王伟": "Wang Wei",
        "李娜": "Li Na",
        "张伟": "Zhang Wei",
    },
}

_HAN_CHARACTERS: dict[str, dict[str, str]] = {
    "ja": {
        "佐": "sa",
        "藤": "to",
        "田": "ta",
        "中": "naka",
        "鈴": "suzu",
        "木": "ki",
        "高": "taka",
        "橋": "hashi",
        "渡": "wata",
        "辺": "nabe",
        "松": "matsu",
        "本": "moto",
        "花": "hana",
        "子": "ko",
        "太": "ta",
        "郎": "ro",
        "美": "mi",
        "咲": "saki",
        "健": "ken",
        "春": "haru",
        "香": "ka",
        "陽": "yo",
        "一": "ichi",
    },
    "zh": {
        "李": "li",
        "博": "bo",
        "王": "wang",
        "娜": "na",
        "刘": "liu",
        "兰": "lan",
        "英": "ying",
        "郭": "guo",
        "黄": "huang",
        "浩": "hao",
        "成": "cheng",
        "顾": "gu",
        "秀": "xiu",
        "华": "hua",
        "霞": "xia",
        "杨": "yang",
        "彬": "bin",
        "谭": "tan",
        "鹏": "peng",
        "钟": "zhong",
        "伟": "wei",
        "谢": "xie",
        "春": "chun",
        "梅": "mei",
        "张": "zhang",
    },
}


def romanize_name(text: str, *, lang: str | None = None) -> str:
    """Return a deterministic Latin rendering of a non-Latin name.

    The helper preserves whitespace and punctuation, removes Latin combining
    marks, and emits only ASCII for supported non-Latin scripts.  ``lang`` is
    used solely to disambiguate the small Han-name table (``"ja"`` or
    ``"zh"``); all other scripts are inferred with
    :func:`openmed.core.script_detect.detect_script`.

    Unknown Han readings become stable ``uXXXX`` tokens.  That fallback is
    intentionally explicit because guessing a pronunciation would be less
    deterministic than acknowledging the stdlib's lack of Unihan readings.
    """

    if not text:
        return text

    normalized = unicodedata.normalize("NFKC", text)
    dominant_script = detect_script(normalized)
    phrase = _phrase_romanization(normalized, lang)
    if phrase is not None:
        return phrase

    if dominant_script == UNKNOWN_SCRIPT:
        return _romanize_latin(normalized)

    rendered = [
        _romanize_segment(normalized[start:end], script, lang=lang)
        for start, end, script in segment_by_script(normalized)
    ]
    return "".join(rendered)


def _phrase_romanization(text: str, lang: str | None) -> str | None:
    if lang in _PHRASES:
        return _PHRASES[lang].get(text)
    for phrases in _PHRASES.values():
        if text in phrases:
            return phrases[text]
    return None


def _romanize_segment(text: str, script: str, *, lang: str | None) -> str:
    if script == "Latin" or script == UNKNOWN_SCRIPT:
        return _romanize_latin(text)
    if script == "Cyrillic":
        return _romanize_mapped(text, _CYRILLIC)
    if script == "Greek":
        return _romanize_mapped(text, _GREEK)
    if script == "Hangul":
        return "".join(_romanize_hangul_char(char) for char in text)
    if script == "Hiragana/Katakana":
        return _romanize_kana(text)
    if script == "Han":
        return _romanize_han(text, lang)
    return "".join(_ascii_fallback(char) for char in text)


def _romanize_latin(text: str) -> str:
    rendered: list[str] = []
    replacements = {"Æ": "AE", "æ": "ae", "Ø": "O", "ø": "o", "ß": "ss"}
    for char in unicodedata.normalize("NFKD", text):
        if unicodedata.category(char) == "Mn":
            continue
        rendered.append(replacements.get(char, _ascii_fallback(char)))
    return "".join(rendered)


def _romanize_mapped(text: str, mapping: Mapping[str, str]) -> str:
    rendered: list[str] = []
    for char in unicodedata.normalize("NFD", text):
        if unicodedata.category(char) == "Mn":
            continue
        rendered.append(mapping.get(char, _ascii_fallback(char)))
    return "".join(rendered)


def _romanize_hangul_char(char: str) -> str:
    codepoint = ord(char)
    if not 0xAC00 <= codepoint <= 0xD7A3:
        return _ascii_fallback(char)

    index = codepoint - 0xAC00
    initial = index // 588
    vowel = (index % 588) // 28
    final = index % 28
    return _HANGUL_INITIALS[initial] + _HANGUL_VOWELS[vowel] + _HANGUL_FINALS[final]


def _romanize_kana(text: str) -> str:
    hiragana = "".join(_hiragana_char(char) for char in text)
    rendered: list[str] = []
    geminate = False
    index = 0
    while index < len(hiragana):
        char = hiragana[index]
        if char == "っ":
            geminate = True
            index += 1
            continue
        if char == "ー":
            index += 1
            continue

        pair = hiragana[index : index + 2]
        syllable = _KANA_DIGRAPHS.get(pair)
        if syllable is not None:
            index += 2
        else:
            syllable = _KANA.get(char, _ascii_fallback(char))
            index += 1
        if geminate and syllable and syllable[0] not in "aeiou":
            syllable = syllable[0] + syllable
        geminate = False
        rendered.append(syllable)
    return "".join(rendered)


def _hiragana_char(char: str) -> str:
    codepoint = ord(char)
    if 0x30A1 <= codepoint <= 0x30F6:
        return chr(codepoint - 0x60)
    return char


def _romanize_han(text: str, lang: str | None) -> str:
    readings = _HAN_CHARACTERS.get(lang or "", {})
    return "".join(readings.get(char, _ascii_fallback(char)) for char in text)


def _ascii_fallback(char: str) -> str:
    if char.isascii():
        return char
    if char in {"・", "·", "　"}:
        return " "
    if unicodedata.category(char).startswith(("L", "M")):
        return f"u{ord(char):04x}"
    return "-" if unicodedata.category(char).startswith("P") else ""


__all__ = ["romanize_name"]
