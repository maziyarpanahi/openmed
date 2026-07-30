"""Unicode script detection helpers for mixed-script PII routing.

The helpers in this module are intentionally lightweight and stdlib-only. They
use explicit Unicode block ranges plus :mod:`unicodedata` character categories
to identify dominant scripts and preserve exact offsets while segmenting text
into script-oriented runs.

The curated confusable mappings are derived from Unicode UTS #39
``confusables.txt`` version 17.0.0. Unicode data files are distributed under
the Unicode License v3 (SPDX: ``Unicode-3.0``). The full data file is not
embedded: only mappings needed for the supported Latin/Cyrillic/Greek/CJK PII
evasion defense are retained.
"""

from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn

from .language_pack_catalog import SCRIPT_LANGUAGE_HINTS

if TYPE_CHECKING:
    from ..processing.legacy_encoding import LegacyFontMap

UNKNOWN_SCRIPT = "Unknown"

logger = logging.getLogger(__name__)

MAX_INGESTION_BYTES = 16 * 1024 * 1024

_INGESTION_ENCODING_ALIASES = {
    "ascii": "ascii",
    "big5": "big5",
    "cp1252": "cp1252",
    "euc-jp": "euc_jp",
    "euc-kr": "euc_kr",
    "eucjp": "euc_jp",
    "euckr": "euc_kr",
    "gb18030": "gb18030",
    "iso-8859-1": "latin-1",
    "latin-1": "latin-1",
    "latin1": "latin-1",
    "shift-jis": "shift_jis",
    "shiftjis": "shift_jis",
    "sjis": "shift_jis",
    "utf-16-be": "utf-16-be",
    "utf-16-le": "utf-16-le",
    "utf-8": "utf-8",
    "utf-8-sig": "utf-8-sig",
    "utf8": "utf-8",
    "windows-1252": "cp1252",
}
ALLOWED_INGESTION_ENCODINGS = frozenset(_INGESTION_ENCODING_ALIASES.values())

CJK_SCRIPTS = frozenset({"Han", "Hiragana/Katakana", "Hangul"})
INDIC_SCRIPTS = frozenset(
    {
        "Devanagari",
        "Bengali",
        "Gurmukhi",
        "Gujarati",
        "Odia",
        "Tamil",
        "Telugu",
        "Kannada",
        "Malayalam",
    }
)
CONFUSABLE_DATA_VERSION = "17.0.0"
CONFUSABLE_DATA_URL = "https://www.unicode.org/Public/17.0.0/security/confusables.txt"
CONFUSABLE_DATA_LICENSE = "Unicode-3.0"


class ChineseScriptVariant(str, Enum):
    """Estimated Chinese character variant used in a text."""

    SIMPLIFIED = "simplified"
    TRADITIONAL = "traditional"
    MIXED = "mixed"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ChineseScriptEstimate:
    """Ratio-based Simplified/Traditional estimate for synthetic text."""

    variant: ChineseScriptVariant
    simplified_count: int
    traditional_count: int
    simplified_ratio: float
    traditional_ratio: float

    @property
    def script(self) -> ChineseScriptVariant:
        """Alias for the estimated predominant variant."""

        return self.variant

    @property
    def mixed(self) -> bool:
        """Return whether both Simplified and Traditional evidence appears."""

        return self.simplified_count > 0 and self.traditional_count > 0


SUPPORTED_SCRIPTS = (
    "Latin",
    "Arabic",
    "Ethiopic",
    "Han",
    "Hiragana/Katakana",
    "Hangul",
    "Cyrillic",
    "Devanagari",
    "Bengali",
    "Gurmukhi",
    "Gujarati",
    "Odia",
    "Tamil",
    "Telugu",
    "Kannada",
    "Malayalam",
    "Greek",
    "Hebrew",
    "Thai",
)

ZERO_WIDTH_CHARS = frozenset(
    {
        "\u200b",  # zero width space
        "\u200c",  # zero width non-joiner
        "\u200d",  # zero width joiner
        "\u2060",  # word joiner
        "\ufeff",  # zero width no-break space
    }
)

_CONFUSABLE_FOLD: dict[str, str] = {
    "\u0391": "A",
    "\u0392": "B",
    "\u0395": "E",
    "\u0397": "H",
    "\u0399": "I",
    "\u039a": "K",
    "\u039c": "M",
    "\u039d": "N",
    "\u039f": "O",
    "\u03a1": "P",
    "\u03a4": "T",
    "\u03a7": "X",
    "\u03b1": "a",
    "\u03b5": "e",
    "\u03b7": "n",
    "\u03b9": "i",
    "\u03ba": "k",
    "\u03bc": "u",
    "\u03bf": "o",
    "\u03c1": "p",
    "\u03c4": "t",
    "\u03c5": "u",
    "\u03c7": "x",
    "\u0410": "A",
    "\u0412": "B",
    "\u0415": "E",
    "\u041a": "K",
    "\u041c": "M",
    "\u041d": "H",
    "\u041e": "O",
    "\u0420": "P",
    "\u0421": "C",
    "\u0422": "T",
    "\u0425": "X",
    "\u0430": "a",
    "\u0435": "e",
    "\u043e": "o",
    "\u0440": "p",
    "\u0441": "c",
    "\u0445": "x",
    "\u0456": "i",
    "\u3007": "O",
}

INDIAN_NAME_LANGUAGES = frozenset({"hi", "ta"})
INDIAN_NAME_SCRIPTS = frozenset({"Devanagari", "Tamil"})

_DEVANAGARI_CONSONANTS = {
    "क": "k",
    "ख": "kh",
    "ग": "g",
    "घ": "gh",
    "ङ": "n",
    "च": "ch",
    "छ": "chh",
    "ज": "j",
    "झ": "jh",
    "ञ": "n",
    "ट": "t",
    "ठ": "th",
    "ड": "d",
    "ढ": "dh",
    "ण": "n",
    "त": "t",
    "थ": "th",
    "द": "d",
    "ध": "dh",
    "न": "n",
    "प": "p",
    "फ": "ph",
    "ब": "b",
    "भ": "bh",
    "म": "m",
    "य": "y",
    "र": "r",
    "ल": "l",
    "व": "v",
    "श": "sh",
    "ष": "sh",
    "स": "s",
    "ह": "h",
    "क़": "k",
    "ख़": "kh",
    "ग़": "g",
    "ज़": "j",
    "ड़": "d",
    "ढ़": "dh",
    "फ़": "f",
}
_DEVANAGARI_VOWELS = {
    "अ": "a",
    "आ": "aa",
    "इ": "i",
    "ई": "ii",
    "उ": "u",
    "ऊ": "uu",
    "ऋ": "r",
    "ए": "e",
    "ऐ": "ai",
    "ओ": "o",
    "औ": "au",
}
_DEVANAGARI_VOWEL_SIGNS = {
    "ा": "aa",
    "ि": "i",
    "ी": "ii",
    "ु": "u",
    "ू": "uu",
    "ृ": "r",
    "े": "e",
    "ै": "ai",
    "ो": "o",
    "ौ": "au",
}

_TAMIL_CONSONANTS = {
    "க": "k",
    "ங": "n",
    "ச": "s",
    "ஞ": "n",
    "ட": "t",
    "ண": "n",
    "த": "t",
    "ந": "n",
    "ப": "p",
    "ம": "m",
    "ய": "y",
    "ர": "r",
    "ல": "l",
    "வ": "v",
    "ழ": "l",
    "ள": "l",
    "ற": "r",
    "ன": "n",
    "ஜ": "j",
    "ஷ": "sh",
    "ஸ": "s",
    "ஹ": "h",
}
_TAMIL_VOWELS = {
    "அ": "a",
    "ஆ": "aa",
    "இ": "i",
    "ஈ": "ii",
    "உ": "u",
    "ஊ": "uu",
    "எ": "e",
    "ஏ": "e",
    "ஐ": "ai",
    "ஒ": "o",
    "ஓ": "o",
    "ஔ": "au",
}
_TAMIL_VOWEL_SIGNS = {
    "ா": "aa",
    "ி": "i",
    "ீ": "ii",
    "ு": "u",
    "ூ": "uu",
    "ெ": "e",
    "ே": "e",
    "ை": "ai",
    "ொ": "o",
    "ோ": "o",
    "ௌ": "au",
}

_DEVANAGARI_RENDER_CONSONANTS = {
    "kh": "ख",
    "gh": "घ",
    "ch": "च",
    "jh": "झ",
    "th": "थ",
    "dh": "ध",
    "ph": "फ",
    "bh": "भ",
    "sh": "श",
    "k": "क",
    "g": "ग",
    "c": "च",
    "j": "ज",
    "t": "त",
    "d": "द",
    "n": "न",
    "p": "प",
    "b": "ब",
    "m": "म",
    "y": "य",
    "r": "र",
    "l": "ल",
    "v": "व",
    "w": "व",
    "s": "स",
    "h": "ह",
    "f": "फ",
    "q": "क",
    "x": "क्स",
    "z": "ज",
}
_DEVANAGARI_RENDER_VOWELS = {
    "a": ("अ", ""),
    "i": ("इ", "ि"),
    "u": ("उ", "ु"),
    "e": ("ए", "े"),
    "o": ("ओ", "ो"),
    "ai": ("ऐ", "ै"),
    "au": ("औ", "ौ"),
}

_TAMIL_RENDER_CONSONANTS = {
    "kh": "க",
    "gh": "க",
    "ch": "ச",
    "jh": "ஜ",
    "th": "த",
    "dh": "த",
    "ph": "ப",
    "bh": "ப",
    "sh": "ஷ",
    "k": "க",
    "g": "க",
    "c": "ச",
    "j": "ஜ",
    "t": "த",
    "d": "த",
    "n": "ந",
    "p": "ப",
    "b": "ப",
    "m": "ம",
    "y": "ய",
    "r": "ர",
    "l": "ல",
    "v": "வ",
    "w": "வ",
    "s": "ஸ",
    "h": "ஹ",
    "f": "ஃப",
    "q": "க",
    "x": "க்ஸ",
    "z": "ஜ",
}
_TAMIL_RENDER_VOWELS = {
    "a": ("அ", ""),
    "i": ("இ", "ி"),
    "u": ("உ", "ு"),
    "e": ("எ", "ெ"),
    "o": ("ஒ", "ொ"),
    "ai": ("ஐ", "ை"),
    "au": ("ஔ", "ௌ"),
}

_PHONETIC_VOWEL_FOLDS = (
    ("ee", "i"),
    ("ii", "i"),
    ("oo", "u"),
    ("uu", "u"),
    ("aa", "a"),
)


@dataclass(frozen=True)
class MixedScriptSpan:
    """An identifier-like source span containing more than one script."""

    start: int
    end: int
    scripts: tuple[str, ...]
    confusable_count: int = 0
    invisible_count: int = 0

    def to_metadata(self) -> dict[str, object]:
        """Return a raw-text-free representation suitable for audit metadata."""
        return {
            "confusable_count": self.confusable_count,
            "end": self.end,
            "invisible_count": self.invisible_count,
            "scripts": list(self.scripts),
            "start": self.start,
        }


# Unicode assigns both Simplified and Traditional Han characters to the same
# script blocks. These curated, common one-sided variants provide deterministic
# evidence without bundling a language model or copying OpenCC dictionaries.
_SIMPLIFIED_VARIANT_RAW = frozenset(
    "头药医门发后里网软台历叶万与东丝两严个临为乐书买乱云产亲仅从仓们"
    "众会体余传伤伦儿兰关兴养写军农冲决况冻净减几凤击刘创别动务华"
    "单卫厂厅压厌县双变号听员园圆场声处复备够夹夺奖妇妈孙学宁宝实"
    "审宽对寻导将层岁岛岭帐帮广庄庆库应开张归当录忆怀总态恋恶惊惯"
    "愿护报担拟拥拦拨择挂挤挥损换据摆敌数无旧时显晓术机杀杂权条来"
    "杨极构枪柜树样档桥梦检欢欧残气汉汤沟泪泻泽洁浅测济浓涛涡润涨"
    "湿滚满滤灭灯灵点炼热爱爷牵状犹独狮猫猪环现电画疗疮疯痒瘫盘盐"
    "监盖盗矿码砖确础礼祷离种积称稳穷窍竞笔笼筑签简粮紧红纤约级纪"
    "纯线练组细织终经结给络绝统绢继续绿编缘缝罗罚翘职联聪肃肠肤肿"
    "胀胜胶脉脏脑脸艺节芜苹范茧荐荡荣莱莲获营萧萨蓝虑虚虫虽虾蚁蚕"
    "补装见观规视览觉触计订认讲许论设访证评诉诊词译试诗诚话该详语"
    "误说请诸读课调谈谢谣谱贝财责贤账货质购贵费贺资赌赏赔赖赞赢赵"
    "车转轮轻载轿较辅辆辈输辑边达迁过运还这进远连迟适选递逻邓郑酱"
    "释鉴钟钢钥钱铁铜铝银销锁锅锋锐错锡锣锦键锯镜长闪闭问闯闲间闷"
    "闸闹闻阁阅阔队阳阴阵阶际陆陈险随隐难雾静韩页顶项顺须顾顿领颈"
    "频题颜额风飞饭饮馆马驱驶骑验鱼鲜鲸鸟鸡鸭鹅鹤鹰黄齐齿龄龙龟"
)
_TRADITIONAL_VARIANT_RAW = frozenset(
    "頭藥醫門發髮後裡裏網軟臺歷葉萬與東絲兩嚴個臨為樂書買亂雲產親"
    "僅從倉們眾會體餘傳傷倫兒蘭關興養寫軍農衝決況凍淨減幾鳳擊劉創"
    "別動務華單衛廠廳壓厭縣雙變號聽員園圓場聲處復備夠夾奪獎婦媽孫"
    "學寧寶實審寬對尋導將層歲島嶺帳幫廣莊慶庫應開張歸當錄憶懷總態"
    "戀惡驚慣願護報擔擬擁攔撥擇掛擠揮損換據擺敵數無舊時顯曉術機殺"
    "雜權條來楊極構槍櫃樹樣檔橋夢檢歡歐殘氣漢湯溝淚瀉澤潔淺測濟濃"
    "濤渦潤漲濕滾滿濾滅燈靈點煉熱愛爺牽狀猶獨獅貓豬環現電畫療瘡瘋"
    "癢癱盤鹽監蓋盜礦碼磚確礎禮禱離種積稱穩窮竅競筆籠築簽簡糧緊紅"
    "纖約級紀純線練組細織終經結給絡絕統絹繼續綠編緣縫羅罰翹職聯聰"
    "肅腸膚腫脹勝膠脈臟腦臉藝節蕪蘋範繭薦蕩榮萊蓮獲營蕭薩藍慮虛蟲"
    "雖蝦蟻蠶補裝見觀規視覽覺觸計訂認講許論設訪證評訴診詞譯試詩誠"
    "話該詳語誤說請諸讀課調談謝謠譜貝財責賢賬貨質購貴費賀資賭賞賠"
    "賴贊贏趙車轉輪輕載轎較輔輛輩輸輯邊達遷過運還這進遠連遲適選遞"
    "邏鄧鄭醬釋鑒鐘鋼鑰錢鐵銅鋁銀銷鎖鍋鋒銳錯錫鑼錦鍵鋸鏡長閃閉問"
    "闖閒間悶閘鬧聞閣閱闊隊陽陰陣階際陸陳險隨隱難霧靜韓頁頂項順須"
    "顧頓領頸頻題顏額風飛飯飲館馬驅駛騎驗魚鮮鯨鳥雞鴨鵝鶴鷹黃齊齒"
    "齡龍龜"
)
_SHARED_VARIANT_EVIDENCE = _SIMPLIFIED_VARIANT_RAW & _TRADITIONAL_VARIANT_RAW
SIMPLIFIED_VARIANT_CHARS = _SIMPLIFIED_VARIANT_RAW - _SHARED_VARIANT_EVIDENCE
TRADITIONAL_VARIANT_CHARS = _TRADITIONAL_VARIANT_RAW - _SHARED_VARIANT_EVIDENCE


class EncodingIngestionError(ValueError):
    """Base class for fail-closed multilingual byte-decoding errors."""

    reason = "encoding_rejected"


class UnsupportedIngestionEncodingError(EncodingIngestionError):
    """Raised before codec lookup when an encoding is not allow-listed."""

    reason = "encoding_not_allowed"


class EncodingInputLimitError(EncodingIngestionError):
    """Raised before decoding when a byte input crosses the configured cap."""

    reason = "encoding_size_limit"


class EncodingConversionError(EncodingIngestionError):
    """Raised when strict decoding fails for an allow-listed encoding."""

    reason = "encoding_conversion"


class ConfusableIngestionWarning(UserWarning):
    """Warn that decoded text needs confusable or mixed-script review."""


@dataclass(frozen=True)
class DecodedIngestionText:
    """Strictly decoded text plus content-free script warning codes."""

    text: str
    encoding: str
    byte_length: int
    warning_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class DetectionNormalization:
    """Offset-preserving Unicode normalization for PII detection."""

    text: str
    original_length: int
    offset_starts: tuple[int, ...]
    offset_ends: tuple[int, ...]
    removed_zero_width: int = 0
    stripped_combining_marks: int = 0
    folded_confusables: int = 0
    folded_native_digits: int = 0
    indic_changes: int = 0
    indic_scripts: tuple[str, ...] = ()
    converted_legacy_bytes: int = 0
    legacy_encoding: str = "unicode"
    scripts: tuple[str, ...] = ()
    mixed_script: bool = False
    chinese_variant_normalized: bool = False
    chinese_target_script: str | None = None
    opencc_available: bool | None = None

    @property
    def changed(self) -> bool:
        """Return whether the normalized text differs structurally."""
        return (
            self.removed_zero_width > 0
            or self.stripped_combining_marks > 0
            or self.folded_confusables > 0
            or self.folded_native_digits > 0
            or self.indic_changes > 0
            or self.converted_legacy_bytes > 0
            or self.chinese_variant_normalized
        )

    def remap_span(self, start: int, end: int) -> tuple[int, int]:
        """Map normalized-text offsets back to original-text offsets."""
        safe_start = max(0, min(int(start), len(self.text)))
        safe_end = max(safe_start, min(int(end), len(self.text)))
        if not self.offset_starts:
            return 0, 0
        if safe_start >= len(self.offset_starts):
            original_start = self.original_length
        else:
            original_start = self.offset_starts[safe_start]
        if safe_end <= 0:
            original_end = original_start
        elif safe_end - 1 >= len(self.offset_ends):
            original_end = self.original_length
        else:
            original_end = self.offset_ends[safe_end - 1]
        return original_start, max(original_start, original_end)

    def to_metadata(self) -> dict[str, object]:
        """Return PHI-free normalization metadata."""
        return {
            "changed": self.changed,
            "chinese_target_script": self.chinese_target_script,
            "chinese_variant_normalized": self.chinese_variant_normalized,
            "folded_confusables": self.folded_confusables,
            "folded_native_digits": self.folded_native_digits,
            "indic_changes": self.indic_changes,
            "indic_scripts": list(self.indic_scripts),
            "converted_legacy_bytes": self.converted_legacy_bytes,
            "legacy_encoding": self.legacy_encoding,
            "mixed_script": self.mixed_script,
            "opencc_available": self.opencc_available,
            "removed_zero_width": self.removed_zero_width,
            "scripts": list(self.scripts),
            "stripped_combining_marks": self.stripped_combining_marks,
        }


@dataclass(frozen=True)
class ScriptDetectionWindow:
    """One offset-preserving inference window around a Unicode script run.

    ``start`` and ``end`` delimit the context-bearing text sent to a detector,
    while ``core_start`` and ``core_end`` retain the exact run that caused the
    route to be selected. The source text is deliberately not stored.
    """

    start: int
    end: int
    core_start: int
    core_end: int
    script: str

    def extract(self, text: str) -> str:
        """Return this window's exact slice from ``text``."""

        return text[self.start : self.end]


_SCRIPT_RANGES: tuple[tuple[str, tuple[tuple[int, int], ...]], ...] = (
    (
        "Latin",
        (
            (0x0041, 0x005A),
            (0x0061, 0x007A),
            (0x00C0, 0x00FF),
            (0x0100, 0x017F),
            (0x0180, 0x024F),
            # IPA Extensions contains lowercase Hausa hooked consonants
            # U+0253 (ɓ) and U+0257 (ɗ); their uppercase forms live in
            # Latin Extended-B above.
            (0x0250, 0x02AF),
            (0x1E00, 0x1EFF),
            (0x2C60, 0x2C7F),
            (0xA720, 0xA7FF),
            (0xAB30, 0xAB6F),
            (0xFF21, 0xFF3A),
            (0xFF41, 0xFF5A),
        ),
    ),
    (
        "Arabic",
        (
            (0x0600, 0x06FF),
            (0x0750, 0x077F),
            (0x08A0, 0x08FF),
            (0xFB50, 0xFDFF),
            (0xFE70, 0xFEFF),
        ),
    ),
    (
        "Ethiopic",
        (
            (0x1200, 0x137F),
            (0x1380, 0x139F),
            (0x2D80, 0x2DDF),
            (0xAB00, 0xAB2F),
            (0x1E7E0, 0x1E7FF),
        ),
    ),
    (
        "Han",
        (
            (0x3400, 0x4DBF),
            (0x4E00, 0x9FFF),
            (0xF900, 0xFAFF),
            (0x20000, 0x2A6DF),
            (0x2A700, 0x2B73F),
            (0x2B740, 0x2B81F),
            (0x2B820, 0x2CEAF),
            (0x2CEB0, 0x2EBEF),
            (0x30000, 0x3134F),
            (0x31350, 0x323AF),
        ),
    ),
    (
        "Hiragana/Katakana",
        (
            (0x3040, 0x309F),
            (0x30A0, 0x30FF),
            (0x31F0, 0x31FF),
            (0x1B000, 0x1B16F),
            (0xFF65, 0xFF9F),
        ),
    ),
    (
        "Hangul",
        (
            (0x1100, 0x11FF),
            (0x3130, 0x318F),
            (0xA960, 0xA97F),
            (0xAC00, 0xD7AF),
            (0xD7B0, 0xD7FF),
        ),
    ),
    (
        "Cyrillic",
        (
            (0x0400, 0x04FF),
            (0x0500, 0x052F),
            (0x1C80, 0x1C8F),
            (0x2DE0, 0x2DFF),
            (0xA640, 0xA69F),
        ),
    ),
    (
        "Devanagari",
        (
            (0x0900, 0x097F),
            (0xA8E0, 0xA8FF),
            (0x11B00, 0x11B5F),
        ),
    ),
    ("Bengali", ((0x0980, 0x09FF),)),
    ("Gurmukhi", ((0x0A00, 0x0A7F),)),
    ("Gujarati", ((0x0A80, 0x0AFF),)),
    ("Odia", ((0x0B00, 0x0B7F),)),
    ("Tamil", ((0x0B80, 0x0BFF),)),
    ("Telugu", ((0x0C00, 0x0C7F),)),
    ("Kannada", ((0x0C80, 0x0CFF),)),
    ("Malayalam", ((0x0D00, 0x0D7F),)),
    (
        "Greek",
        (
            (0x0370, 0x03FF),
            (0x1F00, 0x1FFF),
        ),
    ),
    (
        "Hebrew",
        (
            (0x0590, 0x05FF),
            (0xFB1D, 0xFB4F),
        ),
    ),
    ("Thai", ((0x0E00, 0x0E7F),)),
)


def indian_name_script(text: str, lang: str = "hi") -> str | None:
    """Return the rendering script for an Indian name surface, if in scope.

    Native Devanagari and Tamil surfaces are unambiguous. Latin surfaces opt in
    through a Hindi or Tamil language hint so unrelated Latin names retain their
    existing exact vault key behavior.
    """

    script = detect_script(text)
    if script in INDIAN_NAME_SCRIPTS:
        return script
    base_lang = str(lang or "").strip().replace("-", "_").split("_", 1)[0]
    if script == "Latin" and base_lang.casefold() in INDIAN_NAME_LANGUAGES:
        return script
    return None


def canonical_indian_name(text: str) -> str:
    """Fold a Devanagari, Tamil, or Romanized name to one phonetic key.

    This is a deterministic transliteration fold, not fuzzy entity resolution.
    It handles common long-vowel Roman variants and Indic consonant spellings
    while retaining the rest of each name, so similar but distinct names do not
    merge merely because they share a prefix.
    """

    script = detect_script(text)
    if script == "Devanagari":
        transliterated = _indic_to_latin(
            text,
            consonants=_DEVANAGARI_CONSONANTS,
            vowels=_DEVANAGARI_VOWELS,
            vowel_signs=_DEVANAGARI_VOWEL_SIGNS,
            virama="्",
            nasal_marks=frozenset({"ं", "ँ"}),
            ignored_marks=frozenset({"़"}),
        )
    elif script == "Tamil":
        transliterated = _indic_to_latin(
            text,
            consonants=_TAMIL_CONSONANTS,
            vowels=_TAMIL_VOWELS,
            vowel_signs=_TAMIL_VOWEL_SIGNS,
            virama="்",
            nasal_marks=frozenset(),
            ignored_marks=frozenset(),
        )
    else:
        words = []
        for word in text.split():
            if word.endswith("a") and any(not char.isascii() for char in word):
                word = word[:-1]
            words.append(word)
        transliterated = " ".join(words)
    return _fold_indian_romanization(transliterated)


def render_indian_name(canonical_name: str, script: str) -> str:
    """Render one canonical Indian surrogate identity in ``script``."""

    if script == "Devanagari":
        return _latin_to_indic(
            canonical_name,
            consonants=_DEVANAGARI_RENDER_CONSONANTS,
            vowels=_DEVANAGARI_RENDER_VOWELS,
            virama="्",
        )
    if script == "Tamil":
        return _latin_to_indic(
            canonical_name,
            consonants=_TAMIL_RENDER_CONSONANTS,
            vowels=_TAMIL_RENDER_VOWELS,
            virama="்",
        )
    return " ".join(part.capitalize() for part in canonical_name.split())


def _indic_to_latin(
    text: str,
    *,
    consonants: dict[str, str],
    vowels: dict[str, str],
    vowel_signs: dict[str, str],
    virama: str,
    nasal_marks: frozenset[str],
    ignored_marks: frozenset[str],
) -> str:
    output: list[str] = []
    index = 0
    while index < len(text):
        char = text[index]
        consonant = consonants.get(char)
        if consonant is not None:
            following = text[index + 1] if index + 1 < len(text) else ""
            if following == virama:
                output.append(consonant)
                index += 2
                continue
            vowel_sign = vowel_signs.get(following)
            if vowel_sign is not None:
                output.append(consonant + vowel_sign)
                index += 2
                continue
            output.append(consonant + "a")
        elif char in vowels:
            output.append(vowels[char])
        elif char in nasal_marks:
            output.append("n")
        elif char in ignored_marks or char == virama:
            pass
        elif char.isascii() or char.isspace():
            output.append(char)
        index += 1

    words = "".join(output).split()
    return " ".join(word[:-1] if word.endswith("a") else word for word in words)


def _fold_indian_romanization(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text).casefold()
    ascii_letters = "".join(
        char
        for char in decomposed
        if not unicodedata.combining(char) and (char.isascii() or char.isspace())
    )
    folded = re.sub(r"[^a-z]+", " ", ascii_letters).strip()
    folded = folded.replace("sh", "s")
    for source, replacement in _PHONETIC_VOWEL_FOLDS:
        folded = folded.replace(source, replacement)
    return " ".join(folded.split())


def _latin_to_indic(
    text: str,
    *,
    consonants: dict[str, str],
    vowels: dict[str, tuple[str, str]],
    virama: str,
) -> str:
    consonant_tokens = sorted(consonants, key=len, reverse=True)
    vowel_tokens = sorted(vowels, key=len, reverse=True)
    output: list[str] = []
    index = 0
    previous_was_consonant = False

    while index < len(text):
        char = text[index]
        if not char.isalpha():
            if previous_was_consonant:
                output.append(virama)
            output.append(char)
            previous_was_consonant = False
            index += 1
            continue

        consonant = next(
            (token for token in consonant_tokens if text.startswith(token, index)),
            None,
        )
        if consonant is not None:
            if previous_was_consonant:
                output.append(virama)
            output.append(consonants[consonant])
            previous_was_consonant = True
            index += len(consonant)
            continue

        vowel = next(
            (token for token in vowel_tokens if text.startswith(token, index)),
            None,
        )
        if vowel is not None:
            independent, sign = vowels[vowel]
            output.append(sign if previous_was_consonant else independent)
            previous_was_consonant = False
            index += len(vowel)
            continue

        output.append(char)
        previous_was_consonant = False
        index += 1

    if previous_was_consonant:
        output.append(virama)
    return "".join(output)


def detect_script(text: str) -> str:
    """Return the dominant Unicode script in ``text``.

    Neutral characters such as whitespace, punctuation, symbols, and digits do
    not affect the decision. If no supported script-bearing code point is
    present, ``"Unknown"`` is returned.
    """

    counts: dict[str, int] = {}
    first_seen: dict[str, int] = {}

    for index, char in enumerate(text):
        script = _script_for_char(char)
        if script is None:
            continue
        counts[script] = counts.get(script, 0) + 1
        first_seen.setdefault(script, index)

    if not counts:
        return UNKNOWN_SCRIPT

    return max(counts, key=lambda script: (counts[script], -first_seen[script]))


def is_han_dominant(text: str) -> bool:
    """Return whether Han is the dominant supported script in ``text``."""

    counts = _script_counts(text)
    han_count = counts.get("Han", 0)
    return han_count > 0 and all(
        han_count > count for script, count in counts.items() if script != "Han"
    )


def detect_chinese_script(text: str) -> ChineseScriptEstimate:
    """Estimate the predominant Simplified/Traditional Chinese variant.

    Unicode blocks cannot distinguish the variants, so the estimate uses
    ratios over common characters that are exclusive to one form. Characters
    shared by both forms are ignored. A tie with evidence on both sides is
    reported as mixed; otherwise the larger evidence ratio is predominant.
    """

    simplified_count = sum(char in SIMPLIFIED_VARIANT_CHARS for char in text)
    traditional_count = sum(char in TRADITIONAL_VARIANT_CHARS for char in text)
    evidence_count = simplified_count + traditional_count
    if evidence_count == 0:
        variant = ChineseScriptVariant.UNKNOWN
        simplified_ratio = 0.0
        traditional_ratio = 0.0
    else:
        simplified_ratio = simplified_count / evidence_count
        traditional_ratio = traditional_count / evidence_count
        if simplified_count == traditional_count:
            variant = ChineseScriptVariant.MIXED
        elif simplified_count > traditional_count:
            variant = ChineseScriptVariant.SIMPLIFIED
        else:
            variant = ChineseScriptVariant.TRADITIONAL

    return ChineseScriptEstimate(
        variant=variant,
        simplified_count=simplified_count,
        traditional_count=traditional_count,
        simplified_ratio=simplified_ratio,
        traditional_ratio=traditional_ratio,
    )


def segment_by_script(text: str) -> Iterator[tuple[int, int, str]]:
    """Yield contiguous ``(start, end, script)`` runs covering ``text``.

    Neutral characters are assigned to the surrounding run: leading neutral
    characters attach to the first detected script, and later neutral characters
    attach to the preceding script. This keeps offsets exact while avoiding
    stand-alone whitespace or punctuation runs.
    """

    if not text:
        return

    run_start = 0
    current_script: str | None = None

    for index, char in enumerate(text):
        script = _script_for_char(char)
        if script is None:
            continue
        if current_script is None:
            current_script = script
            continue
        if script == current_script:
            continue

        yield run_start, index, current_script
        run_start = index
        current_script = script

    if current_script is None:
        yield 0, len(text), UNKNOWN_SCRIPT
        return

    yield run_start, len(text), current_script


def candidate_languages_for_script(script: str) -> tuple[str, ...]:
    """Return candidate language codes for a detected script."""

    return SCRIPT_LANGUAGE_HINTS.get(script, SCRIPT_LANGUAGE_HINTS[UNKNOWN_SCRIPT])


_ASSAMESE_EXCLUSIVE_LETTERS = frozenset({"\u09f0", "\u09f1"})
_ASSAMESE_MONTH_CUES = (
    "জানুৱাৰী",
    "ফেব্ৰুৱাৰী",
    "মাৰ্চ",
    "এপ্ৰিল",
    "আগষ্ট",
    "ছেপ্টেম্বৰ",
    "অক্টোবৰ",
    "নৱেম্বৰ",
    "ডিচেম্বৰ",
)


def assamese_language_evidence(text: str) -> int:
    """Return deterministic Assamese evidence in Bengali-script text.

    The score counts the Assamese ra/wa letters (U+09F0/U+09F1) and gives
    additional weight to Assamese month spellings. It intentionally does not
    treat shared Bengali-Assamese block membership as language evidence.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    ra_wa_count = sum(char in _ASSAMESE_EXCLUSIVE_LETTERS for char in text)
    month_count = sum(text.count(month) for month in _ASSAMESE_MONTH_CUES)
    return ra_wa_count + (2 * month_count)


def candidate_languages_for_text(
    text: str,
    script: str | None = None,
) -> tuple[str, ...]:
    """Return script candidates reordered by deterministic lexical evidence.

    Bengali and Assamese share a Unicode block. Assamese-exclusive ra/wa
    letters and Assamese month spellings move ``"as"`` ahead of ``"bn"``;
    without that evidence the catalog's established Bengali-first order is
    preserved.
    """

    detected_script = detect_script(text) if script is None else script
    candidates = candidate_languages_for_script(detected_script)
    if (
        detected_script != "Bengali"
        or "as" not in candidates
        or "bn" not in candidates
        or assamese_language_evidence(text) == 0
    ):
        return candidates
    return ("as", *(code for code in candidates if code != "as"))


def confusable_skeleton(text: str) -> str:
    """Return a curated UTS #39-style skeleton for PII matching.

    The mapper is deliberately narrower than general-purpose Unicode
    normalization: it folds the supported cross-script lookalikes and ASCII
    full-width forms, and removes the invisible controls used by the evasion
    generator. It does not case-fold or strip diacritics.
    """

    output: list[str] = []
    for char in text:
        if char in ZERO_WIDTH_CHARS:
            continue
        output.append(_fold_confusable_char(char))
    return "".join(output)


def mixed_script_spans(text: str) -> tuple[MixedScriptSpan, ...]:
    """Return identifier-like spans that mix Unicode scripts.

    Script changes separated by whitespace or ordinary punctuation are normal
    multilingual prose and are not flagged. Invisible controls stay attached
    to the surrounding token so an inserted zero-width character cannot split
    an otherwise suspicious identifier.
    """

    findings: list[MixedScriptSpan] = []
    token_start: int | None = None
    for index in range(len(text) + 1):
        char = text[index] if index < len(text) else ""
        if char and _is_identifier_char(char):
            if token_start is None:
                token_start = index
            continue
        if token_start is None:
            continue

        token = text[token_start:index]
        scripts = tuple(sorted(_script_counts(token)))
        if len(scripts) > 1:
            findings.append(
                MixedScriptSpan(
                    start=token_start,
                    end=index,
                    scripts=scripts,
                    confusable_count=sum(
                        _fold_confusable_char(item) != item for item in token
                    ),
                    invisible_count=sum(item in ZERO_WIDTH_CHARS for item in token),
                )
            )
        token_start = None
    return tuple(findings)


def detect_mixed_script(text: str) -> bool:
    """Return whether an identifier-like span mixes Unicode scripts."""

    return bool(mixed_script_spans(text))


def india_clinical_script_windows(
    text: str,
    lang: str,
    *,
    context_chars: int = 64,
) -> tuple[ScriptDetectionWindow, ...]:
    """Return context-bearing Latin/Indic windows for Indian clinical NER.

    The route activates only when Latin and an Indic script occur in the same
    note. Hindi accepts Devanagari, while Telugu accepts Telugu or Devanagari
    because Indian-English clinical notes can embed Hindi phrases even when
    ``lang="te"`` is the closest configured language. Context is expanded on
    both sides of each run and snapped to token boundaries so PERSON and
    LOCATION spans are not truncated merely because the script changes.

    Args:
        text: Source note whose offsets the returned windows reference.
        lang: OpenMed language code. Only ``"hi"`` and ``"te"`` activate the
            India clinical route.
        context_chars: Maximum context expansion on either side before token
            boundary adjustment.

    Returns:
        Deduplicated inference windows in source order, or an empty tuple when
        the text is not an eligible mixed-script Indian clinical note.
    """

    if lang not in {"hi", "te"} or not text:
        return ()
    if context_chars < 0:
        raise ValueError("context_chars must be non-negative")

    target_scripts = {"Devanagari"}
    if lang == "te":
        target_scripts.add("Telugu")

    runs = list(segment_by_script(text))
    scripts = {script for _start, _end, script in runs}
    if "Latin" not in scripts or not (scripts & target_scripts):
        return ()

    windows: list[ScriptDetectionWindow] = []
    seen: set[tuple[int, int, str]] = set()
    for core_start, core_end, script in runs:
        if script != "Latin" and script not in target_scripts:
            continue
        start, end = _expand_detection_window(
            text,
            core_start,
            core_end,
            context_chars=context_chars,
        )
        key = (start, end, script)
        if key in seen:
            continue
        seen.add(key)
        windows.append(
            ScriptDetectionWindow(
                start=start,
                end=end,
                core_start=core_start,
                core_end=core_end,
                script=script,
            )
        )
    return tuple(windows)


def normalize_for_pii_detection(
    text: str,
    *,
    width_convention: str = "cjk",
    chinese_target_script: str | None = None,
    legacy_font_map: LegacyFontMap | None = None,
) -> DetectionNormalization:
    """Convert legacy Indic text, then apply offset-safe Unicode defenses.

    Likely ISCII-1991 input and caller-mapped legacy-font runs are converted
    before script routing. The resulting offsets are composed back to the
    original source-byte coordinate system; ordinary Unicode text follows the
    existing normalization path unchanged.
    """

    from ..processing.legacy_encoding import convert_legacy_encoding

    legacy_conversion = convert_legacy_encoding(
        text,
        legacy_font_map=legacy_font_map,
    )
    normalized = _normalize_unicode_for_pii_detection(
        legacy_conversion.text,
        width_convention=width_convention,
        chinese_target_script=chinese_target_script,
    )
    if legacy_conversion.encoding == "unicode":
        return normalized

    legacy_origins = legacy_conversion.offset_map.converted_to_original_spans
    starts: list[int] = []
    ends: list[int] = []
    for start, end in zip(normalized.offset_starts, normalized.offset_ends):
        source_start, source_end = _source_span_for_legacy_range(
            legacy_origins,
            start,
            end,
        )
        starts.append(source_start)
        ends.append(source_end)

    converted_sources: set[int] = set()
    converted_by_span: dict[tuple[int, int], list[str]] = {}
    for char, span in zip(legacy_conversion.text, legacy_origins):
        converted_by_span.setdefault(span, []).append(char)
    for (start, end), converted_chars in converted_by_span.items():
        if text[start:end] != "".join(converted_chars):
            converted_sources.update(range(start, end))

    return DetectionNormalization(
        text=normalized.text,
        original_length=len(text),
        offset_starts=tuple(starts),
        offset_ends=tuple(ends),
        removed_zero_width=normalized.removed_zero_width,
        stripped_combining_marks=normalized.stripped_combining_marks,
        folded_confusables=normalized.folded_confusables,
        folded_native_digits=normalized.folded_native_digits,
        indic_changes=normalized.indic_changes,
        indic_scripts=normalized.indic_scripts,
        converted_legacy_bytes=len(converted_sources),
        legacy_encoding=legacy_conversion.encoding,
        scripts=normalized.scripts,
        mixed_script=normalized.mixed_script,
        chinese_variant_normalized=normalized.chinese_variant_normalized,
        chinese_target_script=normalized.chinese_target_script,
        opencc_available=normalized.opencc_available,
    )


def _source_span_for_legacy_range(
    origins: tuple[tuple[int, int], ...],
    start: int,
    end: int,
) -> tuple[int, int]:
    spans = origins[start:end]
    if spans:
        return min(span[0] for span in spans), max(span[1] for span in spans)
    if start < len(origins):
        anchor = origins[start][0]
    elif origins:
        anchor = origins[-1][1]
    else:
        anchor = 0
    return anchor, anchor


def _normalize_unicode_for_pii_detection(
    text: str,
    *,
    width_convention: str = "cjk",
    chinese_target_script: str | None = None,
) -> DetectionNormalization:
    """Fold adversarial Unicode artifacts while preserving offset remapping.

    Indic script runs first receive script-specific NFC canonicalization. The
    defense then strips zero-width controls and standalone non-Indic combining
    marks, while retaining Ethiopic marks attached to a preceding Ethiopic
    grapheme. It folds common Latin-lookalike Greek/Cyrillic/full-width
    characters and Indic decimal digits, and records a script-consistency
    summary without storing source text. ``width_convention`` selects the
    CJK-safe width fold or strict per-character NFKC normalization.
    ``chinese_target_script`` optionally canonicalizes Han variants with OpenCC
    after the Unicode defenses and composes its alignment into the source map.
    """

    # Local imports keep the lightweight script helpers from importing the
    # broader processing package during module initialization.
    from ..processing.text import INDIC_SCRIPTS, IndicNormalizer, fold_indic_digits
    from ..processing.zh_normalize import normalize_chinese_variants, normalize_width

    scripts = tuple(sorted(_script_counts(text)))
    mixed_script = detect_mixed_script(text)
    indic_normalizer = IndicNormalizer()
    routed_chars: list[str] = []
    routed_starts: list[int] = []
    routed_ends: list[int] = []
    indic_changes = 0
    indic_scripts: list[str] = []
    removed_zero_width = 0

    for run_start, run_end, script in segment_by_script(text):
        run = text[run_start:run_end]
        if script in INDIC_SCRIPTS:
            normalized = indic_normalizer.normalize_with_offsets(run, script=script)
            routed_chars.extend(normalized.text)
            routed_starts.extend(
                run_start + offset for offset in normalized.offset_starts
            )
            routed_ends.extend(run_start + offset for offset in normalized.offset_ends)
            indic_changes += normalized.changes
            removed_zero_width += normalized.removed_joiners
            if script not in indic_scripts:
                indic_scripts.append(script)
            continue

        routed_chars.extend(run)
        routed_starts.extend(range(run_start, run_end))
        routed_ends.extend(range(run_start + 1, run_end + 1))

    routed_text = "".join(routed_chars)
    width_normalization = normalize_width(
        routed_text,
        convention=width_convention,
    )
    digit_folding = fold_indic_digits(width_normalization.text)
    output: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    stripped_combining_marks = 0
    normalized_by_routed_source: list[list[str]] = [[] for _ in routed_text]
    for char, (routed_start, _routed_end) in zip(
        width_normalization.text,
        width_normalization.char_origins,
    ):
        normalized_by_routed_source[routed_start].append(char)
    changed_source_indices = {
        routed_starts[index]
        for index, (char, normalized_chars) in enumerate(
            zip(routed_text, normalized_by_routed_source)
        )
        if "".join(normalized_chars) != char
    }
    folded_native_digit_sources = {
        routed_starts[width_normalization.char_origins[index][0]]
        for index, (width_char, folded_char) in enumerate(
            zip(width_normalization.text, digit_folding.text)
        )
        if width_char != folded_char
    }
    cluster_starts, cluster_ends = _base_mark_cluster_maps(text)

    for index, char in enumerate(digit_folding.text):
        routed_start, routed_end = width_normalization.char_origins[index]
        original_start = routed_starts[routed_start]
        original_end = routed_ends[routed_end - 1]
        if char in ZERO_WIDTH_CHARS:
            removed_zero_width += 1
            continue
        category = unicodedata.category(char)
        attached_ethiopic_mark = (
            category == "Mn"
            and _script_for_char(char) == "Ethiopic"
            and original_start > 0
            and _script_for_char(text[original_start - 1]) == "Ethiopic"
        )
        attached_indic_mark = category == "Mn" and _is_attached_indic_mark(
            digit_folding.text,
            index,
        )
        if category == "Mn" and not attached_indic_mark and not attached_ethiopic_mark:
            stripped_combining_marks += 1
            continue

        replacement = _fold_confusable_char(char)
        if replacement != char:
            changed_source_indices.add(original_start)
        for replacement_char in replacement:
            output.append(replacement_char)
            starts.append(cluster_starts[original_start])
            ends.append(cluster_ends[original_end - 1])

    normalized_text = "".join(output)
    chinese_variant_normalized = False
    opencc_available: bool | None = None
    if chinese_target_script is not None:
        conversion = normalize_chinese_variants(
            normalized_text,
            chinese_target_script,
        )
        converted_starts: list[int] = []
        converted_ends: list[int] = []
        for converted_start, converted_end in conversion.char_origins:
            if converted_start < converted_end:
                source_starts = starts[converted_start:converted_end]
                source_ends = ends[converted_start:converted_end]
                converted_starts.append(min(source_starts))
                converted_ends.append(max(source_ends))
            else:
                anchor = (
                    starts[converted_start]
                    if converted_start < len(starts)
                    else len(text)
                )
                converted_starts.append(anchor)
                converted_ends.append(anchor)
        normalized_text = conversion.text
        starts = converted_starts
        ends = converted_ends
        chinese_variant_normalized = conversion.changed
        opencc_available = conversion.opencc_available

    return DetectionNormalization(
        text=normalized_text,
        original_length=len(text),
        offset_starts=tuple(starts),
        offset_ends=tuple(ends),
        removed_zero_width=removed_zero_width,
        stripped_combining_marks=stripped_combining_marks,
        folded_confusables=len(changed_source_indices),
        folded_native_digits=len(folded_native_digit_sources),
        indic_changes=indic_changes,
        indic_scripts=tuple(indic_scripts),
        scripts=scripts,
        mixed_script=mixed_script,
        chinese_variant_normalized=chinese_variant_normalized,
        chinese_target_script=chinese_target_script,
        opencc_available=opencc_available,
    )


def decode_ingestion_bytes(
    data: bytes | bytearray | memoryview,
    *,
    encoding: str,
    source_path: str | Path | None = None,
    max_bytes: int = MAX_INGESTION_BYTES,
    warn_on_confusables: bool = True,
) -> DecodedIngestionText:
    """Decode untrusted multilingual bytes using an explicit codec allow-list.

    Codec names are normalized through a static alias table before Python codec
    lookup, so a caller cannot activate a dynamically registered codec. Decoding
    is strict and bounded. Mixed-script or folded-confusable text produces only
    machine-readable warning codes; no source text or raw path is logged.

    Args:
        data: Bytes to decode.
        encoding: Explicit allow-listed encoding name. Ambiguous or executable
            codecs such as ``utf-7`` and BOM-selected ``utf-16`` are rejected.
        source_path: Optional source path used only to derive a SHA-256 log key.
        max_bytes: Positive input-size cap no greater than
            :data:`MAX_INGESTION_BYTES`, applied before materialization and
            decoding.
        warn_on_confusables: Emit :class:`ConfusableIngestionWarning` when the
            decoded text is mixed-script or contains folded confusables.

    Returns:
        Strictly decoded text and content-free warning metadata.

    Raises:
        EncodingIngestionError: If the codec, size, or byte sequence is invalid.
    """

    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("data must be a bytes-like object")
    if not isinstance(encoding, str):
        raise TypeError("encoding must be text")
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int):
        raise TypeError("max_bytes must be an integer")
    if not 0 < max_bytes <= MAX_INGESTION_BYTES:
        raise ValueError(f"max_bytes must be between 1 and {MAX_INGESTION_BYTES}")

    path_hash = _ingestion_path_hash(source_path)
    payload_size = data.nbytes if isinstance(data, memoryview) else len(data)
    if payload_size > max_bytes:
        _reject_encoding(
            EncodingInputLimitError("Encoded input exceeds the configured limit"),
            path_hash=path_hash,
            size_bytes=payload_size,
        )
    payload = _materialize_ingestion_bytes(data)

    normalized_name = encoding.strip().casefold().replace("_", "-")
    codec_name = _INGESTION_ENCODING_ALIASES.get(normalized_name)
    if codec_name is None:
        _reject_encoding(
            UnsupportedIngestionEncodingError("Encoding is not allow-listed"),
            path_hash=path_hash,
            size_bytes=len(payload),
        )

    try:
        text = payload.decode(codec_name, errors="strict")
    except (LookupError, UnicodeDecodeError):
        _reject_encoding(
            EncodingConversionError("Input cannot be decoded strictly"),
            path_hash=path_hash,
            size_bytes=len(payload),
        )

    warning_codes = list(_ingestion_warning_codes(text))
    if warning_codes and warn_on_confusables:
        warnings.warn(
            "decoded ingestion requires mixed-script/confusable review; "
            f"warning_codes={','.join(warning_codes)}",
            ConfusableIngestionWarning,
            stacklevel=2,
        )

    return DecodedIngestionText(
        text=text,
        encoding=codec_name,
        byte_length=len(payload),
        warning_codes=tuple(warning_codes),
    )


def _materialize_ingestion_bytes(
    data: bytes | bytearray | memoryview,
) -> bytes:
    """Return immutable bytes after the caller has enforced the byte budget."""

    return data if isinstance(data, bytes) else bytes(data)


def _ingestion_warning_codes(text: str) -> tuple[str, ...]:
    """Return content-free warnings without building normalization offset maps."""

    warning_codes: list[str] = []
    if _contains_mixed_script_identifier(text):
        warning_codes.append("mixed_script")
    if any(char == "\u3000" or _fold_confusable_char(char) != char for char in text):
        warning_codes.append("confusable_characters")
    return tuple(warning_codes)


def _contains_mixed_script_identifier(text: str) -> bool:
    """Detect mixed-script identifier runs in one pass and constant space."""

    first_script: str | None = None
    mixed = False
    for char in text:
        if not _is_identifier_char(char):
            if mixed:
                return True
            first_script = None
            mixed = False
            continue
        script = _script_for_char(char)
        if script is None:
            continue
        if first_script is None:
            first_script = script
        elif script != first_script:
            mixed = True
    return mixed


def decode_legacy_text(
    data: bytes | bytearray | memoryview,
    *,
    encoding: str,
    source_path: str | Path | None = None,
    max_bytes: int = MAX_INGESTION_BYTES,
    warn_on_confusables: bool = True,
) -> str:
    """Return text from :func:`decode_ingestion_bytes` for converter adapters."""

    return decode_ingestion_bytes(
        data,
        encoding=encoding,
        source_path=source_path,
        max_bytes=max_bytes,
        warn_on_confusables=warn_on_confusables,
    ).text


def _ingestion_path_hash(path: str | Path | None) -> str:
    if path is None:
        normalized = "<memory>"
    else:
        candidate = Path(path).expanduser()
        try:
            normalized = str(candidate.resolve(strict=False))
        except OSError:
            normalized = str(candidate.absolute())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _reject_encoding(
    error: EncodingIngestionError,
    *,
    path_hash: str,
    size_bytes: int,
) -> NoReturn:
    logger.warning(
        "encoding_ingestion_rejected path_hash=%s size_bytes=%d "
        "entry_count=0 reason=%s",
        path_hash,
        size_bytes,
        error.reason,
    )
    raise error from None


def _base_mark_cluster_maps(text: str) -> tuple[list[int], list[int]]:
    """Return containing base-plus-mark bounds for every source code point."""

    starts = [0] * len(text)
    ends = [0] * len(text)
    cluster_start = 0

    for index, char in enumerate(text):
        if index > 0 and not unicodedata.category(char).startswith("M"):
            for cluster_index in range(cluster_start, index):
                ends[cluster_index] = index
            cluster_start = index
        starts[index] = cluster_start

    for cluster_index in range(cluster_start, len(text)):
        ends[cluster_index] = len(text)
    return starts, ends


def _script_for_char(char: str) -> str | None:
    codepoint = ord(char)
    if codepoint == 0x3007:
        return "Han"
    # Python 3.10's Unicode 13 database predates Ethiopic Extended-B. Route the
    # explicit Unicode block independently of ``unicodedata.category`` so the
    # same text is detected consistently across supported Python versions.
    if 0x1E7E0 <= codepoint <= 0x1E7FF:
        return "Ethiopic"

    category = unicodedata.category(char)
    if category[0] not in {"L", "M"}:
        return None

    for script, ranges in _SCRIPT_RANGES:
        if any(start <= codepoint <= end for start, end in ranges):
            return script
    return None


def _is_attached_indic_mark(text: str, index: int) -> bool:
    char = text[index]
    if not _is_indic_codepoint(char) or index == 0:
        return False
    previous = text[index - 1]
    return _is_indic_codepoint(previous) and unicodedata.category(previous)[0] in {
        "L",
        "M",
    }


def _is_indic_codepoint(char: str) -> bool:
    codepoint = ord(char)
    return any(
        start <= codepoint <= end
        for start, end in (
            (0x0900, 0x097F),
            (0x0980, 0x09FF),
            (0x0A00, 0x0A7F),
            (0x0A80, 0x0AFF),
            (0x0B00, 0x0B7F),
            (0x0B80, 0x0BFF),
            (0x0C00, 0x0C7F),
            (0x0C80, 0x0CFF),
            (0x0D00, 0x0D7F),
        )
    )


def _expand_detection_window(
    text: str,
    core_start: int,
    core_end: int,
    *,
    context_chars: int,
) -> tuple[int, int]:
    """Expand one script run without cutting through adjacent tokens."""

    start = max(0, core_start - context_chars)
    end = min(len(text), core_end + context_chars)

    while start > 0 and not text[start - 1].isspace():
        start -= 1
    while end < len(text) and not text[end].isspace():
        end += 1
    return start, end


def _script_counts(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for char in text:
        script = _script_for_char(char)
        if script is None:
            continue
        counts[script] = counts.get(script, 0) + 1
    return counts


def _fold_confusable_char(char: str) -> str:
    folded = _CONFUSABLE_FOLD.get(char)
    if folded is not None:
        return folded

    codepoint = ord(char)
    if 0xFF01 <= codepoint <= 0xFF5E:
        return chr(codepoint - 0xFEE0)

    return char


def _is_identifier_char(char: str) -> bool:
    if char in ZERO_WIDTH_CHARS:
        return True
    return unicodedata.category(char)[0] in {"L", "M", "N"}


__all__ = [
    "ALLOWED_INGESTION_ENCODINGS",
    "CJK_SCRIPTS",
    "CONFUSABLE_DATA_LICENSE",
    "CONFUSABLE_DATA_URL",
    "CONFUSABLE_DATA_VERSION",
    "ChineseScriptEstimate",
    "ChineseScriptVariant",
    "ConfusableIngestionWarning",
    "DecodedIngestionText",
    "DetectionNormalization",
    "EncodingConversionError",
    "EncodingIngestionError",
    "EncodingInputLimitError",
    "INDIC_SCRIPTS",
    "INDIAN_NAME_LANGUAGES",
    "INDIAN_NAME_SCRIPTS",
    "MixedScriptSpan",
    "MAX_INGESTION_BYTES",
    "ScriptDetectionWindow",
    "SCRIPT_LANGUAGE_HINTS",
    "SIMPLIFIED_VARIANT_CHARS",
    "SUPPORTED_SCRIPTS",
    "TRADITIONAL_VARIANT_CHARS",
    "UNKNOWN_SCRIPT",
    "UnsupportedIngestionEncodingError",
    "ZERO_WIDTH_CHARS",
    "assamese_language_evidence",
    "canonical_indian_name",
    "candidate_languages_for_script",
    "candidate_languages_for_text",
    "confusable_skeleton",
    "detect_chinese_script",
    "detect_mixed_script",
    "detect_script",
    "decode_ingestion_bytes",
    "decode_legacy_text",
    "india_clinical_script_windows",
    "indian_name_script",
    "is_han_dominant",
    "mixed_script_spans",
    "normalize_for_pii_detection",
    "render_indian_name",
    "segment_by_script",
]
