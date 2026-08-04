"""Fabricated multilingual notes used by the de-identification demo.

Every person, organization, address, and identifier in this module is synthetic.
The values are deliberately reserved for examples and do not describe a real
patient or clinician.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SyntheticSample:
    """One fabricated note and the identifiers it contains."""

    text: str
    identifiers: tuple[tuple[str, str], ...]


SAMPLES = {
    "zh": SyntheticSample(
        text=(
            "【完全虚构的合成示例】患者李晓雯，病历号 ZH-709-0001，出生日期 "
            "1990年2月3日。联系电话 +86 10 5555 0709，电子邮箱 "
            "li.xiaowen@example.test，住址北京市海淀区示例路709号。陈志远医生"
            "建议一周后复诊。"
        ),
        identifiers=(
            ("李晓雯", "PERSON"),
            ("陈志远", "PERSON"),
            ("ZH-709-0001", "ID_NUM"),
            ("1990年2月3日", "DATE"),
            ("+86 10 5555 0709", "PHONE"),
            ("li.xiaowen@example.test", "EMAIL"),
            ("北京市海淀区示例路709号", "ADDRESS"),
        ),
    ),
    "hi": SyntheticSample(
        text=(
            "【पूरी तरह काल्पनिक सिंथेटिक उदाहरण】रोगी अनन्या मेहता, मेडिकल रिकॉर्ड "
            "HI-709-0001, जन्म तिथि 3 फ़रवरी 1990। फोन +91 98765 40709, ईमेल "
            "ananya@example.test। डॉ. रोहन कपूर ने एक सप्ताह बाद जाँच की सलाह दी।"
        ),
        identifiers=(
            ("अनन्या मेहता", "PERSON"),
            ("डॉ. रोहन कपूर", "PERSON"),
            ("HI-709-0001", "ID_NUM"),
            ("3 फ़रवरी 1990", "DATE"),
            ("+91 98765 40709", "PHONE"),
            ("ananya@example.test", "EMAIL"),
        ),
    ),
    "en": SyntheticSample(
        text=(
            "Synthetic-only example: patient Alex Example, medical record "
            "EN-709-0001, was seen on February 3, 1990. Call +1 415 555 0709 "
            "or email alex@example.test. Dr. Casey Sample requested follow-up "
            "in one week."
        ),
        identifiers=(
            ("Alex Example", "PERSON"),
            ("Dr. Casey Sample", "PERSON"),
            ("EN-709-0001", "ID_NUM"),
            ("February 3, 1990", "DATE"),
            ("+1 415 555 0709", "PHONE"),
            ("alex@example.test", "EMAIL"),
        ),
    ),
}


__all__ = ["SAMPLES", "SyntheticSample"]
