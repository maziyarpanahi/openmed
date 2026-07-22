#!/usr/bin/env python3
"""De-identify a fabricated Simplified Chinese clinical note end to end.

The note, names, and identifiers in this example are synthetic test data. They
do not identify a real person, clinician, organization, or account.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from openmed import deidentify
from openmed.core.pii_i18n import validate_chinese_resident_identity_card

# Use the registry key so the generic English checkpoint remains an explicit
# fallback instead of being replaced by the ``zh`` placeholder default.
MODEL_KEY = "pii_superclinical_small"
DEFAULT_OUTPUT_PATH = Path("chinese_clinical_note_redacted.txt")

SYNTHETIC_PATIENT_NAME = "李晓雯"
SYNTHETIC_CLINICIAN_NAME = "陈志远"
SYNTHETIC_MEDICAL_RECORD_NUMBER = "ZH-DEMO-708"
SYNTHETIC_DATE_OF_BIRTH = "1990年2月3日"
SYNTHETIC_RESIDENT_ID = "110108198503150018"
SYNTHETIC_PHONE = "+86 10 5555 0708"
SYNTHETIC_EMAIL = "li.xiaowen@example.test"
SYNTHETIC_ADDRESS = "北京市海淀区示例路708号"

# 此病历及其中所有姓名和标识符均为虚构测试数据。
# This note and every name and identifier in it are fabricated test data.
SYNTHETIC_CHINESE_NOTE = (
    "【完全虚构的示例】患者李晓雯，病历号 ZH-DEMO-708，"
    "出生日期 1990年2月3日，身份证号 110108198503150018。"
    "联系电话 +86 10 5555 0708，电子邮箱 li.xiaowen@example.test，"
    "住址 北京市海淀区示例路708号。患者因咳嗽复诊，陈志远医生建议继续观察。"
)

SYNTHETIC_IDENTIFIERS = (
    SYNTHETIC_PATIENT_NAME,
    SYNTHETIC_CLINICIAN_NAME,
    SYNTHETIC_MEDICAL_RECORD_NUMBER,
    SYNTHETIC_DATE_OF_BIRTH,
    SYNTHETIC_RESIDENT_ID,
    SYNTHETIC_PHONE,
    SYNTHETIC_EMAIL,
    SYNTHETIC_ADDRESS,
)

# ``zh`` 路由提供中文规范化和内置规则；显式本地规则保证教程中的虚构值可重复识别。
# The ``zh`` route provides Chinese normalization and built-in patterns. It is
# currently backed by the generic compact detector rather than a dedicated
# Chinese checkpoint, while these explicit local rules make the tutorial's
# fabricated values deterministic.
CHINESE_CUSTOM_RECOGNIZER = {
    "case_sensitive": False,
    "deny": {
        "terms": [
            {"term": SYNTHETIC_PATIENT_NAME, "label": "PERSON"},
            {"term": SYNTHETIC_CLINICIAN_NAME, "label": "PERSON"},
            {"term": SYNTHETIC_MEDICAL_RECORD_NUMBER, "label": "ID_NUM"},
            {"term": SYNTHETIC_DATE_OF_BIRTH, "label": "DATE"},
            {"term": SYNTHETIC_RESIDENT_ID, "label": "ID_NUM"},
            {"term": SYNTHETIC_PHONE, "label": "PHONE"},
            {"term": SYNTHETIC_EMAIL, "label": "EMAIL"},
            {"term": SYNTHETIC_ADDRESS, "label": "ADDRESS"},
        ]
    },
}

Deidentifier = Callable[..., Any]


def assert_synthetic_identifiers_removed(deidentified_text: str) -> None:
    """Fail closed if any fabricated identifier survives de-identification."""

    leaked = [
        identifier
        for identifier in SYNTHETIC_IDENTIFIERS
        if identifier in deidentified_text
    ]
    if leaked:
        raise AssertionError(f"Synthetic identifiers were not redacted: {leaked!r}")


def structured_entities(result: Any) -> list[dict[str, Any]]:
    """Return JSON-serializable rows for the entities in a result."""

    return [
        {
            "label": entity.canonical_label or entity.entity_type or entity.label,
            "text": entity.text,
            "start": entity.start,
            "end": entity.end,
            "confidence": round(float(entity.confidence), 3),
            "replacement": entity.redacted_text,
            "sources": list(entity.sources),
        }
        for entity in result.pii_entities
    ]


def run_chinese_deidentification(
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    *,
    deidentifier: Deidentifier = deidentify,
    loader: Any = None,
) -> Any:
    """De-identify the synthetic note, verify it, and save the redacted text.

    Args:
        output_path: UTF-8 text file to create with the redacted note.
        deidentifier: Injectable ``deidentify``-compatible callable for tests.
        loader: Optional cached or test model loader forwarded to OpenMed.

    Returns:
        The ``DeidentificationResult`` returned by OpenMed.
    """

    # 先校验虚构身份证号的格式和校验位。 / Validate its format and checksum first.
    if not validate_chinese_resident_identity_card(SYNTHETIC_RESIDENT_ID):
        raise AssertionError("The fabricated Chinese resident ID is invalid")

    kwargs: dict[str, Any] = {
        "method": "replace",
        "model_name": MODEL_KEY,
        "lang": "zh",
        "locale": "zh_CN",
        "policy": "china_pipl",
        "use_safety_sweep": True,
        "custom_recognizer": CHINESE_CUSTOM_RECOGNIZER,
        "consistent": True,
        "seed": 708,
    }
    if loader is not None:
        kwargs["loader"] = loader

    # 执行脱敏并在保存前进行零泄漏检查。 / De-identify, then check zero leakage.
    result = deidentifier(SYNTHETIC_CHINESE_NOTE, **kwargs)
    assert_synthetic_identifiers_removed(result.deidentified_text)

    # 以 UTF-8 保存脱敏文本。 / Save the redacted text as UTF-8.
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(result.deidentified_text + "\n", encoding="utf-8")
    return result


def main() -> None:
    """Run the complete Chinese example and print its structured output."""

    result = run_chinese_deidentification()
    print("=== 脱敏文本 / De-identified text ===")
    print(result.deidentified_text)
    print("\n=== 结构化实体 / Structured entities ===")
    print(json.dumps(structured_entities(result), ensure_ascii=False, indent=2))
    print(f"\n已保存 / Saved to: {DEFAULT_OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
