# Simplified Chinese README glossary

Use these canonical renderings when updating `README.zh-CN.md`. Keep product,
library, framework, model, and protocol names in their official form when a
translation would make them harder to identify.

| English term | Canonical Simplified Chinese | Usage note |
|---|---|---|
| de-identification | 去标识化 | Use for the process that removes or replaces identifying information. |
| de-identified | 已去标识化 | Use for output after de-identification. |
| on-device | 设备本地 | Use when computation happens on the user's device. |
| entity extraction | 实体抽取 | Use for extracting biomedical, clinical, or PII entities. |
| local-first | 本地优先 | Use for the architecture and product principle. |
| Privacy Filter | 隐私过滤器 | Keep `Privacy Filter` in English when it is part of an official model-family name. |
| PII detection | PII 检测 | Expand PII only when a new audience needs the definition. |
| redaction | 脱敏 | Use for masking or removing sensitive spans from displayed text. |
| model-backed PII language | 由模型支持的 PII 语言 | Distinguish this from validator-only national-ID coverage. |
| smart entity merging | 智能实体合并 | Use for recombining fragmented token spans. |
| air-gapped | 隔离网络 | The first use may include the English term in parentheses. |
| clinical NER | 临床命名实体识别（临床 NER） | The short form `临床 NER` is acceptable after first use. |
| vendor lock-in | 供应商锁定 | Use `无供应商锁定` for the OpenMed guarantee. |
| checkpoint | 检查点 | Use for a published model checkpoint. |

After reviewing a translation update, refresh and verify the section manifest:

```bash
python scripts/i18n/check_readme_drift.py --update
python scripts/i18n/check_readme_drift.py
```
