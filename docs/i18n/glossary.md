# Translation glossary

Use these canonical renderings across translated READMEs and documentation.
Keep product, package, API, function, model, protocol, command, and code names
in their official form when translation would make them harder to identify.

## Simplified Chinese

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

## Hindi

`README.hi.md` को अपडेट करते समय इन canonical renderings का उपयोग करें। Product,
library, framework, model और protocol के आधिकारिक नाम उसी रूप में रखें जब अनुवाद
से उनकी पहचान कठिन हो सकती है।

| English term | Canonical Hindi | Usage note |
|---|---|---|
| de-identification | डी-आइडेंटिफिकेशन | पहचान करने वाली जानकारी हटाने या बदलने की प्रक्रिया के लिए। |
| de-identified | डी-आइडेंटिफाइड | डी-आइडेंटिफिकेशन के बाद के output के लिए। |
| on-device | डिवाइस पर | जब computation उपयोगकर्ता के device पर होता है। |
| entity extraction | एंटिटी निष्कर्षण | Biomedical, clinical या PII entities निकालने के लिए। |
| local-first | लोकल-फर्स्ट | Architecture और product principle के लिए। |
| Privacy Filter | Privacy Filter | Official model-family name में English रूप बनाए रखें। |
| PII detection | PII पहचान | नए पाठक के लिए आवश्यक होने पर पहली बार PII का विस्तार करें। |
| redaction | PII छिपाना | Sensitive spans को mask करने या हटाने के लिए। |
| model-backed PII language | मॉडल-समर्थित PII भाषा | इसे validator-only national-ID coverage से अलग रखें। |
| smart entity merging | स्मार्ट एंटिटी मर्जिंग | Fragmented token spans को फिर से जोड़ने के लिए। |
| air-gapped | एयर-गैप्ड | पहली बार उपयोग पर English term कोष्ठक में दिया जा सकता है। |
| clinical NER | क्लिनिकल NER | NER को code और model identifiers में English में रखें। |
| vendor lock-in | वेंडर लॉक-इन | OpenMed के आश्वासन के लिए `कोई वेंडर लॉक-इन नहीं` लिखें। |
| checkpoint | चेकपॉइंट | Published model checkpoint के लिए। |

After reviewing any translation update, refresh and verify the section manifest:

```bash
python scripts/i18n/check_readme_drift.py --update
python scripts/i18n/check_readme_drift.py
```

## Documentation-specific terms

| English term | Simplified Chinese (`zh`) | Hindi (`hi`) |
|---|---|---|
| clinical NLP | 临床自然语言处理 | क्लिनिकल NLP |
| personally identifiable information (PII) | 个人身份信息 (PII) | व्यक्तिगत पहचान योग्य जानकारी (PII) |
| protected health information (PHI) | 受保护健康信息 (PHI) | संरक्षित स्वास्थ्य जानकारी (PHI) |
| entity | 实体 | एंटिटी |
| surrogate value | 替代值 | प्रतिस्थापन मान |
| fallback | 回退 | फ़ॉलबैक |
| pipeline | 流水线 | पाइपलाइन |
| batch processing | 批处理 | बैच प्रोसेसिंग |
| model registry | 模型注册表 | मॉडल रजिस्ट्री |
| confidence threshold | 置信度阈值 | कॉन्फ़िडेंस थ्रेशोल्ड |

Prefer clear native-language prose around established technical terms. Retain
widely recognized names such as OpenMed, OpenMedKit, Python, REST, MLX, ONNX,
FHIR, HL7, HIPAA, PyTorch, and Hugging Face.
