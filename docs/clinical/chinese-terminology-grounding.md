# Chinese terminology grounding

OpenMed can ground Chinese `CONDITION` and `MEDICATION` spans against a
dictionary that you supply. Matching is local and deterministic: exact matches
run first, followed by an optional normalized fuzzy match. The loader never
downloads, scrapes, or silently substitutes a Chinese ICD-10 or drug vocabulary.

!!! warning "Not a medical device"

    Chinese terminology grounding is assistive software, not a medical device.
    Codes are user-vocabulary-dependent and are not for clinical decisions.
    Verify every match against your source dictionary and its license.

## Dictionary schema

Provide one UTF-8 CSV, TSV, or JSONL file with these fields:

| Field | Required | Value |
|---|---:|---|
| `label` | yes | `CONDITION` or `MEDICATION` |
| `system` | no | `ICD-10-CN` for conditions or `CN-DRUG` for medications |
| `code` | yes | Code from your licensed/user-managed vocabulary |
| `display` | yes | Preferred Chinese display term |
| `aliases` | no | `|`-separated text in CSV/TSV, or a string array in JSONL |

The repository's
`examples/data/chinese_terminology_synthetic.csv` file contains twelve invented
entries for tests and documentation. Its `SYN-CN-*` codes and terms are not a
real clinical vocabulary.

## Ground an analysis result

```python
from openmed import analyze_text
from openmed.clinical import ChineseTerminologyGrounder

analysis = analyze_text(
    "诊断星芒性热病，予晨露安片。",
    model_name="your-chinese-clinical-ner-model",
    sentence_language="zh",
)

grounder = ChineseTerminologyGrounder.from_path(
    "/secure/local/path/chinese-terminology.csv",
    license_acknowledged=True,
)
grounded = grounder.ground_result(analysis)

for entity in grounded.entities:
    print(entity.text, entity.metadata)
```

The license acknowledgement is mandatory. Omitting the path raises an error
that explains that OpenMed has no bundled fallback. Set `fuzzy=False` when only
exact matches are acceptable, or adjust `min_score` from its conservative
default of `0.8` for a vocabulary-specific validation study.

Dictionary values can be emitted in grounding metadata. Supply terminology
only: do not put patient names, identifiers, or other PHI in `code`, `display`,
or `aliases` fields.

Matched spans gain only `system`, `code`, and `display` metadata. Their text,
start/end offsets, confidence, and any de-identification action remain
unchanged. Result-level metadata emits the medical-device disclaimer and the
user-vocabulary note without recording dictionary contents or source PHI.
