# Multilingual Clinical Relations

OpenMed can extract deterministic Chinese and Hindi clinical relations from
already-extracted NER spans. It does not run or train a second NER model. The
candidate layer uses character offsets rather than whitespace tokens, then
passes typed edges through the same constrained span-graph decoder used by the
other relation backends.

```python
from openmed.clinical import extract_relations

text = "肺炎使用阿莫西林治疗。"
spans = [
    {"label": "CONDITION", "start": 0, "end": 2},
    {"label": "MEDICATION", "start": 4, "end": 8},
]

relations = extract_relations(text, spans, language="zh")
print(relations[0].to_dict())
```

The output retains both relation arguments and their source character offsets.
Relation extraction is assistive support and should be validated before
clinical use.

## Registry Version 1

`MULTILINGUAL_RELATION_REGISTRY_VERSION` is `1`. The Chinese registry preserves
all 44 distinct predicates in the official CMeIE schema and maps them onto
stable lower-snake-case labels. CMeIE defines 53 subject-predicate-object
schemas because some predicates allow more than one entity-type pairing. The
registry includes labels and type compatibility only; OpenMed does not bundle
the CMeIE corpus. See the
[official CBLUE benchmark](https://github.com/CBLUEbenchmark/CBLUE) for the
dataset boundary and evaluation format.

| CMeIE predicate | Canonical label |
|---|---|
| 预防 | `prevention` |
| 内窥镜检查 | `endoscopic_examination` |
| 病理分型 | `pathology_type` |
| 病史 | `history` |
| 阶段 | `stage` |
| 筛查 | `screening` |
| 相关（导致） | `causes` |
| 遗传因素 | `genetic_factor` |
| 就诊科室 | `care_department` |
| 多发群体 | `prevalent_population` |
| 相关（转化） | `transforms_to` |
| 发病机制 | `pathogenesis` |
| 辅助治疗 | `adjuvant_treatment` |
| 发病率 | `incidence` |
| 相关（症状） | `associated_symptom` |
| 病理生理 | `pathophysiology` |
| 化疗 | `chemotherapy` |
| 发病年龄 | `onset_age` |
| 鉴别诊断 | `differential_diagnosis` |
| 药物治疗 | `drug_treatment` |
| 放射治疗 | `radiation_treatment` |
| 多发地区 | `prevalent_region` |
| 临床表现 | `clinical_manifestation` |
| 发病部位 | `affected_body_site` |
| 手术治疗 | `surgical_treatment` |
| 发病性别倾向 | `sex_predisposition` |
| 治疗后症状 | `post_treatment_symptom` |
| 转移部位 | `metastasis_site` |
| 实验室检查 | `laboratory_test` |
| 死亡率 | `mortality` |
| 侵及周围组织转移的症状 | `tissue_invasion_symptom` |
| 外侵部位 | `external_invasion_site` |
| 影像学检查 | `imaging_test` |
| 传播途径 | `transmission_route` |
| 病因 | `etiology` |
| 预后状况 | `prognosis` |
| 辅助检查 | `auxiliary_examination` |
| 多发季节 | `prevalent_season` |
| 高危因素 | `high_risk_factor` |
| 预后生存率 | `survival_rate` |
| 组织学检查 | `histological_examination` |
| 并发症 | `complication` |
| 风险评估因素 | `risk_assessment_factor` |
| 同义词 | `synonym` |

The Hindi Indic subset covers the relation families exercised by the shipped
synthetic clinical fixtures:

| Hindi relation | Canonical label |
|---|---|
| दवा उपचार | `drug_treatment` |
| शल्य उपचार | `surgical_treatment` |
| रोकथाम | `prevention` |
| नैदानिक अभिव्यक्ति | `clinical_manifestation` |
| कारण | `etiology` |
| जटिलता | `complication` |
| प्रयोगशाला जांच | `laboratory_test` |
| इमेजिंग जांच | `imaging_test` |

## Assertion And Evaluation Behavior

Chinese extraction uses the Chinese ConText cue pack and Hindi extraction uses
the Hindi pack. Missing cues default to recent, certain, and affirmed; an
English-looking record marker does not negate a Chinese or Hindi relation.
Refuted and conditional relations are withheld by default, while uncertain
relations remain available as `possible` for review.

The synthetic fixtures live at:

- `openmed/eval/golden/fixtures/i18n/relations_zh.jsonl`
- `openmed/eval/golden/fixtures/i18n/relations_indic.jsonl`

`run_relation_benchmark()` and `ModelScorecard` report strict and relaxed F1
under `relation_extraction.per_language`, so Chinese and Hindi performance
cannot be hidden inside an aggregate relation score.
