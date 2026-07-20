"""Deterministic multilingual clinical relation extraction.

The versioned registry maps the 44 CMeIE predicates and a focused Hindi
subset onto OpenMed canonical relation labels. Extraction reuses existing NER
spans, constructs script-agnostic character-offset candidates, and delegates
selection to the shared span-graph decoder.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from openmed.clinical.context import (
    HYPOTHETICAL,
    NEGATED,
    UNCERTAIN,
    resolve_span_context,
)
from openmed.core.decoding import SpanGraphConstraints, decode_span_graph
from openmed.core.labels import (
    AGE,
    BODY_SITE,
    CONDITION,
    GENDER,
    LAB_TEST,
    MEDICATION,
    MICROORGANISM,
    ORGANIZATION,
    OTHER,
    PROCEDURE,
)

from .assertion_filter import (
    RELATION_CONDITIONAL,
    RELATION_CONFIRMED,
    RELATION_POSSIBLE,
    RELATION_REFUTED,
)
from .candidate import (
    RelationCandidateRule,
    SpanReference,
    build_relation_candidates,
)

MULTILINGUAL_RELATION_REGISTRY_VERSION = 1
MULTILINGUAL_RELATION_ADVISORY = (
    "Multilingual relation extraction is deterministic assistive support over "
    "existing NER spans. Validate extracted relations before clinical use."
)

CMEIE_ENTITY_TYPES: tuple[str, ...] = (
    "疾病",
    "症状",
    "检查",
    "药物",
    "部位",
    "手术治疗",
    "其他治疗",
    "预后",
    "流行病学",
    "社会学",
    "其他",
)

# CMeIE publishes 53 schemas containing 44 distinct predicates. Keep this
# insertion order aligned with the documented mapping table.
CMEIE_RELATION_MAPPING: Mapping[str, str] = MappingProxyType(
    {
        "预防": "prevention",
        "内窥镜检查": "endoscopic_examination",
        "病理分型": "pathology_type",
        "病史": "history",
        "阶段": "stage",
        "筛查": "screening",
        "相关（导致）": "causes",
        "遗传因素": "genetic_factor",
        "就诊科室": "care_department",
        "多发群体": "prevalent_population",
        "相关（转化）": "transforms_to",
        "发病机制": "pathogenesis",
        "辅助治疗": "adjuvant_treatment",
        "发病率": "incidence",
        "相关（症状）": "associated_symptom",
        "病理生理": "pathophysiology",
        "化疗": "chemotherapy",
        "发病年龄": "onset_age",
        "鉴别诊断": "differential_diagnosis",
        "药物治疗": "drug_treatment",
        "放射治疗": "radiation_treatment",
        "多发地区": "prevalent_region",
        "临床表现": "clinical_manifestation",
        "发病部位": "affected_body_site",
        "手术治疗": "surgical_treatment",
        "发病性别倾向": "sex_predisposition",
        "治疗后症状": "post_treatment_symptom",
        "转移部位": "metastasis_site",
        "实验室检查": "laboratory_test",
        "死亡率": "mortality",
        "侵及周围组织转移的症状": "tissue_invasion_symptom",
        "外侵部位": "external_invasion_site",
        "影像学检查": "imaging_test",
        "传播途径": "transmission_route",
        "病因": "etiology",
        "预后状况": "prognosis",
        "辅助检查": "auxiliary_examination",
        "多发季节": "prevalent_season",
        "高危因素": "high_risk_factor",
        "预后生存率": "survival_rate",
        "组织学检查": "histological_examination",
        "并发症": "complication",
        "风险评估因素": "risk_assessment_factor",
        "同义词": "synonym",
    }
)

INDIC_RELATION_MAPPING: Mapping[str, str] = MappingProxyType(
    {
        "दवा उपचार": "drug_treatment",
        "शल्य उपचार": "surgical_treatment",
        "रोकथाम": "prevention",
        "नैदानिक अभिव्यक्ति": "clinical_manifestation",
        "कारण": "etiology",
        "जटिलता": "complication",
        "प्रयोगशाला जांच": "laboratory_test",
        "इमेजिंग जांच": "imaging_test",
    }
)

RELATION_TYPE_REGISTRY: Mapping[str, Mapping[str, str]] = MappingProxyType(
    {
        "hi": INDIC_RELATION_MAPPING,
        "zh": CMEIE_RELATION_MAPPING,
    }
)

_CONDITIONS = frozenset({CONDITION})
_CONDITION_ARGUMENTS = frozenset({CONDITION, MICROORGANISM, OTHER})
_FACTORS = frozenset(
    {AGE, BODY_SITE, CONDITION, GENDER, MEDICATION, MICROORGANISM, OTHER}
)
_TESTS = frozenset({LAB_TEST, PROCEDURE, OTHER})
_TREATMENTS = frozenset({MEDICATION, PROCEDURE, OTHER})

_TYPE_COMPATIBILITY: Mapping[str, tuple[frozenset[str], frozenset[str]]] = {
    "prevention": (_CONDITIONS, _TREATMENTS),
    "endoscopic_examination": (_CONDITIONS, _TESTS),
    "pathology_type": (_CONDITIONS, _CONDITIONS),
    "history": (_CONDITIONS, _FACTORS),
    "stage": (_CONDITIONS, frozenset({OTHER})),
    "screening": (_CONDITIONS, _TESTS),
    "causes": (_CONDITIONS, _CONDITION_ARGUMENTS),
    "genetic_factor": (_CONDITIONS, _FACTORS),
    "care_department": (_CONDITIONS, frozenset({ORGANIZATION, OTHER})),
    "prevalent_population": (_CONDITIONS, _FACTORS),
    "transforms_to": (_CONDITIONS, _CONDITIONS),
    "pathogenesis": (_CONDITIONS, _FACTORS),
    "adjuvant_treatment": (_CONDITIONS, _TREATMENTS),
    "incidence": (_CONDITIONS, frozenset({OTHER})),
    "associated_symptom": (_CONDITIONS, _CONDITIONS),
    "pathophysiology": (_CONDITIONS, _FACTORS),
    "chemotherapy": (_CONDITIONS, _TREATMENTS),
    "onset_age": (_CONDITIONS, frozenset({AGE, OTHER})),
    "differential_diagnosis": (_CONDITIONS, _CONDITIONS),
    "drug_treatment": (_CONDITIONS, frozenset({MEDICATION})),
    "radiation_treatment": (_CONDITIONS, _TREATMENTS),
    "prevalent_region": (_CONDITIONS, frozenset({BODY_SITE, OTHER})),
    "clinical_manifestation": (_CONDITIONS, _CONDITIONS),
    "affected_body_site": (_CONDITIONS, frozenset({BODY_SITE})),
    "surgical_treatment": (_CONDITIONS, frozenset({PROCEDURE, OTHER})),
    "sex_predisposition": (_CONDITIONS, frozenset({GENDER, OTHER})),
    "post_treatment_symptom": (_CONDITIONS, _CONDITIONS),
    "metastasis_site": (_CONDITIONS, frozenset({BODY_SITE})),
    "laboratory_test": (_CONDITIONS, _TESTS),
    "mortality": (_CONDITIONS, frozenset({OTHER})),
    "tissue_invasion_symptom": (_CONDITIONS, _CONDITIONS),
    "external_invasion_site": (_CONDITIONS, frozenset({BODY_SITE})),
    "imaging_test": (_CONDITIONS, _TESTS),
    "transmission_route": (_CONDITIONS, _FACTORS),
    "etiology": (_CONDITIONS, _FACTORS),
    "prognosis": (_CONDITIONS, frozenset({CONDITION, OTHER})),
    "auxiliary_examination": (_CONDITIONS, _TESTS),
    "prevalent_season": (_CONDITIONS, frozenset({OTHER})),
    "high_risk_factor": (_CONDITIONS, _FACTORS),
    "survival_rate": (_CONDITIONS, frozenset({OTHER})),
    "histological_examination": (_CONDITIONS, _TESTS),
    "complication": (_CONDITIONS, _CONDITIONS),
    "risk_assessment_factor": (_CONDITIONS, _FACTORS),
    "synonym": (_CONDITIONS, _CONDITIONS),
}

_CUES_BY_LANGUAGE: Mapping[str, Mapping[str, tuple[str, ...]]] = {
    "zh": {
        "prevention": ("预防", "防止", "疫苗"),
        "drug_treatment": ("药物治疗", "使用", "服用", "给予", "用药"),
        "surgical_treatment": ("手术治疗", "手术", "切除术", "切除"),
        "clinical_manifestation": ("临床表现", "表现为", "症状为"),
        "etiology": ("病因", "由于", "引起", "所致"),
        "complication": ("并发症", "并发"),
        "laboratory_test": ("实验室检查", "化验", "检测"),
        "imaging_test": ("影像学检查", "影像", "CT", "MRI"),
    },
    "hi": {
        "prevention": ("रोकथाम", "बचाव", "रोकने", "टीका"),
        "drug_treatment": (
            "दवा उपचार",
            "दवा से उपचार",
            "इलाज",
            "उपचार",
            "दी गई",
        ),
        "surgical_treatment": (
            "शल्य उपचार",
            "शल्य चिकित्सा",
            "ऑपरेशन",
        ),
        "clinical_manifestation": (
            "नैदानिक अभिव्यक्ति",
            "लक्षण",
            "दिखाता है",
        ),
        "etiology": ("कारण", "के कारण", "से होता"),
        "complication": ("जटिलता", "जटिलताओं"),
        "laboratory_test": ("प्रयोगशाला जांच", "रक्त जांच", "जांच"),
        "imaging_test": ("इमेजिंग जांच", "सीटी", "एमआरआई"),
    },
}


@dataclass(frozen=True)
class MultilingualRelation:
    """One decoded multilingual relation with source offsets and assertion."""

    relation_type: str
    head: SpanReference
    tail: SpanReference
    score: float
    language: str
    source_relation: str
    assertion_status: str

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic dictionary representation."""

        return {
            "type": self.relation_type,
            "head": self.head.to_dict(),
            "tail": self.tail.to_dict(),
            "score": self.score,
            "language": self.language,
            "source_relation": self.source_relation,
            "assertion_status": self.assertion_status,
            "advisory": MULTILINGUAL_RELATION_ADVISORY,
        }


def available_multilingual_relation_languages() -> tuple[str, ...]:
    """Return languages with versioned relation registries."""

    return tuple(sorted(RELATION_TYPE_REGISTRY))


def relation_type_mapping(language: str) -> Mapping[str, str]:
    """Return the versioned source-to-canonical mapping for ``language``."""

    code = _language_code(language)
    try:
        return RELATION_TYPE_REGISTRY[code]
    except KeyError as exc:
        supported = ", ".join(available_multilingual_relation_languages())
        raise ValueError(
            f"unsupported relation language {language!r}; expected: {supported}"
        ) from exc


def map_relation_type(source_relation: str, *, language: str) -> str:
    """Map one source predicate onto an OpenMed canonical relation label."""

    mapping = relation_type_mapping(language)
    try:
        return mapping[source_relation]
    except KeyError as exc:
        raise ValueError(
            f"unknown {language!r} relation type: {source_relation!r}"
        ) from exc


def multilingual_relation_rules(language: str) -> tuple[RelationCandidateRule, ...]:
    """Build deterministic candidate rules for one registered language."""

    code = _language_code(language)
    mapping = relation_type_mapping(code)
    cue_mapping = _CUES_BY_LANGUAGE[code]
    rules: list[RelationCandidateRule] = []
    for source_relation, canonical_relation in mapping.items():
        head_labels, tail_labels = _TYPE_COMPATIBILITY[canonical_relation]
        cues = cue_mapping.get(canonical_relation, (source_relation,))
        rules.append(
            RelationCandidateRule(
                relation_type=canonical_relation,
                source_relation=source_relation,
                head_labels=head_labels,
                tail_labels=tail_labels,
                cues=cues,
            )
        )
    return tuple(rules)


def extract_relations(
    text: str,
    spans: Iterable[Any],
    *,
    language: str,
    min_score: float = 0.5,
    asserted_only: bool = True,
) -> tuple[MultilingualRelation, ...]:
    """Extract Chinese or Hindi relations from existing multilingual NER spans.

    The function never runs a second NER model. Character-offset candidates go
    through the shared span-graph decoder with registry type constraints.
    Refuted and conditional relations are omitted by default; uncertainty is
    retained as ``possible`` for downstream review.

    Args:
        text: Original clinical text.
        spans: Existing NER spans with offsets into ``text``.
        language: ``zh`` for the CMeIE inventory or ``hi`` for the Indic subset.
        min_score: Minimum deterministic candidate score.
        asserted_only: Exclude refuted and conditional relations when true.

    Returns:
        Decoded relations in deterministic graph order.
    """

    code = _language_code(language)
    rules = multilingual_relation_rules(code)
    batch = build_relation_candidates(text, spans, rules, language=code)
    type_compatibility: dict[str, set[tuple[str, str]]] = {}
    for rule in rules:
        pairs = type_compatibility.setdefault(rule.relation_type, set())
        pairs.update(
            (head_label, tail_label)
            for head_label in rule.head_labels
            for tail_label in rule.tail_labels
        )
    graph = decode_span_graph(
        batch.nodes,
        batch.candidates,
        constraints=SpanGraphConstraints(
            allowed_edge_labels={rule.relation_type for rule in rules},
            type_compatibility=type_compatibility,
        ),
        min_edge_score=min_score,
    )

    relations: list[MultilingualRelation] = []
    for edge in graph.edges:
        head = batch.spans_by_node_id[edge.head]
        tail = batch.spans_by_node_id[edge.tail]
        assertion_status = _relation_assertion_status(text, head, tail, code)
        if asserted_only and assertion_status in {
            RELATION_REFUTED,
            RELATION_CONDITIONAL,
        }:
            continue
        relations.append(
            MultilingualRelation(
                relation_type=edge.label,
                head=head,
                tail=tail,
                score=edge.score,
                language=code,
                source_relation=str(edge.metadata["source_relation"]),
                assertion_status=assertion_status,
            )
        )
    return tuple(relations)


def _relation_assertion_status(
    text: str,
    head: SpanReference,
    tail: SpanReference,
    language: str,
) -> str:
    contexts = [
        resolve_span_context(
            {
                "text": reference.text,
                "context": text,
                "start": reference.start,
                "end": reference.end,
            },
            language=language,
        )
        for reference in (head, tail)
    ]
    if any(context.negation == NEGATED for context in contexts):
        return RELATION_REFUTED
    if any(context.temporality == HYPOTHETICAL for context in contexts):
        return RELATION_CONDITIONAL
    if any(context.certainty == UNCERTAIN for context in contexts):
        return RELATION_POSSIBLE
    return RELATION_CONFIRMED


def _language_code(language: str) -> str:
    return language.casefold().replace("_", "-").split("-", 1)[0].strip()


__all__ = [
    "CMEIE_ENTITY_TYPES",
    "CMEIE_RELATION_MAPPING",
    "INDIC_RELATION_MAPPING",
    "MULTILINGUAL_RELATION_ADVISORY",
    "MULTILINGUAL_RELATION_REGISTRY_VERSION",
    "RELATION_TYPE_REGISTRY",
    "MultilingualRelation",
    "available_multilingual_relation_languages",
    "extract_relations",
    "map_relation_type",
    "multilingual_relation_rules",
    "relation_type_mapping",
]
