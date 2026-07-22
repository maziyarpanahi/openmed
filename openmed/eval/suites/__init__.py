"""Benchmark suite registry for the OpenMed eval harness."""

from __future__ import annotations

from typing import Any

from openmed.eval.comparators import (
    ComparatorAdapter,
    ComparatorMatrixReport,
    ComparatorMatrixRow,
    ComparatorUnavailable,
    run_comparator_matrix,
)
from openmed.eval.datasets.biomedical_ner import (
    BIOMEDICAL_NER,
    biomedical_ner_suite_metadata,
    load_biomedical_ner_fixtures,
    run_biomedical_ner_benchmark,
)
from openmed.eval.datasets.drugprot import (
    DRUGPROT,
    drugprot_suite_metadata,
    load_drugprot_fixtures,
)
from openmed.eval.datasets.i2b2 import (
    I2B2,
    I2B2_PATH_ENV,
    I2B2_YEAR_ENV,
    i2b2_suite_metadata,
    load_i2b2_deid,
)
from openmed.eval.datasets.masakhaner import (
    MASAKHANER,
    load_masakhaner_fixtures,
    masakhaner_suite_metadata,
    run_masakhaner_benchmark,
)
from openmed.eval.datasets.multilingual_ner import (
    MULTILINGUAL_NER,
    load_multilingual_ner_fixtures,
    multilingual_ner_suite_metadata,
)
from openmed.eval.golden import load_benchmark_fixtures
from openmed.eval.harness import BenchmarkFixture
from openmed.eval.suites.chinese_clinical_ner import (
    CHINESE_CLINICAL_NER,
    ChineseClinicalNerLeakageError,
    PhiTokenLeakageFinding,
    chinese_clinical_ner_metadata,
    load_chinese_clinical_ner_fixtures,
    run_chinese_clinical_ner_suite,
    run_synthetic_chinese_clinical_ner_smoke,
)
from openmed.eval.suites.chinese_terminology import (
    ChineseTerminologyLeakageReport,
    evaluate_chinese_terminology_leakage,
)
from openmed.eval.suites.code_mixed_routing import (
    CODE_MIXED_ROUTING,
    evaluate_code_mixed_routing,
    load_code_mixed_fixtures,
)
from openmed.eval.suites.india_health_ids import (
    INDIA_HEALTH_ID_LEAKAGE,
    assert_india_health_id_leakage_gate,
    india_health_id_metadata,
    load_india_health_id_fixtures,
    run_india_health_id_leakage_gate,
)
from openmed.eval.suites.multimodal_dicom import (
    MULTIMODAL_DICOM,
    generate_synthetic_dicom_corpus,
    load_multimodal_dicom_fixtures,
    multimodal_dicom_metadata,
    run_multimodal_dicom,
)
from openmed.eval.suites.naamapadam import (
    NAAMAPADAM,
    load_naamapadam_fixtures,
    naamapadam_suite_metadata,
    run_naamapadam,
)
from openmed.eval.suites.policy_compliance import (
    POLICY_COMPLIANCE,
    load_policy_compliance_fixtures,
    policy_compliance_metadata,
)
from openmed.eval.suites.relations import (
    RELATIONS,
    RelationFixture,
    RelationTrap,
    load_relation_fixtures,
    relation_suite_metadata,
    relation_trap_summary,
    score_relation_fixtures,
)
from openmed.eval.suites.shield import (
    SHIELD,
    load_shield_fixtures,
    shield_suite_metadata,
)

GOLDEN = "golden"
N2C2 = "n2c2"

DEFAULT_SUITES: tuple[str, ...] = (
    GOLDEN,
    I2B2,
    N2C2,
    SHIELD,
    DRUGPROT,
    POLICY_COMPLIANCE,
    BIOMEDICAL_NER,
    MULTILINGUAL_NER,
    MASAKHANER,
    NAAMAPADAM,
    CHINESE_CLINICAL_NER,
    MULTIMODAL_DICOM,
    CODE_MIXED_ROUTING,
    INDIA_HEALTH_ID_LEAKAGE,
)


def validate_suite_name(name: str) -> str:
    """Return *name* if it is one of the scaffolded benchmark suites."""
    if name not in DEFAULT_SUITES:
        allowed = ", ".join(DEFAULT_SUITES)
        raise ValueError(
            f"unknown benchmark suite {name!r}; expected one of: {allowed}"
        )
    return name


def load_suite_fixtures(name: str, **kwargs: Any) -> list[Any]:
    """Load benchmark fixtures for a named suite."""
    suite = validate_suite_name(name)
    if suite == GOLDEN:
        return load_benchmark_fixtures(kwargs.get("path"))
    if suite == I2B2:
        return load_i2b2_deid(
            path=kwargs.get("path"),
            year=kwargs.get("year", kwargs.get("corpus_year")),
        )
    if suite == SHIELD:
        return load_shield_fixtures(**kwargs)
    if suite == DRUGPROT:
        return load_drugprot_fixtures(**kwargs)
    if suite == POLICY_COMPLIANCE:
        return load_policy_compliance_fixtures(**kwargs)
    if suite == BIOMEDICAL_NER:
        return load_biomedical_ner_fixtures(**kwargs)
    if suite == MULTILINGUAL_NER:
        paths = kwargs.pop("paths", kwargs.pop("path", None))
        return load_multilingual_ner_fixtures(paths=paths, **kwargs)
    if suite == MASAKHANER:
        paths = kwargs.pop("paths", kwargs.pop("path", None))
        return load_masakhaner_fixtures(paths=paths, **kwargs)
    if suite == NAAMAPADAM:
        path = kwargs.pop("path", None)
        return load_naamapadam_fixtures(**({"path": path} if path else {}))
    if suite == CHINESE_CLINICAL_NER:
        return load_chinese_clinical_ner_fixtures(kwargs.get("path"))
    if suite == MULTIMODAL_DICOM:
        return load_multimodal_dicom_fixtures(**kwargs)
    if suite == CODE_MIXED_ROUTING:
        return load_code_mixed_fixtures(kwargs.get("path"))
    if suite == INDIA_HEALTH_ID_LEAKAGE:
        return load_india_health_id_fixtures(**kwargs)
    raise ValueError(f"benchmark suite {suite!r} does not have a concrete loader yet")


def suite_metadata(name: str, **kwargs: Any) -> dict[str, Any]:
    """Return suite-specific report metadata."""
    suite = validate_suite_name(name)
    if suite == I2B2:
        metadata = i2b2_suite_metadata()
        metadata["path_config"] = kwargs.get("path_config", I2B2_PATH_ENV)
        metadata["year_config"] = kwargs.get("year_config", I2B2_YEAR_ENV)
        return metadata
    if suite == SHIELD:
        return shield_suite_metadata(**kwargs)
    if suite == DRUGPROT:
        return drugprot_suite_metadata(**kwargs)
    if suite == POLICY_COMPLIANCE:
        return policy_compliance_metadata(**kwargs)
    if suite == BIOMEDICAL_NER:
        return biomedical_ner_suite_metadata(**kwargs)
    if suite == MULTILINGUAL_NER:
        return multilingual_ner_suite_metadata(**kwargs)
    if suite == MASAKHANER:
        return masakhaner_suite_metadata(**kwargs)
    if suite == NAAMAPADAM:
        return naamapadam_suite_metadata()
    if suite == CHINESE_CLINICAL_NER:
        return chinese_clinical_ner_metadata()
    if suite == MULTIMODAL_DICOM:
        return multimodal_dicom_metadata(**kwargs)
    if suite == CODE_MIXED_ROUTING:
        return {
            "suite": CODE_MIXED_ROUTING,
            "synthetic": True,
            "gates": {
                "phi_recall_min": 1.0,
                "entity_leakage_max": 0,
            },
        }
    if suite == INDIA_HEALTH_ID_LEAKAGE:
        return india_health_id_metadata(**kwargs)
    return {"suite": suite}


__all__ = [
    "GOLDEN",
    "I2B2",
    "N2C2",
    "SHIELD",
    "DRUGPROT",
    "POLICY_COMPLIANCE",
    "BIOMEDICAL_NER",
    "MULTILINGUAL_NER",
    "MASAKHANER",
    "NAAMAPADAM",
    "CHINESE_CLINICAL_NER",
    "MULTIMODAL_DICOM",
    "CODE_MIXED_ROUTING",
    "INDIA_HEALTH_ID_LEAKAGE",
    "RELATIONS",
    "RelationFixture",
    "RelationTrap",
    "ComparatorAdapter",
    "ComparatorMatrixReport",
    "ComparatorMatrixRow",
    "ComparatorUnavailable",
    "ChineseTerminologyLeakageReport",
    "DEFAULT_SUITES",
    "validate_suite_name",
    "load_benchmark_fixtures",
    "load_suite_fixtures",
    "suite_metadata",
    "run_comparator_matrix",
    "evaluate_chinese_terminology_leakage",
    "load_i2b2_deid",
    "i2b2_suite_metadata",
    "biomedical_ner_suite_metadata",
    "multilingual_ner_suite_metadata",
    "masakhaner_suite_metadata",
    "ChineseClinicalNerLeakageError",
    "PhiTokenLeakageFinding",
    "chinese_clinical_ner_metadata",
    "load_chinese_clinical_ner_fixtures",
    "run_chinese_clinical_ner_suite",
    "run_synthetic_chinese_clinical_ner_smoke",
    "load_drugprot_fixtures",
    "load_biomedical_ner_fixtures",
    "load_multilingual_ner_fixtures",
    "load_masakhaner_fixtures",
    "drugprot_suite_metadata",
    "load_shield_fixtures",
    "shield_suite_metadata",
    "load_policy_compliance_fixtures",
    "policy_compliance_metadata",
    "load_relation_fixtures",
    "relation_suite_metadata",
    "relation_trap_summary",
    "score_relation_fixtures",
    "run_biomedical_ner_benchmark",
    "run_masakhaner_benchmark",
    "load_naamapadam_fixtures",
    "naamapadam_suite_metadata",
    "run_naamapadam",
    "load_multimodal_dicom_fixtures",
    "multimodal_dicom_metadata",
    "run_multimodal_dicom",
    "generate_synthetic_dicom_corpus",
    "evaluate_code_mixed_routing",
    "load_code_mixed_fixtures",
    "assert_india_health_id_leakage_gate",
    "india_health_id_metadata",
    "load_india_health_id_fixtures",
    "run_india_health_id_leakage_gate",
]
