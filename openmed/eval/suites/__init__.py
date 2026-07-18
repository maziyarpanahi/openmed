"""Benchmark suite registry for the OpenMed eval harness."""

from __future__ import annotations

import warnings
from dataclasses import replace
from typing import Any, Mapping

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
from openmed.eval.datasets.cmeee import (
    CMEEE,
    CMEEE_PATH_ENV,
    cmeee_suite_metadata,
    configured_cmeee_path,
    load_cmeee_fixtures,
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
from openmed.eval.datasets.naamapadam import (
    NAAMAPADAM_PATH_ENV,
    configured_naamapadam_path,
)
from openmed.eval.datasets.naamapadam import (
    load_naamapadam_fixtures as load_naamapadam_corpus_fixtures,
)
from openmed.eval.datasets.naamapadam import (
    naamapadam_suite_metadata as naamapadam_corpus_suite_metadata,
)
from openmed.eval.golden import load_benchmark_fixtures
from openmed.eval.harness import BenchmarkFixture, ModelRunner, run_benchmark
from openmed.eval.report import BenchmarkReport
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
    code_mixed_routing_metadata,
    evaluate_code_mixed_routing,
    load_code_mixed_fixtures,
    load_code_mixed_routing_fixtures,
    run_code_mixed_routing,
)
from openmed.eval.suites.india_health_ids import (
    INDIA_HEALTH_ID_LEAKAGE,
    assert_india_health_id_leakage_gate,
    india_health_id_metadata,
    load_india_health_id_fixtures,
    run_india_health_id_leakage_gate,
)
from openmed.eval.suites.indian_ids import (
    INDIAN_MULTI_ID,
    evaluate_indian_id_recognizer,
    indian_id_suite_metadata,
    load_indian_id_fixtures,
    run_indian_id_evaluation,
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
    CMEEE,
    NAAMAPADAM,
    CHINESE_CLINICAL_NER,
    MULTIMODAL_DICOM,
    CODE_MIXED_ROUTING,
    INDIA_HEALTH_ID_LEAKAGE,
    INDIAN_MULTI_ID,
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
    if suite == CMEEE:
        path = kwargs.pop("path", None)
        if configured_cmeee_path(path) is None:
            _warn_skipped_suite(CMEEE, CMEEE_PATH_ENV)
            return []
        return load_cmeee_fixtures(path=path, **kwargs)
    if suite == NAAMAPADAM:
        path = kwargs.pop("path", None)
        if configured_naamapadam_path(path) is None:
            _warn_skipped_suite(NAAMAPADAM, NAAMAPADAM_PATH_ENV)
            return []
        return load_naamapadam_corpus_fixtures(path=path, **kwargs)
    if suite == CHINESE_CLINICAL_NER:
        return load_chinese_clinical_ner_fixtures(kwargs.get("path"))
    if suite == MULTIMODAL_DICOM:
        return load_multimodal_dicom_fixtures(**kwargs)
    if suite == CODE_MIXED_ROUTING:
        return load_code_mixed_fixtures(kwargs.get("path"))
    if suite == INDIA_HEALTH_ID_LEAKAGE:
        return load_india_health_id_fixtures(**kwargs)
    if suite == INDIAN_MULTI_ID:
        return load_indian_id_fixtures(kwargs.get("path"))
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
    if suite == CMEEE:
        return cmeee_suite_metadata(path=kwargs.get("path"))
    if suite == NAAMAPADAM:
        return naamapadam_corpus_suite_metadata(path=kwargs.get("path"))
    if suite == CHINESE_CLINICAL_NER:
        return chinese_clinical_ner_metadata()
    if suite == MULTIMODAL_DICOM:
        return multimodal_dicom_metadata(**kwargs)
    if suite == CODE_MIXED_ROUTING:
        return code_mixed_routing_metadata(fixture_path=kwargs.get("fixture_path"))
    if suite == INDIA_HEALTH_ID_LEAKAGE:
        return india_health_id_metadata(**kwargs)
    if suite == INDIAN_MULTI_ID:
        return indian_id_suite_metadata(**kwargs)
    return {"suite": suite}


def run_script_ner_benchmark(
    name: str,
    *,
    model_name: str,
    device: str = "cpu",
    runner: ModelRunner | None = None,
    load_kwargs: Mapping[str, Any] | None = None,
    generated_at: str | None = None,
) -> BenchmarkReport:
    """Run a CMeEE or Naamapadam suite with micro-F1 per writing script."""

    suite = validate_suite_name(name)
    if suite not in {CMEEE, NAAMAPADAM}:
        raise ValueError("script-aware NER reporting supports cmeee and naamapadam")
    loader_options = dict(load_kwargs or {})
    fixtures = load_suite_fixtures(suite, **loader_options)
    metadata = suite_metadata(suite, path=loader_options.get("path"))
    if not fixtures:
        skip_reason = str(metadata["availability"]["reason"])
        return BenchmarkReport(
            suite=suite,
            model_name=model_name,
            device=device,
            fixture_count=0,
            generated_at=generated_at,
            metrics={
                "micro_f1_by_script": {},
                "skip_reason": skip_reason,
                "skipped": True,
            },
            metadata=metadata,
        )

    overall = run_benchmark(
        fixtures,
        suite=suite,
        model_name=model_name,
        device=device,
        runner=runner,
        generated_at=generated_at,
        metadata=metadata,
    )
    grouped: dict[str, list[BenchmarkFixture]] = {}
    for fixture in fixtures:
        script = str(fixture.metadata.get("script") or "Unknown")
        grouped.setdefault(script, []).append(fixture)
    micro_f1_by_script: dict[str, float] = {}
    for script, script_fixtures in sorted(grouped.items()):
        script_report = run_benchmark(
            script_fixtures,
            suite=suite,
            model_name=model_name,
            device=device,
            runner=runner,
            generated_at=generated_at,
            metadata={**metadata, "script": script},
        )
        micro_f1_by_script[script] = float(script_report.metrics["exact_span_f1"]["f1"])
    metrics = dict(overall.metrics)
    metrics.update(
        {
            "micro_f1": float(overall.metrics["exact_span_f1"]["f1"]),
            "micro_f1_by_script": micro_f1_by_script,
            "skipped": False,
        }
    )
    return replace(overall, metrics=metrics)


def _warn_skipped_suite(suite: str, path_env: str) -> None:
    warnings.warn(
        f"Skipping {suite}: {path_env} is not set and no explicit path was provided",
        UserWarning,
        stacklevel=2,
    )


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
    "CMEEE",
    "NAAMAPADAM",
    "CHINESE_CLINICAL_NER",
    "MULTIMODAL_DICOM",
    "CODE_MIXED_ROUTING",
    "INDIA_HEALTH_ID_LEAKAGE",
    "INDIAN_MULTI_ID",
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
    "run_script_ner_benchmark",
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
    "load_code_mixed_routing_fixtures",
    "code_mixed_routing_metadata",
    "run_code_mixed_routing",
    "assert_india_health_id_leakage_gate",
    "india_health_id_metadata",
    "load_india_health_id_fixtures",
    "run_india_health_id_leakage_gate",
    "evaluate_indian_id_recognizer",
    "indian_id_suite_metadata",
    "load_indian_id_fixtures",
    "run_indian_id_evaluation",
]
