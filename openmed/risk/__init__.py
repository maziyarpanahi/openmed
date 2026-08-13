"""Re-identification risk package for section 4.2.

Intended contents include quasi-identifier detection, uniqueness/k-anonymity
measurement, and adversarial re-identification analysis.
"""

from typing import Any

from .aggregate_dp import (
    AggregateDPBudgetLedger,
    AggregateDPRelease,
    DPAggregateBudgetExceeded,
    DPBudgetComposition,
    DPBudgetExhausted,
    DPBudgetLedger,
    DPBudgetSpend,
    laplace_aggregate,
    release_aggregate,
)
from .audit_diff import AuditDiff, diff_audit_reports
from .budget import (
    CURRENT_EPSILON_POLICY_SCHEMA_VERSION,
    DEFAULT_DP_SURROGATE_SENSITIVITIES,
    DEFAULT_POLICY_BUDGETS,
    DEFAULT_QI_WEIGHTS,
    DEFAULT_RDP_ORDERS,
    DEFAULT_RISK_BUDGET,
    EPSILON_POLICY_CONFIG_RESOURCE,
    BudgetComposition,
    BudgetDecision,
    BudgetExceeded,
    CompositionRule,
    DPGenerationBudgetAccountant,
    DPSurrogateBudget,
    DPSurrogateBudgetExceeded,
    DPSurrogateComposition,
    DPSurrogateSensitivity,
    DPSurrogateSensitivityRegistry,
    DPSurrogateSpend,
    EpsilonPolicy,
    GenerationSpend,
    RiskBudget,
    RiskBudgetExceeded,
    RiskBudgetVerdict,
    RiskBudgetViolation,
    SurrogateDrawKind,
    budget_for_policy,
    epsilon_policy_for,
    evaluate_budget,
    load_epsilon_policies,
)
from .dashboard import (
    render_release_assessment_dashboard,
    render_risk_dashboard,
    write_release_assessment_dashboard,
    write_risk_dashboard,
)
from .differential_privacy import (
    AggregateKind,
    DifferentialPrivacy,
    DPMechanism,
    PrivacyBudget,
    PrivacyBudgetExceeded,
    PrivacyBudgetStatus,
    PrivacySpend,
    UtilityPoint,
    UtilityReport,
    gaussian_mechanism,
    gaussian_noise,
    gaussian_scale,
    gaussian_stddev,
    laplace_mechanism,
    laplace_noise,
    laplace_scale,
    release_aggregate as release_dp_aggregate,
    release_count,
    release_histogram,
    release_mean,
    release_sum,
    utility_report,
    utility_vs_epsilon,
)
from .k_anonymity import (
    EquivalenceClass,
    KAnonymityEngine,
    KAnonymityReport,
    SuppressionProposal,
    analyze_k_anonymity,
    apply_suppression,
    propose_suppression,
)
from .kanon import (
    MemoryCeilingError,
    StreamingKanonDecision,
    StreamingKanonState,
    build_generalization_hierarchies,
    enforce_kanon,
    kanon_report,
)
from .l_diversity import (
    DiversityClass,
    LDiversityChecker,
    LDiversityEngine,
    LDiversityReport,
    analyze_l_diversity,
    check_l_diversity,
    l_diversity_report,
)
from .membership import (
    MembershipSelfTestError,
    MembershipSelfTestResult,
    bounded_membership_inference_self_test,
)
from .membership import (
    membership_inference_self_test as _bounded_membership_inference_self_test,
)
from .membership import (
    run_membership_inference_self_test as _run_bounded_membership_inference_self_test,
)
from .membership_inference import (
    DEFAULT_MEMBERSHIP_ADVANTAGE_BUDGET,
    DEFAULT_RISKIEST_RECORD_COUNT,
    MembershipInferenceReport,
    MembershipInferenceResult,
)
from .membership_inference import (
    membership_inference_self_test as _table_membership_inference_self_test,
)
from .membership_inference import (
    run_membership_inference_self_test as _run_table_membership_inference_self_test,
)
from .population import PopulationRiskAssessment, assess_population_risk
from .qi_profiler import (
    GeneralizationPlan,
    QIColumnProfile,
    QIGeneralization,
    QIProfiler,
    QIProfilerReport,
    QuasiIdentifierProfiler,
    apply_generalization_plan,
    profile_qi,
    profile_quasi_identifier_risk,
    profile_quasi_identifiers,
)
from .reid import (
    LongitudinalCorpus,
    LongitudinalEvidence,
    LongitudinalNote,
    LongitudinalPatient,
    build_longitudinal_corpus,
    cross_modal_linkage_risk_report,
    longitudinal_attack_fingerprint,
    longitudinal_risk_report,
    quasi_identifier_key,
    quasi_identifier_key_bytes,
    risk_report,
)
from .release import (
    AnonymityPolicy,
    AnonymizationResult,
    AttributeDisclosureSummary,
    GeneralizationSummary,
    ReleaseAssessment,
    ReleasedOutputValidation,
    UtilitySummary,
    anonymize_release,
    assess_release,
    release_dataset_digest,
    release_schema_digest,
    safe_risk_summary,
    validate_released_output,
)
from .synthetic_tabular import (
    DEFAULT_CORRELATION_TOLERANCE,
    DEFAULT_MARGINAL_TOLERANCE,
    ColumnDistribution,
    TabularProfile,
    fit_tabular_profile,
    sample_synthetic_table,
    tabular_fidelity_report,
)


def membership_inference_self_test(*args: Any, **kwargs: Any) -> Any:
    """Run the bounded-QI or table membership self-test.

    Calls that declare ``quasi_identifiers`` retain the bounded exact-match
    API. Other calls use the table attack-advantage API.
    """

    if "quasi_identifiers" in kwargs:
        return _bounded_membership_inference_self_test(*args, **kwargs)
    return _table_membership_inference_self_test(*args, **kwargs)


def run_membership_inference_self_test(*args: Any, **kwargs: Any) -> Any:
    """Compatibility dispatcher for both membership self-test APIs."""

    if "quasi_identifiers" in kwargs:
        return _run_bounded_membership_inference_self_test(*args, **kwargs)
    return _run_table_membership_inference_self_test(*args, **kwargs)


__all__ = [
    "CURRENT_EPSILON_POLICY_SCHEMA_VERSION",
    "AggregateDPBudgetLedger",
    "AggregateDPRelease",
    "DPBudgetExhausted",
    "DPBudgetComposition",
    "DPBudgetLedger",
    "DPBudgetSpend",
    "DPAggregateBudgetExceeded",
    "AggregateKind",
    "CompositionRule",
    "DEFAULT_DP_SURROGATE_SENSITIVITIES",
    "DEFAULT_CORRELATION_TOLERANCE",
    "DEFAULT_MARGINAL_TOLERANCE",
    "DEFAULT_MEMBERSHIP_ADVANTAGE_BUDGET",
    "DEFAULT_POLICY_BUDGETS",
    "DEFAULT_QI_WEIGHTS",
    "DEFAULT_RDP_ORDERS",
    "DEFAULT_RISK_BUDGET",
    "EPSILON_POLICY_CONFIG_RESOURCE",
    "BudgetComposition",
    "BudgetDecision",
    "BudgetExceeded",
    "DPGenerationBudgetAccountant",
    "DPSurrogateBudget",
    "DPSurrogateBudgetExceeded",
    "DPSurrogateComposition",
    "DPSurrogateSensitivity",
    "DPSurrogateSensitivityRegistry",
    "DPSurrogateSpend",
    "DiversityClass",
    "DPMechanism",
    "EpsilonPolicy",
    "GenerationSpend",
    "DifferentialPrivacy",
    "ColumnDistribution",
    "EquivalenceClass",
    "GeneralizationPlan",
    "KAnonymityEngine",
    "KAnonymityReport",
    "QIColumnProfile",
    "QIGeneralization",
    "QIProfiler",
    "QIProfilerReport",
    "QuasiIdentifierProfiler",
    "LDiversityChecker",
    "LDiversityEngine",
    "LDiversityReport",
    "RiskBudget",
    "RiskBudgetExceeded",
    "RiskBudgetVerdict",
    "RiskBudgetViolation",
    "LongitudinalCorpus",
    "LongitudinalEvidence",
    "LongitudinalNote",
    "LongitudinalPatient",
    "MembershipSelfTestError",
    "MembershipSelfTestResult",
    "MembershipInferenceReport",
    "MembershipInferenceResult",
    "PopulationRiskAssessment",
    "SurrogateDrawKind",
    "SuppressionProposal",
    "TabularProfile",
    "DEFAULT_RISKIEST_RECORD_COUNT",
    "analyze_k_anonymity",
    "analyze_l_diversity",
    "apply_suppression",
    "apply_generalization_plan",
    "assess_population_risk",
    "budget_for_policy",
    "build_longitudinal_corpus",
    "bounded_membership_inference_self_test",
    "cross_modal_linkage_risk_report",
    "check_l_diversity",
    "epsilon_policy_for",
    "evaluate_budget",
    "load_epsilon_policies",
    "laplace_aggregate",
    "membership_inference_self_test",
    "PrivacyBudget",
    "PrivacyBudgetExceeded",
    "PrivacyBudgetStatus",
    "PrivacySpend",
    "UtilityPoint",
    "UtilityReport",
    "gaussian_mechanism",
    "gaussian_noise",
    "gaussian_scale",
    "gaussian_stddev",
    "laplace_mechanism",
    "laplace_noise",
    "laplace_scale",
    "release_dp_aggregate",
    "release_count",
    "release_histogram",
    "release_mean",
    "release_sum",
    "fit_tabular_profile",
    "longitudinal_attack_fingerprint",
    "longitudinal_risk_report",
    "l_diversity_report",
    "release_aggregate",
    "quasi_identifier_key",
    "quasi_identifier_key_bytes",
    "risk_report",
    "run_membership_inference_self_test",
    "sample_synthetic_table",
    "tabular_fidelity_report",
    "MemoryCeilingError",
    "StreamingKanonDecision",
    "StreamingKanonState",
    "build_generalization_hierarchies",
    "enforce_kanon",
    "kanon_report",
    "propose_suppression",
    "profile_qi",
    "profile_quasi_identifier_risk",
    "profile_quasi_identifiers",
    "diff_audit_reports",
    "AuditDiff",
    "AnonymityPolicy",
    "AnonymizationResult",
    "AttributeDisclosureSummary",
    "GeneralizationSummary",
    "ReleaseAssessment",
    "ReleasedOutputValidation",
    "UtilitySummary",
    "anonymize_release",
    "assess_release",
    "release_dataset_digest",
    "release_schema_digest",
    "render_release_assessment_dashboard",
    "render_risk_dashboard",
    "safe_risk_summary",
    "validate_released_output",
    "write_release_assessment_dashboard",
    "write_risk_dashboard",
    "utility_report",
    "utility_vs_epsilon",
]
