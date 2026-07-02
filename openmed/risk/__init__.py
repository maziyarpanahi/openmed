"""Re-identification risk package for section 4.2."""

from .audit_diff import AuditDiff, diff_audit_reports
from .budget import (
    DEFAULT_POLICY_BUDGETS,
    DEFAULT_QI_WEIGHTS,
    DEFAULT_RISK_BUDGET,
    RiskBudget,
    RiskBudgetExceeded,
    RiskBudgetVerdict,
    RiskBudgetViolation,
    budget_for_policy,
    evaluate_budget,
)
from .dashboard import render_risk_dashboard, write_risk_dashboard
from .kanon import build_generalization_hierarchies, enforce_kanon, kanon_report
from .reid import risk_report

__all__ = [
    "DEFAULT_POLICY_BUDGETS",
    "DEFAULT_QI_WEIGHTS",
    "DEFAULT_RISK_BUDGET",
    "RiskBudget",
    "RiskBudgetExceeded",
    "RiskBudgetVerdict",
    "RiskBudgetViolation",
    "budget_for_policy",
    "evaluate_budget",
    "risk_report",
    "build_generalization_hierarchies",
    "enforce_kanon",
    "kanon_report",
    "diff_audit_reports",
    "AuditDiff",
    "render_risk_dashboard",
    "write_risk_dashboard",
]
