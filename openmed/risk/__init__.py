"""Re-identification risk package for section 4.2.

Intended contents include quasi-identifier detection, uniqueness/k-anonymity
measurement, and adversarial re-identification analysis.
"""

from .audit_diff import AuditDiff, diff_audit_reports
from .dashboard import render_risk_dashboard, write_risk_dashboard
from .kanon import kanon_report
from .reid import risk_report

__all__ = [
    "risk_report",
    "kanon_report",
    "diff_audit_reports",
    "AuditDiff",
    "render_risk_dashboard",
    "write_risk_dashboard",
]
