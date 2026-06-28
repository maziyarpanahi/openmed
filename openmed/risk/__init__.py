"""Re-identification risk package for section 4.2.

Intended contents include quasi-identifier detection, uniqueness/k-anonymity
measurement, and adversarial re-identification analysis.
"""

from .audit_diff import AuditDiff, diff_audit_reports
from .kanon import kanon_report
from .reid import risk_report

__all__ = ["risk_report", "kanon_report", "diff_audit_reports", "AuditDiff"]
