"""Re-identification risk package for section 4.2.

Intended contents include quasi-identifier detection, uniqueness/k-anonymity
measurement, and adversarial re-identification analysis.
"""

from .dashboard import render_risk_dashboard, write_risk_dashboard
from .kanon import kanon_report
from .reid import risk_report

__all__ = [
    "risk_report",
    "kanon_report",
    "render_risk_dashboard",
    "write_risk_dashboard",
]
