"""Re-identification risk package for section 4.2.

Intended contents include quasi-identifier detection, uniqueness/k-anonymity
measurement, and adversarial re-identification analysis.
"""

from .reid import risk_report

__all__ = ["risk_report"]
