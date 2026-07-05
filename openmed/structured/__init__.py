"""Structured data privacy package for section 4.2.

Intended contents include column classification, k-anonymity, l-diversity,
t-closeness, and differential privacy capabilities.
"""

from .qi_detect import (
    ROLE_DIRECT_ID,
    ROLE_QUASI_ID,
    ROLE_SAFE,
    ROLE_SENSITIVE,
    scan_table,
)

__all__ = [
    "ROLE_DIRECT_ID",
    "ROLE_QUASI_ID",
    "ROLE_SAFE",
    "ROLE_SENSITIVE",
    "scan_table",
]
