"""Structured data privacy package for section 4.2.

Intended contents include column classification, k-anonymity, l-diversity,
t-closeness, and differential privacy capabilities.
"""

from .discharge_summary import (
    REQUIRED_DISCHARGE_SLOTS,
    DischargeSlotName,
    DischargeSummary,
    DischargeSummarySection,
    canonical_discharge_slot,
    structure_discharge_summary,
)
from .flowsheet import (
    FLOWSHEET_ADVISORY,
    Flowsheet,
    ParameterSeries,
    TimeSeriesPoint,
    structure_flowsheet,
)
from .hierarchies import (
    COLUMN_TYPE_AGE,
    COLUMN_TYPE_DATE,
    COLUMN_TYPE_ZIP,
    HIERARCHY_SCHEMA_VERSION,
    SUPPORTED_COLUMN_TYPES,
    GeneralizationLevel,
    Hierarchy,
    HierarchyError,
    describe_level,
    generalize_value,
    get_hierarchy,
    max_level,
)
from .lab_panels import (
    LAB_PANEL_ADVISORY,
    PANEL_ORDER,
    AnalyteRow,
    LabPanel,
    canonical_analyte,
    parse_lab_report,
    structure_lab_panels,
)
from .qi_detect import (
    ROLE_DIRECT_ID,
    ROLE_FREE_TEXT,
    ROLE_INTERNAL_LINKAGE,
    ROLE_QUASI_ID,
    ROLE_SAFE,
    ROLE_SENSITIVE,
    DiscoveryConfigurationError,
    scan_table,
)
from .table_io import SUPPORTED_TABLE_SUFFIXES, read_table, write_table

__all__ = [
    "COLUMN_TYPE_AGE",
    "COLUMN_TYPE_DATE",
    "COLUMN_TYPE_ZIP",
    "FLOWSHEET_ADVISORY",
    "HIERARCHY_SCHEMA_VERSION",
    "LAB_PANEL_ADVISORY",
    "PANEL_ORDER",
    "REQUIRED_DISCHARGE_SLOTS",
    "SUPPORTED_COLUMN_TYPES",
    "AnalyteRow",
    "DischargeSlotName",
    "DischargeSummary",
    "DischargeSummarySection",
    "DiscoveryConfigurationError",
    "Flowsheet",
    "GeneralizationLevel",
    "Hierarchy",
    "HierarchyError",
    "LabPanel",
    "ParameterSeries",
    "ROLE_DIRECT_ID",
    "ROLE_FREE_TEXT",
    "ROLE_INTERNAL_LINKAGE",
    "ROLE_QUASI_ID",
    "ROLE_SAFE",
    "ROLE_SENSITIVE",
    "SUPPORTED_TABLE_SUFFIXES",
    "TimeSeriesPoint",
    "canonical_analyte",
    "canonical_discharge_slot",
    "describe_level",
    "generalize_value",
    "get_hierarchy",
    "max_level",
    "parse_lab_report",
    "read_table",
    "scan_table",
    "structure_discharge_summary",
    "structure_flowsheet",
    "structure_lab_panels",
    "write_table",
]
