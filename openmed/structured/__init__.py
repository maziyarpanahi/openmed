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
from .relational import (
    RELATIONAL_ADVISORY,
    ColumnRef,
    DanglingForeignKeyError,
    DateColumn,
    ForeignKey,
    KeySpace,
    RelationalDeidentificationResult,
    RelationalSchema,
    RelationalSchemaError,
    SurrogateManifest,
    deidentify_linked_tables,
)
from .scan import (
    ColumnClassification,
    ColumnRole,
    RoleOverrideError,
    TableRoleScan,
)
from .scan import scan_table as scan_column_roles
from .table_io import SUPPORTED_TABLE_SUFFIXES, read_table, write_table
from .tables import (
    TABLE_ADVISORY,
    Table,
    TableCell,
    TableToken,
    cell_at,
    structure_table,
)

__all__ = [
    "FLOWSHEET_ADVISORY",
    "LAB_PANEL_ADVISORY",
    "PANEL_ORDER",
    "RELATIONAL_ADVISORY",
    "REQUIRED_DISCHARGE_SLOTS",
    "AnalyteRow",
    "ColumnClassification",
    "ColumnRef",
    "ColumnRole",
    "DanglingForeignKeyError",
    "DateColumn",
    "DischargeSlotName",
    "DischargeSummary",
    "DischargeSummarySection",
    "DiscoveryConfigurationError",
    "Flowsheet",
    "ForeignKey",
    "KeySpace",
    "LabPanel",
    "ParameterSeries",
    "ROLE_DIRECT_ID",
    "ROLE_FREE_TEXT",
    "ROLE_INTERNAL_LINKAGE",
    "ROLE_QUASI_ID",
    "ROLE_SAFE",
    "ROLE_SENSITIVE",
    "RelationalDeidentificationResult",
    "RelationalSchema",
    "RelationalSchemaError",
    "RoleOverrideError",
    "SUPPORTED_TABLE_SUFFIXES",
    "SurrogateManifest",
    "TABLE_ADVISORY",
    "Table",
    "TableCell",
    "TableToken",
    "TableRoleScan",
    "TimeSeriesPoint",
    "canonical_analyte",
    "cell_at",
    "canonical_discharge_slot",
    "deidentify_linked_tables",
    "parse_lab_report",
    "read_table",
    "scan_column_roles",
    "scan_table",
    "structure_discharge_summary",
    "structure_flowsheet",
    "structure_lab_panels",
    "structure_table",
    "write_table",
]
