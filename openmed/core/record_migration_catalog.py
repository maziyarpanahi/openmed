"""Built-in stored-record migrators for the default migration registry.

Each entry adapts an existing public migration function; no migration logic
lives here. The owning modules are imported only when one of their migrators
runs, so identifying a record never loads the clinical, structured or risk
packages.

Registered edges:

- ``clinical_review_packet`` 1.0.0 to 1.1.0, lossless, through
  ``openmed.clinical.review_packet_migrations.migrate_review_packet``.
- ``clinical_review_packet_mapping`` 1 to 2, lossless, through
  ``openmed.clinical.review_packet_mapping_migrations.migrate_review_packet``.
- ``decision_result`` 1.0.0 to 1.1.0 and 1.1.0 to 1.2.0, lossless, through
  ``openmed.structured.decision.migrate_decision_result``.
- ``dp_budget_ledger`` unversioned to ``openmed.dp_budget_ledger.v1``, lossy,
  through ``openmed.risk.budget_migration.BudgetLedgerSnapshot`` and checked
  with ``enforce_budget_migration``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from .errors import LegacyRootedError
from .record_migrations import Migrator, MigratorRegistry

CLINICAL_REVIEW_PACKET: Final = "clinical_review_packet"
"""Semver clinical review packets (``openmed.clinical.review_transitions``)."""

CLINICAL_REVIEW_PACKET_MAPPING: Final = "clinical_review_packet_mapping"
"""Integer-versioned review packet mappings."""

DECISION_RESULT: Final = "decision_result"
"""Fixed-option decision results (``openmed.structured.decision``)."""

DP_BUDGET_LEDGER: Final = "dp_budget_ledger"
"""Differential-privacy budget snapshots (``openmed.risk.budget_migration``)."""

DP_BUDGET_LEDGER_VERSION: Final = "openmed.dp_budget_ledger.v1"
"""Canonical budget ledger snapshot version."""

UNVERSIONED: Final = "unversioned"
"""Version assigned to budget snapshots that carry no ``schema_version``."""

_REVIEW_PACKET_FIELDS = frozenset({"packet_id", "conflict_id", "fact_ids", "state"})
_DECISION_RESULT_FIELDS = frozenset(
    {"advisory", "autonomous_action", "mode", "option_scores", "state"}
)


class BuiltinMigrationError(ValueError, LegacyRootedError):
    """An adapted migration function reported a non-success outcome.

    The message is a controlled outcome code from the owning module and never
    contains record values.
    """


def register_record_migrations(registry: MigratorRegistry) -> None:
    """Register the built-in schemas and migration edges.

    Args:
        registry: Registry to populate, normally the default registry.
    """

    registry.register_schema(
        CLINICAL_REVIEW_PACKET,
        current_version="1.1.0",
        matcher=_is_clinical_review_packet,
    )
    registry.register_migrator(
        CLINICAL_REVIEW_PACKET,
        "1.0.0",
        "1.1.0",
        _clinical_review_packet_to_1_1_0,
        lossless=True,
        fixture="clinical_review_packet/1.0.0-to-1.1.0",
    )
    registry.register_schema(CLINICAL_REVIEW_PACKET_MAPPING, current_version=2)
    registry.register_migrator(
        CLINICAL_REVIEW_PACKET_MAPPING,
        1,
        2,
        _review_packet_mapping_to_2,
        lossless=True,
        fixture="clinical_review_packet_mapping/1-to-2",
    )
    registry.register_schema(
        DECISION_RESULT,
        current_version="1.2.0",
        matcher=_is_decision_result,
    )
    for source, target in (("1.0.0", "1.1.0"), ("1.1.0", "1.2.0")):
        registry.register_migrator(
            DECISION_RESULT,
            source,
            target,
            _decision_result_to(target),
            lossless=True,
            fixture=f"decision_result/{source}-to-{target}",
        )
    registry.register_schema(
        DP_BUDGET_LEDGER,
        current_version=DP_BUDGET_LEDGER_VERSION,
        version_when_missing=UNVERSIONED,
        matcher=_is_dp_budget_snapshot,
    )
    registry.register_migrator(
        DP_BUDGET_LEDGER,
        UNVERSIONED,
        DP_BUDGET_LEDGER_VERSION,
        _dp_budget_snapshot_to_ledger,
        lossless=False,
        fixture="dp_budget_ledger/unversioned-to-v1",
    )


def _clinical_review_packet_to_1_1_0(record: dict[str, Any]) -> dict[str, Any]:
    from openmed.clinical.review_packet_migrations import migrate_review_packet

    result = migrate_review_packet(record, target_version="1.1.0")
    value = getattr(result, "value", None)
    if not getattr(result, "ok", False) or value is None:
        raise BuiltinMigrationError(str(getattr(result, "code", "failed")))
    return value.packet.to_dict()


def _review_packet_mapping_to_2(record: dict[str, Any]) -> dict[str, Any]:
    from openmed.clinical.review_packet_mapping_migrations import (
        migrate_review_packet,
    )

    return dict(migrate_review_packet(record, target_version=2).packet)


def _decision_result_to(target: str) -> Migrator:
    def migrate(record: dict[str, Any]) -> dict[str, Any]:
        from openmed.structured.decision import migrate_decision_result

        return migrate_decision_result(record, target_version=target)

    migrate.__name__ = f"decision_result_to_{target.replace('.', '_')}"
    return migrate


def _dp_budget_snapshot_to_ledger(record: dict[str, Any]) -> dict[str, Any]:
    from openmed.risk.budget_migration import (
        BudgetLedgerSnapshot,
        enforce_budget_migration,
    )

    migrated = BudgetLedgerSnapshot.from_mapping(record).to_dict()
    enforce_budget_migration(record, migrated)
    return migrated


def _is_clinical_review_packet(record: Mapping[str, Any]) -> bool:
    return isinstance(record.get("schema_version"), str) and (
        _REVIEW_PACKET_FIELDS <= record.keys()
    )


def _is_decision_result(record: Mapping[str, Any]) -> bool:
    return isinstance(record.get("schema_version"), str) and (
        _DECISION_RESULT_FIELDS <= record.keys()
    )


def _is_dp_budget_snapshot(record: Mapping[str, Any]) -> bool:
    version = record.get("schema_version")
    if version is None:
        return {"compositions", "policies"} <= record.keys()
    return isinstance(version, str) and version.startswith("openmed.dp_budget_ledger.")


__all__ = [
    "CLINICAL_REVIEW_PACKET",
    "CLINICAL_REVIEW_PACKET_MAPPING",
    "DECISION_RESULT",
    "DP_BUDGET_LEDGER",
    "DP_BUDGET_LEDGER_VERSION",
    "UNVERSIONED",
    "BuiltinMigrationError",
    "register_record_migrations",
]
