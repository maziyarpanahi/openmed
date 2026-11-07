"""Offline ``openmed recovery`` commands for encrypted service-state backups.

Every command prints a value-free inventory or receipt: component categories,
counts, sizes, digests, schema versions, opaque recovery-point identifiers,
and controlled reason codes.  Paths, file names, key identifiers, and key
material are never printed.

Keys are read from caller-owned files given as ``KEY_ID=FILE``.  A key file
holds exactly 32 raw bytes or 64 hexadecimal characters.  The first key given
is the active key; any further keys are decrypt-only (older keys a chain may
still need after rotation).
"""

from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ._output import EXIT_USAGE, CliError, emit

if TYPE_CHECKING:
    from openmed.core.key_provider import InMemoryKeyProvider
    from openmed.service.recovery import (
        RecoveryReceipt,
        ServiceStateInventory,
        ServiceStateSources,
    )

_KEY_FILE_LIMIT = 4096
_KEY_BYTES = 32
_KEY_HELP = (
    "Backup key as KEY_ID=FILE (32 raw bytes or 64 hex characters). Repeat for "
    "older decrypt-only keys; the first key is active."
)


def add_recovery_command(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``openmed recovery`` command group."""

    parser = subparsers.add_parser(
        "recovery",
        help="Back up, verify, restore, and activate encrypted service state.",
    )
    commands = parser.add_subparsers(dest="recovery_command")

    inventory = commands.add_parser(
        "inventory",
        help="List durable service-state categories without values.",
    )
    _add_source_arguments(inventory)
    inventory.set_defaults(handler=_handle_inventory)

    backup = commands.add_parser(
        "backup",
        help="Capture an encrypted, integrity-checked recovery point.",
    )
    backup.add_argument("output", type=Path, help="New recovery-point directory.")
    _add_source_arguments(backup)
    _add_key_argument(backup)
    backup.add_argument(
        "--parent",
        type=Path,
        default=None,
        help="Existing recovery point to extend with an incremental point.",
    )
    backup.add_argument(
        "--writers-stopped",
        action="store_true",
        help=(
            "Assert that admission and every writer are stopped; required "
            "because the CLI has no drain hook."
        ),
    )
    backup.add_argument(
        "--retention-days",
        type=int,
        default=None,
        help="Days after which restore refuses the recovery point.",
    )
    backup.set_defaults(handler=_handle_backup)

    verify = commands.add_parser(
        "verify",
        help="Decrypt and verify a recovery point and its chain in memory.",
    )
    verify.add_argument("point", type=Path, help="Recovery-point directory.")
    _add_key_argument(verify)
    _add_chain_arguments(verify)
    verify.set_defaults(handler=_handle_verify)

    restore = commands.add_parser(
        "restore",
        help="Restore a recovery point into a new, inactive directory.",
    )
    restore.add_argument("point", type=Path, help="Recovery-point directory.")
    restore.add_argument("target", type=Path, help="New inactive target directory.")
    _add_key_argument(restore)
    _add_chain_arguments(restore)
    restore.add_argument(
        "--live",
        type=Path,
        default=None,
        help="Live state link; a target overlapping it is refused.",
    )
    restore.set_defaults(handler=_handle_restore)

    activate = commands.add_parser(
        "activate",
        help="Verify a restored directory and atomically repoint the live link.",
    )
    activate.add_argument("staged", type=Path, help="Restored target directory.")
    activate.add_argument("live", type=Path, help="Live state symlink to repoint.")
    _add_key_argument(activate)
    journey = activate.add_mutually_exclusive_group()
    journey.add_argument(
        "--journey-key",
        action="append",
        default=[],
        metavar="KEY_ID=FILE",
        help=(
            "Journey store key used to open and verify the restored store; "
            "repeat for older keys."
        ),
    )
    journey.add_argument(
        "--journey-plaintext-dev",
        action="store_true",
        help="Verify a plaintext development Journey store (refused in prod).",
    )
    activate.set_defaults(handler=_handle_activate)

    discard = commands.add_parser(
        "discard",
        help="Remove an inactive restored directory.",
    )
    discard.add_argument("staged", type=Path, help="Restored target directory.")
    discard.add_argument(
        "--live",
        type=Path,
        default=None,
        help="Live state link whose current target must not be removed.",
    )
    discard.set_defaults(handler=_handle_discard)


def _add_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--state-root",
        type=Path,
        default=None,
        help="State root using the canonical journey/jobs/registry/audit layout.",
    )
    parser.add_argument(
        "--journey-root",
        type=Path,
        default=None,
        help="Local Journey store directory (artifacts/ and journey.sqlite3).",
    )
    parser.add_argument(
        "--jobs-store",
        type=Path,
        default=None,
        help="Async job metadata store JSON file.",
    )
    parser.add_argument(
        "--registry-state",
        type=Path,
        default=None,
        help="Model-registry slot and pointer state JSON file.",
    )
    parser.add_argument(
        "--audit-chain",
        type=Path,
        action="append",
        default=[],
        help="Audit-chain JSON file; repeat in a stable order.",
    )


def _add_key_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--key",
        action="append",
        required=True,
        metavar="KEY_ID=FILE",
        help=_KEY_HELP,
    )


def _add_chain_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--ancestor",
        type=Path,
        action="append",
        default=[],
        help="Parent recovery point an incremental chain needs; repeatable.",
    )
    parser.add_argument(
        "--max-age-hours",
        type=int,
        default=None,
        help="Refuse recovery points older than this many hours.",
    )


def _sources(args: argparse.Namespace) -> ServiceStateSources:
    from openmed.service.recovery import ServiceStateSources

    explicit = (
        args.journey_root is not None
        or args.jobs_store is not None
        or args.registry_state is not None
        or bool(args.audit_chain)
    )
    if args.state_root is not None and explicit:
        raise CliError(
            "Use either --state-root or explicit component sources.",
            code="sources_conflict",
            exit_code=EXIT_USAGE,
        )
    if args.state_root is not None:
        if not args.state_root.is_dir():
            raise CliError(
                "The state root is not a directory.",
                code="source_unavailable",
                exit_code=EXIT_USAGE,
            )
        return ServiceStateSources.from_state_root(args.state_root)
    if not explicit:
        raise CliError(
            "Name the service state with --state-root or component options.",
            code="sources_required",
            exit_code=EXIT_USAGE,
        )
    return ServiceStateSources(
        journey_root=args.journey_root,
        job_store=args.jobs_store,
        registry_state=args.registry_state,
        audit_chains=tuple(args.audit_chain),
    )


def _read_key_file(path: Path) -> bytes:
    try:
        with open(path, "rb") as handle:
            raw = handle.read(_KEY_FILE_LIMIT + 1)
    except OSError:
        raise CliError(
            "A key file could not be read.",
            code="key_file_unavailable",
            exit_code=EXIT_USAGE,
        ) from None
    if len(raw) == _KEY_BYTES:
        return raw
    text = raw.strip()
    if len(text) == 2 * _KEY_BYTES:
        try:
            return bytes.fromhex(text.decode("ascii"))
        except (UnicodeDecodeError, ValueError):
            pass
    raise CliError(
        "A key file must hold 32 raw bytes or 64 hexadecimal characters.",
        code="key_file_invalid",
        exit_code=EXIT_USAGE,
    )


def _key_provider(specs: list[str], *, purpose: str) -> InMemoryKeyProvider:
    from openmed.core.key_provider import (
        EncryptionError,
        InMemoryKeyProvider,
        KeyState,
    )

    provider = InMemoryKeyProvider()
    for index, spec in enumerate(specs):
        key_id, separator, location = spec.partition("=")
        if not separator or not key_id or not location:
            raise CliError(
                "Keys must be given as KEY_ID=FILE.",
                code="key_spec_invalid",
                exit_code=EXIT_USAGE,
            )
        material = _read_key_file(Path(location))
        try:
            provider.add_key(
                key_id,
                material,
                purpose=purpose,
                state=KeyState.ACTIVE if index == 0 else KeyState.DECRYPT_ONLY,
            )
        except (EncryptionError, ValueError):
            raise CliError(
                "A key specification is invalid or repeated.",
                code="key_spec_invalid",
                exit_code=EXIT_USAGE,
            ) from None
    return provider


def _backup_keys(args: argparse.Namespace) -> InMemoryKeyProvider:
    from openmed.service.recovery import BACKUP_KEY_PURPOSE

    return _key_provider(list(args.key), purpose=BACKUP_KEY_PURPOSE)


def _max_age(args: argparse.Namespace) -> timedelta | None:
    if args.max_age_hours is None:
        return None
    if args.max_age_hours < 1:
        raise CliError(
            "--max-age-hours must be a positive integer.",
            code="max_age_invalid",
            exit_code=EXIT_USAGE,
        )
    return timedelta(hours=args.max_age_hours)


def _handle_inventory(args: argparse.Namespace) -> int:
    from openmed.service.recovery import RecoveryError, inventory_service_state

    try:
        inventory = inventory_service_state(_sources(args))
    except RecoveryError as exc:
        raise CliError(
            f"Service state could not be inventoried ({exc.code}).",
            code=exc.code,
        ) from None
    return emit(args, inventory.to_dict(), human=_format_inventory(inventory))


def _handle_backup(args: argparse.Namespace) -> int:
    from openmed.service.recovery import create_recovery_point

    if args.retention_days is not None and args.retention_days < 1:
        raise CliError(
            "--retention-days must be a positive integer.",
            code="retention_invalid",
            exit_code=EXIT_USAGE,
        )
    receipt = create_recovery_point(
        _sources(args),
        args.output,
        key_provider=_backup_keys(args),
        parent=args.parent,
        writers_stopped=bool(args.writers_stopped),
        retention_days=args.retention_days,
    )
    return _emit_receipt(args, receipt)


def _handle_verify(args: argparse.Namespace) -> int:
    from openmed.service.recovery import verify_recovery_point

    receipt = verify_recovery_point(
        args.point,
        key_provider=_backup_keys(args),
        ancestors=tuple(args.ancestor),
        max_age=_max_age(args),
    )
    return _emit_receipt(args, receipt)


def _handle_restore(args: argparse.Namespace) -> int:
    from openmed.service.recovery import restore_recovery_point

    receipt = restore_recovery_point(
        args.point,
        args.target,
        key_provider=_backup_keys(args),
        ancestors=tuple(args.ancestor),
        live=args.live,
        max_age=_max_age(args),
    )
    return _emit_receipt(args, receipt)


def _handle_activate(args: argparse.Namespace) -> int:
    from openmed.service.recovery import (
        JOURNEY_DIRECTORY,
        activate_restored_state,
        journey_store_check,
    )
    from openmed.structured.store import JOURNEY_KEY_PURPOSE, JourneyStoreEncryption

    checks = []
    if args.journey_key:
        encryption = JourneyStoreEncryption.encrypted(
            _key_provider(list(args.journey_key), purpose=JOURNEY_KEY_PURPOSE)
        )
        checks.append(journey_store_check(encryption))
    elif args.journey_plaintext_dev:
        checks.append(
            journey_store_check(JourneyStoreEncryption.development_plaintext())
        )
    elif (args.staged / JOURNEY_DIRECTORY).is_dir():
        raise CliError(
            "The restored state has a Journey store; pass --journey-key so it "
            "is opened and verified before activation.",
            code="journey_check_required",
            exit_code=EXIT_USAGE,
        )
    receipt = activate_restored_state(
        args.staged,
        args.live,
        key_provider=_backup_keys(args),
        checks=checks,
    )
    return _emit_receipt(args, receipt)


def _handle_discard(args: argparse.Namespace) -> int:
    from openmed.service.recovery import discard_restored_state

    return _emit_receipt(args, discard_restored_state(args.staged, live=args.live))


def _emit_receipt(args: argparse.Namespace, receipt: RecoveryReceipt) -> int:
    return emit(
        args,
        receipt.to_dict(),
        human=_format_receipt(receipt),
        outcome="completed" if receipt.completed else "refused",
    )


def _format_inventory(inventory: ServiceStateInventory) -> str:
    lines = []
    for entry in inventory.entries:
        line = f"{entry.category}: {entry.disposition.value}"
        if entry.disposition.value == "authoritative":
            line += f", {entry.objects} object(s), {entry.bytes} bytes"
            if entry.schema_version is not None:
                line += f", schema {entry.schema_version}"
        lines.append(line)
    return "\n".join(lines)


def _format_receipt(receipt: RecoveryReceipt) -> str:
    head = f"recovery {receipt.operation} {receipt.outcome.value}"
    if receipt.reason is not None:
        head += f": {receipt.reason}"
    lines = [head]
    if receipt.recovery_point_id is not None:
        point = f"recovery point {receipt.recovery_point_id}"
        if receipt.kind is not None:
            point += f" ({receipt.kind}"
            if receipt.sequence is not None:
                point += f", sequence {receipt.sequence}"
            point += ")"
        lines.append(point)
    for component in receipt.components:
        line = (
            f"  {component.category}: {component.objects} object(s), "
            f"{component.bytes} bytes"
        )
        if component.schema_version is not None:
            line += f", schema {component.schema_version}"
        lines.append(line)
    details: dict[str, Any] = dict(receipt.details)
    if details:
        lines.append(
            "  " + ", ".join(f"{key}={value}" for key, value in sorted(details.items()))
        )
    return "\n".join(lines)
