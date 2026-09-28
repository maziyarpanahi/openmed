"""Shared operational contract for production Journey deployments.

The entry point intentionally emits only controlled state, schema versions,
counts, and reason codes.  Connection strings, artifact contents, clinical
values, and exception messages never enter its output or HTTP responses.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from enum import Enum
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Mapping, Sequence

from openmed.clinical.journey_contracts import ClinicalArtifact, sha256_digest
from openmed.structured.store import (
    FsspecArtifactStore,
    MigrationHealth,
    PostgresJourneyStore,
    StoreState,
)

DEPLOYMENT_HEALTH_SCHEMA_VERSION = "openmed.journey.deployment-health.v1"
DEPLOYMENT_COMPATIBILITY_MAJOR = 1
GOLDEN_CONTRACT_VERSION = "openmed.journey.golden.v1"
DEFAULT_SCHEMA = "openmed_journey"
DEFAULT_ARTIFACT_ROOT = "/artifacts"
DEFAULT_WORKER_HOST = "0.0.0.0"
DEFAULT_WORKER_PORT = 8091
DEFAULT_PROBE_TIMEOUT_SECONDS = 3.0

_GOLDEN_CONTENT = b"openmed synthetic deployment golden journey v1"
_GOLDEN_RECORDED_AT = "2026-01-01T00:00:00Z"
_ARTIFACT_PROBE_CONTENT = b"openmed synthetic artifact readiness v1"


class ComponentState(str, Enum):
    """Typed readiness state shared by CLI and HTTP health output."""

    READY = "ready"
    PENDING = "pending"
    UNAVAILABLE = "unavailable"
    INCOMPATIBLE = "incompatible"
    DISABLED = "disabled"


@dataclass(frozen=True, slots=True)
class ComponentHealth:
    """Value-free state for one deployment component."""

    state: ComponentState
    code: str
    schema_version: int | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "state": self.state.value,
            "code": self.code,
        }
        if self.schema_version is not None:
            payload["schema_version"] = self.schema_version
        return payload


@dataclass(frozen=True, slots=True)
class DeploymentHealth:
    """Stable deployment health response with fixed component names."""

    components: Mapping[str, ComponentHealth]

    @property
    def ready(self) -> bool:
        return all(
            component.state in {ComponentState.READY, ComponentState.DISABLED}
            for component in self.components.values()
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": DEPLOYMENT_HEALTH_SCHEMA_VERSION,
            "compatibility_major": DEPLOYMENT_COMPATIBILITY_MAJOR,
            "state": "ready" if self.ready else "not_ready",
            "components": {
                name: self.components[name].to_dict()
                for name in ("migration", "store", "artifact", "worker", "model")
            },
        }


def _open_store(
    dsn: str,
    *,
    schema: str,
) -> tuple[PostgresJourneyStore | None, ComponentHealth, ComponentHealth]:
    result = PostgresJourneyStore.connect(
        dsn,
        schema=schema,
        connect_options={"connect_timeout": 3},
    )
    if not result.ok or result.value is None:
        state = (
            ComponentState.INCOMPATIBLE
            if result.state is StoreState.UNSUPPORTED
            else ComponentState.UNAVAILABLE
        )
        code = result.code or "store_open_failed"
        return None, ComponentHealth(state, code), ComponentHealth(state, code)

    store = result.value
    migration = store.migration_report
    if migration.state is not MigrationHealth.HEALTHY:
        state = (
            ComponentState.INCOMPATIBLE
            if migration.state in {MigrationHealth.UNSUPPORTED, MigrationHealth.DRIFTED}
            else ComponentState.PENDING
        )
        return (
            store,
            ComponentHealth(
                state,
                f"migration_{migration.state.value}",
                migration.current_version,
            ),
            ComponentHealth(ComponentState.PENDING, "migration_not_ready"),
        )

    try:
        current_version = store.schema_version
    except RuntimeError:
        store_health = ComponentHealth(
            ComponentState.UNAVAILABLE,
            "store_probe_failed",
        )
    else:
        store_health = (
            ComponentHealth(ComponentState.READY, "store_ready")
            if current_version == migration.target_version
            else ComponentHealth(ComponentState.PENDING, "store_schema_pending")
        )
    return (
        store,
        ComponentHealth(
            ComponentState.READY,
            "migration_current",
            migration.current_version,
        ),
        store_health,
    )


def _probe_artifact(root: str) -> ComponentHealth:
    try:
        store = FsspecArtifactStore(root)
    except (RuntimeError, TypeError, ValueError):
        return ComponentHealth(ComponentState.UNAVAILABLE, "artifact_store_unavailable")
    if store.protocol != "file":
        return ComponentHealth(ComponentState.INCOMPATIBLE, "artifact_protocol_denied")
    probe = ClinicalArtifact(
        artifact_id="artifact_deploymentprobe0001",
        artifact_type="operational_probe",
        media_type="application/octet-stream",
        content_hash=sha256_digest(_ARTIFACT_PROBE_CONTENT),
        byte_size=len(_ARTIFACT_PROBE_CONTENT),
        source_id="source_deploymentprobe0001",
        recorded_at=_GOLDEN_RECORDED_AT,
        attributes={"fixture": "synthetic", "purpose": "readiness"},
    )
    written = store.put_bytes(probe, _ARTIFACT_PROBE_CONTENT)
    if not written.ok:
        return ComponentHealth(
            ComponentState.UNAVAILABLE,
            written.code or "artifact_write_failed",
        )
    read = store.get_bytes(probe.content_hash)
    if not read.ok or read.value != _ARTIFACT_PROBE_CONTENT:
        return ComponentHealth(
            ComponentState.UNAVAILABLE,
            read.code or "artifact_read_failed",
        )
    return ComponentHealth(ComponentState.READY, "artifact_store_ready")


def _probe_http(url: str | None, *, component: str) -> ComponentHealth:
    if not url:
        return ComponentHealth(ComponentState.DISABLED, f"{component}_disabled")
    try:
        with urllib.request.urlopen(  # noqa: S310 - deployment-owned URL only
            url,
            timeout=DEFAULT_PROBE_TIMEOUT_SECONDS,
        ) as response:
            if response.status < 200 or response.status >= 300:
                return ComponentHealth(
                    ComponentState.UNAVAILABLE,
                    f"{component}_not_ready",
                )
    except (OSError, urllib.error.URLError, ValueError):
        return ComponentHealth(
            ComponentState.UNAVAILABLE,
            f"{component}_unavailable",
        )
    return ComponentHealth(ComponentState.READY, f"{component}_ready")


def probe_deployment(
    dsn: str,
    *,
    artifact_root: str = DEFAULT_ARTIFACT_ROOT,
    schema: str = DEFAULT_SCHEMA,
    worker_url: str | None = None,
    model_url: str | None = None,
) -> DeploymentHealth:
    """Probe all configured deployment planes without exposing stored values."""

    store, migration, store_health = _open_store(dsn, schema=schema)
    if store is not None:
        store.close()
    return DeploymentHealth(
        components={
            "migration": migration,
            "store": store_health,
            "artifact": _probe_artifact(artifact_root),
            "worker": _probe_http(worker_url, component="worker"),
            "model": _probe_http(model_url, component="model"),
        }
    )


def run_migrations(dsn: str, *, schema: str = DEFAULT_SCHEMA) -> DeploymentHealth:
    """Apply and verify the append-only PostgreSQL migration chain."""

    store, migration, store_health = _open_store(dsn, schema=schema)
    if store is not None:
        store.close()
    return DeploymentHealth(
        components={
            "migration": migration,
            "store": store_health,
            "artifact": ComponentHealth(ComponentState.DISABLED, "artifact_disabled"),
            "worker": ComponentHealth(ComponentState.DISABLED, "worker_disabled"),
            "model": ComponentHealth(ComponentState.DISABLED, "model_disabled"),
        }
    )


def run_golden_journey(
    dsn: str,
    *,
    artifact_root: str = DEFAULT_ARTIFACT_ROOT,
    schema: str = DEFAULT_SCHEMA,
) -> dict[str, Any]:
    """Commit and read back one idempotent synthetic deployment artifact."""

    store_result = PostgresJourneyStore.connect(
        dsn,
        schema=schema,
        connect_options={"connect_timeout": 3},
    )
    if not store_result.ok or store_result.value is None:
        return _golden_result("failed", store_result.code or "store_open_failed")

    store = store_result.value
    try:
        try:
            objects = FsspecArtifactStore(artifact_root)
        except (RuntimeError, TypeError, ValueError):
            return _golden_result("failed", "artifact_store_unavailable")

        artifact = ClinicalArtifact(
            artifact_id="artifact_deploymentgolden001",
            artifact_type="clinical_note",
            media_type="text/plain",
            content_hash=sha256_digest(_GOLDEN_CONTENT),
            byte_size=len(_GOLDEN_CONTENT),
            source_id="source_deploymentgolden001",
            recorded_at=_GOLDEN_RECORDED_AT,
            attributes={"fixture": "synthetic", "contract": GOLDEN_CONTRACT_VERSION},
        )
        object_result = objects.put_bytes(artifact, _GOLDEN_CONTENT)
        if not object_result.ok:
            return _golden_result(
                "failed",
                object_result.code or "artifact_write_failed",
            )
        metadata_result = store.put_artifact(
            artifact,
            committed_at=_GOLDEN_RECORDED_AT,
        )
        if not metadata_result.ok:
            if object_result.created:
                objects.discard_if_created(artifact.content_hash)
            return _golden_result(
                "failed",
                metadata_result.code or "metadata_write_failed",
            )
        stored = store.get_artifact(artifact.artifact_id)
        content = objects.get_bytes(artifact.content_hash)
        if (
            not stored.ok
            or stored.value != artifact
            or not content.ok
            or content.value != _GOLDEN_CONTENT
        ):
            return _golden_result("failed", "golden_readback_failed")
        return {
            **_golden_result("passed", "golden_journey_verified"),
            "created": bool(metadata_result.created),
            "artifact_count": 1,
        }
    finally:
        store.close()


def _golden_result(state: str, code: str) -> dict[str, Any]:
    return {
        "schema_version": GOLDEN_CONTRACT_VERSION,
        "compatibility_major": DEPLOYMENT_COMPATIBILITY_MAJOR,
        "state": state,
        "code": code,
        "synthetic": True,
    }


class _WorkerState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._report = DeploymentHealth(
            components={
                "migration": ComponentHealth(ComponentState.PENDING, "probe_pending"),
                "store": ComponentHealth(ComponentState.PENDING, "probe_pending"),
                "artifact": ComponentHealth(ComponentState.PENDING, "probe_pending"),
                "worker": ComponentHealth(ComponentState.READY, "worker_live"),
                "model": ComponentHealth(ComponentState.DISABLED, "model_disabled"),
            }
        )

    def update(self, report: DeploymentHealth) -> None:
        components = dict(report.components)
        components["worker"] = ComponentHealth(ComponentState.READY, "worker_ready")
        components["model"] = ComponentHealth(ComponentState.DISABLED, "model_disabled")
        with self._lock:
            self._report = DeploymentHealth(components=components)

    def report(self) -> DeploymentHealth:
        with self._lock:
            return self._report


def _serve_worker(
    dsn: str,
    *,
    artifact_root: str,
    schema: str,
    host: str,
    port: int,
    interval_seconds: float,
) -> int:
    state = _WorkerState()
    stopped = threading.Event()

    def refresh() -> None:
        while not stopped.is_set():
            state.update(
                probe_deployment(
                    dsn,
                    artifact_root=artifact_root,
                    schema=schema,
                )
            )
            stopped.wait(interval_seconds)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract
            report = state.report()
            if self.path == "/livez":
                body = {
                    "schema_version": DEPLOYMENT_HEALTH_SCHEMA_VERSION,
                    "state": "live",
                }
                status = 200
            elif self.path in {"/readyz", "/healthz"}:
                body = report.to_dict()
                status = 200 if report.ready else 503
            else:
                body = {
                    "schema_version": DEPLOYMENT_HEALTH_SCHEMA_VERSION,
                    "state": "not_found",
                }
                status = 404
            encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, format: str, *args: object) -> None:
            del format, args

    thread = threading.Thread(target=refresh, name="journey-worker-probe", daemon=True)
    thread.start()
    server = ThreadingHTTPServer((host, port), Handler)

    def stop(_signum: int, _frame: Any) -> None:
        stopped.set()
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        stopped.set()
        server.server_close()
        thread.join(timeout=max(interval_seconds, 1.0) + 1.0)
    return 0


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"{name} is required")
    return value


def _print_json(payload: Mapping[str, Any]) -> None:
    print(json.dumps(dict(payload), sort_keys=True, separators=(",", ":")))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="openmed-journey")
    parser.add_argument(
        "--dsn-env",
        default="OPENMED_JOURNEY_POSTGRES_DSN",
        help="Environment variable containing the PostgreSQL DSN.",
    )
    parser.add_argument(
        "--schema", default=os.getenv("OPENMED_JOURNEY_SCHEMA", DEFAULT_SCHEMA)
    )
    parser.add_argument(
        "--artifact-root",
        default=os.getenv("OPENMED_JOURNEY_ARTIFACT_ROOT", DEFAULT_ARTIFACT_ROOT),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("migrate")
    commands.add_parser("golden")
    probe = commands.add_parser("probe")
    probe.add_argument("--worker-url")
    probe.add_argument("--model-url")
    worker = commands.add_parser("worker")
    worker.add_argument("--host", default=DEFAULT_WORKER_HOST)
    worker.add_argument("--port", type=int, default=DEFAULT_WORKER_PORT)
    worker.add_argument("--interval-seconds", type=float, default=10.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run migrations, deployment probes, the worker, or the golden journey."""

    args = _parser().parse_args(argv)
    try:
        dsn = _required_env(args.dsn_env)
    except ValueError:
        _print_json(
            {
                "schema_version": DEPLOYMENT_HEALTH_SCHEMA_VERSION,
                "state": "not_ready",
                "code": "dsn_not_configured",
            }
        )
        return 2

    if args.command == "migrate":
        report = run_migrations(dsn, schema=args.schema)
        _print_json(report.to_dict())
        return 0 if report.ready else 1
    if args.command == "golden":
        result = run_golden_journey(
            dsn,
            artifact_root=args.artifact_root,
            schema=args.schema,
        )
        _print_json(result)
        return 0 if result["state"] == "passed" else 1
    if args.command == "probe":
        report = probe_deployment(
            dsn,
            artifact_root=args.artifact_root,
            schema=args.schema,
            worker_url=args.worker_url,
            model_url=args.model_url,
        )
        _print_json(report.to_dict())
        return 0 if report.ready else 1
    if args.command == "worker":
        if args.interval_seconds <= 0 or not 1 <= args.port <= 65535:
            _print_json(
                {
                    "schema_version": DEPLOYMENT_HEALTH_SCHEMA_VERSION,
                    "state": "not_ready",
                    "code": "worker_configuration_invalid",
                }
            )
            return 2
        return _serve_worker(
            dsn,
            artifact_root=args.artifact_root,
            schema=args.schema,
            host=args.host,
            port=args.port,
            interval_seconds=args.interval_seconds,
        )
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
