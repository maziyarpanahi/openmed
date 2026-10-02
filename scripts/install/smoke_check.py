#!/usr/bin/env python3
"""Run a deterministic, offline smoke check against an installed OpenMed."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

SCHEMA_VERSION = 1
COMMAND_TIMEOUT_SECONDS = 30

_OFFLINE_FLAGS = {
    "OPENMED_OFFLINE": "1",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
}
_MANIFEST_ROW_RE = re.compile(r"\(([1-9][0-9]{0,8}) rows? checked\)")
_MAX_VERSION_LENGTH = 64
_SAFE_VERSION_PATTERN = (
    r"[0-9]+(?:\.[0-9]+){2}"
    r"(?:(?:a|b|rc)[0-9]+)?"
    r"(?:\.post[0-9]+)?"
    r"(?:\.dev[0-9]+)?"
    r"(?:\+[0-9A-Za-z]+(?:[._-][0-9A-Za-z]+)*)?"
)
_VERSION_RE = re.compile(rf"^openmed ({_SAFE_VERSION_PATTERN})$")
_SAFE_VERSION_RE = re.compile(rf"^{_SAFE_VERSION_PATTERN}$")
_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
BRIEF_PROFILES = ("core", "adapters")
BRIEF_RESOURCE = "core/schemas/json/clinical_review_packet.schema.json"
BRIEF_CHECKS = (
    "installed_origin",
    "packaged_resources",
    "brief_python",
    "missing_runtime",
    "privacy_sentinel",
    "brief_cli",
    "brief_rest_client",
    "brief_mcp",
    "no_network",
)

# The probe carries only a synthetic marker. It emits hashes and offsets, never
# the marker itself, so captured child output remains safe for release logs.
_SYNTHETIC_PROBE = r"""
import json
from importlib.metadata import distribution
from types import SimpleNamespace

from openmed import redaction_preview

text = "Synthetic install marker OM-SMOKE-0001"
marker = "OM-SMOKE-0001"
start = text.index(marker)
result = SimpleNamespace(
    deidentified_text=text.replace(marker, "[ID_NUM]"),
    method="mask",
    pii_entities=[
        SimpleNamespace(
            action="mask",
            end=start + len(marker),
            label="ID_NUM",
            redacted_text="[ID_NUM]",
            start=start,
        )
    ],
)
preview = redaction_preview(text, result)
installed_distribution = distribution("openmed")
entry_point_declared = any(
    item.name == "openmed" and item.value == "openmed.cli:main"
    for item in installed_distribution.entry_points
)
if not entry_point_declared or preview["change_count"] != 1:
    raise SystemExit(1)
change = preview["changes"][0]
print(
    json.dumps(
        {
            "change_count": preview["change_count"],
            "document_hash": preview["document_hash"],
            "entry_point_declared": entry_point_declared,
            "package_version": installed_distribution.version,
            "surface_hash": change["surface_hash"],
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
)
"""


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


def _brief_fixture():
    """Create synthetic reviewed evidence, never model-quality evidence."""
    import hashlib
    from datetime import datetime

    from openmed.clinical.brief import BriefContext, BriefFact, brief_policy_fingerprint
    from openmed.clinical.evidence_packet import (
        build_evidence_packet,
        fingerprint_evidence_review,
    )
    from openmed.clinical.nli_gate import NLIThresholds
    from openmed.clinical.review_state_machine import (
        ReviewState,
        ReviewStateMachine,
        make_opaque_event_id,
    )
    from openmed.core.pii import DeidentificationResult

    sentences = (
        "The admission problem was dehydration.",
        "The discharge diagnosis was dehydration.",
        "Symptoms improved after fluids.",
    )
    text = " ".join(sentences)
    fields = ("admission_reason", "discharge_diagnoses", "hospital_course")
    facts = tuple(
        BriefFact(
            f"synthetic:ref-{i}", name, "affirmed", "certain", "recent", "patient"
        )
        for i, name in enumerate(fields)
    )
    policy = brief_policy_fingerprint(text, facts)
    rows = []
    for fact, sentence in zip(facts, sentences):
        start = text.index(sentence)
        row = dict(
            reference_id=fact.reference_id,
            source_id="synthetic:install-document",
            start=start,
            end=start + len(sentence),
            policy_fingerprint=policy,
        )
        fingerprint = fingerprint_evidence_review(**row)
        review = ReviewStateMachine()
        for state in (ReviewState.IN_REVIEW, ReviewState.APPROVED):
            review.transition(
                state,
                make_opaque_event_id((fact.reference_id, state.value)),
                fingerprint,
            )
        rows.append(
            dict(
                row,
                synthetic=True,
                verified=True,
                review_state="approved",
                review_transitions=review.transitions,
            )
        )
    calibration = "synthetic-install-only"
    context = BriefContext(
        build_evidence_packet(rows, policy_fingerprint=policy),
        "sha256:"
        + hashlib.sha256(json.dumps(text, ensure_ascii=True).encode()).hexdigest(),
        facts,
        lambda p, h: dict(
            entailment=1.0, contradiction=0.0, neutral=0.0, calibration_id=calibration
        ),
        NLIThresholds(
            calibration_id=calibration, calibration_method="synthetic-fixture"
        ),
        lambda _: [],
    )
    return DeidentificationResult(text, text, [], "mask", datetime(2026, 1, 1)), context


def _probe_cli(entry, value, provider, expected) -> None:
    """Drive the installed console entry point, including trusted injection."""
    import stat
    from types import ModuleType

    module = ModuleType("install_fixture_provider")
    module.factory = lambda: provider
    sys.modules[module.__name__] = module
    with tempfile.TemporaryDirectory(prefix="brief-fixture-") as directory:
        root = Path(directory)
        source, summary, audit = (root / name for name in ("note", "summary", "audit"))
        source.write_text(value.original_text, encoding="utf-8")
        args = [
            "brief",
            str(source),
            "--model",
            "extractive",
            "--json",
            "--summary-output",
            str(summary),
            "--review-output",
            str(audit),
        ]
        reviewed = [
            *args,
            "--review-id",
            "a" * 64,
            "--context-factory",
            "install_fixture_provider:factory",
        ]
        try:
            with contextlib.redirect_stdout(io.StringIO()) as output:
                assert entry(reviewed) == 0
            assert "dehydration" not in output.getvalue()
            assert summary.read_text(encoding="utf-8") == expected["summary"]
            audit_text = audit.read_text(encoding="utf-8")
            assert "summary" not in json.loads(audit_text)
            assert "dehydration" not in audit_text
            for path in (summary, audit):
                mode = stat.S_IMODE(path.stat().st_mode)
                if os.name == "posix":
                    assert mode == 0o600
                else:
                    # Windows stat exposes read/write attributes, not Unix ACLs.
                    assert mode & stat.S_IREAD and mode & stat.S_IWRITE
                assert path.is_file() and not path.is_symlink()
            # Exclusive-create safety is required on every OS, including Windows.
            assert entry(reviewed) != 0
            assert summary.read_text(encoding="utf-8") == expected["summary"]
            assert audit.read_text(encoding="utf-8") == audit_text
            refused = [
                "brief",
                str(source),
                "--model",
                "extractive",
                "--review-id",
                "a" * 64,
                "--json",
                "--summary-output",
                str(root / "refused-summary"),
                "--review-output",
                str(root / "refused-audit"),
            ]
            assert entry(refused) == 1
            assert (root / "refused-summary").read_text(encoding="utf-8") == ""
            assert (
                json.loads((root / "refused-audit").read_text(encoding="utf-8"))[
                    "refusal_reason"
                ]
                == "review_required"
            )
        finally:
            del sys.modules[module.__name__]


def _probe_rest(value, provider, expected, loop) -> None:
    """Use ASGI in memory; no listener, HTTP socket, or model is started."""
    import httpx

    from openmed.service.app import create_app
    from openmed.service.client import OpenMedClient

    app = create_app()
    app.state.brief_context_provider = provider
    transport = httpx.ASGITransport(app=app)

    async def rest():
        async with httpx.AsyncClient(
            transport=transport, base_url="http://127.0.0.1"
        ) as client:
            payload = dict(
                text=value.original_text, model="extractive", review_id="a" * 64
            )
            response = await client.post("/brief", json=payload)
            assert response.status_code == 200 and response.json() == expected
            response = await client.post(
                "/brief", json=dict(payload, review_id="invalid")
            )
            assert response.status_code == 422
            assert "dehydration" not in response.text

    loop.run_until_complete(rest())

    class InMemoryTransport(httpx.BaseTransport):
        def handle_request(self, request):
            async def send():
                response = await transport.handle_async_request(request)
                return httpx.Response(
                    response.status_code,
                    headers=response.headers,
                    content=await response.aread(),
                    request=request,
                )

            return loop.run_until_complete(send())

    with OpenMedClient(transport=InMemoryTransport()) as client:
        assert (
            client.brief(value.original_text, model="extractive", review_id="a" * 64)
            == expected
        )
    loop.run_until_complete(transport.aclose())


def _run_brief_probe(profile: str, source_root: Path) -> SmokeReport:
    """Run only in an isolated child against a non-editable installation."""
    import asyncio
    import importlib.util
    import logging
    import socket
    import traceback
    from dataclasses import replace
    from importlib.metadata import distribution
    from importlib.resources import files
    from types import SimpleNamespace
    from unittest.mock import patch

    checks = []
    active = "installed_origin"
    network_attempts = []
    captured = io.StringIO()

    def blocked(*args, **kwargs):
        network_attempts.append(True)
        raise RuntimeError("network_forbidden")

    # Windows may use sockets for its event-loop self-pipe. Construct it before
    # the outbound guard; all application operations below remain guarded.
    loop = asyncio.new_event_loop()
    handler = logging.StreamHandler(captured)
    logging.getLogger().addHandler(handler)
    try:
        with contextlib.ExitStack() as stack:
            stack.enter_context(contextlib.redirect_stdout(captured))
            stack.enter_context(contextlib.redirect_stderr(captured))
            for target in (
                "socket.socket.connect",
                "socket.socket.connect_ex",
                "socket.create_connection",
                "socket.getaddrinfo",
            ):
                stack.enter_context(patch(target, blocked))
            import openmed
            from openmed.clinical.brief import BriefRefusal, build_clinical_brief
            from openmed.clinical.summarize_backends import resolve_summarizer_backend
            from openmed.core.capabilities import MissingOptionalDependencyError
            from openmed.ner.infer import _convert_gliner_entity
            from openmed.service.brief import brief_response

            installed = distribution("openmed")
            root = Path(installed.locate_file("openmed")).resolve()
            assert Path(openmed.__file__).resolve() == root / "__init__.py"
            assert root != source_root.resolve() / "openmed"
            assert source_root.resolve() not in {Path(p).resolve() for p in sys.path}
            direct = json.loads(installed.read_text("direct_url.json") or "{}")
            assert not direct.get("dir_info", {}).get("editable", False)
            checks.append(_passed(active))

            active = "packaged_resources"
            for resource in (BRIEF_RESOURCE, "core/thresholds.json"):
                assert isinstance(
                    json.loads(
                        files("openmed").joinpath(resource).read_text(encoding="utf-8")
                    ),
                    dict,
                )
            vocabulary = files("openmed").joinpath(
                "eval/golden/fixtures/grounding_vocab_synthetic.jsonl"
            )
            assert all(
                isinstance(json.loads(row), dict)
                for row in vocabulary.read_text(encoding="utf-8").splitlines()
            )
            checks.append(_passed(active))

            active = "brief_python"
            value, context = _brief_fixture()
            provider = lambda text, review_id: (value, context)
            brief = build_clinical_brief(value, model="extractive", context=context)
            assert (
                brief.summary == value.deidentified_text and len(brief.citations) == 3
            )
            assert brief.refusal_reason is None
            assert (
                brief.envelope["requires_human_review"]
                and not brief.envelope["is_diagnostic"]
            )
            expected = brief_response(
                value.original_text,
                model="extractive",
                review_id="a" * 64,
                context_provider=provider,
            )
            assert expected == brief.to_response()
            assert (
                build_clinical_brief(value, model="extractive").refusal_reason
                is BriefRefusal.REVIEW_REQUIRED
            )
            checks.append(_passed(active))

            active = "missing_runtime"
            assert importlib.util.find_spec("mlx") is None
            try:
                resolve_summarizer_backend("mlx").summarize(value.deidentified_text)
            except MissingOptionalDependencyError as error:
                assert error.package == error.extra == "mlx"
            else:
                raise AssertionError("missing runtime accepted")
            refused = build_clinical_brief(value, model="mlx", context=context)
            assert (
                refused.refusal_reason is BriefRefusal.STAGE_FAILED
                and refused.summary == ""
            )
            for module in ("torch", "transformers", "mlx"):
                assert importlib.util.find_spec(module) is None
            for module in ("fastapi", "mcp"):
                assert (importlib.util.find_spec(module) is not None) == (
                    profile == "adapters"
                )
            checks.append(_passed(active))

            active = "privacy_sentinel"
            marker = "synthetic-private-5550199"
            entity = dict(text=marker, label="Drug", score=0.9)
            try:
                _convert_gliner_entity(entity)
            except KeyError as error:
                # Keep the full chain, including Python 3.13's source excerpts.
                assert marker not in "".join(traceback.format_exception(error))
            else:
                raise AssertionError("malformed entity accepted")

            def faulty_provider(*args):
                raise ValueError(marker)

            refused = build_clinical_brief(
                value,
                model="extractive",
                context=replace(context, nli_predict=faulty_provider),
            )
            assert refused.summary == "" and refused.refusal_reason is not None
            assert marker not in json.dumps(refused.to_dict())
            assert "dehydration" not in json.dumps(brief.to_dict())
            checks.append(_passed(active))

            active = "brief_cli"
            entry = next(
                item for item in installed.entry_points if item.name == "openmed"
            )
            _probe_cli(entry.load(), value, provider, expected)
            checks.append(_passed(active))
            if profile == "adapters":
                active = "brief_rest_client"
                _probe_rest(value, provider, expected, loop)
                checks.append(_passed(active))
                active = "brief_mcp"
                from openmed.mcp.server import openmed_brief
                from openmed.mcp.tool_registry import TOOL_REGISTRY

                assert (
                    openmed_brief(
                        value.original_text,
                        model="extractive",
                        review_id="a" * 64,
                        runtime_provider=lambda: SimpleNamespace(
                            brief_context_provider=provider
                        ),
                    )
                    == expected
                )
                assert (
                    TOOL_REGISTRY.get("openmed_brief").annotations()["readOnlyHint"]
                    is True
                )
                checks.append(_passed(active))
            active = "no_network"
            assert not network_attempts
            assert (
                marker not in captured.getvalue()
                and "dehydration" not in captured.getvalue()
            )
            # Imported package modules must also resolve to the installed tree.
            assert all(
                Path(module.__file__).resolve().is_relative_to(root)
                for name, module in tuple(sys.modules.items())
                if name.startswith("openmed.") and getattr(module, "__file__", None)
            )
            checks.append(_passed(active))
    except Exception:
        checks.append(_failed(active, "contract_failed"))
    finally:
        logging.getLogger().removeHandler(handler)
        loop.close()
    return SmokeReport(
        "failed" if checks[-1].status == "failed" else "passed", tuple(checks)
    )


@dataclass(frozen=True)
class CheckResult:
    """One privacy-safe smoke-check result."""

    name: str
    status: str
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the stable JSON representation for this check."""
        return {"name": self.name, "status": self.status, **dict(self.details)}


@dataclass(frozen=True)
class SmokeReport:
    """Compact, machine-readable evidence for one install smoke check."""

    status: str
    checks: tuple[CheckResult, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a stable, JSON-serializable report."""
        return {
            "checks": [check.to_dict() for check in self.checks],
            "offline": True,
            "schema_version": SCHEMA_VERSION,
            "status": self.status,
        }

    def to_json(self) -> str:
        """Render the report as one compact JSON document."""
        return json.dumps(
            self.to_dict(),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )


def _passed(name: str, **details: Any) -> CheckResult:
    return CheckResult(name=name, status="passed", details=details)


def _failed(name: str, reason: str) -> CheckResult:
    return CheckResult(name=name, status="failed", details={"reason": reason})


def _clean_environment(home: Path) -> dict[str, str]:
    """Build a minimal child environment with network and user state disabled."""
    temporary = home / "tmp"
    cache = home / "cache"
    config = home / "config"
    temporary.mkdir()
    cache.mkdir()
    config.mkdir()

    environment = {
        "HOME": str(home),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": os.environ.get("PATH", os.defpath),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "TMPDIR": str(temporary),
        "TEMP": str(temporary),
        "TMP": str(temporary),
        "XDG_CACHE_HOME": str(cache),
        "XDG_CONFIG_HOME": str(config),
    }
    environment.update(_OFFLINE_FLAGS)
    if os.name == "nt":  # pragma: no cover - Windows-only environment names.
        environment["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", "")
        environment["USERPROFILE"] = str(home)
    return environment


def _python_path(
    python_executable: str,
    *,
    path: str,
) -> Path | None:
    """Resolve the selected interpreter without dereferencing venv symlinks."""
    discovered = shutil.which(python_executable, path=path)
    if discovered is None:
        return None
    candidate = Path(discovered)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return candidate


def _entry_point_path(python_path: Path) -> Path | None:
    """Find the console script installed beside the selected interpreter."""
    script_name = "openmed.exe" if os.name == "nt" else "openmed"
    adjacent = python_path.parent / script_name
    if adjacent.is_file():
        return adjacent
    return None


def _run_captured(
    command: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    runner: CommandRunner,
) -> subprocess.CompletedProcess[str] | None:
    """Run a child command without exposing its stdout or stderr."""
    try:
        return runner(
            list(command),
            capture_output=True,
            check=False,
            cwd=cwd,
            env=dict(environment),
            text=True,
            timeout=COMMAND_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None


def _parse_version(output: str | None) -> str | None:
    if not output or len(output) > len("openmed ") + _MAX_VERSION_LENGTH + 1:
        return None
    match = _VERSION_RE.fullmatch(output.strip())
    return match.group(1) if match else None


def _manifest_check(
    result: subprocess.CompletedProcess[str] | None,
) -> CheckResult:
    if result is None or result.returncode != 0:
        return _failed("bundled_manifest", "command_failed")
    try:
        payload = json.loads(result.stdout or "")
    except (TypeError, json.JSONDecodeError):
        return _failed("bundled_manifest", "invalid_json")

    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(payload, dict) or payload.get("ok") is not True:
        return _failed("bundled_manifest", "invalid_result")
    if not isinstance(data, dict) or data.get("ok") is not True:
        return _failed("bundled_manifest", "asset_validation_failed")
    if data.get("violation_count") != 0:
        return _failed("bundled_manifest", "asset_validation_failed")

    row_count = None
    messages = data.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, str):
                continue
            match = _MANIFEST_ROW_RE.search(message)
            if match:
                row_count = int(match.group(1))
                break

    if row_count is None:
        return _failed("bundled_manifest", "invalid_result")
    return _passed("bundled_manifest", manifest_rows=row_count)


def _synthetic_check(
    python_executable: str,
    *,
    expected_version: str,
    cwd: Path,
    environment: Mapping[str, str],
    runner: CommandRunner,
) -> CheckResult:
    command = [python_executable, "-I", "-c", _SYNTHETIC_PROBE]
    first = _run_captured(
        command,
        cwd=cwd,
        environment=environment,
        runner=runner,
    )
    second = _run_captured(
        command,
        cwd=cwd,
        environment=environment,
        runner=runner,
    )
    if (
        first is None
        or second is None
        or first.returncode != 0
        or second.returncode != 0
    ):
        return _failed("synthetic_offline_command", "command_failed")
    if first.stdout != second.stdout:
        return _failed("synthetic_offline_command", "non_deterministic_output")

    try:
        payload = json.loads(first.stdout or "")
    except (TypeError, json.JSONDecodeError):
        return _failed("synthetic_offline_command", "invalid_json")
    if not isinstance(payload, dict):
        return _failed("synthetic_offline_command", "invalid_result")
    if payload.get("entry_point_declared") is not True:
        return _failed("synthetic_offline_command", "entry_point_not_declared")
    package_version = payload.get("package_version")
    if (
        not isinstance(package_version, str)
        or len(package_version) > _MAX_VERSION_LENGTH
        or _SAFE_VERSION_RE.fullmatch(package_version) is None
    ):
        return _failed("synthetic_offline_command", "invalid_package_version")
    if package_version != expected_version:
        return _failed("synthetic_offline_command", "version_mismatch")
    if payload.get("change_count") != 1:
        return _failed("synthetic_offline_command", "unexpected_change_count")
    if not all(
        isinstance(payload.get(key), str) and _HASH_RE.fullmatch(payload[key])
        for key in ("document_hash", "surface_hash")
    ):
        return _failed("synthetic_offline_command", "unsafe_or_invalid_hash")

    return _passed(
        "synthetic_offline_command",
        change_count=1,
        document_hash=payload["document_hash"],
        surface_hash=payload["surface_hash"],
    )


def run_smoke_check(
    *,
    python_executable: str = sys.executable,
    runner: CommandRunner = subprocess.run,
    brief_profile: str | None = None,
) -> SmokeReport:
    """Run the installed entry point and deterministic offline runtime probe.

    The selected interpreter must already contain an OpenMed installation. The
    checker never invokes a package installer or a network client; its clean
    temporary home only prevents host configuration, caches, and credentials
    from affecting the evidence.
    """
    if brief_profile is not None and brief_profile not in BRIEF_PROFILES:
        raise ValueError("invalid smoke profile")
    with tempfile.TemporaryDirectory(prefix="openmed-install-smoke-") as raw_home:
        home = Path(raw_home)
        environment = _clean_environment(home)
        workdir = home / "work"
        workdir.mkdir()
        python_path = _python_path(
            python_executable,
            path=environment["PATH"],
        )
        if python_path is None:
            return SmokeReport(
                status="failed",
                checks=(_failed("entry_point", "python_not_found"),),
            )
        # Resolve a bare interpreter name using the caller's PATH, then remove
        # every unrelated environment from the child executable search path.
        environment["PATH"] = str(python_path.parent)
        entry_point = _entry_point_path(python_path)
        if entry_point is None:
            return SmokeReport(
                status="failed",
                checks=(_failed("entry_point", "not_installed"),),
            )

        version_result = _run_captured(
            [str(entry_point), "--version"],
            cwd=workdir,
            environment=environment,
            runner=runner,
        )
        version = _parse_version(
            version_result.stdout if version_result is not None else None
        )
        if version_result is None or version_result.returncode != 0:
            return SmokeReport(
                status="failed",
                checks=(_failed("entry_point", "command_failed"),),
            )
        if version is None:
            return SmokeReport(
                status="failed",
                checks=(_failed("entry_point", "invalid_version_output"),),
            )

        # Do not copy even a version-shaped child value into the report until
        # the isolated metadata probe confirms that it belongs to this install.
        checks = [_passed("entry_point")]
        manifest_result = _run_captured(
            [str(entry_point), "models", "validate", "--json"],
            cwd=workdir,
            environment=environment,
            runner=runner,
        )
        manifest_check = _manifest_check(manifest_result)
        checks.append(manifest_check)
        if manifest_check.status != "passed":
            return SmokeReport(status="failed", checks=tuple(checks))

        synthetic_check = _synthetic_check(
            str(python_path),
            expected_version=version,
            cwd=workdir,
            environment=environment,
            runner=runner,
        )
        checks.append(synthetic_check)
        status = "passed" if synthetic_check.status == "passed" else "failed"
        if status == "passed":
            checks[0] = _passed("entry_point", version=version)
        if status == "passed" and brief_profile is not None:
            script = workdir / "probe.py"
            shutil.copyfile(Path(__file__), script)
            result = _run_captured(
                [
                    str(python_path),
                    "-I",
                    str(script),
                    "--_brief-probe",
                    brief_profile,
                    "--source-root",
                    str(Path(__file__).resolve().parents[2]),
                ],
                cwd=workdir,
                environment=environment,
                runner=runner,
            )
            checks.extend(_brief_probe_results(result, brief_profile))
            status = "passed" if all(c.status == "passed" for c in checks) else "failed"
        return SmokeReport(status=status, checks=tuple(checks))


def _brief_probe_results(result, profile: str) -> tuple[CheckResult, ...]:
    """Allow-list child fields; never relay arbitrary output or exceptions."""
    if result is None:
        return (_failed("brief_probe", "command_failed"),)
    try:
        payload = json.loads(result.stdout)
        expected = list(BRIEF_CHECKS)
        if profile == "core":
            expected.remove("brief_rest_client")
            expected.remove("brief_mcp")
        rows = payload["checks"]
        if not isinstance(rows, list) or not rows or len(rows) > len(expected):
            raise ValueError
        checks = []
        for row, name in zip(rows, expected):
            if row == {"name": name, "status": "passed"}:
                checks.append(_passed(name))
            elif row == {"name": name, "status": "failed", "reason": "contract_failed"}:
                checks.append(_failed(name, "contract_failed"))
            else:
                raise ValueError
        passed = len(checks) == len(expected) and all(
            c.status == "passed" for c in checks
        )
        if (result.returncode == 0) != passed or payload["status"] != (
            "passed" if passed else "failed"
        ):
            raise ValueError
        if not passed and checks[-1].status != "failed":
            raise ValueError
        return tuple(checks)
    except (ValueError, KeyError, TypeError):
        return (_failed("brief_probe", "invalid_result"),)


def _omit_brief_resource(artifact: Path, destination: Path) -> Path:
    """Create a damaged artifact in a disposable directory for a negative control."""
    import tarfile
    import zipfile

    target = destination / artifact.name
    member_suffix = "openmed/" + BRIEF_RESOURCE
    omitted = 0
    if artifact.suffix == ".whl":
        with (
            zipfile.ZipFile(artifact) as source,
            zipfile.ZipFile(target, "w") as output,
        ):
            for member in source.infolist():
                if member.filename == member_suffix:
                    omitted += 1
                else:
                    output.writestr(member, source.read(member))
    else:
        with (
            tarfile.open(artifact, "r:gz") as source,
            tarfile.open(target, "w:gz") as output,
        ):
            for member in source.getmembers():
                if member.name.endswith("/" + member_suffix):
                    omitted += 1
                else:
                    output.addfile(
                        member, source.extractfile(member) if member.isfile() else None
                    )
    if omitted != 1:
        raise ValueError("invalid resource inventory")
    return target


def run_artifact_checks(
    directory: Path, *, python_executable: str, constraints: Path | None = None
) -> dict[str, Any]:
    """Install both artifacts in disposable core/extras environments, then probe.

    Only this explicit mode invokes uv and may download build/package dependencies.
    Each runtime probe still runs offline, outside the source and without weights.
    """
    artifacts = [list(directory.glob(pattern)) for pattern in ("*.whl", "*.tar.gz")]
    if any(len(matches) != 1 for matches in artifacts):
        raise ValueError("expected one wheel and one sdist")
    uv = shutil.which("uv")
    if uv is None:
        raise ValueError("uv required")
    rows = []
    for artifact_type, matches in zip(("wheel", "sdist"), artifacts):
        artifact = matches[0].resolve()
        for profile in (*BRIEF_PROFILES, "missing_resource"):
            row = dict(artifact=artifact_type, profile=profile, status="failed")
            with tempfile.TemporaryDirectory(prefix="openmed-artifact-") as temporary:
                root = Path(temporary)
                env = root / "venv"
                python = env / (
                    "Scripts/python.exe" if os.name == "nt" else "bin/python"
                )
                selected = artifact
                if profile == "missing_resource":
                    selected = _omit_brief_resource(artifact, root)
                requirement = str(selected) + (
                    "[cli,service,mcp]" if profile == "adapters" else ""
                )
                install = [uv, "pip", "install", "--python", str(python), requirement]
                if constraints is not None:
                    install.extend(["--constraint", str(constraints.resolve())])
                prepared = True
                for command in (
                    [uv, "venv", "--python", python_executable, str(env)],
                    install,
                ):
                    try:
                        result = subprocess.run(
                            command,
                            cwd=root,
                            capture_output=True,
                            text=True,
                            timeout=300,
                            check=False,
                        )
                        if result.returncode != 0:
                            prepared = False
                            break
                    except (OSError, subprocess.TimeoutExpired):
                        prepared = False
                        break
                if not prepared:
                    row["reason"] = "installation_failed"
                else:
                    report = run_smoke_check(
                        python_executable=str(python),
                        brief_profile="core"
                        if profile == "missing_resource"
                        else profile,
                    )
                    if profile == "missing_resource":
                        failed = [c for c in report.checks if c.status == "failed"]
                        passed = (
                            report.status == "failed"
                            and len(failed) == 1
                            and failed[0].name == "packaged_resources"
                        )
                        row["status"] = "passed" if passed else "failed"
                        row["reason"] = (
                            "missing_resource_rejected"
                            if passed
                            else "negative_control_failed"
                        )
                    else:
                        row["status"] = report.status
                        row["checks"] = report.to_dict()["checks"]
            rows.append(row)
            # Controlled progress only; never print installer output or paths.
            print(json.dumps(row, sort_keys=True), flush=True)
    return dict(
        schema_version=1,
        status="passed" if all(r["status"] == "passed" for r in rows) else "failed",
        checks=rows,
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--python",
        dest="python_executable",
        default=sys.executable,
        help="Python executable for the installed OpenMed environment.",
    )
    parser.add_argument("--brief-profile", choices=BRIEF_PROFILES)
    parser.add_argument(
        "--artifacts",
        type=Path,
        help="Explicitly install and check a wheel/sdist directory using uv.",
    )
    parser.add_argument(
        "--constraints",
        type=Path,
        help="Frozen constraints for artifact dependency installation.",
    )
    parser.add_argument("--report", type=Path, help="Write the value-free JSON report.")
    parser.add_argument(
        "--_brief-probe", choices=BRIEF_PROFILES, help=argparse.SUPPRESS
    )
    parser.add_argument("--source-root", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Print one smoke report and return its release-gate status."""
    args = _parse_args(argv)
    try:
        if args._brief_probe:
            report = _run_brief_probe(args._brief_probe, args.source_root)
        elif args.artifacts:
            payload = run_artifact_checks(
                args.artifacts,
                python_executable=args.python_executable,
                constraints=args.constraints,
            )
            if args.report:
                args.report.write_text(
                    json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8"
                )
            return 0 if payload["status"] == "passed" else 1
        else:
            report = run_smoke_check(
                python_executable=args.python_executable,
                brief_profile=args.brief_profile,
            )
    except Exception:  # pragma: no cover - defensive privacy boundary.
        report = SmokeReport(
            status="failed",
            checks=(_failed("smoke_check", "internal_error"),),
        )
    print(report.to_json())
    if args.report:
        args.report.write_text(report.to_json() + "\n", encoding="utf-8")
    return 0 if report.status == "passed" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
