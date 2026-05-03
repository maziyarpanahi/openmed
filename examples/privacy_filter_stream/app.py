"""OpenMed Privacy Filter Stream — delayed redaction demo.

A browser example for agent-trace sharing: the raw trace streams immediately
to the left pane, while a share-safe version follows on the right after a
configurable delay. The right pane masks or fakes secrets, contact details,
people, addresses, clinical IDs, and operational identifiers before the trace
would be stored or shared.

Run from the repository root:

    uvicorn examples.privacy_filter_stream.app:app --reload --port 8771
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from openmed.core.anonymizer import Anonymizer


ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"

Mode = Literal["mask", "replace"]


@dataclass(frozen=True)
class SensitiveHint:
    label: str
    literal: str


@dataclass(frozen=True)
class TraceScenario:
    id: str
    title: str
    blurb: str
    hints: tuple[SensitiveHint, ...]
    lines: tuple[str, ...]

    @property
    def text(self) -> str:
        return "".join(self.lines)

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "blurb": self.blurb,
            "lineCount": len(self.lines),
            "charCount": len(self.text),
        }


def _support_handoff_lines() -> tuple[str, ...]:
    return (
        "[00:00.000] agent.start trace_id=trc_prod_01HV9M8Y5RN7VZ user='Sarah Johnson' account=acct_9f73a1\n",
        "[00:00.044] input.message channel=web text='My bill still shows the pneumonia follow-up copay.'\n",
        "[00:00.091] memory.fetch profile email=sarah.johnson@example.com phone=(415) 555-7012 plan=enterprise\n",
        "[00:00.137] tool.crm.lookup customer_id=CUS-448102 mrn=MRN-4872910 dob=03/15/1985\n",
        "[00:00.188] crm.result attending='Dr. Michael Chen' facility='Cedars Medical Center' next_visit=05/06/2026\n",
        "[00:00.241] address.verify mailing='1742 Ocean View Ave, Apt 14, San Diego CA 92107' confidence=0.98\n",
        "[00:00.294] policy.classify labels=payment,clinical,contact action=redact_before_trace_export\n",
        "[00:00.352] retriever.query q='refund workflow pneumonia copay exception' index=support-playbooks\n",
        "[00:00.416] retriever.hit id=kb_4421 title='Clinical billing handoff' snippet='Do not expose MRN or DOB in shared traces.'\n",
        "[00:00.481] tool.http.request GET https://api.billing.example/v1/invoices/inv_2026_04_22_008712\n",
        "[00:00.538] tool.http.header Authorization: Bearer sk-live-8ahWJD2lQ3pX9pTk1mZ-demo\n",
        "[00:00.602] tool.http.response 200 card=4111-1111-1111-7392 bank_routing=021000089 balance_due=$42.10\n",
        "[00:00.676] tool.billing.adjustment candidate=writeoff account=ACCT-0091-7782 reason='duplicate follow-up copay'\n",
        "[00:00.743] agent.observe risk='invoice id and card appear in tool output' needs_redaction=true\n",
        "[00:00.811] tool.calendar.lookup owner=michael.chen@hospital.example patient='Sarah Johnson' slot='May 6, 2026 09:30'\n",
        "[00:00.884] policy.gate allow_response=true allow_trace_share=false until=sanitized\n",
        "[00:00.947] assistant.draft sentence='Hi Sarah, I checked invoice inv_2026_04_22_008712 and found the duplicate copay.'\n",
        "[00:01.016] evaluator.check criterion='no raw MRN, DOB, card, account, email, phone, address, or key in exported trace'\n",
        "[00:01.088] tool.ticket.update id=TCK-55721 requester='Casey Rivera' note='Billing will reverse the duplicate line.'\n",
        "[00:01.159] notification.preview to=sarah.johnson@example.com cc=casey.rivera@hospital.example\n",
        "[00:01.226] store.trace.prepare destination=s3://support-traces-prod/acct_9f73a1/trc_prod_01HV9M8Y5RN7VZ.jsonl mode=shareable\n",
        "[00:01.299] audit.emit reviewer=michael.chen@hospital.example ip=10.42.118.27 session=sess_live_44128\n",
        "[00:01.367] agent.end status=complete tokens_in=3584 tokens_out=612\n",
    )


def _security_triage_lines() -> tuple[str, ...]:
    return (
        "[00:00.000] agent.start trace_id=trc_sec_7HF2KQ workspace=prod-secops user='Linda Park'\n",
        "[00:00.041] ingest.alert source=siem host=billing-svc ip=10.42.118.27 mac=4c:bb:58:9a:11:7e severity=high\n",
        "[00:00.099] identity.resolve employee_id=EMP-447128 email=linda.park@hospital.example phone=312-555-0144\n",
        "[00:00.152] vault.read path=secret/data/prod/billing token=ghp_7N3kV0x9kT2bQ4mP1sR8demo\n",
        "[00:00.211] tool.kube.describe pod=billing-svc-849ff namespace=prod env.OPENAI_API_KEY=sk-proj-zYw8fakel0ngSecret4711\n",
        "[00:00.276] tool.netflow.summary src=10.42.118.27 dst=172.18.4.21 bytes=88420 window=15m\n",
        "[00:00.337] log.sample service=auth-gateway principal=linda.park@hospital.example failures=9\n",
        "[00:00.398] log.sample service=claims-worker principal=service-account/billing failures=2 token=sk-live-incident-01-R8zQpFake\n",
        "[00:00.462] file.peek path=/runbooks/finance.md line='fallback account ACCT-0091-7782 belongs to Finance rotation queue'\n",
        "[00:00.529] tool.slack.lookup channel=#incident-finance requester='Marcus Reilly'\n",
        "[00:00.590] identity.resolve manager='Nina Torres' email=nina.torres@hospital.example escalation=SRE\n",
        "[00:00.658] policy.classify labels=credential,employee,network action=redact_before_postmortem\n",
        "[00:00.726] tool.kms.rotate key_alias=prod/billing-api actor=EMP-447128 approval=required\n",
        "[00:00.798] agent.observe exposed_api_key=true exposed_employee_contact=true exposed_network_ioc=true\n",
        "[00:00.871] detector.ioc ip=10.42.118.27 mac=4c:bb:58:9a:11:7e status=internal_workstation\n",
        "[00:00.936] ticket.create incident=SEC-2026-044 owner=Marcus Reilly notify=linda.park@hospital.example\n",
        "[00:01.004] assistant.draft summary='Credential material was present in billing pod environment and vault output.'\n",
        "[00:01.078] containment.step revoke token ghp_7N3kV0x9kT2bQ4mP1sR8demo and expire session sess_live_44128\n",
        "[00:01.151] containment.step rotate sk-proj-zYw8fakel0ngSecret4711 and sk-live-incident-01-R8zQpFake\n",
        "[00:01.218] evidence.bundle path=s3://security-reviews/prod-secops/trc_sec_7HF2KQ.jsonl mode=redacted\n",
        "[00:01.287] assistant.final rotate keys, expire session sess_live_44128, notify Linda Park and CISO office\n",
        "[00:01.354] agent.end status=escalated severity=high\n",
    )


SCENARIOS: tuple[TraceScenario, ...] = (
    TraceScenario(
        id="support-handoff",
        title="Support Handoff",
        blurb="Agent trace with customer, billing, and clinical identifiers.",
        lines=_support_handoff_lines(),
        hints=(
            SensitiveHint("person", "Sarah Johnson"),
            SensitiveHint("person", "Sarah"),
            SensitiveHint("person", "Michael Chen"),
            SensitiveHint("person", "Dr. Michael Chen"),
            SensitiveHint("person", "Casey Rivera"),
            SensitiveHint("email", "sarah.johnson@example.com"),
            SensitiveHint("email", "michael.chen@hospital.example"),
            SensitiveHint("email", "casey.rivera@hospital.example"),
            SensitiveHint("phone", "(415) 555-7012"),
            SensitiveHint("street_address", "1742 Ocean View Ave, Apt 14, San Diego CA 92107"),
            SensitiveHint("id_num", "CUS-448102"),
            SensitiveHint("id_num", "MRN-4872910"),
            SensitiveHint("date_of_birth", "03/15/1985"),
            SensitiveHint("account_number", "ACCT-0091-7782"),
            SensitiveHint("account_number", "acct_9f73a1"),
            SensitiveHint("id_num", "trc_prod_01HV9M8Y5RN7VZ"),
            SensitiveHint("id_num", "sess_live_44128"),
            SensitiveHint("api_key", "sk-live-8ahWJD2lQ3pX9pTk1mZ-demo"),
            SensitiveHint("credit_card", "4111-1111-1111-7392"),
        ),
    ),
    TraceScenario(
        id="security-triage",
        title="Security Triage",
        blurb="Incident trace with keys, employee data, hosts, and account IDs.",
        lines=_security_triage_lines(),
        hints=(
            SensitiveHint("person", "Linda Park"),
            SensitiveHint("person", "Marcus Reilly"),
            SensitiveHint("person", "Nina Torres"),
            SensitiveHint("email", "linda.park@hospital.example"),
            SensitiveHint("email", "nina.torres@hospital.example"),
            SensitiveHint("phone", "312-555-0144"),
            SensitiveHint("id_num", "EMP-447128"),
            SensitiveHint("account_number", "ACCT-0091-7782"),
            SensitiveHint("id_num", "trc_sec_7HF2KQ"),
            SensitiveHint("id_num", "sess_live_44128"),
            SensitiveHint("api_key", "ghp_7N3kV0x9kT2bQ4mP1sR8demo"),
            SensitiveHint("api_key", "sk-proj-zYw8fakel0ngSecret4711"),
        ),
    ),
)


REGEX_HINTS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("phone", re.compile(r"\b(?:\+?1[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}\b")),
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("credit_card", re.compile(r"\b(?:\d[ -]*?){13,19}\b")),
    ("ip_address", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    ("mac_address", re.compile(r"\b[0-9a-fA-F]{2}(?::[0-9a-fA-F]{2}){5}\b")),
    ("api_key", re.compile(r"\b(?:sk|ghp|xox[baprs]|AKIA)[A-Za-z0-9_-]{12,}\b")),
    ("api_key", re.compile(r"\bBearer\s+[A-Za-z0-9._-]{16,}\b")),
    ("account_number", re.compile(r"\b(?:acct|account|ACCT|inv|sess|trc)[A-Za-z0-9_-]{6,}\b")),
    ("id_num", re.compile(r"\b(?:MRN|CUS|EMP)-[A-Z0-9-]{4,}\b")),
)

LABEL_TITLES = {
    "api_key": "API key",
    "account_number": "Account",
    "credit_card": "Card",
    "date_of_birth": "DOB",
    "email": "Email",
    "id_num": "Identifier",
    "ip_address": "IP address",
    "mac_address": "MAC address",
    "person": "Person",
    "phone": "Phone",
    "ssn": "SSN",
    "street_address": "Address",
}

LABEL_GROUPS = {
    "person": "identity",
    "email": "contact",
    "phone": "contact",
    "street_address": "address",
    "date_of_birth": "dates",
    "ssn": "govid",
    "id_num": "identifier",
    "account_number": "financial",
    "credit_card": "financial",
    "api_key": "secret",
    "ip_address": "network",
    "mac_address": "network",
}


def _find_scenario(scenario_id: str) -> TraceScenario:
    for scenario in SCENARIOS:
        if scenario.id == scenario_id:
            return scenario
    return SCENARIOS[0]


def _dedupe_entities(entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for entity in sorted(
        entities,
        key=lambda item: (item["start"], -(item["end"] - item["start"]), item["label"]),
    ):
        overlaps = any(
            not (entity["end"] <= existing["start"] or entity["start"] >= existing["end"])
            for existing in selected
        )
        if not overlaps:
            selected.append(entity)
    return sorted(selected, key=lambda item: (item["start"], item["end"], item["label"]))


def _detect_entities(text: str, scenario: TraceScenario) -> list[dict[str, Any]]:
    entities: list[dict[str, Any]] = []
    for hint in scenario.hints:
        start = 0
        while True:
            found = text.find(hint.literal, start)
            if found == -1:
                break
            entities.append(
                {
                    "label": hint.label,
                    "start": found,
                    "end": found + len(hint.literal),
                    "text": hint.literal,
                    "score": 1.0,
                }
            )
            start = found + len(hint.literal)

    for label, pattern in REGEX_HINTS:
        for match in pattern.finditer(text):
            value = match.group(0).rstrip(".,;")
            if not value:
                continue
            entities.append(
                {
                    "label": label,
                    "start": match.start(),
                    "end": match.start() + len(value),
                    "text": value,
                    "score": 0.92,
                }
            )

    return _dedupe_entities(entities)


class RedactorSession:
    def __init__(self, *, mode: Mode, seed: int = 8771) -> None:
        self.mode = mode
        self.anonymizer = Anonymizer(lang="en", consistent=True, seed=seed)
        self.surrogate_map: dict[str, str] = {}

    def _replacement(self, label: str, original: str) -> str:
        if self.mode == "mask":
            return f"[{LABEL_TITLES.get(label, label).upper()}]"
        key = f"{label}|{original}"
        if key not in self.surrogate_map:
            self.surrogate_map[key] = self.anonymizer.surrogate(original, label)
        return self.surrogate_map[key]

    def redact(self, text: str, scenario: TraceScenario) -> dict[str, Any]:
        entities = _detect_entities(text, scenario)
        cursor = 0
        segments: list[dict[str, Any]] = []
        for entity in entities:
            if entity["start"] < cursor:
                continue
            if entity["start"] > cursor:
                segments.append({"kind": "plain", "text": text[cursor:entity["start"]]})
            replacement = self._replacement(entity["label"], entity["text"])
            segments.append(
                {
                    "kind": "entity",
                    "text": replacement,
                    "label": entity["label"],
                    "labelTitle": LABEL_TITLES.get(entity["label"], entity["label"]),
                    "group": LABEL_GROUPS.get(entity["label"], "other"),
                    "original": entity["text"],
                }
            )
            cursor = entity["end"]
        if cursor < len(text):
            segments.append({"kind": "plain", "text": text[cursor:]})
        redacted = "".join(segment["text"] for segment in segments)
        return {
            "redacted": redacted,
            "segments": segments,
            "entities": [
                {
                    "label": entity["label"],
                    "labelTitle": LABEL_TITLES.get(entity["label"], entity["label"]),
                    "group": LABEL_GROUPS.get(entity["label"], "other"),
                    "text": entity["text"],
                }
                for entity in entities
            ],
        }


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n"


async def _trace_event_stream(
    scenario: TraceScenario,
    *,
    mode: Mode,
    delay_ms: int,
    speed_ms: int,
) -> Any:
    redactor = RedactorSession(mode=mode)
    delay_s = delay_ms / 1000
    speed_s = speed_ms / 1000
    started = time.monotonic()
    pending: list[tuple[float, int, str]] = []
    raw_chars = 0
    shared_chars = 0
    total_entities = 0

    yield _sse(
        "meta",
        {
            "scenario": scenario.to_public_dict(),
            "mode": mode,
            "delayMs": delay_ms,
            "speedMs": speed_ms,
            "startedAt": started,
        },
    )

    for index, line in enumerate(scenario.lines):
        now = time.monotonic()
        due, pending = (
            [item for item in pending if item[0] <= now],
            [item for item in pending if item[0] > now],
        )
        for _, redacted_index, queued_line in due:
            result = redactor.redact(queued_line, scenario)
            shared_chars += len(result["redacted"])
            total_entities += len(result["entities"])
            yield _sse(
                "redacted",
                {
                    "index": redacted_index,
                    "chunk": queued_line,
                    "redacted": result["redacted"],
                    "segments": result["segments"],
                    "entities": result["entities"],
                    "sharedChars": shared_chars,
                    "entityCount": total_entities,
                    "lagMs": delay_ms,
                },
            )

        raw_chars += len(line)
        yield _sse(
            "source",
            {
                "index": index,
                "chunk": line,
                "rawChars": raw_chars,
                "elapsedMs": round((time.monotonic() - started) * 1000),
            },
        )
        pending.append((time.monotonic() + delay_s, index, line))
        await asyncio.sleep(speed_s)

    while pending:
        pending.sort(key=lambda item: item[0])
        due_at, redacted_index, queued_line = pending.pop(0)
        wait_s = max(0.0, due_at - time.monotonic())
        if wait_s:
            await asyncio.sleep(wait_s)
        result = redactor.redact(queued_line, scenario)
        shared_chars += len(result["redacted"])
        total_entities += len(result["entities"])
        yield _sse(
            "redacted",
            {
                "index": redacted_index,
                "chunk": queued_line,
                "redacted": result["redacted"],
                "segments": result["segments"],
                "entities": result["entities"],
                "sharedChars": shared_chars,
                "entityCount": total_entities,
                "lagMs": delay_ms,
            },
        )

    yield _sse(
        "done",
        {
            "rawChars": raw_chars,
            "sharedChars": shared_chars,
            "entityCount": total_entities,
            "elapsedMs": round((time.monotonic() - started) * 1000),
        },
    )


app = FastAPI(
    title="OpenMed Privacy Filter Stream",
    description="Streaming agent-trace redaction demo with delayed share-safe output.",
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/favicon.ico")
def favicon() -> FileResponse:
    return FileResponse(STATIC_DIR / "assets" / "logo.svg", media_type="image/svg+xml")


@app.get("/api/scenarios")
def api_scenarios() -> dict[str, Any]:
    return {
        "scenarios": [scenario.to_public_dict() for scenario in SCENARIOS],
        "defaults": {
            "scenario": SCENARIOS[0].id,
            "mode": "replace",
            "delayMs": 1000,
            "speedMs": 80,
        },
    }


@app.get("/api/stream")
async def api_stream(
    scenario: str = Query(default=SCENARIOS[0].id),
    mode: Mode = Query(default="replace"),
    delay_ms: int = Query(default=1000, ge=250, le=4000),
    speed_ms: int = Query(default=80, ge=20, le=400),
) -> StreamingResponse:
    trace = _find_scenario(scenario)
    return StreamingResponse(
        _trace_event_stream(trace, mode=mode, delay_ms=delay_ms, speed_ms=speed_ms),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
