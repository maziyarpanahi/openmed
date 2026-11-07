"""Keyed span, document and webhook URL digests in async job records."""

from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest
from fastapi.testclient import TestClient

import openmed
from openmed.core.audit import hash_text
from openmed.service import jobs
from openmed.service import runtime as service_runtime
from openmed.service.app import create_app
from openmed.service.job_ownership import JOB_OWNER_DIGEST_DOMAIN
from openmed.service.keyed_digest import (
    KEYED_DIGEST_ALGORITHM,
    SERVICE_DIGEST_KEY_ENV_VAR,
    SERVICE_DIGEST_KEY_ID_ENV_VAR,
    ServiceDigestKey,
    encode_digest_fields,
)
from openmed.service.schemas import (
    DeidentifyJobDocument,
    DeidentifyJobRequest,
    JobWebhookRequest,
)
from openmed.service.webhooks import WebhookDeliveryResult

LOOPBACK_BASE_URL = "http://127.0.0.1"
SYNTHETIC_DIGEST_KEY = "synthetic-digest-key-" + "c" * 40
SYNTHETIC_ROTATED_KEY = "synthetic-digest-key-" + "d" * 40
SYNTHETIC_KEY_ID = "synthetic-jobs-1"
SYNTHETIC_ROTATED_KEY_ID = "synthetic-jobs-2"
# Low-entropy synthetic identifiers: each sits inside a small, enumerable space.
SYNTHETIC_MRN = "4821"
SYNTHETIC_DATE = "1970-03-14"
SYNTHETIC_PHONE = "555-0142"
SYNTHETIC_TEXT = (
    f"Synthetic note MRN {SYNTHETIC_MRN} born {SYNTHETIC_DATE} phone {SYNTHETIC_PHONE}."
)
SYNTHETIC_IDENTIFIERS = {
    "ID": SYNTHETIC_MRN,
    "DATE": SYNTHETIC_DATE,
    "PHONE": SYNTHETIC_PHONE,
}
SYNTHETIC_WEBHOOK_URL = "https://callbacks.example.test/openmed/jobs"
SYNTHETIC_WEBHOOK_SECRET = "synthetic-webhook-secret"
_HEX64 = re.compile(r"[0-9a-f]{64}")
_SERVICE_ENV_VARS = (
    "OPENMED_PROFILE",
    "OPENMED_SERVICE_PRELOAD_MODELS",
    "OPENMED_SERVICE_RATE_LIMIT_RPS",
    "OPENMED_SERVICE_RATE_LIMIT_BURST",
    "OPENMED_SERVICE_RATE_LIMIT_MAX_CONCURRENCY",
    "OPENMED_SERVICE_AUTH_ENABLED",
    "OPENMED_SERVICE_AUTH_API_KEYS",
    "OPENMED_SERVICE_MTLS_ENABLED",
    "OPENMED_SERVICE_JOBS_STORE_PATH",
    "OPENMED_SERVICE_JOBS_TTL_SECONDS",
    "OPENMED_SERVICE_JOBS_WORKERS",
    SERVICE_DIGEST_KEY_ENV_VAR,
    SERVICE_DIGEST_KEY_ID_ENV_VAR,
)


class FakeLoader:
    """Minimal model loader double; no model weights are loaded."""

    def __init__(self, config: Any) -> None:
        self.config = config

    def resolve_model_name(self, model_name: str) -> str:
        return model_name

    def loaded_models(self) -> dict[str, dict[str, int]]:
        return {}


def _synthetic_entities(text: str) -> list[SimpleNamespace]:
    return [
        SimpleNamespace(
            label=label,
            text=value,
            start=text.index(value),
            end=text.index(value) + len(value),
            confidence=0.9,
        )
        for label, value in SYNTHETIC_IDENTIFIERS.items()
        if value in text
    ]


def _fake_deidentify(text: str, **_: Any) -> SimpleNamespace:
    return SimpleNamespace(pii_entities=_synthetic_entities(text))


@pytest.fixture(autouse=True)
def clean_service_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for env_var in _SERVICE_ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)
    monkeypatch.setattr(service_runtime, "ModelLoader", FakeLoader)
    monkeypatch.setattr(openmed, "deidentify", _fake_deidentify)


def _enumeration_candidates() -> set[str]:
    """Return an attacker's dictionary that contains every synthetic value."""
    candidates = {f"{number:04d}" for number in range(10_000)}
    start = date(1970, 1, 1)
    candidates.update((start + timedelta(days=day)).isoformat() for day in range(365))
    candidates.update(f"555-{number:04d}" for number in range(1_000))
    candidates.update({SYNTHETIC_TEXT, SYNTHETIC_WEBHOOK_URL})
    assert set(SYNTHETIC_IDENTIFIERS.values()) <= candidates
    return candidates


_CANDIDATE_SHA256 = {
    hashlib.sha256(candidate.encode("utf-8")).hexdigest()
    for candidate in _enumeration_candidates()
}


def _enumerated(rendered: str) -> set[str]:
    """Return plain SHA-256 digests of dictionary values found in *rendered*."""
    return set(_HEX64.findall(rendered)) & _CANDIDATE_SHA256


def _sentinel_leaks(rendered: str) -> list[str]:
    """Return raw values, secrets or legacy hash markers present in *rendered*."""
    # Token boundaries keep short numeric identifiers from matching inside hex.
    leaks = [
        value
        for value in SYNTHETIC_IDENTIFIERS.values()
        if re.search(rf"(?<![0-9A-Za-z]){re.escape(value)}(?![0-9A-Za-z])", rendered)
    ]
    sentinels = [
        "Synthetic note",
        SYNTHETIC_WEBHOOK_URL,
        SYNTHETIC_WEBHOOK_SECRET,
        SYNTHETIC_DIGEST_KEY,
        SYNTHETIC_ROTATED_KEY,
        "sha256:",
        '"text_hash"',
        '"url_hash"',
    ]
    return leaks + [sentinel for sentinel in sentinels if sentinel in rendered]


def _key(secret: str = SYNTHETIC_DIGEST_KEY, key_id: str = SYNTHETIC_KEY_ID):
    return ServiceDigestKey.from_secret(secret.encode("utf-8"), key_id=key_id)


def _payload(*, webhook: bool = True) -> DeidentifyJobRequest:
    return DeidentifyJobRequest(
        documents=[DeidentifyJobDocument(id="synthetic-doc", text=SYNTHETIC_TEXT)],
        webhook=(
            JobWebhookRequest(
                url=SYNTHETIC_WEBHOOK_URL,
                secret=SYNTHETIC_WEBHOOK_SECRET,
                max_attempts=1,
                backoff_seconds=0,
            )
            if webhook
            else None
        ),
    )


def _run_queue_job(
    store_path: Path,
    digests: jobs.JobDigestPolicy,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run one synthetic job to completion and return its record and webhook."""
    store = jobs.LocalJobStore(store_path)
    sent: list[dict[str, Any]] = []

    def sender(_url: str, payload: dict[str, Any], **_: Any) -> WebhookDeliveryResult:
        sent.append(payload)
        return WebhookDeliveryResult(success=True, attempts=1, status_code=204)

    queue = jobs.DeidentifyJobQueue(
        SimpleNamespace(),
        store=store,
        webhook_sender=sender,
        digests=digests,
    )
    payload = _payload()
    record = queue._new_record(payload)
    store.create(record)

    def process(_payload: Any, document: DeidentifyJobDocument) -> SimpleNamespace:
        return _fake_deidentify(document.text)

    queue._deidentify_document = process  # type: ignore[method-assign]
    try:
        queue._run_job(jobs._JobWorkItem(record["id"], payload))
    finally:
        queue.shutdown()
    final = store.get(record["id"])
    assert final is not None and len(sent) == 1
    return final, sent[0]


def _wait(predicate: Callable[[], bool]) -> None:
    for _ in range(300):
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition was not reached")


def _run_api_job(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    profile: str,
    digest_key: str | None,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Submit one job through the REST app; return response, webhook, store."""
    store_path = tmp_path / "jobs.json"
    monkeypatch.setenv("OPENMED_PROFILE", profile)
    monkeypatch.setenv("OPENMED_SERVICE_JOBS_STORE_PATH", str(store_path))
    monkeypatch.setenv("OPENMED_SERVICE_JOBS_WORKERS", "1")
    if digest_key is not None:
        monkeypatch.setenv(SERVICE_DIGEST_KEY_ENV_VAR, digest_key)
        monkeypatch.setenv(SERVICE_DIGEST_KEY_ID_ENV_VAR, SYNTHETIC_KEY_ID)
    callbacks: list[dict[str, Any]] = []

    def sender(_url: str, payload: dict[str, Any], **_: Any) -> WebhookDeliveryResult:
        callbacks.append(payload)
        return WebhookDeliveryResult(success=True, attempts=1, status_code=204)

    app = create_app()
    with TestClient(app, base_url=LOOPBACK_BASE_URL) as client:
        app.state.job_queue.webhook_sender = sender
        created = client.post(
            "/jobs",
            json={
                "documents": [{"id": "synthetic-doc", "text": SYNTHETIC_TEXT}],
                "webhook": {
                    "url": SYNTHETIC_WEBHOOK_URL,
                    "secret": SYNTHETIC_WEBHOOK_SECRET,
                    "max_attempts": 1,
                    "backoff_seconds": 0,
                },
            },
        )
        assert created.status_code == 202, created.text
        path = f"/jobs/{created.json()['id']}"
        _wait(lambda: client.get(path).json()["webhook_delivery"] is not None)
        done = client.get(path).json()
    assert len(callbacks) == 1
    return done, callbacks[0], store_path.read_text(encoding="utf-8")


# --- policy ----------------------------------------------------------------


def test_digest_policy_is_domain_separated_and_carries_markers() -> None:
    key = _key()
    policy = jobs.JobDigestPolicy(key=key)
    material = encode_digest_fields(SYNTHETIC_MRN)
    by_domain = {
        domain: policy.digest(domain, SYNTHETIC_MRN)
        for domain in (
            jobs.SPAN_TEXT_DIGEST_DOMAIN,
            jobs.DOCUMENT_TEXT_DIGEST_DOMAIN,
            jobs.WEBHOOK_URL_DIGEST_DOMAIN,
        )
    }

    values = {digest["value"] for digest in by_domain.values() if digest}
    assert len(values) == 3
    assert key.digest(JOB_OWNER_DIGEST_DOMAIN, material).value not in values
    for domain, digest in by_domain.items():
        assert digest is not None
        assert set(digest) == {"alg", "key_id", "value"}
        assert digest["alg"] == KEYED_DIGEST_ALGORITHM
        assert digest["key_id"] == SYNTHETIC_KEY_ID
        assert digest["value"] == key.digest(domain, material).value
    assert policy.descriptor() == {
        "mode": jobs.DIGEST_MODE_KEYED,
        "alg": KEYED_DIGEST_ALGORITHM,
        "key_id": SYNTHETIC_KEY_ID,
        "reason": None,
    }
    assert _enumerated(json.dumps(by_domain)) == set()
    assert SYNTHETIC_DIGEST_KEY not in repr(policy)


def test_digests_are_stable_for_one_key_and_change_on_rotation() -> None:
    policy = jobs.JobDigestPolicy(key=_key())
    same_key = jobs.JobDigestPolicy(key=_key())
    rotated = jobs.JobDigestPolicy(
        key=_key(SYNTHETIC_ROTATED_KEY, SYNTHETIC_ROTATED_KEY_ID)
    )
    same_id_new_secret = jobs.JobDigestPolicy(key=_key(SYNTHETIC_ROTATED_KEY))
    domain = jobs.SPAN_TEXT_DIGEST_DOMAIN

    first = policy.digest(domain, SYNTHETIC_PHONE)
    assert first == policy.digest(domain, SYNTHETIC_PHONE)
    assert first == same_key.digest(domain, SYNTHETIC_PHONE)
    assert first != policy.digest(domain, "555-0143")
    after_rotation = rotated.digest(domain, SYNTHETIC_PHONE)
    assert after_rotation is not None and first is not None
    assert after_rotation["key_id"] == SYNTHETIC_ROTATED_KEY_ID
    assert after_rotation["value"] != first["value"]
    # A new secret changes every value even if the operator reuses the key id.
    moved = same_id_new_secret.digest(domain, SYNTHETIC_PHONE)
    assert moved is not None and moved["value"] != first["value"]


def test_digest_policy_omits_digests_in_prod_without_a_key(subtests) -> None:
    configured = {
        SERVICE_DIGEST_KEY_ENV_VAR: SYNTHETIC_DIGEST_KEY,
        SERVICE_DIGEST_KEY_ID_ENV_VAR: SYNTHETIC_KEY_ID,
    }
    for environ, keyed, key_id_prefix in (
        ({}, False, None),
        ({"OPENMED_PROFILE": "prod"}, False, None),
        ({"OPENMED_PROFILE": "low_resource"}, False, None),
        ({"OPENMED_PROFILE": "prod", **configured}, True, SYNTHETIC_KEY_ID),
        ({"OPENMED_PROFILE": "dev"}, True, "ephemeral-"),
        ({"OPENMED_PROFILE": "test"}, True, "ephemeral-"),
    ):
        with subtests.test(profile=environ.get("OPENMED_PROFILE"), keyed=keyed):
            policy = jobs.JobDigestPolicy.from_env(environ=environ)
            assert policy.keyed is keyed
            digest = policy.digest(jobs.SPAN_TEXT_DIGEST_DOMAIN, SYNTHETIC_MRN)
            if keyed:
                assert digest is not None
                assert digest["key_id"].startswith(key_id_prefix)
                assert policy.descriptor()["mode"] == jobs.DIGEST_MODE_KEYED
            else:
                assert digest is None
                assert policy.descriptor() == {
                    "mode": jobs.DIGEST_MODE_OMITTED,
                    "alg": None,
                    "key_id": None,
                    "reason": jobs.DIGEST_KEY_UNCONFIGURED,
                }


# --- job records and webhook payloads --------------------------------------


def test_plain_sha256_enumeration_matches_no_job_digest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    done, webhook, store_text = _run_api_job(
        monkeypatch, tmp_path, profile="prod", digest_key=SYNTHETIC_DIGEST_KEY
    )

    rendered = {
        "store": store_text,
        "response": json.dumps(done),
        "webhook": json.dumps(webhook),
    }
    for surface, text in rendered.items():
        assert _enumerated(text) == set(), surface
        assert _sentinel_leaks(text) == [], surface

    key = _key()
    assert done["schema_version"] == jobs.JOB_STORE_SCHEMA_VERSION
    assert webhook["schema_version"] == jobs.WEBHOOK_PAYLOAD_SCHEMA_VERSION
    assert (
        done["digests"]
        == webhook["digests"]
        == jobs.JobDigestPolicy(key=key).descriptor()
    )
    assert done["spans"] == webhook["spans"]
    assert done["documents"] == webhook["documents"]
    assert {span["label"] for span in done["spans"]} == set(SYNTHETIC_IDENTIFIERS)
    for span in done["spans"]:
        value = SYNTHETIC_IDENTIFIERS[span["label"]]
        assert SYNTHETIC_TEXT[span["start"] : span["end"]] == value
        assert (
            span["text_digest"]
            == (
                key.digest(jobs.SPAN_TEXT_DIGEST_DOMAIN, encode_digest_fields(value))
            ).to_dict()
        )
        assert span["text_digest"]["value"] != hash_text(value).split(":", 1)[1]
    (document,) = done["documents"]
    assert document["text_digest"]["key_id"] == SYNTHETIC_KEY_ID
    assert done["webhook"]["url_digest"]["key_id"] == SYNTHETIC_KEY_ID
    stored = json.loads(store_text)
    assert stored["schema_version"] == jobs.JOB_STORE_SCHEMA_VERSION
    (record,) = stored["jobs"].values()
    assert record["spans"] == done["spans"]


def test_prod_without_digest_key_omits_digests_with_marker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    done, webhook, store_text = _run_api_job(
        monkeypatch, tmp_path, profile="prod", digest_key=None
    )

    marker = {
        "mode": jobs.DIGEST_MODE_OMITTED,
        "alg": None,
        "key_id": None,
        "reason": jobs.DIGEST_KEY_UNCONFIGURED,
    }
    assert done["digests"] == webhook["digests"] == marker
    assert len(done["spans"]) == len(SYNTHETIC_IDENTIFIERS)
    assert all(span["text_digest"] is None for span in webhook["spans"])
    assert all(doc["text_digest"] is None for doc in webhook["documents"])
    assert done["webhook"]["url_digest"] is None
    for text in (store_text, json.dumps(done), json.dumps(webhook)):
        # No digest of any kind is emitted, keyed or not.
        assert _HEX64.findall(text) == []
        assert _sentinel_leaks(text) == []


def test_dev_profile_without_key_uses_ephemeral_keyed_digests(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    done, webhook, store_text = _run_api_job(
        monkeypatch, tmp_path, profile="dev", digest_key=None
    )

    assert done["digests"]["mode"] == jobs.DIGEST_MODE_KEYED
    assert done["digests"]["key_id"].startswith("ephemeral-")
    assert {span["text_digest"]["key_id"] for span in webhook["spans"]} == {
        done["digests"]["key_id"]
    }
    for text in (store_text, json.dumps(done), json.dumps(webhook)):
        assert _enumerated(text) == set()
        assert _sentinel_leaks(text) == []


def test_job_queue_shares_the_ownership_digest_key(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("OPENMED_PROFILE", "dev")
    monkeypatch.setenv("OPENMED_SERVICE_JOBS_STORE_PATH", str(tmp_path / "jobs.json"))

    app = create_app()
    with TestClient(app, base_url=LOOPBACK_BASE_URL):
        # Even the per-process dev key is one key for owner and span digests.
        assert app.state.job_queue.digests.key is app.state.job_ownership.key


def test_job_digests_are_stable_across_jobs_and_change_after_rotation(
    tmp_path: Path,
) -> None:
    first, first_webhook = _run_queue_job(
        tmp_path / "first.json", jobs.JobDigestPolicy(key=_key())
    )
    second, _ = _run_queue_job(
        tmp_path / "second.json", jobs.JobDigestPolicy(key=_key())
    )
    rotated, rotated_webhook = _run_queue_job(
        tmp_path / "rotated.json",
        jobs.JobDigestPolicy(key=_key(SYNTHETIC_ROTATED_KEY, SYNTHETIC_ROTATED_KEY_ID)),
    )

    assert first["spans"] == second["spans"]
    assert first["documents"] == second["documents"]
    assert first["webhook"]["url_digest"] == second["webhook"]["url_digest"]
    assert rotated["digests"]["key_id"] == SYNTHETIC_ROTATED_KEY_ID
    old_values = {span["text_digest"]["value"] for span in first["spans"]}
    new_values = {span["text_digest"]["value"] for span in rotated["spans"]}
    assert len(old_values) == len(new_values) == len(SYNTHETIC_IDENTIFIERS)
    assert old_values.isdisjoint(new_values)
    assert {span["text_digest"]["key_id"] for span in rotated["spans"]} == {
        SYNTHETIC_ROTATED_KEY_ID
    }
    for payload in (first, first_webhook, rotated, rotated_webhook):
        rendered = json.dumps(payload)
        assert _enumerated(rendered) == set()
        assert _sentinel_leaks(rendered) == []

    # Records keyed before a rotation stay as written: still keyed, still
    # naming their key id, and never rewritten on load.
    before = (tmp_path / "first.json").read_bytes()
    reloaded = jobs.LocalJobStore(tmp_path / "first.json").get(first["id"])
    assert reloaded == first
    assert (tmp_path / "first.json").read_bytes() == before


# --- legacy migration -------------------------------------------------------


def _legacy_record(job_id: str, *, version: int) -> dict[str, Any]:
    record: dict[str, Any] = {
        "id": job_id,
        "status": "done",
        "progress_percent": 100.0,
        "document_count": 1,
        "processed_count": 1,
        "failed_count": 0,
        "label_histogram": {"ID": 1, "PHONE": 1},
        "spans": [
            {
                "document_id": "synthetic-doc",
                "start": SYNTHETIC_TEXT.index(value),
                "end": SYNTHETIC_TEXT.index(value) + len(value),
                "label": label,
                "text_hash": hash_text(value),
                "confidence": 0.9,
            }
            for label, value in (("ID", SYNTHETIC_MRN), ("PHONE", SYNTHETIC_PHONE))
        ],
        "documents": [
            {
                "id": "synthetic-doc",
                "length": len(SYNTHETIC_TEXT),
                "text_hash": hash_text(SYNTHETIC_TEXT),
            }
        ],
        "error": None,
        "webhook": {
            "configured": True,
            "url_hash": hash_text(SYNTHETIC_WEBHOOK_URL),
            "max_attempts": 1,
            "backoff_seconds": 0.0,
        },
        "webhook_delivery": None,
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
        "started_at": "2026-01-01T00:00:00Z",
        "completed_at": "2026-01-01T00:00:01Z",
        "expires_at": "2999-01-01T00:00:00Z",
    }
    if version >= 2:
        record[jobs.OWNER_DIGEST_FIELD] = {
            "alg": KEYED_DIGEST_ALGORITHM,
            "key_id": SYNTHETIC_KEY_ID,
            "value": "1" * 64,
        }
    return record


def _write_legacy_store(path: Path, *, version: int) -> str:
    job_id = f"{version}" * 32
    contents: dict[str, Any] = {
        "jobs": {job_id: _legacy_record(job_id, version=version)}
    }
    if version >= 2:
        contents["schema_version"] = version
    path.write_text(json.dumps(contents), encoding="utf-8")
    return job_id


def test_legacy_unkeyed_hashes_are_stripped_and_rewritten_on_load(
    subtests,
    tmp_path: Path,
) -> None:
    for version in (1, 2):
        with subtests.test(version=version):
            store_path = tmp_path / f"v{version}.json"
            job_id = _write_legacy_store(store_path, version=version)
            legacy_text = store_path.read_text(encoding="utf-8")
            # Positive control: the enumeration recovers the legacy hashes.
            assert _enumerated(legacy_text) == {
                hash_text(value).split(":", 1)[1]
                for value in (
                    SYNTHETIC_MRN,
                    SYNTHETIC_PHONE,
                    SYNTHETIC_TEXT,
                    SYNTHETIC_WEBHOOK_URL,
                )
            }

            store = jobs.LocalJobStore(store_path)
            migrated_text = store_path.read_text(encoding="utf-8")
            loaded = store.get(job_id)

            assert _enumerated(migrated_text) == set()
            assert _sentinel_leaks(migrated_text) == []
            assert json.loads(migrated_text)["schema_version"] == 3
            assert loaded is not None
            assert loaded["schema_version"] == jobs.JOB_STORE_SCHEMA_VERSION
            assert loaded["digests"] == {
                "mode": jobs.DIGEST_MODE_OMITTED,
                "alg": None,
                "key_id": None,
                "reason": jobs.LEGACY_DIGESTS_REMOVED,
            }
            legacy = _legacy_record(job_id, version=version)
            for span, original in zip(loaded["spans"], legacy["spans"]):
                assert span["text_digest"] is None
                assert {k: span[k] for k in ("start", "end", "label")} == {
                    k: original[k] for k in ("start", "end", "label")
                }
            assert loaded["documents"][0]["text_digest"] is None
            assert loaded["documents"][0]["length"] == len(SYNTHETIC_TEXT)
            assert loaded["webhook"]["url_digest"] is None
            expected_owner = legacy.get(jobs.OWNER_DIGEST_FIELD)
            assert loaded[jobs.OWNER_DIGEST_FIELD] == expected_owner
            # A second load finds nothing to migrate and leaves the file alone.
            jobs.LocalJobStore(store_path)
            assert store_path.read_text(encoding="utf-8") == migrated_text


def test_legacy_store_is_served_without_unkeyed_hashes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store_path = tmp_path / "jobs.json"
    job_id = _write_legacy_store(store_path, version=2)
    monkeypatch.setenv("OPENMED_PROFILE", "test")
    monkeypatch.setenv("OPENMED_SERVICE_JOBS_STORE_PATH", str(store_path))

    app = create_app()
    with TestClient(app, base_url=LOOPBACK_BASE_URL) as client:
        response = client.get(f"/jobs/{job_id}")

    assert response.status_code == 200
    body = response.json()
    assert body["digests"]["reason"] == jobs.LEGACY_DIGESTS_REMOVED
    assert _enumerated(response.text) == set()
    assert _sentinel_leaks(response.text) == []
    assert _enumerated(store_path.read_text(encoding="utf-8")) == set()


def test_payload_builders_never_echo_legacy_hashes() -> None:
    legacy = _legacy_record("0" * 32, version=1)

    for payload in (jobs.webhook_payload(legacy), jobs.job_response_payload(legacy)):
        rendered = json.dumps(payload)
        assert _enumerated(rendered) == set()
        assert _sentinel_leaks(rendered) == []
        assert payload["digests"]["reason"] == jobs.LEGACY_DIGESTS_REMOVED
    assert jobs.webhook_payload(legacy)["schema_version"] == 2
    # The caller's record is not mutated by building a payload.
    assert "text_hash" in legacy["spans"][0]


def test_failed_migration_rewrite_leaves_legacy_store_untouched(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store_path = tmp_path / "jobs.json"
    _write_legacy_store(store_path, version=2)
    original = store_path.read_bytes()

    def fail(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("synthetic publication failure")

    monkeypatch.setattr(jobs.os, "replace", fail)
    with pytest.raises(OSError, match="synthetic publication failure"):
        jobs.LocalJobStore(store_path)

    assert store_path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [store_path]
