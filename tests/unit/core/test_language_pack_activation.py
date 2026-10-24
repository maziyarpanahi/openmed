"""Tests for atomic activation of verified offline language bundles."""

from __future__ import annotations

import copy
import hashlib
import hmac
import importlib.metadata
import json
import socket
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from openmed.core.language_pack import LanguagePack, LanguagePackRegistry
from openmed.core.language_pack_activation import (
    ActivationRefusal,
    LanguageBundleActivation,
    LanguageBundleActivationError,
    LanguageBundleActivator,
    LanguageReadinessEvidence,
    LocalModelHandle,
    language_bundle_digest,
    language_bundle_signing_payload,
)

NOW = datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc)
KEY = b"synthetic-bundle-signing-key"
KEY_ID = "synthetic-release-key"
LT_REFERENCE = "/synthetic-models/lt-pii"
SL_REFERENCE = "/synthetic-models/sl-pii"


def _sha(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode("utf-8")).hexdigest()


LT_MODEL = _sha("synthetic lt model")
SL_MODEL = _sha("synthetic sl model")


class HmacVerifier:
    """Synthetic trust anchor; real deployments inject their own verifier."""

    def __init__(self, key: bytes = KEY) -> None:
        self.key = key
        self.calls = 0

    def verify(
        self, payload: bytes, *, scheme: str, key_id: str, signature: str
    ) -> bool:
        self.calls += 1
        expected = hmac.new(self.key, payload, hashlib.sha256).hexdigest()
        return (
            scheme == "hmac-sha256"
            and key_id == KEY_ID
            and hmac.compare_digest(expected, signature)
        )


def _sign(manifest: dict[str, Any], key: bytes = KEY) -> dict[str, Any]:
    signed = copy.deepcopy(manifest)
    signed.pop("signature", None)
    payload = language_bundle_signing_payload(signed)
    signed["signature"] = {
        "scheme": "hmac-sha256",
        "key_id": KEY_ID,
        "value": hmac.new(key, payload, hashlib.sha256).hexdigest(),
    }
    return signed


_LOCALES = {"af": "af_ZA", "is": "is_IS", "lt": "lt_LT", "sl": "sl_SI"}


def _pack_spec(language: str = "lt", **overrides: Any) -> dict[str, Any]:
    code = language.split("-")[0]
    spec: dict[str, Any] = {
        "language": language,
        "model": f"{code}-pii",
        "tasks": ["deidentification", "ner"],
        "scripts": ["Latin"],
        "segmenter_id": "pysbd",
        "recognizers": ["builtin-patterns", "model"],
        "surrogate_locale": _LOCALES[code],
        "policy_overrides": {"profile": "strict_no_leak"},
    }
    spec.update(overrides)
    return spec


def _unsigned(*packs: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    packs = packs or (_pack_spec(),)
    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "bundle_id": "org.example.baltic-clinical",
        "bundle_version": "1.0.0",
        "packs": list(packs),
        "artifacts": {
            "lt-pii": {"kind": "model", "sha256": LT_MODEL},
            "sl-pii": {"kind": "model", "sha256": SL_MODEL},
            "lt-lexicon": {"kind": "lexicon", "sha256": _sha("synthetic lexicon")},
        },
    }
    manifest.update(overrides)
    return manifest


def _manifest(*packs: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    return _sign(_unsigned(*packs, **overrides))


def _readiness(
    language: str = "lt",
    *,
    state: str = "supported",
    tasks: tuple[str, ...] = ("deidentification", "ner"),
    age: timedelta = timedelta(days=1),
    model_digest: str | None = None,
) -> LanguageReadinessEvidence:
    return LanguageReadinessEvidence(
        language=language,
        state=state,
        qualified_tasks=tasks,
        issued_at=NOW - age,
        evidence_digest=_sha(f"synthetic readiness {language} {state}"),
        model_digest=model_digest,
    )


def _handles(**extra: LocalModelHandle) -> dict[str, LocalModelHandle]:
    handles = {
        "lt-pii": LocalModelHandle(reference=LT_REFERENCE, sha256=LT_MODEL),
        "sl-pii": LocalModelHandle(reference=SL_REFERENCE, sha256=SL_MODEL),
    }
    handles.update(extra)
    return handles


def _user_pack(code: str = "en", **overrides: Any) -> LanguagePack:
    values: dict[str, Any] = {
        "code": code,
        "scripts": ("Latin",),
        "default_model": "/user-models/en-pii",
        "segmenter_id": "pysbd",
        "recognizers": ("model",),
        "surrogate_locale": "en_US",
    }
    values.update(overrides)
    return LanguagePack(**values)


class _Fixture:
    def __init__(self) -> None:
        self.registry = LanguagePackRegistry()
        self.user_pack = self.registry.register(_user_pack())
        self.notifications: list[tuple[str, ...]] = []
        self.registry._add_listener(self._record, replay=False)
        self.verifier = HmacVerifier()
        self.activator = LanguageBundleActivator(
            verifier=self.verifier,
            registry=self.registry,
            clock=lambda: NOW,
        )

    def _record(self) -> None:
        self.notifications.append(tuple(self.registry.iter_codes()))

    def state(self) -> tuple[object, ...]:
        generation, packs = self.registry._snapshot()
        return (
            generation,
            dict(packs),
            len(self.notifications),
            self.activator.active_activations(),
        )


@pytest.fixture
def env() -> _Fixture:
    return _Fixture()


def _refusal(excinfo: pytest.ExceptionInfo[LanguageBundleActivationError]) -> str:
    return excinfo.value.reason.value


# --- activation of a qualified bundle -------------------------------------


def test_registers_only_declared_qualified_tasks(env: _Fixture) -> None:
    manifest = _manifest(
        _pack_spec(tasks=["deidentification", "ner", "summarization"]),
    )
    readiness = [_readiness(tasks=("deidentification", "ner", "translation"))]

    record = env.activator.activate(manifest, readiness, _handles())

    (language,) = record.languages
    assert language.language == "lt"
    assert language.code == "lt"
    assert language.tasks == ("deidentification", "ner")
    assert language.withheld_tasks == ("summarization",)
    assert language.claim == "supported"
    assert language.locale_status == "native"
    assert language.model_digest == LT_MODEL
    assert env.activator.qualified_tasks("lt") == ("deidentification", "ner")
    assert "translation" not in env.activator.qualified_tasks("lt")
    assert env.activator.clinical_claim("lt") == "supported"

    pack = env.registry.get("lt")
    assert pack.default_model == LT_REFERENCE
    assert pack.surrogate_locale == "lt_LT"
    assert env.registry.get("en") is env.user_pack
    assert record.manifest_digest == language_bundle_digest(manifest)
    assert record.previous_activation_id is None
    assert record.activated_at == NOW


def test_multi_pack_bundle_is_committed_as_one_generation(env: _Fixture) -> None:
    generation = env.registry.generation
    manifest = _manifest(_pack_spec("lt"), _pack_spec("sl"))

    record = env.activator.activate(
        manifest, [_readiness("lt"), _readiness("sl")], _handles()
    )

    assert env.registry.generation == generation + 1
    # Listeners observe both packs together, never a partial bundle.
    assert env.notifications == [("en", "lt", "sl")]
    assert [item.language for item in record.languages] == ["lt", "sl"]


def test_record_and_inputs_survive_serialization_and_reload(env: _Fixture) -> None:
    manifest = _manifest(_pack_spec("lt"), _pack_spec("sl"))
    readiness = [_readiness("lt"), _readiness("sl", state="preview")]
    record = env.activator.activate(manifest, readiness, _handles())

    saved_manifest = json.dumps(manifest, ensure_ascii=False)
    saved_readiness = json.dumps([item.to_dict() for item in readiness])
    saved_record = record.to_json()

    restored = LanguageBundleActivation.from_dict(json.loads(saved_record))
    assert restored == record
    assert restored.to_json() == saved_record

    fresh = _Fixture()
    reloaded = fresh.activator.activate(
        json.loads(saved_manifest),
        json.loads(saved_readiness),
        _handles(),
        expected_activation_id=restored.activation_id,
    )

    assert reloaded.to_json() == saved_record
    assert fresh.registry.get("lt") == env.registry.get("lt")
    assert fresh.registry.get("sl") == env.registry.get("sl")
    assert fresh.activator.qualified_tasks("sl") == env.activator.qualified_tasks("sl")
    assert fresh.activator.clinical_claim("sl") == "preview"


def test_reload_with_drifted_inputs_fails_closed(env: _Fixture) -> None:
    record = env.activator.activate(_manifest(), [_readiness()], _handles())
    fresh = _Fixture()
    before = fresh.state()

    with pytest.raises(LanguageBundleActivationError) as excinfo:
        fresh.activator.activate(
            _manifest(),
            [_readiness(tasks=("deidentification",))],
            _handles(),
            expected_activation_id=record.activation_id,
        )

    assert _refusal(excinfo) == "activation_id_mismatch"
    assert fresh.state() == before


def test_tampered_serialized_record_is_rejected(env: _Fixture) -> None:
    record = env.activator.activate(_manifest(), [_readiness()], _handles())
    data = json.loads(record.to_json())
    data["languages"][0]["tasks"].append("summarization")

    with pytest.raises(LanguageBundleActivationError) as excinfo:
        LanguageBundleActivation.from_dict(data)
    assert _refusal(excinfo) == "activation_id_mismatch"

    upgraded = json.loads(record.to_json())
    upgraded["languages"][0]["locale_status"] = "approximated"
    with pytest.raises(LanguageBundleActivationError) as excinfo:
        LanguageBundleActivation.from_dict(upgraded)
    assert _refusal(excinfo) == "activation_record_malformed"


def test_serialized_record_contains_no_private_values(env: _Fixture) -> None:
    manifest = _manifest()
    record = env.activator.activate(manifest, [_readiness()], _handles())
    serialized = record.to_json()

    assert LT_REFERENCE not in serialized
    assert manifest["signature"]["value"] not in serialized
    assert KEY_ID not in serialized
    assert LT_REFERENCE not in repr(_handles()["lt-pii"])


def test_signing_payload_is_canonical_and_excludes_signature() -> None:
    manifest = _unsigned()
    reordered = dict(reversed(list(manifest.items())))

    assert language_bundle_signing_payload(manifest) == (
        language_bundle_signing_payload(reordered)
    )
    assert language_bundle_digest(manifest) == language_bundle_digest(_sign(manifest))


def test_signing_payload_matches_golden_canonical_form() -> None:
    # Packaging tools must sign these exact bytes; a change here breaks every
    # previously signed bundle and needs a new manifest schema version.
    manifest = {
        "signature": {"scheme": "x", "key_id": "k", "value": "v"},
        "schema_version": "1.0",
        "packs": [
            {
                "tasks": ["ner"],
                "routing_markers": ["pacientė"],
                "recall_floor_overrides": {"person": 0.95},
                "model": "m",
                "language": "lt",
            }
        ],
        "bundle_version": "0.1.0",
        "bundle_id": "org.example.golden",
        "artifacts": {"m": {"sha256": "sha256:" + "0" * 64, "kind": "model"}},
    }

    assert language_bundle_signing_payload(manifest) == (
        '{"artifacts":{"m":{"kind":"model","sha256":"sha256:'
        + "0" * 64
        + '"}},"bundle_id":"org.example.golden","bundle_version":"0.1.0",'
        '"packs":[{"language":"lt","model":"m",'
        '"recall_floor_overrides":{"person":0.95},'
        '"routing_markers":["pacientė"],"tasks":["ner"]}],'
        '"schema_version":"1.0"}'
    ).encode("utf-8")
    assert language_bundle_digest(manifest) == (
        "sha256:602cf366dbf509630163c822a48763b1b4405fe60de64480dbe033f15a1ef8bf"
    )


def test_signing_payload_rejects_non_json_manifests() -> None:
    for manifest in (
        ["not", "a", "mapping"],
        {"schema_version": "1.0", "nan": float("nan")},
        {"schema_version": "1.0", "set": {"a"}},
        {1: "non-string key"},
    ):
        with pytest.raises(LanguageBundleActivationError) as excinfo:
            language_bundle_signing_payload(manifest)  # type: ignore[arg-type]
        assert _refusal(excinfo) == "manifest_malformed"


# --- refusals leave the previous registry state unchanged -----------------

Case = Callable[[_Fixture], tuple[dict[str, Any], list[Any], dict[str, Any]]]


def _case_bad_signature(env: _Fixture) -> tuple[Any, Any, Any]:
    manifest = _manifest()
    manifest["signature"]["value"] = "0" * 64
    return manifest, [_readiness()], _handles()


def _case_tampered_after_signing(env: _Fixture) -> tuple[Any, Any, Any]:
    manifest = _manifest()
    manifest["packs"][0]["tasks"].append("summarization")
    return (
        manifest,
        [_readiness(tasks=("deidentification", "summarization"))],
        (_handles()),
    )


def _case_wrong_key(env: _Fixture) -> tuple[Any, Any, Any]:
    return _sign(_unsigned(), key=b"other-key"), [_readiness()], _handles()


def _case_unsigned(env: _Fixture) -> tuple[Any, Any, Any]:
    return _unsigned(), [_readiness()], _handles()


def _case_verifier_raises(env: _Fixture) -> tuple[Any, Any, Any]:
    def explode(*args: Any, **kwargs: Any) -> bool:
        raise RuntimeError("synthetic verifier outage")

    env.verifier.verify = explode  # type: ignore[method-assign]
    return _manifest(), [_readiness()], _handles()


def _case_verifier_truthy(env: _Fixture) -> tuple[Any, Any, Any]:
    env.verifier.verify = lambda *args, **kwargs: 1  # type: ignore[method-assign]
    return _manifest(), [_readiness()], _handles()


def _case_stale(env: _Fixture) -> tuple[Any, Any, Any]:
    return _manifest(), [_readiness(age=timedelta(days=31))], _handles()


def _case_future(env: _Fixture) -> tuple[Any, Any, Any]:
    return _manifest(), [_readiness(age=-timedelta(hours=1))], _handles()


def _case_unknown_top_level(env: _Fixture) -> tuple[Any, Any, Any]:
    return (
        _manifest(clinical_use_restriction="research-only"),
        [_readiness()],
        _handles(),
    )


def _case_unknown_critical_extension(env: _Fixture) -> tuple[Any, Any, Any]:
    manifest = _manifest(
        extensions={"org.example.dosage-limits": {"enabled": True}},
        critical=["org.example.dosage-limits"],
    )
    return manifest, [_readiness()], _handles()


def _case_unknown_pack_field(env: _Fixture) -> tuple[Any, Any, Any]:
    return (
        _manifest(_pack_spec(auto_route_unknown_text=True)),
        [_readiness()],
        _handles(),
    )


def _case_unknown_artifact_field(env: _Fixture) -> tuple[Any, Any, Any]:
    manifest = _unsigned()
    manifest["artifacts"]["lt-pii"]["remote_url"] = "https://example.invalid/m"
    return _sign(manifest), [_readiness()], _handles()


def _case_unsupported_schema(env: _Fixture) -> tuple[Any, Any, Any]:
    return _manifest(schema_version="2.0"), [_readiness()], _handles()


def _case_conflicting_user_pack(env: _Fixture) -> tuple[Any, Any, Any]:
    env.registry.register(_user_pack("lt", surrogate_locale="lt_LT"))
    return _manifest(), [_readiness()], _handles()


def _case_partial_conflict(env: _Fixture) -> tuple[Any, Any, Any]:
    env.registry.register(_user_pack("sl", surrogate_locale="sl_SI"))
    manifest = _manifest(_pack_spec("lt"), _pack_spec("sl"))
    return manifest, [_readiness("lt"), _readiness("sl")], _handles()


def _case_duplicate_code(env: _Fixture) -> tuple[Any, Any, Any]:
    manifest = _manifest(_pack_spec("lt"), _pack_spec("lt-LT"))
    return manifest, [_readiness("lt"), _readiness("lt-LT")], _handles()


def _case_priority_capability(env: _Fixture) -> tuple[Any, Any, Any]:
    manifest = _manifest(_pack_spec(candidate_priority={"Latin": 10}))
    return manifest, [_readiness()], _handles()


def _case_marker_capability(env: _Fixture) -> tuple[Any, Any, Any]:
    env.registry.register(
        _user_pack("lv", surrogate_locale="lv_LV", routing_markers=("pacientas",))
    )
    manifest = _manifest(_pack_spec(routing_markers=["Pacientas"]))
    return manifest, [_readiness()], _handles()


def _case_region_readiness_only(env: _Fixture) -> tuple[Any, Any, Any]:
    return _manifest(), [_readiness("lt-LT")], _handles()


def _case_base_readiness_for_region(env: _Fixture) -> tuple[Any, Any, Any]:
    manifest = _manifest(_pack_spec("lt-LT"))
    return manifest, [_readiness("lt")], _handles()


def _case_experimental(env: _Fixture) -> tuple[Any, Any, Any]:
    return _manifest(), [_readiness(state="experimental")], _handles()


def _case_no_qualified_task(env: _Fixture) -> tuple[Any, Any, Any]:
    return _manifest(), [_readiness(tasks=("translation",))], _handles()


def _case_readiness_for_other_model(env: _Fixture) -> tuple[Any, Any, Any]:
    return _manifest(), [_readiness(model_digest=SL_MODEL)], _handles()


def _case_digest_mismatch(env: _Fixture) -> tuple[Any, Any, Any]:
    handles = _handles(**{"lt-pii": LocalModelHandle(LT_REFERENCE, SL_MODEL)})
    return _manifest(), [_readiness()], handles


def _case_missing_handle(env: _Fixture) -> tuple[Any, Any, Any]:
    return _manifest(), [_readiness()], {}


def _case_remote_handle(env: _Fixture) -> tuple[Any, Any, Any]:
    handle = LocalModelHandle("https://example.invalid/lt-pii", LT_MODEL)
    return _manifest(), [_readiness()], _handles(**{"lt-pii": handle})


def _case_hub_handle(env: _Fixture) -> tuple[Any, Any, Any]:
    handle = LocalModelHandle("OpenMed/synthetic-lt-pii", LT_MODEL)
    return _manifest(), [_readiness()], _handles(**{"lt-pii": handle})


def _case_undeclared_approximation(env: _Fixture) -> tuple[Any, Any, Any]:
    manifest = _manifest(_pack_spec(surrogate_locale="en_US"))
    return manifest, [_readiness()], _handles()


def _case_incoherent_pack(env: _Fixture) -> tuple[Any, Any, Any]:
    manifest = _manifest(_pack_spec(segmenter_id="synthetic-segmenter"))
    return manifest, [_readiness()], _handles()


def _case_invalid_pack_declaration(env: _Fixture) -> tuple[Any, Any, Any]:
    manifest = _manifest(_pack_spec(recall_floor_overrides={"person": 1.5}))
    return manifest, [_readiness()], _handles()


REFUSALS: list[tuple[str, Case]] = [
    ("signature_invalid", _case_bad_signature),
    ("signature_invalid", _case_tampered_after_signing),
    ("signature_invalid", _case_wrong_key),
    ("signature_missing", _case_unsigned),
    ("signature_invalid", _case_verifier_raises),
    ("signature_invalid", _case_verifier_truthy),
    ("readiness_stale", _case_stale),
    ("readiness_not_yet_valid", _case_future),
    ("unknown_critical_field", _case_unknown_top_level),
    ("unknown_critical_field", _case_unknown_critical_extension),
    ("unknown_critical_field", _case_unknown_pack_field),
    ("unknown_critical_field", _case_unknown_artifact_field),
    ("manifest_schema_unsupported", _case_unsupported_schema),
    ("pack_identity_conflict", _case_conflicting_user_pack),
    ("pack_identity_conflict", _case_partial_conflict),
    ("pack_identity_conflict", _case_duplicate_code),
    ("capability_conflict", _case_priority_capability),
    ("capability_conflict", _case_marker_capability),
    ("readiness_missing", _case_region_readiness_only),
    ("readiness_missing", _case_base_readiness_for_region),
    ("readiness_insufficient", _case_experimental),
    ("tasks_not_qualified", _case_no_qualified_task),
    ("readiness_subject_mismatch", _case_readiness_for_other_model),
    ("artifact_digest_mismatch", _case_digest_mismatch),
    ("artifact_handle_missing", _case_missing_handle),
    ("artifact_handle_not_local", _case_remote_handle),
    ("artifact_handle_not_local", _case_hub_handle),
    ("locale_approximation_undeclared", _case_undeclared_approximation),
    ("pack_incoherent", _case_incoherent_pack),
    ("manifest_malformed", _case_invalid_pack_declaration),
]


@pytest.mark.parametrize(
    ("reason", "case"),
    REFUSALS,
    ids=[case.__name__.removeprefix("_case_") for _reason, case in REFUSALS],
)
def test_refusal_leaves_previous_registry_unchanged(
    env: _Fixture, reason: str, case: Case
) -> None:
    # Start from a committed generation so "unchanged" covers bundle state too.
    baseline = env.activator.activate(
        _manifest(
            _pack_spec("is", model="sl-pii"),
            bundle_id="org.example.nordic-clinical",
        ),
        [_readiness("is")],
        _handles(),
    )
    manifest, readiness, handles = case(env)
    before = env.state()
    packs_before = {code: env.registry.get(code) for code in env.registry.iter_codes()}

    with pytest.raises(LanguageBundleActivationError) as excinfo:
        env.activator.activate(manifest, readiness, handles)

    assert _refusal(excinfo) == reason
    assert str(excinfo.value) == reason
    assert env.state() == before
    for code, pack in packs_before.items():
        assert env.registry.get(code) is pack
    assert env.activator.active_activations() == (baseline,)
    assert env.activator.qualified_tasks("lt") == ()
    assert env.activator.clinical_claim("lt") is None


def test_refusal_messages_never_echo_private_values(env: _Fixture) -> None:
    manifest = _manifest()
    secret_signature = "f" * 64
    manifest["signature"]["value"] = secret_signature
    remote = LocalModelHandle("https://example.invalid/private/lt", LT_MODEL)

    for args in (
        (manifest, [_readiness()], _handles()),
        (_manifest(), [_readiness()], _handles(**{"lt-pii": remote})),
    ):
        with pytest.raises(LanguageBundleActivationError) as excinfo:
            env.activator.activate(*args)
        rendered = f"{excinfo.value!s} {excinfo.value!r}"
        assert secret_signature not in rendered
        assert "example.invalid" not in rendered
        assert excinfo.value.__cause__ is None


def test_stale_and_unknown_field_checks_cover_mapping_inputs(env: _Fixture) -> None:
    before = env.state()
    stale = _readiness(age=timedelta(days=45)).to_dict()

    with pytest.raises(LanguageBundleActivationError) as excinfo:
        env.activator.activate(_manifest(), [stale], _handles())
    assert _refusal(excinfo) == "readiness_stale"

    unknown = {**_readiness().to_dict(), "reviewer": "synthetic"}
    with pytest.raises(LanguageBundleActivationError) as excinfo:
        env.activator.activate(_manifest(), [unknown], _handles())
    assert _refusal(excinfo) == "readiness_malformed"
    assert env.state() == before


def test_readiness_freshness_window_is_configurable(env: _Fixture) -> None:
    activator = LanguageBundleActivator(
        verifier=HmacVerifier(),
        registry=env.registry,
        clock=lambda: NOW,
        max_readiness_age=timedelta(days=90),
    )

    record = activator.activate(
        _manifest(), [_readiness(age=timedelta(days=60))], _handles()
    )

    assert record.languages[0].tasks == ("deidentification", "ner")


def test_listener_failure_restores_previous_registry(env: _Fixture) -> None:
    failures = {"remaining": 1}

    def flaky() -> None:
        if failures["remaining"]:
            failures["remaining"] -= 1
            raise RuntimeError("synthetic adapter failure")

    env.registry._add_listener(flaky, replay=False)
    _generation, packs_before = env.registry._snapshot()

    with pytest.raises(LanguageBundleActivationError) as excinfo:
        env.activator.activate(_manifest(), [_readiness()], _handles())

    assert _refusal(excinfo) == "registry_refresh_failed"
    assert dict(env.registry._snapshot()[1]) == dict(packs_before)
    assert env.activator.active_activations() == ()
    assert env.notifications[-1] == ("en",)


def test_concurrent_registry_change_aborts_commit(env: _Fixture) -> None:
    snapshot = env.registry._snapshot
    racing_pack = _user_pack("fi", surrogate_locale="fi_FI")

    def racing_snapshot() -> Any:
        result = snapshot()
        env.registry.register(racing_pack)
        return result

    env.registry._snapshot = racing_snapshot  # type: ignore[method-assign]

    with pytest.raises(LanguageBundleActivationError) as excinfo:
        env.activator.activate(_manifest(), [_readiness()], _handles())

    assert _refusal(excinfo) == "registry_changed"
    assert env.registry.find("lt") is None
    assert env.registry.get("fi") is racing_pack
    assert env.activator.active_activations() == ()


def test_activation_performs_no_discovery_or_network(
    env: _Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("network or discovery is forbidden")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    monkeypatch.setattr(importlib.metadata, "entry_points", forbidden)

    record = env.activator.activate(_manifest(), [_readiness()], _handles())

    assert record.languages[0].tasks == ("deidentification", "ner")


# --- approximate locales never become supported claims --------------------


def test_declared_locale_approximation_caps_claim_at_preview(env: _Fixture) -> None:
    manifest = _manifest(
        _pack_spec(
            surrogate_locale="en_US",
            surrogate_locale_approximation=(
                "Faker has no native Lithuanian-like locale in this synthetic case"
            ),
        )
    )

    record = env.activator.activate(manifest, [_readiness()], _handles())

    (language,) = record.languages
    assert language.readiness_state == "supported"
    assert language.locale_status == "approximated"
    assert language.claim == "preview"
    assert env.activator.clinical_claim("lt") == "preview"


def test_backend_locale_approximation_caps_claim_at_preview(env: _Fixture) -> None:
    # Faker has no af_ZA locale; OpenMed's resolver backs it with another
    # locale, so the coherence report marks it approximated even though the
    # bundle did not say so.
    manifest = _manifest(_pack_spec("af", model="lt-pii", surrogate_locale="af_ZA"))

    record = env.activator.activate(manifest, [_readiness("af")], _handles())

    assert record.languages[0].locale_status == "approximated"
    assert env.activator.clinical_claim("af") == "preview"


def test_claims_and_tasks_match_exact_language_tags(env: _Fixture) -> None:
    manifest = _manifest(_pack_spec("lt-LT"))

    record = env.activator.activate(manifest, [_readiness("lt-LT")], _handles())

    assert record.languages[0].language == "lt-LT"
    assert record.languages[0].code == "lt"
    assert env.activator.clinical_claim("lt-LT") == "supported"
    assert env.activator.qualified_tasks("lt-LT") == ("deidentification", "ner")
    assert env.activator.clinical_claim("lt") is None
    assert env.activator.qualified_tasks("lt") == ()
    assert env.activator.clinical_claim("lt-LV") is None


def test_preview_readiness_never_reports_supported(env: _Fixture) -> None:
    env.activator.activate(_manifest(), [_readiness(state="preview")], _handles())

    assert env.activator.clinical_claim("lt") == "preview"


# --- rollback -------------------------------------------------------------


def test_rollback_restores_previous_generation_and_user_packs(
    env: _Fixture,
) -> None:
    _generation, packs_before = env.registry._snapshot()
    record = env.activator.activate(_manifest(), [_readiness()], _handles())
    later_user_pack = env.registry.register(_user_pack("fi", surrogate_locale="fi_FI"))
    notifications = len(env.notifications)

    assert env.activator.rollback(record.activation_id) is None

    assert env.registry.find("lt") is None
    assert env.registry.get("en") is packs_before["en"]
    assert env.registry.get("fi") is later_user_pack
    assert len(env.notifications) == notifications + 1
    assert env.activator.active_activations() == ()
    assert env.activator.clinical_claim("lt") is None


def test_upgrade_supersedes_and_rollback_restores_previous_version(
    env: _Fixture,
) -> None:
    v1 = env.activator.activate(_manifest(), [_readiness()], _handles())
    v1_pack = env.registry.get("lt")
    v2_manifest = _manifest(
        _pack_spec("lt", tasks=["deidentification"]),
        _pack_spec("sl"),
        bundle_version="1.1.0",
    )

    v2 = env.activator.activate(
        v2_manifest, [_readiness("lt"), _readiness("sl")], _handles()
    )

    assert v2.previous_activation_id == v1.activation_id
    assert env.activator.qualified_tasks("lt") == ("deidentification",)
    assert env.registry.find("sl") is not None
    assert env.activator.active_activations() == (v2,)

    assert env.activator.rollback(v2.activation_id) == v1
    assert env.registry.get("lt") is v1_pack
    assert env.registry.find("sl") is None
    assert env.activator.qualified_tasks("lt") == ("deidentification", "ner")

    with pytest.raises(LanguageBundleActivationError) as excinfo:
        env.activator.rollback(v2.activation_id)
    assert _refusal(excinfo) == "rollback_not_active"

    assert env.activator.rollback(v1.activation_id) is None
    assert env.registry.find("lt") is None


def test_same_or_older_version_cannot_supersede(env: _Fixture) -> None:
    env.activator.activate(
        _manifest(bundle_version="2.0.0"), [_readiness()], _handles()
    )
    before = env.state()

    for version in ("2.0.0", "1.9.9"):
        with pytest.raises(LanguageBundleActivationError) as excinfo:
            env.activator.activate(
                _manifest(bundle_version=version), [_readiness()], _handles()
            )
        assert _refusal(excinfo) == "bundle_version_not_newer"
    assert env.state() == before


def test_rollback_refuses_when_other_code_replaced_the_pack(env: _Fixture) -> None:
    record = env.activator.activate(_manifest(), [_readiness()], _handles())
    replacement = env.registry.register(
        _user_pack("lt", surrogate_locale="lt_LT"), replace=True
    )
    before = env.state()

    with pytest.raises(LanguageBundleActivationError) as excinfo:
        env.activator.rollback(record.activation_id)

    assert _refusal(excinfo) == "rollback_conflict"
    assert env.state() == before
    assert env.registry.get("lt") is replacement


def test_separate_activators_treat_each_other_as_foreign(env: _Fixture) -> None:
    env.activator.activate(_manifest(), [_readiness()], _handles())
    other = LanguageBundleActivator(
        verifier=HmacVerifier(), registry=env.registry, clock=lambda: NOW
    )

    with pytest.raises(LanguageBundleActivationError) as excinfo:
        other.activate(_manifest(bundle_version="9.0.0"), [_readiness()], _handles())

    assert _refusal(excinfo) == "pack_identity_conflict"


# --- plain-data inputs and registry primitives ----------------------------


def test_readiness_evidence_round_trips_and_validates() -> None:
    evidence = _readiness("pt-br", model_digest=LT_MODEL)

    assert evidence.language == "pt-BR"
    assert LanguageReadinessEvidence.from_dict(evidence.to_dict()) == evidence
    assert evidence.to_dict()["issued_at"] == "2026-08-31T12:00:00Z"

    invalid: list[dict[str, Any]] = [
        {"issued_at": datetime(2026, 8, 31, 12, 0)},
        {"state": "certified"},
        {"evidence_digest": "not-a-digest"},
        {"qualified_tasks": "deidentification"},
        {"language": "pt_BR"},
    ]
    for override in invalid:
        values = {
            "language": "lt",
            "state": "supported",
            "qualified_tasks": ("deidentification",),
            "issued_at": NOW,
            "evidence_digest": _sha("synthetic"),
            **override,
        }
        with pytest.raises(LanguageBundleActivationError) as excinfo:
            LanguageReadinessEvidence(**values)
        assert _refusal(excinfo) == "readiness_malformed"


def test_activator_rejects_invalid_configuration(env: _Fixture) -> None:
    with pytest.raises(TypeError):
        LanguageBundleActivator(verifier=object())  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        LanguageBundleActivator(verifier=HmacVerifier(), max_readiness_age=timedelta(0))
    naive = LanguageBundleActivator(
        verifier=HmacVerifier(),
        registry=env.registry,
        clock=lambda: datetime(2026, 9, 1),
    )
    with pytest.raises(TypeError):
        naive.activate(_manifest(), [_readiness()], _handles())


def test_registry_generation_tracks_every_commit() -> None:
    registry = LanguagePackRegistry()
    assert registry.generation == 0
    registry.register(_user_pack())
    assert registry.generation == 1

    generation, packs = registry._snapshot()
    assert registry._commit_generation({}, expected_generation=generation - 1) is None
    assert registry.find("en") is not None

    assert registry._commit_generation({}, expected_generation=generation) == 2
    assert registry.find("en") is None
    with pytest.raises(ValueError, match="must match"):
        registry._commit_generation(
            {"fr": _user_pack()}, expected_generation=registry.generation
        )
    assert dict(packs) == {"en": _user_pack()}


def test_refusal_reasons_are_controlled_codes() -> None:
    assert all(
        reason.value == reason.value.lower() and " " not in reason.value
        for reason in ActivationRefusal
    )
