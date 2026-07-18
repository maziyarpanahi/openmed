"""Tests for WHO ICD-API snapshot building and offline grounding."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from urllib import parse as urlparse
from urllib import request as urlrequest

import pytest

from openmed.cli import main_module
from openmed.clinical.exporters.code_provenance import (
    CODE_SYSTEM_VERSION_SOURCE_EXTENSION_URL,
)
from openmed.interop import RedactionMapping, assert_redacted
from openmed.interop.icd11_api import (
    ICD11_MMS_SYSTEM,
    ICD11APIClient,
    SnapshotIntegrityError,
    build_snapshot,
    ground_mention,
    ground_to_codeable_concept,
    load_snapshot,
    snapshot_path,
)

FIXTURES = Path(__file__).parent / "fixtures" / "icd11"
SYNTHETIC_SNAPSHOT = FIXTURES / "synthetic-snapshot.json"


class _FixtureResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def __enter__(self) -> _FixtureResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self._payload


class _RecordedFixtureOpener:
    def __init__(self) -> None:
        self.requests: list[urlrequest.Request] = []

    def __call__(
        self,
        request: urlrequest.Request,
        *,
        timeout: float,
    ) -> _FixtureResponse:
        assert timeout == 7.0
        self.requests.append(request)
        parsed = urlparse.urlparse(request.full_url)

        if parsed.netloc == "icdaccessmanagement.who.int":
            fixture = "token.json"
        elif parsed.path.endswith("/search"):
            fixture = "search.json"
        elif parsed.path.endswith("/lookup"):
            fixture = "lookup.json"
        elif parsed.path.endswith("/mms"):
            fixture = "release.json"
        elif parsed.path.endswith("/1000"):
            fixture = "chapter-05.json"
        elif parsed.path.endswith("/2001"):
            fixture = "entity-diabetes.json"
        elif parsed.path.endswith("/2002"):
            fixture = "entity-malaria.json"
        else:  # pragma: no cover - makes unexpected network shapes explicit
            raise AssertionError(f"unhandled fixture request: {request.full_url}")
        return _FixtureResponse((FIXTURES / fixture).read_bytes())


def _client(opener: _RecordedFixtureOpener) -> ICD11APIClient:
    return ICD11APIClient(
        "fixture-client",
        "fixture-secret",
        timeout=7.0,
        opener=opener,
    )


def test_client_uses_oauth_search_and_pinned_lookup_endpoints() -> None:
    opener = _RecordedFixtureOpener()
    client = _client(opener)

    search = client.search("type 2 diabetes", release="2026-01", chapters=["05"])
    lookup = client.lookup(
        "http://id.who.int/icd/entity/2001",
        release="2026-01",
    )

    assert search["destinationEntities"][0]["theCode"] == "TEST-5A11"
    assert lookup["code"] == "TEST-5A11"
    assert len(opener.requests) == 3

    token_request, search_request, lookup_request = opener.requests
    assert token_request.method == "POST"
    authorization = token_request.get_header("Authorization")
    assert authorization is not None
    encoded_credentials = authorization.removeprefix("Basic ")
    assert base64.b64decode(encoded_credentials).decode() == (
        "fixture-client:fixture-secret"
    )
    assert urlparse.parse_qs(token_request.data.decode()) == {
        "grant_type": ["client_credentials"],
        "scope": ["icdapi_access"],
    }

    search_query = urlparse.parse_qs(urlparse.urlparse(search_request.full_url).query)
    assert search_query["q"] == ["type 2 diabetes"]
    assert search_query["chapterFilter"] == ["05"]
    assert search_query["highlightingEnabled"] == ["false"]
    assert search_request.get_header("Api-version") == "v2"
    assert search_request.get_header("Accept-language") == "en"
    lookup_url = urlparse.urlparse(lookup_request.full_url)
    assert lookup_url.path.endswith("/2026-01/mms/lookup")
    assert urlparse.parse_qs(lookup_url.query)["foundationUri"] == [
        "http://id.who.int/icd/entity/2001"
    ]


def test_client_rejects_lookup_outside_pinned_who_release() -> None:
    client = _client(_RecordedFixtureOpener())

    with pytest.raises(ValueError, match="must identify one WHO ICD entity"):
        client.lookup("https://example.test/entity/1", release="2026-01")
    with pytest.raises(ValueError, match="outside the pinned WHO MMS release"):
        client.get_entity(
            "https://id.who.int/icd/release/11/2025-01/mms/1",
            release="2026-01",
        )


def test_release_and_chapter_pins_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="release month"):
        snapshot_path(release="2026-13", chapters=["05"])
    with pytest.raises(ValueError, match="chapter codes"):
        snapshot_path(release="2026-01", chapters=["99"])


def test_builder_writes_byte_identical_snapshot_and_manifest(
    tmp_path: Path,
) -> None:
    opener = _RecordedFixtureOpener()
    client = _client(opener)

    first = build_snapshot(
        client,
        release="2026-01",
        chapters=["05"],
        cache_dir=tmp_path / "first",
    )
    second = build_snapshot(
        client,
        release="2026-01",
        chapters=["05"],
        cache_dir=tmp_path / "second",
    )

    assert first.entity_count == 2
    assert first.snapshot_path.read_bytes() == second.snapshot_path.read_bytes()
    assert first.manifest_path.read_bytes() == second.manifest_path.read_bytes()
    assert first.snapshot_sha256 == second.snapshot_sha256

    snapshot = load_snapshot(first.snapshot_path)
    assert snapshot.release == "2026-01"
    assert snapshot.chapters == ("05",)
    assert [entity.code for entity in snapshot.entities] == [
        "TEST-1F40",
        "TEST-5A11",
    ]
    assert all(
        request.full_url.startswith(
            (
                "https://id.who.int/",
                "https://icdaccessmanagement.who.int/",
            )
        )
        for request in opener.requests
    )


def test_round_trip_grounding_exports_fhir_coding_and_provenance() -> None:
    snapshot = load_snapshot(SYNTHETIC_SNAPSHOT)

    entity = ground_mention("  Type-2   Diabetes ", snapshot)
    concept = ground_to_codeable_concept("T2DM", snapshot)

    assert entity is not None
    assert entity.code == "TEST-5A11"
    assert concept is not None
    assert concept["text"] == "Type 2 diabetes mellitus"
    assert "T2DM" not in json.dumps(concept)
    assert concept["coding"] == [
        {
            "system": ICD11_MMS_SYSTEM,
            "code": "TEST-5A11",
            "display": "Type 2 diabetes mellitus",
            "_score": 1.0,
            "version": "2026-01",
            "extension": [
                {
                    "url": CODE_SYSTEM_VERSION_SOURCE_EXTENSION_URL,
                    "valueString": (
                        "ICD-11 MMS snapshot sha256:"
                        "3b147f95cc5ed6fb86ebabad4a3eaa38145f94fe6406520a29efdeda44a702a4"
                    ),
                }
            ],
        }
    ]


def test_runtime_grounding_is_offline_and_does_not_copy_phi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("runtime grounding attempted network access")

    monkeypatch.setattr(urlrequest, "urlopen", fail_network)
    snapshot = load_snapshot(SYNTHETIC_SNAPSHOT)
    note = "Patient Casey Example has type 2 diabetes. Email casey@example.test."
    mention_start = note.index("type 2 diabetes")
    mention = note[mention_start : mention_start + len("type 2 diabetes")]

    concept = ground_to_codeable_concept(mention, snapshot)

    assert concept is not None
    mapping = RedactionMapping(
        {
            "[NAME]": "Casey Example",
            "[EMAIL]": "casey@example.test",
        }
    )
    assert_redacted(json.dumps(concept, sort_keys=True), mapping)
    assert "Casey Example" not in json.dumps(concept)
    assert "casey@example.test" not in json.dumps(concept)


def test_snapshot_integrity_failure_is_closed(tmp_path: Path) -> None:
    snapshot_copy = tmp_path / "synthetic-snapshot.json"
    manifest_copy = tmp_path / "synthetic-snapshot.manifest.json"
    snapshot_copy.write_bytes(SYNTHETIC_SNAPSHOT.read_bytes() + b" ")
    manifest_copy.write_bytes(
        SYNTHETIC_SNAPSHOT.with_suffix(".manifest.json").read_bytes()
    )

    with pytest.raises(SnapshotIntegrityError, match="sha256"):
        load_snapshot(snapshot_copy)


def test_cli_registers_release_pinned_snapshot_builder() -> None:
    args = main_module.build_parser().parse_args(
        [
            "icd11",
            "build-snapshot",
            "--release",
            "2026-01",
            "--chapter",
            "05",
        ]
    )

    assert args.command == "icd11"
    assert args.icd11_command == "build-snapshot"
    assert args.release == "2026-01"
    assert args.chapters == ["05"]
