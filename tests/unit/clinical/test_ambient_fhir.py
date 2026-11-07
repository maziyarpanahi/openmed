"""Synthetic offline ambient document contract and privacy negative controls."""

import copy
import json
from pathlib import Path

import pytest

from openmed.clinical.exporters.fhir.ambient import (
    AmbientDocumentError,
    ambient_draft_digest,
    export_ambient_document,
    import_ambient_document,
    validate_ambient_document,
)
from openmed.interop.fhir.validation import validation_result

FIXTURE = Path(__file__).parents[2] / "fixtures/clinical/ambient_fhir.json"
SECRET = "SyntheticTranscriptOnlyIdentifier-7395"


@pytest.fixture
def draft():
    return json.loads(FIXTURE.read_text())["draft"]


def export(draft, **kwargs):
    return export_ambient_document(
        draft,
        current_evidence_digest=kwargs.pop("current_evidence_digest", "1" * 64),
        recorded_at=kwargs.pop("recorded_at", "2026-01-02T03:04:05Z"),
        **kwargs,
    )


def test_golden_reviewed_roundtrip(draft):
    fixture = json.loads(FIXTURE.read_text())
    digest = ambient_draft_digest(draft)
    assert digest == fixture["digest"]
    result = export(draft, confirmed_digest=digest)
    assert result.bundle == fixture["final_document"]
    assert validation_result(result.bundle).valid
    assert validate_ambient_document(result.bundle).valid
    assert import_ambient_document(result.bundle) == draft
    assert result.losses == (
        ("transcript_payload", 1),
        ("audio_payload", 1),
        ("review_history", 2),
    )
    assert draft["sections"][0]["note"] not in repr(result)
    entries = result.bundle["entry"]
    assert entries[1]["resource"]["target"][0]["reference"] == entries[0]["fullUrl"]
    assert all("request" not in entry for entry in entries)


def test_reviewed_defaults_preliminary_and_is_detached(draft):
    before = copy.deepcopy(draft)
    result = export(draft)
    assert result.bundle["entry"][0]["resource"]["status"] == "preliminary"
    assert import_ambient_document(result.bundle) == before
    draft["sections"][0]["note"] = SECRET
    assert SECRET not in json.dumps(result.bundle)


@pytest.mark.parametrize("final", [False, True])
@pytest.mark.parametrize(
    "state,code",
    [
        ("unreviewed", "unreviewed_draft"),
        ("stale_review", "stale_review"),
        ("stale_evidence", "stale_evidence"),
        ("pending", "correction_pending"),
    ],
)
def test_refuse_unsafe_states(draft, state, code, final):
    kwargs = {"confirmed_digest": draft["review"]["digest"]} if final else {}
    if state == "unreviewed":
        draft["review"] = None
    elif state == "stale_review":
        draft["sections"][0]["note"] += SECRET
    elif state == "stale_evidence":
        kwargs["current_evidence_digest"] = "2" * 64
    else:
        draft["correction_pending"] = True
    with pytest.raises(AmbientDocumentError, match=f"^{code}$") as caught:
        export(draft, **kwargs)
    assert SECRET not in str(caught.value)
    assert caught.value.__cause__ is None


@pytest.mark.parametrize(
    "change", ["order", "note", "reference", "speaker", "start", "end", "loss"]
)
def test_review_commits_every_supported_field(draft, change):
    if change == "order":
        draft["sections"].reverse()
    elif change == "note":
        draft["sections"][0]["note"] += "changed"
    elif change == "loss":
        draft["loss_counts"][0]["count"] += 1
    elif change in {"reference", "speaker"}:
        draft["sections"][0]["evidence"][0][change] = (
            "urn:uuid:50000000-0000-4000-8000-000000000001"
        )
    else:
        draft["sections"][0]["evidence"][0][change] += 1
    with pytest.raises(AmbientDocumentError, match="^stale_review$"):
        export(draft)


@pytest.mark.parametrize("value", [SECRET, "2" * 64, True, 1])
def test_confirmation_must_bind_exact_snapshot(draft, value):
    with pytest.raises(AmbientDocumentError, match="^confirmation_mismatch$"):
        export(draft, confirmed_digest=value)


@pytest.mark.parametrize("location", ["draft", "section", "evidence", "review", "loss"])
def test_unrecognized_source_fields_never_echo(draft, location):
    target = {
        "draft": draft,
        "section": draft["sections"][0],
        "evidence": draft["sections"][0]["evidence"][0],
        "review": draft["review"],
        "loss": draft["loss_counts"][0],
    }[location]
    target[SECRET] = SECRET
    with pytest.raises(AmbientDocumentError, match="^invalid_draft$") as caught:
        export(draft)
    assert SECRET not in repr(caught.value)


@pytest.mark.parametrize(
    "value",
    [
        SECRET,
        "/private/secret.wav",
        "https://example.test/transcript",
        "urn:uuid:invalid",
        None,
    ],
)
def test_evidence_must_be_opaque(draft, value):
    draft["sections"][0]["evidence"][0]["reference"] = value
    with pytest.raises(AmbientDocumentError, match="^invalid_draft$"):
        export(draft)


@pytest.mark.parametrize("value", [True, -1, 1.5, SECRET, 2**31])
def test_offset_contract(draft, value):
    draft["sections"][0]["evidence"][0]["start"] = value
    with pytest.raises(AmbientDocumentError, match="^invalid_draft$"):
        export(draft)


@pytest.mark.parametrize(
    "kind",
    [
        "attachment",
        "narrative",
        "extension",
        "section",
        "source",
        "request",
        "status",
        "digest",
        "confirmation",
        "float_schema",
    ],
)
def test_roundtrip_rejects_tampering_with_value_free_errors(draft, kind):
    bundle = export(draft).bundle
    composition = bundle["entry"][0]["resource"]
    if kind == "attachment":
        composition["content"] = [{"attachment": {"data": SECRET, "title": SECRET}}]
    elif kind == "narrative":
        composition["text"]["div"] += SECRET
    elif kind == "extension":
        composition["extension"].append({"url": SECRET, "valueString": SECRET})
    elif kind == "section":
        composition["section"][0]["text"]["div"] += SECRET
    elif kind == "source":
        bundle["entry"][1]["resource"]["entity"][0]["what"]["display"] = SECRET
    elif kind == "request":
        bundle["entry"][0]["request"] = {"method": "POST", "url": SECRET}
    elif kind == "status":
        composition["status"] = "final"
    elif kind == "digest":
        composition["extension"][1]["valueString"] = "2" * 64
    elif kind == "float_schema":
        composition["extension"][0]["valueUnsignedInt"] = 1.0
    else:
        composition["extension"][4]["valueBoolean"] = 0
    result = validate_ambient_document(bundle)
    assert not result.valid
    assert SECRET not in repr(result)
    with pytest.raises(AmbientDocumentError, match="^invalid_document$") as caught:
        import_ambient_document(bundle)
    assert caught.value.__cause__ is None


@pytest.mark.parametrize(
    "payload", [None, {}, {"entry": []}, {"entry": [None]}, {"entry": SECRET}]
)
def test_malformed_import_never_raises_or_echoes(payload):
    assert not validate_ambient_document(payload).valid


@pytest.mark.parametrize(
    "instant",
    [SECRET, "2026-02-30T00:00:00Z", "2026-01-01", "2026-01-01T00:00:00+00:00"],
)
def test_injected_clock_is_strict(draft, instant):
    with pytest.raises(AmbientDocumentError, match="^invalid_recorded_time$"):
        export(draft, recorded_at=instant)


def test_xml_escape_controls_and_multilingual_note(draft):
    draft["sections"][0]["note"] = "合成 & <script>alert('x')</script> &lt; café 🩺"
    draft["review"]["digest"] = ambient_draft_digest(draft)
    assert import_ambient_document(export(draft).bundle) == draft
    draft["sections"][0]["note"] = "unsafe\x00"
    with pytest.raises(AmbientDocumentError, match="^invalid_draft$"):
        ambient_draft_digest(draft)


@pytest.mark.parametrize(
    "change", ["unknown", "duplicate", "zero", "bool", "float", "negative", "large"]
)
def test_controlled_loss_categories_and_counts(draft, change):
    row = draft["loss_counts"][0]
    if change == "unknown":
        row["category"] = SECRET
    elif change == "duplicate":
        draft["loss_counts"].append(copy.deepcopy(row))
    else:
        row["count"] = {
            "zero": 0,
            "bool": True,
            "float": 1.0,
            "negative": -1,
            "large": 2**31,
        }[change]
    with pytest.raises(AmbientDocumentError, match="^invalid_draft$"):
        export(draft)


def test_empty_loss_report_is_not_fabricated(draft):
    draft["loss_counts"] = []
    draft["review"]["digest"] = ambient_draft_digest(draft)
    result = export(draft)
    assert result.losses == ()
    assert import_ambient_document(result.bundle) == draft


@pytest.mark.parametrize(
    "change",
    [
        "empty_sections",
        "empty_evidence",
        "reversed_span",
        "unknown_section",
        "too_long",
        "too_many_sections",
    ],
)
def test_bounded_section_and_evidence_contract(draft, change):
    if change == "empty_sections":
        draft["sections"] = []
    elif change == "empty_evidence":
        draft["sections"][0]["evidence"] = []
    elif change == "reversed_span":
        draft["sections"][0]["evidence"][0]["start"] = 12
    elif change == "unknown_section":
        draft["sections"][0]["code"] = SECRET
    elif change == "too_long":
        draft["sections"][0]["note"] = "a" * 16385
    else:
        draft["sections"] = draft["sections"] * 17
    with pytest.raises(AmbientDocumentError, match="^invalid_draft$"):
        export(draft)
