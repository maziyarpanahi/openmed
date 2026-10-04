"""Tests for the offline streaming-ASR provider contract."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from openmed.multimodal.streaming import speech
from openmed.multimodal.streaming.speech import providers

FINGERPRINT = "sha256:" + "a" * 64
PRIVATE_TEXT = "participant-5550199-said-something-private"
PRIVATE_PAYLOAD = b"private-audio-5550199" * 4
SENTINELS = (
    PRIVATE_TEXT,
    "5550199",
    "private-audio",
    "clinic/host-7",
    "Participant@example.com",
    "123-45-6789",
)


def _provenance(**overrides: object) -> providers.ProviderProvenance:
    values: dict[str, object] = {
        "provider_id": "whisper-local",
        "provider_revision": "1.2.0",
        "model_fingerprint": FINGERPRINT,
    }
    values.update(overrides)
    return providers.ProviderProvenance(**values)  # type: ignore[arg-type]


def _limits(**overrides: object) -> providers.SpeechStreamLimits:
    values: dict[str, object] = {
        "max_buffered_chunks": 4,
        "max_buffered_byte_count": 4096,
        "max_buffered_duration_ms": 1000,
        "max_stream_duration_ms": 10000,
        "max_partial_hypotheses": 8,
        "max_segments": 4,
    }
    values.update(overrides)
    return providers.SpeechStreamLimits(**values)  # type: ignore[arg-type]


def _declaration(**overrides: object) -> providers.SpeechProviderDeclaration:
    values: dict[str, object] = {
        "provenance": _provenance(),
        "languages": ("de", "en"),
        "sample_rates_hz": (8000, 16000),
        "limits": _limits(),
    }
    values.update(overrides)
    return providers.SpeechProviderDeclaration(**values)  # type: ignore[arg-type]


def _chunk(
    sequence: int = 0,
    *,
    byte_count: int = 1024,
    duration_ms: int = 64,
    sample_rate_hz: int = 16000,
    channel_count: int = 1,
    payload: Any = None,
) -> providers.AudioChunk:
    if payload is None:
        repeats = byte_count // len(PRIVATE_PAYLOAD) + 1
        body: Any = (PRIVATE_PAYLOAD * repeats)[:byte_count]
    elif type(payload) is bytes and len(payload) != byte_count:
        body = (payload + bytes(byte_count))[:byte_count]
    else:
        body = payload
    return providers.AudioChunk(
        sequence=sequence,
        payload=body,
        sample_rate_hz=sample_rate_hz,
        channel_count=channel_count,
        duration_ms=duration_ms,
    )


def _token(
    index: int = 0, *, start_ms: int = 0, end_ms: int = 10
) -> providers.TokenTime:
    return providers.TokenTime(
        token_index=index,
        start_ms=start_ms,
        end_ms=end_ms,
        confidence_ppm=900000,
        token="h",
    )


def _hypothesis(revision: int = 0, **overrides: object) -> providers.PartialHypothesis:
    values: dict[str, object] = {
        "revision": revision,
        "text": PRIVATE_TEXT,
        "is_final": False,
        "token_times": (_token(0),),
        "language": providers.LanguageConfidence("en", 950000),
    }
    values.update(overrides)
    return providers.PartialHypothesis(**values)  # type: ignore[arg-type]


def _segment(index: int = 0, **overrides: object) -> providers.FinalizedSegment:
    values: dict[str, object] = {
        "segment_index": index,
        "text": PRIVATE_TEXT,
        "start_ms": 0,
        "end_ms": 64,
        "token_times": (_token(0),),
        "language": providers.LanguageConfidence("en", 950000),
    }
    values.update(overrides)
    return providers.FinalizedSegment(**values)  # type: ignore[arg-type]


def _assert_error(category: str, call) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(providers.SpeechProviderError) as raised:
        call()
    assert raised.value.category == category
    assert str(raised.value) == category


def test_reason_codes_are_unique_and_ordered() -> None:
    codes = providers.SPEECH_PROVIDER_REASON_CODES
    assert isinstance(codes, tuple)
    assert codes
    assert len(set(codes)) == len(codes)
    assert all(isinstance(code, str) and code for code in codes)


def test_public_surface_matches_provider_module() -> None:
    assert speech.__all__ == providers.__all__
    assert len(set(speech.__all__)) == len(speech.__all__)
    for name in providers.__all__:
        assert getattr(speech, name) is getattr(providers, name)


def test_module_declares_no_network_imports() -> None:
    source = providers.__file__
    assert source is not None
    with open(source, encoding="utf-8") as handle:
        text = handle.read()
    for forbidden in (
        "import socket",
        "import urllib",
        "import requests",
        "http.client",
        "boto3",
        "import ssl",
    ):
        assert forbidden not in text


def test_chunk_manifest_is_content_free() -> None:
    chunk = _chunk(3)
    assert chunk.byte_count == 1024
    assert chunk.to_dict() == {
        "sequence": 3,
        "byte_count": 1024,
        "sample_rate_hz": 16000,
        "channel_count": 1,
        "duration_ms": 64,
    }
    assert PRIVATE_TEXT not in repr(chunk)
    assert "private-audio" not in repr(chunk)
    assert "payload" not in repr(chunk)


def test_token_and_language_manifests_are_content_free() -> None:
    token = providers.TokenTime(
        token_index=2,
        start_ms=30,
        end_ms=55,
        confidence_ppm=750000,
        token=PRIVATE_TEXT,
    )
    assert token.to_dict() == {
        "token_index": 2,
        "start_ms": 30,
        "end_ms": 55,
        "confidence_ppm": 750000,
        "token_length": len(PRIVATE_TEXT),
    }
    assert PRIVATE_TEXT not in repr(token)
    language = providers.LanguageConfidence("en-US", 500000)
    assert language.to_dict() == {"language": "en-US", "confidence_ppm": 500000}


def test_hypothesis_and_segment_manifests_are_content_free() -> None:
    hypothesis = _hypothesis(4, is_final=True)
    assert hypothesis.to_dict() == {
        "revision": 4,
        "text_length": len(PRIVATE_TEXT),
        "is_final": True,
        "token_count": 1,
        "language": "en",
    }
    assert PRIVATE_TEXT not in repr(hypothesis)
    segment = _segment(2, start_ms=64, end_ms=128)
    assert segment.to_dict() == {
        "segment_index": 2,
        "text_length": len(PRIVATE_TEXT),
        "start_ms": 64,
        "end_ms": 128,
        "duration_ms": 64,
        "token_count": 1,
        "language": "en",
    }
    assert PRIVATE_TEXT not in repr(segment)


def test_provenance_and_declaration_are_deterministic() -> None:
    provenance = _provenance()
    assert provenance.to_dict() == {
        "provider_id": "whisper-local",
        "provider_revision": "1.2.0",
        "model_fingerprint": FINGERPRINT,
        "requires_network": False,
    }
    declaration = _declaration()
    assert declaration.provider_id == "whisper-local"
    assert declaration.to_dict() == {
        "provenance": provenance.to_dict(),
        "languages": ["de", "en"],
        "sample_rates_hz": [8000, 16000],
        "limits": _limits().to_dict(),
    }
    first = declaration.to_json()
    assert first == declaration.to_json()
    assert first == json.dumps(
        declaration.to_dict(), sort_keys=True, separators=(",", ":")
    )
    assert "whisper-local" in first


def test_golden_session_report_is_exact() -> None:
    session = providers.StreamingAsrSession(_declaration())
    assert session.state is providers.StreamState.OPEN
    assert session.push_chunk(_chunk(0)) == 1
    assert session.push_chunk(_chunk(1)) == 2
    assert session.buffered_byte_count == 2048
    assert session.buffered_duration_ms == 128
    assert session.drain() == 2
    assert session.buffered_chunks == 0
    assert session.buffered_byte_count == 0
    assert session.buffered_duration_ms == 0
    assert session.stream_duration_ms == 128
    assert session.push_chunk(_chunk(2, byte_count=512, duration_ms=32)) == 1
    assert session.submit_partial(_hypothesis(0)) == 1
    assert session.submit_partial(_hypothesis(1, is_final=True)) == 2
    assert session.submit_segment(_segment(0)) == 1
    report = session.finalize()
    assert session.state is providers.StreamState.FINALIZED
    expected = {
        "schema_version": providers.SPEECH_PROVIDER_CONTRACT_VERSION,
        "provider_id": "whisper-local",
        "provider_revision": "1.2.0",
        "state": "finalized",
        "buffered_chunks": 1,
        "buffered_byte_count": 512,
        "buffered_duration_ms": 32,
        "stream_duration_ms": 160,
        "partial_hypotheses": 2,
        "finalized_segments": 1,
        "reason_code": None,
        "requires_network": False,
    }
    assert report.to_dict() == expected
    assert report.to_json() == json.dumps(
        expected, sort_keys=True, separators=(",", ":")
    )
    assert session.report().to_json() == report.to_json()
    assert report.reason_code is None


def test_identical_sessions_produce_identical_reports() -> None:
    def run() -> str:
        session = providers.StreamingAsrSession(_declaration())
        session.push_chunk(_chunk(0))
        session.submit_partial(_hypothesis(0))
        session.submit_segment(_segment(0))
        return session.finalize().to_json()

    assert run() == run()


def test_report_carries_no_raw_values() -> None:
    session = providers.StreamingAsrSession(_declaration())
    session.push_chunk(_chunk(0))
    session.submit_partial(_hypothesis(0))
    session.submit_segment(_segment(0))
    payload = session.finalize().to_json() + repr(session.report())
    for sentinel in SENTINELS:
        assert sentinel not in payload


def test_session_finalize_accepts_a_last_segment() -> None:
    session = providers.StreamingAsrSession(_declaration())
    report = session.finalize(_segment(0, start_ms=0, end_ms=32))
    assert report.state == "finalized"
    assert report.finalized_segments == 1
    assert session.finalized_segments == 1


def test_cancel_is_terminal_and_reported() -> None:
    session = providers.StreamingAsrSession(_declaration())
    session.push_chunk(_chunk(0))
    report = session.cancel()
    assert session.state is providers.StreamState.CANCELLED
    assert report.state == "cancelled"
    assert report.reason_code == "session_cancelled"
    assert report.buffered_chunks == 1
    _assert_error("session_cancelled", lambda: session.cancel())
    _assert_error("session_cancelled", lambda: session.push_chunk(_chunk(1)))
    _assert_error("session_cancelled", lambda: session.submit_partial(_hypothesis(0)))
    _assert_error("session_cancelled", lambda: session.submit_segment(_segment(0)))
    _assert_error("session_cancelled", lambda: session.finalize())


def test_finalized_session_rejects_further_input() -> None:
    session = providers.StreamingAsrSession(_declaration())
    session.finalize()
    _assert_error("session_closed", lambda: session.finalize())
    _assert_error("session_closed", lambda: session.cancel())
    _assert_error("session_closed", lambda: session.push_chunk(_chunk(0)))
    _assert_error("session_closed", lambda: session.submit_partial(_hypothesis(0)))
    _assert_error("session_closed", lambda: session.submit_segment(_segment(0)))


def test_drain_is_allowed_after_the_session_closes() -> None:
    session = providers.StreamingAsrSession(_declaration())
    session.push_chunk(_chunk(0))
    session.finalize()
    assert session.drain() == 1
    assert session.report().buffered_chunks == 0


@pytest.mark.parametrize(
    ("label", "factory", "category"),
    [
        ("bool", lambda: _chunk(True), "chunk_sequence_invalid"),
        ("float", lambda: _chunk(1.0), "chunk_sequence_invalid"),
        ("negative", lambda: _chunk(-1), "chunk_sequence_out_of_range"),
        (
            "huge",
            lambda: _chunk(providers.MAX_CHUNK_SEQUENCE + 1),
            "chunk_sequence_out_of_range",
        ),
        ("text payload", lambda: _chunk(payload="x"), "audio_chunk_payload_invalid"),
        (
            "bytearray payload",
            lambda: _chunk(payload=bytearray(b"xy")),
            "audio_chunk_payload_invalid",
        ),
        (
            "empty payload",
            lambda: _chunk(byte_count=0),
            "audio_chunk_payload_out_of_range",
        ),
        (
            "oversized payload",
            lambda: _chunk(byte_count=providers.MAX_CHUNK_BYTE_COUNT + 1),
            "audio_chunk_payload_out_of_range",
        ),
        (
            "zero rate",
            lambda: _chunk(sample_rate_hz=0),
            "chunk_sample_rate_hz_out_of_range",
        ),
        (
            "huge rate",
            lambda: _chunk(sample_rate_hz=providers.MAX_SAMPLE_RATE_HZ + 1),
            "chunk_sample_rate_hz_out_of_range",
        ),
        (
            "bool rate",
            lambda: _chunk(sample_rate_hz=True),
            "chunk_sample_rate_hz_invalid",
        ),
        (
            "zero channels",
            lambda: _chunk(channel_count=0),
            "chunk_channel_count_out_of_range",
        ),
        (
            "many channels",
            lambda: _chunk(channel_count=providers.MAX_CHANNEL_COUNT + 1),
            "chunk_channel_count_out_of_range",
        ),
        (
            "zero duration",
            lambda: _chunk(duration_ms=0),
            "chunk_duration_ms_out_of_range",
        ),
        (
            "long duration",
            lambda: _chunk(duration_ms=providers.MAX_CHUNK_DURATION_MS + 1),
            "chunk_duration_ms_out_of_range",
        ),
    ],
)
def test_audio_chunk_validation(label, factory, category) -> None:  # type: ignore[no-untyped-def]
    _assert_error(category, factory)


@pytest.mark.parametrize(
    ("label", "factory", "category"),
    [
        ("negative index", lambda: _token(-1), "token_index_out_of_range"),
        ("float index", lambda: _token(1.5), "token_index_invalid"),
        (
            "backwards range",
            lambda: providers.TokenTime(0, 20, 10, 1),
            "token_time_range_invalid",
        ),
        (
            "negative start",
            lambda: providers.TokenTime(0, -1, 10, 1),
            "token_start_ms_out_of_range",
        ),
        (
            "loud confidence",
            lambda: providers.TokenTime(0, 0, 1, 1_000_001),
            "token_confidence_ppm_out_of_range",
        ),
        (
            "text token",
            lambda: providers.TokenTime(0, 0, 1, 1, token=b"x"),
            "token_text_invalid",
        ),
        (
            "long token",
            lambda: providers.TokenTime(
                0, 0, 1, 1, token="t" * (providers.MAX_HYPOTHESIS_CHARS + 1)
            ),
            "token_text_invalid",
        ),
        (
            "bad language",
            lambda: providers.LanguageConfidence("EN", 1),
            "provider_language_invalid",
        ),
        (
            "short language",
            lambda: providers.LanguageConfidence("e", 1),
            "provider_language_invalid",
        ),
        (
            "language int",
            lambda: providers.LanguageConfidence(1, 1),
            "provider_language_invalid",
        ),
        (
            "loud language",
            lambda: providers.LanguageConfidence("en", 1_000_001),
            "language_confidence_ppm_out_of_range",
        ),
    ],
)
def test_token_and_language_validation(label, factory, category) -> None:  # type: ignore[no-untyped-def]
    _assert_error(category, factory)


@pytest.mark.parametrize(
    ("label", "factory", "category"),
    [
        ("empty id", lambda: _provenance(provider_id=""), "provider_id_invalid"),
        ("upper id", lambda: _provenance(provider_id="Whisper"), "provider_id_invalid"),
        ("int id", lambda: _provenance(provider_id=1), "provider_id_invalid"),
        (
            "empty revision",
            lambda: _provenance(provider_revision=""),
            "provider_revision_invalid",
        ),
        (
            "spaced revision",
            lambda: _provenance(provider_revision="1.2 beta"),
            "provider_revision_invalid",
        ),
        (
            "long revision",
            lambda: _provenance(
                provider_revision="r" * (providers.MAX_PROVIDER_REVISION_CHARS + 2)
            ),
            "provider_revision_invalid",
        ),
        (
            "bare digest",
            lambda: _provenance(model_fingerprint="a" * 64),
            "provider_model_fingerprint_invalid",
        ),
        (
            "short digest",
            lambda: _provenance(model_fingerprint="sha256:" + "a" * 63),
            "provider_model_fingerprint_invalid",
        ),
        (
            "int flag",
            lambda: _provenance(requires_network=1),
            "provider_network_flag_invalid",
        ),
        (
            "text flag",
            lambda: _provenance(requires_network="no"),
            "provider_network_flag_invalid",
        ),
    ],
)
def test_provenance_validation(label, factory, category) -> None:  # type: ignore[no-untyped-def]
    _assert_error(category, factory)


@pytest.mark.parametrize(
    ("label", "factory", "category"),
    [
        (
            "zero chunks",
            lambda: _limits(max_buffered_chunks=0),
            "limit_max_buffered_chunks_out_of_range",
        ),
        (
            "many chunks",
            lambda: _limits(max_buffered_chunks=providers.MAX_BUFFERED_CHUNKS + 1),
            "limit_max_buffered_chunks_out_of_range",
        ),
        (
            "zero bytes",
            lambda: _limits(max_buffered_byte_count=0),
            "limit_max_buffered_byte_count_out_of_range",
        ),
        (
            "zero buffered duration",
            lambda: _limits(max_buffered_duration_ms=0),
            "limit_max_buffered_duration_ms_out_of_range",
        ),
        (
            "zero stream duration",
            lambda: _limits(max_stream_duration_ms=0),
            "limit_max_stream_duration_ms_out_of_range",
        ),
        (
            "zero partials",
            lambda: _limits(max_partial_hypotheses=0),
            "limit_max_partial_hypotheses_out_of_range",
        ),
        (
            "zero segments",
            lambda: _limits(max_segments=0),
            "limit_max_segments_out_of_range",
        ),
        (
            "bool limit",
            lambda: _limits(max_segments=True),
            "limit_max_segments_invalid",
        ),
    ],
)
def test_limit_validation(label, factory, category) -> None:  # type: ignore[no-untyped-def]
    _assert_error(category, factory)


@pytest.mark.parametrize(
    ("label", "factory", "category"),
    [
        (
            "network declaration",
            lambda: _declaration(provenance=_provenance(requires_network=True)),
            "provider_requires_network",
        ),
        (
            "text provenance",
            lambda: _declaration(provenance="local"),
            "provider_provenance_type_invalid",
        ),
        ("text limits", lambda: _declaration(limits=1), "provider_limits_type_invalid"),
        (
            "list languages",
            lambda: _declaration(languages=["en"]),
            "provider_languages_invalid",
        ),
        (
            "empty languages",
            lambda: _declaration(languages=()),
            "provider_languages_empty",
        ),
        (
            "too many languages",
            lambda: _declaration(
                languages=tuple(
                    f"l{index}" for index in range(providers.MAX_PROVIDER_LANGUAGES + 1)
                )
            ),
            "provider_languages_out_of_range",
        ),
        (
            "unsorted languages",
            lambda: _declaration(languages=("en", "de")),
            "provider_languages_unsorted",
        ),
        (
            "duplicate languages",
            lambda: _declaration(languages=("en", "en")),
            "provider_languages_unsorted",
        ),
        (
            "bad language tag",
            lambda: _declaration(languages=("english",)),
            "provider_language_invalid",
        ),
        (
            "list rates",
            lambda: _declaration(sample_rates_hz=[16000]),
            "provider_sample_rates_invalid",
        ),
        (
            "empty rates",
            lambda: _declaration(sample_rates_hz=()),
            "provider_sample_rates_empty",
        ),
        (
            "unsorted rates",
            lambda: _declaration(sample_rates_hz=(16000, 8000)),
            "provider_sample_rates_unsorted",
        ),
        (
            "huge rate",
            lambda: _declaration(sample_rates_hz=(providers.MAX_SAMPLE_RATE_HZ + 1,)),
            "provider_sample_rates_out_of_range",
        ),
        (
            "zero rate",
            lambda: _declaration(sample_rates_hz=(0,)),
            "provider_sample_rates_out_of_range",
        ),
    ],
)
def test_declaration_validation(label, factory, category) -> None:  # type: ignore[no-untyped-def]
    _assert_error(category, factory)


@pytest.mark.parametrize(
    ("label", "factory", "category"),
    [
        (
            "negative revision",
            lambda: _hypothesis(-1),
            "hypothesis_revision_out_of_range",
        ),
        ("float revision", lambda: _hypothesis(1.0), "hypothesis_revision_invalid"),
        ("byte text", lambda: _hypothesis(0, text=b"x"), "hypothesis_text_invalid"),
        (
            "long text",
            lambda: _hypothesis(0, text="t" * (providers.MAX_HYPOTHESIS_CHARS + 1)),
            "hypothesis_text_invalid",
        ),
        (
            "int flag",
            lambda: _hypothesis(0, is_final=1),
            "hypothesis_final_flag_invalid",
        ),
        (
            "list tokens",
            lambda: _hypothesis(0, token_times=[_token(0)]),
            "token_times_invalid",
        ),
        (
            "text token entry",
            lambda: _hypothesis(0, token_times=("t",)),
            "token_times_invalid",
        ),
        (
            "unsorted tokens",
            lambda: _hypothesis(0, token_times=(_token(1), _token(0))),
            "token_times_unsorted",
        ),
        (
            "duplicate tokens",
            lambda: _hypothesis(0, token_times=(_token(0), _token(0))),
            "token_times_unsorted",
        ),
        (
            "bad language",
            lambda: _hypothesis(0, language="en"),
            "hypothesis_language_type_invalid",
        ),
    ],
)
def test_hypothesis_validation(label, factory, category) -> None:  # type: ignore[no-untyped-def]
    _assert_error(category, factory)


@pytest.mark.parametrize(
    ("label", "factory", "category"),
    [
        ("negative index", lambda: _segment(-1), "segment_index_out_of_range"),
        ("byte text", lambda: _segment(0, text=b"x"), "segment_text_invalid"),
        (
            "reversed times",
            lambda: _segment(0, start_ms=64, end_ms=32),
            "segment_time_range_invalid",
        ),
        (
            "negative start",
            lambda: _segment(0, start_ms=-1),
            "segment_start_ms_out_of_range",
        ),
        ("list tokens", lambda: _segment(0, token_times=[]), "token_times_invalid"),
        (
            "bad language",
            lambda: _segment(0, language=1),
            "segment_language_type_invalid",
        ),
    ],
)
def test_segment_validation(label, factory, category) -> None:  # type: ignore[no-untyped-def]
    _assert_error(category, factory)


@pytest.mark.parametrize(
    ("label", "factory", "category"),
    [
        (
            "text declaration",
            lambda: providers.RegisteredSpeechProvider("local", "openmed.local:build"),
            "registered_declaration_type_invalid",
        ),
        (
            "bare entrypoint",
            lambda: providers.RegisteredSpeechProvider(_declaration(), "build"),
            "registered_entrypoint_invalid",
        ),
        (
            "upper entrypoint",
            lambda: providers.RegisteredSpeechProvider(
                _declaration(), "openmed.Local:build"
            ),
            "registered_entrypoint_invalid",
        ),
        (
            "int entrypoint",
            lambda: providers.RegisteredSpeechProvider(_declaration(), 1),
            "registered_entrypoint_invalid",
        ),
        (
            "long entrypoint",
            lambda: providers.RegisteredSpeechProvider(
                _declaration(),
                "openmed." + "a" * providers.MAX_ENTRYPOINT_CHARS + ".build",
            ),
            "registered_entrypoint_invalid",
        ),
    ],
)
def test_registration_validation(label, factory, category) -> None:  # type: ignore[no-untyped-def]
    _assert_error(category, factory)


def test_registration_sorts_and_rejects_duplicates() -> None:
    first = providers.RegisteredSpeechProvider(_declaration(), "openmed.speech.build")
    second = providers.RegisteredSpeechProvider(
        _declaration(
            provenance=_provenance(provider_id="alphavoice", provider_revision="0.1.0")
        ),
        "openmed.alpha.build",
    )
    registered = providers.register_speech_providers([first, second])
    assert isinstance(registered, tuple)
    assert [item.declaration.provider_id for item in registered] == [
        "alphavoice",
        "whisper-local",
    ]
    assert registered[1].to_dict() == {
        "provider_id": "whisper-local",
        "entrypoint": "openmed.speech.build",
        "declaration": _declaration().to_dict(),
    }
    _assert_error(
        "duplicate_provider_id",
        lambda: providers.register_speech_providers([first, first]),
    )
    _assert_error(
        "provider_registration_type_invalid",
        lambda: providers.register_speech_providers("whisper-local"),
    )
    _assert_error(
        "provider_registration_type_invalid",
        lambda: providers.register_speech_providers({"whisper-local": first}),
    )
    _assert_error(
        "provider_registration_entry_invalid",
        lambda: providers.register_speech_providers([first, "second"]),
    )
    assert providers.register_speech_providers([]) == ()


def test_session_rejects_untyped_input() -> None:
    session = providers.StreamingAsrSession(_declaration())
    assert session.state is providers.StreamState.OPEN
    _assert_error(
        "session_declaration_type_invalid",
        lambda: providers.StreamingAsrSession("x"),  # type: ignore[arg-type]
    )
    _assert_error(
        "session_limits_type_invalid",
        lambda: providers.StreamingAsrSession(
            _declaration(),
            limits="tight",  # type: ignore[arg-type]
        ),
    )


@pytest.mark.parametrize(
    ("label", "session_step", "category"),
    [
        (
            "text chunk",
            lambda session: session.push_chunk("x"),
            "audio_chunk_type_invalid",
        ),
        (
            "skipped sequence",
            lambda session: session.push_chunk(_chunk(1)),
            "chunk_sequence_out_of_order",
        ),
    ],
)
def test_session_step_validation(label, session_step, category) -> None:  # type: ignore[no-untyped-def]
    session = providers.StreamingAsrSession(_declaration())
    _assert_error(category, lambda: session_step(session))


def test_session_repeats_and_out_of_order_sequences() -> None:
    session = providers.StreamingAsrSession(_declaration())
    session.push_chunk(_chunk(0))
    _assert_error("chunk_sequence_out_of_order", lambda: session.push_chunk(_chunk(0)))
    _assert_error("chunk_sequence_out_of_order", lambda: session.push_chunk(_chunk(2)))
    assert session.push_chunk(_chunk(1)) == 2


def test_buffered_limits_are_enforced_per_dimension() -> None:
    session = providers.StreamingAsrSession(
        _declaration(limits=_limits(max_buffered_chunks=1))
    )
    session.push_chunk(_chunk(0))
    _assert_error("buffer_limit_exceeded", lambda: session.push_chunk(_chunk(1)))
    assert session.drain() == 1
    assert session.push_chunk(_chunk(1)) == 1

    byte_session = providers.StreamingAsrSession(
        _declaration(limits=_limits(max_buffered_byte_count=1500))
    )
    byte_session.push_chunk(_chunk(0, byte_count=1024))
    _assert_error(
        "buffer_limit_exceeded",
        lambda: byte_session.push_chunk(_chunk(1, byte_count=1024)),
    )
    assert byte_session.push_chunk(_chunk(1, byte_count=400)) == 2

    duration_session = providers.StreamingAsrSession(
        _declaration(limits=_limits(max_buffered_duration_ms=100))
    )
    duration_session.push_chunk(_chunk(0, duration_ms=64))
    _assert_error(
        "buffer_limit_exceeded",
        lambda: duration_session.push_chunk(_chunk(1, duration_ms=64)),
    )
    assert duration_session.push_chunk(_chunk(1, duration_ms=32)) == 2


def test_stream_duration_limit_is_enforced() -> None:
    session = providers.StreamingAsrSession(
        _declaration(limits=_limits(max_stream_duration_ms=100))
    )
    session.push_chunk(_chunk(0, duration_ms=64))
    assert session.drain() == 1
    _assert_error(
        "stream_duration_exceeded",
        lambda: session.push_chunk(_chunk(1, duration_ms=64)),
    )
    assert session.stream_duration_ms == 64
    assert session.push_chunk(_chunk(1, duration_ms=36)) == 1
    _assert_error(
        "stream_duration_exceeded",
        lambda: session.push_chunk(_chunk(2, duration_ms=1)),
    )


def test_partial_hypothesis_ordering_and_limit() -> None:
    session = providers.StreamingAsrSession(
        _declaration(limits=_limits(max_partial_hypotheses=2))
    )
    _assert_error(
        "partial_hypothesis_type_invalid",
        lambda: session.submit_partial("guess"),
    )
    assert session.submit_partial(_hypothesis(0)) == 1
    _assert_error(
        "hypothesis_revision_out_of_order",
        lambda: session.submit_partial(_hypothesis(0)),
    )
    assert session.submit_partial(_hypothesis(1)) == 2
    _assert_error(
        "partial_hypothesis_limit_exceeded",
        lambda: session.submit_partial(_hypothesis(2)),
    )
    assert session.report().partial_hypotheses == 2


def test_segment_ordering_and_limit() -> None:
    session = providers.StreamingAsrSession(
        _declaration(limits=_limits(max_segments=2))
    )
    _assert_error(
        "finalized_segment_type_invalid",
        lambda: session.submit_segment("segment"),
    )
    assert session.submit_segment(_segment(0, start_ms=0, end_ms=64)) == 1
    _assert_error(
        "segment_index_out_of_order",
        lambda: session.submit_segment(_segment(0, start_ms=64, end_ms=128)),
    )
    _assert_error(
        "segment_times_out_of_order",
        lambda: session.submit_segment(_segment(1, start_ms=32, end_ms=96)),
    )
    assert session.submit_segment(_segment(1, start_ms=64, end_ms=128)) == 2
    _assert_error(
        "segment_limit_exceeded",
        lambda: session.submit_segment(_segment(2, start_ms=128, end_ms=160)),
    )


def test_finalize_propagates_segment_errors() -> None:
    session = providers.StreamingAsrSession(_declaration())
    session.submit_segment(_segment(0, start_ms=0, end_ms=64))
    _assert_error(
        "segment_times_out_of_order",
        lambda: session.finalize(_segment(1, start_ms=32, end_ms=96)),
    )
    assert session.state is providers.StreamState.OPEN


def test_frozen_types_reject_mutation() -> None:
    frozen_values = (
        (_provenance(), "provider_id", "other"),
        (_declaration(), "languages", ("fr",)),
        (_limits(), "max_segments", 1),
        (_chunk(0), "sequence", 9),
        (_token(0), "token_index", 9),
        (providers.LanguageConfidence("en", 1), "language", "fr"),
        (_hypothesis(0), "revision", 9),
        (_segment(0), "segment_index", 9),
        (
            providers.RegisteredSpeechProvider(_declaration(), "openmed.speech.build"),
            "entrypoint",
            "openmed.other.build",
        ),
        (
            providers.StreamingAsrSession(_declaration()).finalize(),
            "state",
            "open",
        ),
    )
    for value, name, replacement in frozen_values:
        with pytest.raises(FrozenInstanceError):
            setattr(value, name, replacement)


def test_session_properties_track_declared_state() -> None:
    declaration = _declaration()
    session = providers.StreamingAsrSession(declaration)
    assert session.declaration is declaration
    assert session.limits is declaration.limits
    assert session.reason_code is None
    report = session.report()
    assert report.provider_id == declaration.provider_id
    assert report.provider_revision == declaration.provenance.provider_revision
    assert report.requires_network is False
    assert report.schema_version == providers.SPEECH_PROVIDER_CONTRACT_VERSION
    assert report.state == "open"
    assert report.buffered_chunks == 0
    assert report.buffered_byte_count == 0
    assert report.buffered_duration_ms == 0
    assert report.stream_duration_ms == 0
    assert report.partial_hypotheses == 0
    assert report.finalized_segments == 0


def test_module_limits_are_consistent() -> None:
    assert providers.MAX_BUFFERED_DURATION_MS <= providers.MAX_STREAM_DURATION_MS
    assert providers.MAX_BUFFERED_BYTE_COUNT > providers.MAX_CHUNK_BYTE_COUNT
    assert providers.MAX_BUFFERED_CHUNKS > 1
    assert providers.MAX_PARTIAL_HYPOTHESES > 1
    assert providers.MAX_SEGMENTS > 1
    assert providers.MAX_TOKEN_TIMES > 1
    assert providers.MAX_PROVIDER_LANGUAGES > 1
