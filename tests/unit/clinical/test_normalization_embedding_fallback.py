"""Tests for the opt-in multilingual embedding fallback (OM-751b).

The fallback is disabled and dependency-light by default: importing OpenMed and
running lexical grounding must not import the optional embedding library. When
enabled it uses only a local/offline embedder and re-ranks the zero-alias
candidate pool by multilingual similarity; when disabled or unavailable it
degrades gracefully to the lexical ordering without raising.
"""

from __future__ import annotations

import importlib
import sys

from openmed.clinical.normalization import (
    SYNTHETIC_CONCEPTS,
    EmbeddingFallbackResult,
    MultilingualEmbeddingBackend,
    embedding_fallback,
)

_EMBEDDING_DEPENDENCY = "sentence_transformers"


class _StubEmbedder:
    """Deterministic offline embedder mapping known surfaces to fixed vectors.

    Unknown surfaces map to a zero vector, so only the intended concept aligns
    with the queried mention. No network or model download occurs.
    """

    def __init__(self, vectors: dict[str, tuple[float, ...]], width: int = 3) -> None:
        self._vectors = vectors
        self._width = width
        self.calls: list[tuple[str, ...]] = []

    def encode(self, texts):
        self.calls.append(tuple(texts))
        return [self._vectors.get(text, (0.0,) * self._width) for text in texts]


def _concepts():
    return SYNTHETIC_CONCEPTS[:3]


def test_default_lexical_grounding_never_imports_embedding_dependency():
    # A default (disabled) backend must not import the optional library, even
    # when its fallback is exercised on a zero-alias mention.
    already_imported = _EMBEDDING_DEPENDENCY in sys.modules
    result = embedding_fallback("fiebre", _concepts(), lexical_candidates=())

    assert isinstance(result, EmbeddingFallbackResult)
    assert result.used_embedding is False
    assert result.status.available is False
    # Concepts are returned unchanged (lexical-only ordering preserved).
    assert result.concepts == _concepts()
    if not already_imported:
        assert _EMBEDDING_DEPENDENCY not in sys.modules


def test_importing_normalization_backend_does_not_import_dependency():
    # Fresh import of the backend module must not pull in the optional library.
    sys.modules.pop("sentence_transformers", None)
    module = importlib.import_module("openmed.clinical.normalization.backend")
    importlib.reload(module)

    assert "sentence_transformers" not in sys.modules


def test_disabled_backend_reports_status_and_keeps_lexical_order():
    backend = MultilingualEmbeddingBackend()  # disabled by default

    assert backend.enabled is False
    assert backend.available is False
    result = backend.rank("fiebre", _concepts())

    assert result.used_embedding is False
    assert result.concepts == _concepts()
    assert "disabled" in result.status.reason


def test_unavailable_dependency_degrades_to_lexical_without_raising():
    # Enabled but pointing at a dependency that is not installed: no exception,
    # a clear fallback result, and lexical order preserved.
    backend = MultilingualEmbeddingBackend(
        enabled=True,
        model_path="/nonexistent/offline/model",
        dependency="openmed_absent_embedding_backend_xyz",
    )

    assert backend.available is False
    result = backend.fallback("fiebre", _concepts(), lexical_candidates=())

    assert result.used_embedding is False
    assert result.concepts == _concepts()
    assert "not installed" in result.status.reason


def test_enabled_injected_embedder_reranks_zero_alias_pool():
    # Mention embeds close to the third concept; a stub avoids any model load.
    target = SYNTHETIC_CONCEPTS[2]
    vectors = {
        "fiebre": (0.0, 0.0, 1.0),
        target.normalized_terms[0]: (0.0, 0.0, 1.0),
    }
    backend = MultilingualEmbeddingBackend(
        enabled=True,
        embedder=_StubEmbedder(vectors),
    )

    assert backend.available is True
    result = backend.fallback("fiebre", _concepts(), lexical_candidates=())

    assert result.used_embedding is True
    assert result.status.available is True
    assert result.concepts[0].code == target.code
    # Every input concept survives the re-ranking (no drops).
    assert {concept.code for concept in result.concepts} == {
        concept.code for concept in _concepts()
    }


def test_fallback_defers_to_lexical_candidates_when_alias_present():
    # A target-language alias already matched: the embedder is never consulted.
    embedder = _StubEmbedder({})
    backend = MultilingualEmbeddingBackend(enabled=True, embedder=embedder)
    lexical = (SYNTHETIC_CONCEPTS[0],)

    result = backend.fallback("aster fever", _concepts(), lexical_candidates=lexical)

    assert result.used_embedding is False
    assert result.concepts == lexical
    assert embedder.calls == []
    assert "lexical alias present" in result.status.reason


def test_injected_embedder_never_triggers_dependency_import():
    sys.modules.pop("sentence_transformers", None)
    backend = MultilingualEmbeddingBackend(
        enabled=True,
        embedder=_StubEmbedder({"fiebre": (1.0, 0.0, 0.0)}),
    )

    backend.rank("fiebre", _concepts())

    assert "sentence_transformers" not in sys.modules


def test_remote_model_path_is_not_loaded_or_downloaded():
    # Offline-only, enforced in code: an enabled backend given a non-local model
    # path (e.g. a remote model id) must NOT load/download -- it degrades to
    # lexical and never imports the embedding dependency.
    already = "sentence_transformers" in sys.modules
    backend = MultilingualEmbeddingBackend(
        enabled=True,
        model_path="OpenMed/some-remote-model-id",  # not a local filesystem path
    )

    result = backend.fallback("fiebre", _concepts(), lexical_candidates=())

    assert result.used_embedding is False
    assert result.concepts == _concepts()
    if not already:
        assert "sentence_transformers" not in sys.modules


def test_ragged_embedder_output_degrades_to_lexical_without_dropping():
    # An embedder returning fewer rows than inputs must not silently drop
    # concepts via zip truncation; degrade to lexical instead.
    class _RaggedEmbedder:
        def encode(self, texts):
            return [(0.0, 0.0, 1.0)]  # a single row for many inputs

    backend = MultilingualEmbeddingBackend(enabled=True, embedder=_RaggedEmbedder())

    result = backend.fallback("fiebre", _concepts(), lexical_candidates=())

    assert result.used_embedding is False
    assert result.concepts == _concepts()  # no drops, lexical order preserved


def test_source_language_defaults_to_english_when_not_threaded():
    # No source language threaded: the grounded record exposes the English
    # default, so existing English behavior and serialized output are unchanged.
    result = embedding_fallback("fiebre", _concepts(), lexical_candidates=())

    assert result.source_language == "en"


def test_source_language_default_is_repr_identical_to_explicit_english():
    # Passing no source language must be byte-for-byte identical to passing the
    # English default, so the English path stays stable.
    default = embedding_fallback("fiebre", _concepts(), lexical_candidates=())
    english = embedding_fallback(
        "fiebre", _concepts(), lexical_candidates=(), source_language="en"
    )

    assert repr(default) == repr(english)


def test_source_language_propagates_into_reranked_result():
    # An enabled, available backend stamps the resolved language on the grounded
    # record it re-ranks, so a non-English mention's language is auditable.
    target = SYNTHETIC_CONCEPTS[2]
    vectors = {
        "fiebre": (0.0, 0.0, 1.0),
        target.normalized_terms[0]: (0.0, 0.0, 1.0),
    }
    backend = MultilingualEmbeddingBackend(
        enabled=True, embedder=_StubEmbedder(vectors)
    )

    result = backend.fallback(
        "fiebre", _concepts(), lexical_candidates=(), source_language="es"
    )

    assert result.used_embedding is True
    assert result.concepts[0].code == target.code
    assert result.source_language == "es"


def test_source_language_propagates_on_lexical_present_path():
    # When a lexical alias already matched the embedder is never consulted, but
    # the resolved source language is still exposed on the grounded record.
    embedder = _StubEmbedder({})
    backend = MultilingualEmbeddingBackend(enabled=True, embedder=embedder)
    lexical = (SYNTHETIC_CONCEPTS[0],)

    result = backend.fallback(
        "aster fever", _concepts(), lexical_candidates=lexical, source_language="fr"
    )

    assert result.used_embedding is False
    assert result.concepts == lexical
    assert embedder.calls == []
    assert result.source_language == "fr"


def test_source_language_propagates_when_backend_disabled():
    # Even on the graceful lexical-only degrade path, the language is threaded
    # through to the grounded record's provenance.
    result = embedding_fallback(
        "fiebre", _concepts(), lexical_candidates=(), source_language="de"
    )

    assert result.used_embedding is False
    assert result.concepts == _concepts()
    assert result.source_language == "de"


def test_source_language_normalization_matches_grounding_contract():
    # The fallback must resolve a source-language tag to the same code the
    # sparse candidate generator stamps, so the language flows end-to-end with a
    # single, consistent contract across the generation and fallback surfaces.
    from openmed.clinical.grounding import normalize_language

    for tag in (None, "", "en", "ES", "fr-FR", "pt_BR", "zh-Hans", "  De  "):
        result = embedding_fallback(
            "fiebre", _concepts(), lexical_candidates=(), source_language=tag
        )
        assert result.source_language == normalize_language(tag)
