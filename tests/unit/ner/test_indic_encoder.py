from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from openmed.core import model_registry
from openmed.core.pii_i18n import DEFAULT_PII_MODELS
from openmed.ner.families import indic


class _FakeTokenizer:
    def __call__(self, text: str, **kwargs):
        assert kwargs["return_offsets_mapping"] is True
        assert kwargs["return_tensors"] == "pt"
        return {
            "input_ids": [[101, 42, 102]],
            "attention_mask": [[1, 1, 1]],
            "offset_mapping": [[(0, 0), (0, len(text)), (0, 0)]],
        }


class _FakeModel:
    def __init__(self):
        self.eval_called = False

    def eval(self):
        self.eval_called = True

    def __call__(self, **kwargs):
        assert kwargs["return_dict"] is True
        return SimpleNamespace(
            last_hidden_state=[
                [[0.0, 0.0], [0.5, 1.0], [0.0, 0.0]],
            ]
        )


class _Loader:
    def __init__(self, value):
        self.value = value
        self.calls = []

    def from_pretrained(self, source, **kwargs):
        self.calls.append((source, kwargs))
        return self.value


@pytest.fixture(autouse=True)
def _clear_registry_config():
    model_registry.clear_indic_encoder_config()
    yield
    model_registry.clear_indic_encoder_config()


def test_loader_skips_before_importing_when_no_weights_are_configured(monkeypatch):
    def unexpected_import(name):
        raise AssertionError(f"unexpected optional import: {name}")

    monkeypatch.setattr(indic, "_optional_module", unexpected_import)
    result = indic.load_indic_encoder(None)

    assert result.available is False
    assert result.handle is None
    assert result.skip_reason == "no Indic encoder weights are configured"


def test_loader_reports_missing_optional_dependency(monkeypatch):
    monkeypatch.setattr(indic, "_optional_module", lambda name: None)

    result = indic.load_indic_encoder("google/muril-base-cased")

    assert result.available is False
    assert result.metadata is not None
    assert result.metadata.license_id == "Apache-2.0"
    assert "transformers" in result.skip_reason


def test_loader_treats_broken_optional_runtime_as_unavailable(monkeypatch):
    def broken_import(name):
        raise RuntimeError(f"broken optional runtime: {name}")

    monkeypatch.setattr(indic.importlib, "import_module", broken_import)

    assert indic.is_indic_encoder_available() is False
    result = indic.load_indic_encoder("google/muril-base-cased")
    assert result.available is False
    assert "transformers" in result.skip_reason


def test_loader_rejects_invalid_source_type():
    with pytest.raises(TypeError, match="source"):
        indic.load_indic_encoder(123)


def test_loader_accepts_path_source_with_explicit_family(monkeypatch):
    tokenizer_loader = _Loader(_FakeTokenizer())
    model_loader = _Loader(_FakeModel())
    transformers = SimpleNamespace(
        AutoTokenizer=tokenizer_loader,
        AutoModel=model_loader,
    )
    monkeypatch.setattr(
        indic,
        "_optional_module",
        lambda name: transformers if name == "transformers" else SimpleNamespace(),
    )

    source = Path("/models/muril")
    result = indic.load_indic_encoder(source, family="muril")

    assert result.available is True
    assert result.handle.source == str(source)


def test_loader_rejects_non_integer_offsets(monkeypatch):
    class InvalidOffsetTokenizer(_FakeTokenizer):
        def __call__(self, text: str, **kwargs):
            encoded = super().__call__(text, **kwargs)
            encoded["offset_mapping"] = [[(0, 0), (0.0, len(text)), (0, 0)]]
            return encoded

    transformers = SimpleNamespace(
        AutoTokenizer=_Loader(InvalidOffsetTokenizer()),
        AutoModel=_Loader(_FakeModel()),
    )
    monkeypatch.setattr(
        indic,
        "_optional_module",
        lambda name: transformers if name == "transformers" else SimpleNamespace(),
    )
    result = indic.load_indic_encoder("google/muril-base-cased")

    with pytest.raises(ValueError, match="integer boundaries"):
        result.handle.encode("Asha")


def test_loader_exposes_aligned_hidden_state_contract_without_raw_logs(
    monkeypatch,
    caplog,
):
    tokenizer_loader = _Loader(_FakeTokenizer())
    model = _FakeModel()
    model_loader = _Loader(model)
    transformers = SimpleNamespace(
        AutoTokenizer=tokenizer_loader,
        AutoModel=model_loader,
    )
    torch = SimpleNamespace()
    monkeypatch.setattr(
        indic,
        "_optional_module",
        lambda name: transformers if name == "transformers" else torch,
    )
    raw_text = "Patient Asha Verma ka record dekhiye."

    with caplog.at_level(logging.DEBUG, logger=indic.__name__):
        result = indic.load_indic_encoder(
            "google/muril-base-cased",
            local_files_only=True,
        )
        encoded = result.handle.encode(raw_text)

    assert result.available is True
    assert model.eval_called is True
    assert encoded.input_ids == [[101, 42, 102]]
    assert encoded.attention_mask == [[1, 1, 1]]
    assert encoded.offset_mapping == ((0, 0), (0, len(raw_text)), (0, 0))
    assert encoded.token_count == 3
    assert encoded.hidden_size == 2
    assert encoded.text_sha256 == hashlib.sha256(raw_text.encode()).hexdigest()
    assert raw_text not in caplog.text
    assert encoded.text_sha256 in caplog.text
    assert tokenizer_loader.calls[0][1]["local_files_only"] is True
    assert tokenizer_loader.calls[0][1]["trust_remote_code"] is False


def test_registry_adds_configured_encoder_without_replacing_defaults(monkeypatch):
    baseline_ids = {
        info.model_id
        for info in model_registry.get_pii_models_by_language("hi").values()
    }
    monkeypatch.setattr(
        model_registry,
        "_is_indic_encoder_runtime_available",
        lambda: True,
    )

    config = model_registry.configure_indic_encoder(
        "/models/muril",
        family="muril",
    )
    models = model_registry.get_pii_models_by_language("hi")

    assert config.languages == ("hi", "te")
    assert baseline_ids <= {info.model_id for info in models.values()}
    assert DEFAULT_PII_MODELS["hi"] in {info.model_id for info in models.values()}
    adapter = models["pii_hi_muril_encoder"]
    assert adapter.model_id == "/models/muril"
    assert adapter.license == "Apache-2.0"
    assert adapter.provenance == {
        "source": "/models/muril",
        "weights": "user-supplied",
        "bundled": False,
        "revision": None,
    }


def test_registry_can_opt_an_additional_family_language_in(monkeypatch):
    monkeypatch.setattr(
        model_registry,
        "_is_indic_encoder_runtime_available",
        lambda: True,
    )
    model_registry.configure_indic_encoder(
        "ai4bharat/indic-bert",
        family="indicbert",
        languages=("bn",),
    )

    models = model_registry.get_pii_models_by_language("bn")

    assert list(models) == ["pii_bn_indicbert_encoder"]
    assert models["pii_bn_indicbert_encoder"].license == "MIT"


def test_registry_hides_adapter_when_dependencies_are_unavailable(monkeypatch):
    model_registry.configure_indic_encoder(
        "google/muril-base-cased",
        family="muril",
    )
    monkeypatch.setattr(
        model_registry,
        "_is_indic_encoder_runtime_available",
        lambda: False,
    )

    models = model_registry.get_pii_models_by_language("hi")

    assert "pii_hi_muril_encoder" not in models


def test_registry_rejects_language_not_supported_by_family():
    with pytest.raises(ValueError, match="does not support"):
        model_registry.configure_indic_encoder(
            "ai4bharat/indic-bert",
            family="indicbert",
            languages=("ur",),
        )


def test_registry_only_infers_family_for_official_sources():
    assert (
        model_registry.get_indic_encoder_spec(source="google/muril-base-cased").family
        == "muril"
    )
    with pytest.raises(ValueError, match="unrecognized source"):
        model_registry.get_indic_encoder_spec(source="someone/muril-lookalike")


def test_registry_metadata_and_token_representation_are_safe():
    config = model_registry.configure_indic_encoder(
        "google/muril-base-cased",
        token="private-token",
    )

    assert "private-token" not in repr(config)
    with pytest.raises(TypeError):
        model_registry.INDIC_ENCODER_SPECS["other"] = (  # type: ignore[index]
            model_registry.INDIC_ENCODER_SPECS["muril"]
        )
