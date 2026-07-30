"""Tests for the process-local language-pack contract and registry."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

import openmed.core.language_pack as language_pack_module
from openmed.core import LanguagePack, LanguagePackRegistry, register_language_pack


def _pack(code: str = "xx", **overrides: object) -> LanguagePack:
    values: dict[str, object] = {
        "code": code,
        "scripts": ["Synthetic"],
        "default_model": "OpenMed/synthetic-pii",
        "segmenter_id": "unicode-sentence",
        "recognizers": ["regex", "model"],
        "surrogate_locale": "en_US",
        "national_id_providers": {"ssn": "en_US"},
        "policy_overrides": {"profile": "strict_no_leak"},
        "recall_floor_overrides": {"person": 0.99},
    }
    values.update(overrides)
    return LanguagePack(**values)  # type: ignore[arg-type]


def test_one_call_public_registration_uses_process_local_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = LanguagePackRegistry()
    monkeypatch.setattr(language_pack_module, "LANGUAGE_PACK_REGISTRY", registry)
    pack = _pack()

    assert register_language_pack(pack) is pack
    assert registry.get("xx") is pack
    assert tuple(registry.iter_codes()) == ("xx",)


def test_registry_iteration_is_a_sorted_snapshot() -> None:
    registry = LanguagePackRegistry()
    registry.register(_pack("zz"))
    codes = registry.iter_codes()
    registry.register(_pack("aa"))

    assert tuple(codes) == ("zz",)
    assert tuple(registry.iter_codes()) == ("aa", "zz")


def test_duplicate_registration_requires_explicit_replacement() -> None:
    registry = LanguagePackRegistry()
    original = registry.register(_pack())
    replacement = _pack(default_model="OpenMed/replacement-pii")

    with pytest.raises(ValueError, match="already registered"):
        registry.register(replacement)

    assert registry.get("xx") is original
    assert registry.register(replacement, replace=True) is replacement
    assert registry.get("xx") is replacement


def test_registry_rejects_non_pack_values() -> None:
    registry = LanguagePackRegistry()

    with pytest.raises(TypeError, match="must be a LanguagePack"):
        registry.register(object())  # type: ignore[arg-type]


def test_registry_get_raises_for_unknown_code() -> None:
    registry = LanguagePackRegistry()

    with pytest.raises(KeyError, match="xx"):
        registry.get("xx")


def test_pack_copies_collection_inputs_into_immutable_values() -> None:
    scripts = ["Latin"]
    recognizers = ["regex"]
    providers = {"ssn": "en_US"}
    policy = {"profile": "balanced"}
    recall_floors = {"person": 0.95}
    pack = _pack(
        scripts=scripts,
        recognizers=recognizers,
        national_id_providers=providers,
        policy_overrides=policy,
        recall_floor_overrides=recall_floors,
    )

    scripts.append("Cyrillic")
    recognizers.append("model")
    providers["npi"] = "en_US"
    policy["profile"] = "strict_no_leak"
    recall_floors["person"] = 0.5

    assert pack.scripts == ("Latin",)
    assert pack.recognizers == ("regex",)
    assert dict(pack.national_id_providers) == {"ssn": "en_US"}
    assert dict(pack.policy_overrides) == {"profile": "balanced"}
    assert dict(pack.recall_floor_overrides) == {"person": 0.95}

    with pytest.raises(TypeError):
        pack.policy_overrides["profile"] = "permissive"  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        pack.code = "zz"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("overrides", "error", "message"),
    (
        ({"code": "EN"}, ValueError, "lowercase ISO 639-1"),
        ({"scripts": []}, ValueError, "scripts must contain"),
        ({"scripts": "Latin"}, TypeError, "iterable of strings"),
        ({"scripts": ["Latin", "Latin"]}, ValueError, "duplicate"),
        ({"default_model": ""}, ValueError, "default_model"),
        ({"segmenter_id": ""}, ValueError, "segmenter_id"),
        ({"recognizers": []}, ValueError, "recognizers must contain"),
        ({"surrogate_locale": ""}, ValueError, "surrogate_locale"),
        (
            {"national_id_providers": {"ssn": ""}},
            ValueError,
            "national_id_providers value",
        ),
        (
            {"recall_floor_overrides": {"person": 1.1}},
            ValueError,
            "finite probabilities",
        ),
        (
            {"recall_floor_overrides": {"person": True}},
            TypeError,
            "numeric probabilities",
        ),
    ),
)
def test_incomplete_or_malformed_packs_fail_loudly(
    overrides: dict[str, object],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        _pack(**overrides)
