"""Security regression tests for the privacy-filter loader.

Covers CVE-2026-47117 — an unauthenticated attacker who controls the
``model_name`` request parameter could supply ``attacker/foo-privacy-filter-bar``
and have OpenMed load it with ``trust_remote_code=True``, executing arbitrary
Python from ``auto_map`` entries in the repo's ``config.json``.

These tests exercise the three defense layers added in 1.5.2:

1. The identifier matcher (``_looks_like_privacy_filter_identifier``) only
   routes requests through the privacy-filter dispatcher for first-party
   ``openai/privacy-filter`` and ``OpenMed/privacy-filter-*`` repos.
2. The ``PrivacyFilterTorchPipeline`` constructor refuses to pass
   ``trust_remote_code=True`` for any model outside the allowlist.
3. ``create_privacy_filter_pipeline`` only opts in to ``trust_remote_code``
   for resolved names that pass the allowlist check.
"""

from __future__ import annotations

import json
import sys
import types
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Layer 1: identifier matching
# ---------------------------------------------------------------------------


class TestPrivacyFilterIdentifierMatching:
    """``_looks_like_privacy_filter_identifier`` must not be tricked by
    arbitrary HuggingFace repository names that merely contain the substring
    ``privacy-filter``."""

    @pytest.mark.parametrize(
        "attacker_name",
        [
            "attacker/foo-privacy-filter-bar",
            "attacker/privacy-filter-rce",
            "evil-org/my-privacy-filter",
            "some/privacy-filter-evil",
            "username/privacy-filter",  # third-party org, not openai/OpenMed
            "OpenMed/not-privacy-filter-but-similar",  # missing trailing-prefix
        ],
    )
    def test_attacker_controlled_names_are_not_recognized(self, attacker_name):
        from openmed.core.pii import _looks_like_privacy_filter_identifier
        assert _looks_like_privacy_filter_identifier(attacker_name) is False

    @pytest.mark.parametrize(
        "trusted_name",
        [
            "openai/privacy-filter",
            "OpenAI/Privacy-Filter",  # case-insensitive normalization
            "OpenMed/privacy-filter-multilingual",
            "OpenMed/privacy-filter-nemotron",
            "OpenMed/privacy-filter-mlx",
            "OpenMed/privacy-filter-mlx-8bit",
            "OpenMed/privacy-filter-multilingual-mlx",
            "OpenMed/privacy-filter-multilingual-mlx-8bit",
            "OpenMed/privacy-filter-nemotron-mlx",
            "OpenMed/privacy-filter-nemotron-mlx-8bit",
        ],
    )
    def test_first_party_names_are_still_recognized(self, trusted_name):
        from openmed.core.pii import _looks_like_privacy_filter_identifier
        assert _looks_like_privacy_filter_identifier(trusted_name) is True

    @pytest.mark.parametrize(
        "alias",
        ["privacy-filter", "privacy_filter", "openai-privacy-filter"],
    )
    def test_family_aliases_are_recognized(self, alias):
        from openmed.core.pii import _looks_like_privacy_filter_identifier
        assert _looks_like_privacy_filter_identifier(alias) is True

    @pytest.mark.parametrize("falsy", ["", None, "   "])
    def test_blank_inputs_are_not_recognized(self, falsy):
        from openmed.core.pii import _looks_like_privacy_filter_identifier
        assert _looks_like_privacy_filter_identifier(falsy) is False


# ---------------------------------------------------------------------------
# Layer 2: PrivacyFilterTorchPipeline gate
# ---------------------------------------------------------------------------


class TestPrivacyFilterTorchPipelineGate:
    """Direct instantiation must refuse untrusted models even when callers
    explicitly opt in to ``trust_remote_code=True``."""

    def test_attacker_name_with_trust_remote_code_raises(self):
        from openmed.torch.privacy_filter import PrivacyFilterTorchPipeline
        with pytest.raises(ValueError, match="trusted-remote-code allowlist"):
            PrivacyFilterTorchPipeline(
                "attacker/foo-privacy-filter-bar",
                trust_remote_code=True,
            )

    def test_default_trust_remote_code_is_false(self):
        """If a caller does not opt in, the constructor must not load with
        ``trust_remote_code=True`` even for a trusted name."""
        captured = {}
        with _patched_transformers(captured):
            from openmed.torch.privacy_filter import PrivacyFilterTorchPipeline
            PrivacyFilterTorchPipeline("openai/privacy-filter")
        assert captured["tokenizer_kwargs"]["trust_remote_code"] is False
        assert captured["model_kwargs"]["trust_remote_code"] is False

    def test_trusted_model_with_explicit_opt_in_loads(self):
        captured = {}
        with _patched_transformers(captured):
            from openmed.torch.privacy_filter import PrivacyFilterTorchPipeline
            PrivacyFilterTorchPipeline(
                "openai/privacy-filter",
                trust_remote_code=True,
            )
        assert captured["tokenizer_kwargs"]["trust_remote_code"] is True
        assert captured["model_kwargs"]["trust_remote_code"] is True

    def test_case_variant_of_trusted_model_loads(self):
        """Issue #205 — lowercase variant of an allowlisted model must
        also pass the trust_remote_code gate."""
        captured = {}
        with _patched_transformers(captured):
            from openmed.torch.privacy_filter import PrivacyFilterTorchPipeline
            PrivacyFilterTorchPipeline(
                "openmed/privacy-filter-multilingual",
                trust_remote_code=True,
            )
        assert captured["tokenizer_kwargs"]["trust_remote_code"] is True


class TestIsTrustedForRemoteCode:
    """The allowlist function backs the gate."""

    @pytest.mark.parametrize(
        "trusted",
        [
            "openai/privacy-filter",
            "OpenMed/privacy-filter-multilingual",
            "OpenMed/privacy-filter-nemotron",
            # Case-insensitive matches (issue #205)
            "openmed/privacy-filter-multilingual",
            "OPENMED/PRIVACY-FILTER-NEMOTRON",
            "OpenAI/Privacy-Filter",
        ],
    )
    def test_hardcoded_first_party_models_are_trusted(self, trusted):
        from openmed.torch.privacy_filter import is_trusted_for_remote_code
        assert is_trusted_for_remote_code(trusted) is True

    @pytest.mark.parametrize(
        "attacker",
        [
            "attacker/foo-privacy-filter-bar",
            "some-org/privacy-filter",
            "OpenMed/privacy-filter-mlx",  # MLX-only repo, not in HF code path
            "",
            "openai/privacy-filter-evil",  # not exact match, not in allowlist
        ],
    )
    def test_other_models_are_not_trusted(self, attacker, monkeypatch):
        monkeypatch.delenv("OPENMED_TRUSTED_REMOTE_CODE_MODELS", raising=False)
        from openmed.torch.privacy_filter import is_trusted_for_remote_code
        assert is_trusted_for_remote_code(attacker) is False

    def test_env_var_extends_allowlist(self, monkeypatch):
        from openmed.torch.privacy_filter import is_trusted_for_remote_code
        monkeypatch.setenv(
            "OPENMED_TRUSTED_REMOTE_CODE_MODELS",
            "my-org/my-fork,other-org/another-fork ,  ,",
        )
        assert is_trusted_for_remote_code("my-org/my-fork") is True
        assert is_trusted_for_remote_code("other-org/another-fork") is True
        # Unrelated names are still rejected.
        assert is_trusted_for_remote_code("attacker/foo-privacy-filter") is False

    def test_env_var_matches_case_insensitive(self, monkeypatch):
        """Issue #205 — env-var allowlist entries should match regardless
        of the case the caller supplies."""
        from openmed.torch.privacy_filter import is_trusted_for_remote_code
        monkeypatch.setenv(
            "OPENMED_TRUSTED_REMOTE_CODE_MODELS",
            "My-Org/My-Fork",
        )
        assert is_trusted_for_remote_code("my-org/my-fork") is True
        assert is_trusted_for_remote_code("MY-ORG/MY-FORK") is True

    def test_env_var_unset_does_not_trust_extras(self, monkeypatch):
        monkeypatch.delenv("OPENMED_TRUSTED_REMOTE_CODE_MODELS", raising=False)
        from openmed.torch.privacy_filter import is_trusted_for_remote_code
        assert is_trusted_for_remote_code("my-org/my-fork") is False

    def test_local_privacy_filter_artifact_is_trusted(self, tmp_path):
        """A local directory whose config.json declares the privacy-filter
        family should be loadable with custom code, since the file system
        is already under the operator's control."""
        artifact = tmp_path / "local-privacy-filter"
        artifact.mkdir()
        (artifact / "config.json").write_text(
            json.dumps({"model_type": "openai-privacy-filter"})
        )

        from openmed.torch.privacy_filter import is_trusted_for_remote_code
        # Clear the lru_cache so the tmp_path is actually probed.
        from openmed.core import pii as _pii
        _pii._is_privacy_filter_artifact_path.cache_clear()

        assert is_trusted_for_remote_code(str(artifact)) is True

    def test_local_unrelated_artifact_is_not_trusted(self, tmp_path):
        artifact = tmp_path / "not-a-privacy-filter"
        artifact.mkdir()
        (artifact / "config.json").write_text(
            json.dumps({"model_type": "bert"})
        )
        from openmed.torch.privacy_filter import is_trusted_for_remote_code
        from openmed.core import pii as _pii
        _pii._is_privacy_filter_artifact_path.cache_clear()

        assert is_trusted_for_remote_code(str(artifact)) is False


# ---------------------------------------------------------------------------
# Layer 3: create_privacy_filter_pipeline opt-in
# ---------------------------------------------------------------------------


class TestCreatePrivacyFilterPipelineRemoteCodeOptIn:
    """The dispatcher should pass ``trust_remote_code=True`` only for
    resolved names that pass the allowlist check."""

    def test_trusted_model_passes_trust_remote_code_true(self):
        from openmed.core.backends import create_privacy_filter_pipeline
        with patch("openmed.core.backends.MLXBackend.is_available", return_value=False), \
             patch("openmed.torch.privacy_filter.PrivacyFilterTorchPipeline") as MockPF:
            MockPF.return_value = lambda _text: []
            create_privacy_filter_pipeline("openai/privacy-filter")
            MockPF.assert_called_once_with(
                "openai/privacy-filter",
                trust_remote_code=True,
            )

    def test_untrusted_model_passes_trust_remote_code_false(self):
        """Defense in depth: even if Layer 1 ever routes an untrusted name
        here, the dispatcher must not opt in to trust_remote_code."""
        from openmed.core.backends import create_privacy_filter_pipeline
        with patch("openmed.core.backends.MLXBackend.is_available", return_value=False), \
             patch("openmed.torch.privacy_filter.PrivacyFilterTorchPipeline") as MockPF:
            MockPF.return_value = lambda _text: []
            create_privacy_filter_pipeline("attacker/foo-privacy-filter-bar")
            MockPF.assert_called_once_with(
                "attacker/foo-privacy-filter-bar",
                trust_remote_code=False,
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _patched_transformers(captured):
    """Patch the ``transformers``/``torch`` modules just deep enough for
    ``PrivacyFilterTorchPipeline.__init__`` to run without hitting the Hub.

    Captures the kwargs passed to ``AutoTokenizer.from_pretrained`` and
    ``AutoModelForTokenClassification.from_pretrained`` into *captured*.
    """
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)

    class _FakeTokenizer:
        @staticmethod
        def from_pretrained(model_name, **kwargs):
            captured["tokenizer_model"] = model_name
            captured["tokenizer_kwargs"] = kwargs
            return MagicMock()

    class _FakeModel:
        @staticmethod
        def from_pretrained(model_name, **kwargs):
            captured["model_model"] = model_name
            captured["model_kwargs"] = kwargs
            m = MagicMock()
            m.to.return_value = m
            return m

    def _fake_pipeline(*args, **kwargs):
        return lambda text: []

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = _FakeTokenizer
    fake_transformers.AutoModelForTokenClassification = _FakeModel
    fake_transformers.pipeline = _fake_pipeline

    return patch.dict(
        sys.modules,
        {"torch": fake_torch, "transformers": fake_transformers},
    )
