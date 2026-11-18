"""Offline synthetic controls for local BiomedCLIP embedding (no weights)."""

import pickle
import socket

import numpy as np
import pytest

from openmed.multimodal.vision.retrieval import biomedclip as mod
from openmed.multimodal.vision.retrieval.biomedclip import (
    BiomedCLIPAssets,
    BiomedCLIPEmbedder,
    BiomedCLIPError,
    ProtectedEmbeddings,
    compute_assets_digest,
)

DIM = 8


class FakeBackend:
    dimension = DIM

    def __init__(self, hook=None):
        self.hook = hook

    def _vecs(self, n):
        rng = np.random.default_rng(0)
        return rng.normal(size=(n, DIM)) * 5.0 + 1.0

    def encode_images(self, images):
        if self.hook:
            self.hook()
        return self._vecs(len(images))

    def encode_texts(self, texts):
        if self.hook:
            self.hook()
        return self._vecs(len(texts))


@pytest.fixture
def assets(tmp_path):
    d = tmp_path / "ckpt"
    d.mkdir()
    (d / "open_clip_config.json").write_text("{}")
    (d / "weights.bin").write_bytes(b"synthetic-weights")
    return BiomedCLIPAssets(d, compute_assets_digest(d), b"MIT synthetic license")


def make(assets, hook=None):
    return BiomedCLIPEmbedder(assets, lambda _a: FakeBackend(hook))


def test_embeddings_have_declared_dimension_and_unit_length(assets):
    emb = make(assets)
    for out, n in (
        (emb.embed_images([object(), object(), object()]), 3),
        (emb.embed_texts(["a", "b"]), 2),
    ):
        arr = out.unprotect()
        assert arr.shape == (n, DIM) == (out.count, out.dimension)
        np.testing.assert_allclose(np.linalg.norm(arr, axis=1), 1.0, atol=1e-5)
        assert not arr.flags.writeable


def test_digest_mismatch_is_refused(tmp_path):
    d = tmp_path / "ckpt"
    d.mkdir()
    (d / "weights.bin").write_bytes(b"x")
    with pytest.raises(BiomedCLIPError):
        BiomedCLIPAssets(d, "0" * 64, b"license")


def test_tampered_checkpoint_after_verification_is_refused(assets):
    emb = make(assets)
    (assets.directory / "weights.bin").write_bytes(b"tampered")
    with pytest.raises(BiomedCLIPError):
        emb.embed_texts(["a"])


def test_symlink_asset_rejected(tmp_path):
    d = tmp_path / "ckpt"
    d.mkdir()
    real = tmp_path / "real.bin"
    real.write_bytes(b"x")
    (d / "weights.bin").symlink_to(real)
    with pytest.raises(BiomedCLIPError):
        compute_assets_digest(d)


def test_network_blocked_during_embedding_and_restored_after(assets):
    def dial():
        socket.create_connection(("203.0.113.1", 80), timeout=0.1)

    emb = make(assets, hook=dial)
    original = socket.create_connection
    with pytest.raises(BiomedCLIPError, match="network access blocked"):
        emb.embed_texts(["a"])
    assert socket.create_connection is original


def test_protected_embeddings_leak_no_values(assets):
    out = make(assets).embed_texts(["secret phi text"])
    values = out.unprotect()[0]
    rendered = repr(out) + out.to_json() + make(assets).provenance_json()
    assert str(values[0])[:6] not in rendered
    assert "secret phi text" not in rendered and str(assets.directory) not in rendered
    with pytest.raises(TypeError):
        pickle.dumps(out)


def test_bad_backend_output_rejected(assets):
    class Bad(FakeBackend):
        def encode_texts(self, texts):
            return np.zeros((len(texts), DIM))

    with pytest.raises(BiomedCLIPError):
        BiomedCLIPEmbedder(assets, lambda _a: Bad()).embed_texts(["a"])
    with pytest.raises(BiomedCLIPError):
        ProtectedEmbeddings(np.array([[np.nan] * DIM]))


def test_empty_and_non_string_inputs_rejected(assets):
    emb = make(assets)
    with pytest.raises(BiomedCLIPError):
        emb.embed_images([])
    with pytest.raises(BiomedCLIPError):
        emb.embed_texts([1])  # type: ignore[list-item]


def test_missing_extra_gives_install_hint(assets, monkeypatch):
    import builtins

    real = builtins.__import__

    def fake(name, *a, **k):
        if name in ("open_clip", "torch"):
            raise ImportError(name)
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake)
    with pytest.raises(BiomedCLIPError, match=r"openmed\[biomedclip\]"):
        BiomedCLIPEmbedder(assets)
