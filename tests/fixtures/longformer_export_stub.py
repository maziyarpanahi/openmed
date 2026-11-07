"""Offline tiny Longformer token classifier for export qualification tests.

The checkpoint is built in-test from seeded generators: no downloads and no
committed weights. It reuses the character-level tokenizer, label map,
synthetic fixtures, sentinels, and network guard of the ModernBERT stub. The
classifier is wired so uppercase letters decode as ``I-NAME`` and digits as
``I-ID_NUM`` while the local and global attention projections keep random
weights, so a broken window, global-attention, or padding mask shows up as
logit drift. Layers use different attention windows (8, 16, 8 tokens), and
``max_position_embeddings=66`` with ``pad_token_id=0`` gives a 65-token
position limit, so the declared buckets are 16, 32, 48, and 64.
"""

from __future__ import annotations

import string
from pathlib import Path

from tests.fixtures.modernbert_export_stub import (
    LABELS,
    SENTINELS,
    NetworkGuard,
    _vocabulary,
    build_tokenizer,
    files_containing_sentinels,
    rule_fixture_rows,
    text_contains_sentinels,
    write_rule_fixtures,
)

SEED = 3637
ATTENTION_WINDOWS = (8, 16, 8)
MAX_POSITION_EMBEDDINGS = 66
MAX_BUCKET = 64


def build_tiny_longformer(
    directory: Path,
    *,
    seed: int = SEED,
    attention_windows: tuple[int, ...] = ATTENTION_WINDOWS,
) -> Path:
    """Write a deterministic tiny Longformer token classifier to *directory*.

    Returns:
        The checkpoint directory, loadable with ``from_pretrained``.
    """

    import torch
    from transformers import LongformerConfig, LongformerForTokenClassification

    directory.mkdir(parents=True, exist_ok=True)
    build_tokenizer(directory)
    vocab = _vocabulary()
    config = LongformerConfig(
        vocab_size=len(vocab),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=len(attention_windows),
        num_attention_heads=4,
        max_position_embeddings=MAX_POSITION_EMBEDDINGS,
        attention_window=list(attention_windows),
        type_vocab_size=1,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        num_labels=len(LABELS),
        id2label=dict(LABELS),
        label2id={label: index for index, label in LABELS.items()},
        pad_token_id=vocab["[PAD]"],
        bos_token_id=vocab["[CLS]"],
        eos_token_id=vocab["[SEP]"],
        sep_token_id=vocab["[SEP]"],
    )
    torch.manual_seed(seed)
    model = LongformerForTokenClassification(config).eval()
    generator = torch.Generator().manual_seed(seed)
    hidden = config.hidden_size

    def noise(shape: tuple[int, ...], scale: float) -> torch.Tensor:
        return torch.randn(shape, generator=generator) * scale

    with torch.no_grad():
        embedding = noise((len(vocab), hidden), 0.05)
        for symbol, index in vocab.items():
            if symbol in string.ascii_uppercase:
                embedding[index, 0] += 3.0
            elif symbol in string.digits:
                embedding[index, 1] += 3.0
            else:
                embedding[index, 2] += 3.0
        embeddings = model.longformer.embeddings
        embeddings.word_embeddings.weight.copy_(embedding)
        embeddings.position_embeddings.weight.copy_(
            noise(tuple(embeddings.position_embeddings.weight.shape), 0.05)
        )
        for layer in model.longformer.encoder.layer:
            attention = layer.attention.self
            for linear, scale in (
                (attention.query, 0.3),
                (attention.key, 0.3),
                (attention.value, 0.3),
                (attention.query_global, 0.3),
                (attention.key_global, 0.3),
                (attention.value_global, 0.3),
                (layer.attention.output.dense, 0.05),
                (layer.intermediate.dense, 0.3),
                (layer.output.dense, 0.03),
            ):
                linear.weight.copy_(noise(tuple(linear.weight.shape), scale))
        classifier = torch.zeros(len(LABELS), hidden)
        classifier[0, 2] = 1.0
        classifier[1, 0] = 1.0
        classifier[2, 1] = 1.0
        model.classifier.weight.copy_(classifier)
        model.classifier.bias.zero_()
    model.save_pretrained(str(directory))
    return directory


__all__ = [
    "ATTENTION_WINDOWS",
    "LABELS",
    "MAX_BUCKET",
    "MAX_POSITION_EMBEDDINGS",
    "SENTINELS",
    "NetworkGuard",
    "build_tiny_longformer",
    "files_containing_sentinels",
    "rule_fixture_rows",
    "text_contains_sentinels",
    "write_rule_fixtures",
]
