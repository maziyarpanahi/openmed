"""Parity tests for the ANE BERT token-classification rewrite."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")


def test_ane_bert_logits_match_huggingface_tiny():
    from transformers import BertConfig, BertForTokenClassification

    from openmed.coreml.ane_bert import ANEBertForTokenClassification

    config = BertConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=32,
        type_vocab_size=2,
        num_labels=3,
        hidden_act="gelu",
        layer_norm_eps=1e-5,
    )
    hf = BertForTokenClassification(config)
    hf.eval()
    ane = ANEBertForTokenClassification.from_hf(hf)

    torch.manual_seed(0)
    input_ids = torch.randint(0, 64, (2, 16))
    attention_mask = torch.ones(2, 16, dtype=torch.long)
    attention_mask[1, 12:] = 0

    with torch.no_grad():
        reference = hf(input_ids=input_ids, attention_mask=attention_mask).logits
        candidate = ane(input_ids, attention_mask)

    assert candidate.shape == reference.shape
    assert torch.allclose(candidate, reference, atol=2e-4, rtol=2e-4)
