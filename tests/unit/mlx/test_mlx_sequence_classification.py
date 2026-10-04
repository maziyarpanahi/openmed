"""Native, download-free BERT sequence-classifier weight/attention parity."""

from __future__ import annotations

import json

import pytest

mx = pytest.importorskip("mlx.core")
torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from openmed.mlx.convert import remap_key
from openmed.mlx.models.bert_sc import BertForSequenceClassification, load_model


def _config():
    return transformers.BertConfig(
        vocab_size=128,
        hidden_size=24,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=48,
        num_labels=3,
        hidden_dropout_prob=0,
        attention_probs_dropout_prob=0,
    )


def test_real_mlx_logits_match_torch_on_padded_segment_pairs(tmp_path):
    torch.manual_seed(3236)
    config = _config()
    reference = transformers.BertForSequenceClassification(config).eval()
    weights = {}
    for name, value in reference.state_dict().items():
        target = (
            name.replace("bert.pooler.dense.", "pooler.")
            if name.startswith("bert.pooler.dense.")
            else remap_key(name, "bert")
        )
        weights[target] = mx.array(value.detach().numpy())
    payload = config.to_dict() | {
        "num_labels": 3,
        "_mlx_task": "sequence-classification",
    }
    (tmp_path / "config.json").write_text(json.dumps(payload))
    mx.save_safetensors(str(tmp_path / "weights.safetensors"), weights)
    measured = load_model(tmp_path)
    ids = [[1, 8, 2, 9, 2, 0], [1, 7, 2, 2, 0, 0]]
    mask = [[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 0, 0]]
    types = [[0, 0, 0, 1, 1, 0], [0, 0, 0, 1, 0, 0]]
    with torch.inference_mode():
        expected = reference(
            input_ids=torch.tensor(ids),
            attention_mask=torch.tensor(mask),
            token_type_ids=torch.tensor(types),
        ).logits.numpy()
    actual = measured(mx.array(ids), mx.array(types), mx.array(mask))
    mx.eval(actual)
    assert actual.shape == (2, 3)
    assert mx.max(mx.abs(actual - mx.array(expected))).item() < 1e-5
    assert not measured.training


@pytest.mark.parametrize(
    "change",
    [
        {"model_type": "roberta"},
        {"hidden_act": "relu"},
        {"num_labels": 4},
        {"position_embedding_type": "relative_key"},
    ],
)
def test_unsupported_architecture_never_silently_loads(change):
    with pytest.raises(ValueError):
        BertForSequenceClassification(_config().to_dict() | {"num_labels": 3} | change)


def test_token_classification_artifact_is_not_accepted_as_sequence_classification(
    tmp_path,
):
    (tmp_path / "config.json").write_text(json.dumps(_config().to_dict()))
    with pytest.raises(ValueError, match="not a sequence classifier"):
        load_model(tmp_path)
