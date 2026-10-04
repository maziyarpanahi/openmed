"""Local BERT sequence classification with a weight-compatible MLX pooler."""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from openmed.mlx.models.bert_tc import BertEmbeddings, BertEncoder


class BertForSequenceClassification(nn.Module):
    """Classify a complete premise/hypothesis pair using the pooled CLS state."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        if config.get("model_type") != "bert" or config.get("hidden_act") != "gelu":
            raise ValueError("MLX sequence classification requires BERT with GELU")
        if config.get("position_embedding_type", "absolute") != "absolute":
            raise ValueError("unsupported sequence-classifier positions")
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = nn.Linear(config["hidden_size"], config["hidden_size"])
        self.dropout = nn.Dropout(
            config.get("classifier_dropout")
            if config.get("classifier_dropout") is not None
            else config.get("hidden_dropout_prob", 0.1)
        )
        labels = config.get("num_labels", len(config.get("id2label", {})))
        if labels != 3:
            raise ValueError("clinical NLI requires three classifier outputs")
        self.classifier = nn.Linear(config["hidden_size"], labels)

    def __call__(self, input_ids, token_type_ids=None, attention_mask=None):
        """Return batch-by-three logits without caching or logging input text."""

        hidden = self.embeddings(input_ids, token_type_ids)
        mask = (
            (1.0 - attention_mask[:, None, None, :]) * -1e9
            if attention_mask is not None
            else None
        )
        hidden = self.encoder(hidden, mask)
        pooled = mx.tanh(self.pooler(hidden[:, 0]))
        return self.classifier(self.dropout(pooled))


def load_model(model_path: str | Path) -> BertForSequenceClassification:
    """Load only a local, explicitly sequence-classification MLX artifact.

    Args:
        model_path: Local MLX subdirectory with config and safetensors weights.

    Returns:
        Strictly loaded, evaluation-mode BERT sequence classifier.
    """

    directory = Path(model_path)
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    if config.get("_mlx_task") != "sequence-classification":
        raise ValueError("MLX artifact is not a sequence classifier")
    model = BertForSequenceClassification(config)
    model.load_weights(str(directory / "weights.safetensors"), strict=True)
    model.eval()
    mx.eval(model.parameters())
    return model
