"""Offline tiny GLiNER span and GLiClass artifacts for export tests.

Everything is built in-test from seeded generators: no downloads and no
committed weights. Each builder writes an OpenMed MLX artifact directory
(``config.json``, ``weights.safetensors`` with MLX parameter names, and a
character-level ``tokenizer.json`` whose prompt markers are added tokens), so
the same files drive the MLX runtime, the PyTorch reference, and the exported
ONNX graph.

The GLiNER stub can be *wired*: word embeddings encode "starts uppercase",
"starts with a digit", and two label keys (``p`` for ``person``, ``i`` for
``identifier``); a hand-set bidirectional LSTM turns those into "previous
word" and "next word" features; and the span head scores a span highly only
when it starts a run of capitalized (or digit) words and ends that run. The
DeBERTa layers keep small random weights, so exported graphs still exercise
disentangled relative attention, padding, and the LSTM, and any broken
gather, mask, or reversal shows up as logit drift. Names in the synthetic
fixtures are spelled in capitals and identifiers as digit runs, so the
reference reaches full direct-identifier recall.

The GLiClass stub keeps random weights: its gates are parity gates.
"""

from __future__ import annotations

import json
import string
from pathlib import Path
from typing import Any

SEED = 3639
SPECIAL_TOKENS = ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]")
GLINER_MARKERS = ("<<ENT>>", "<<SEP>>")
GLICLASS_MARKERS = ("<<LABEL>>", "<<SEP>>", "<<EXAMPLE>>")
PUNCTUATION = ".,:;-/#()'"
GLINER_LABELS = ("person", "identifier")

# Synthetic sentinels: surface forms that must never reach logs, errors,
# manifests, reports, or any other artifact written during export.
_GLINER_ROWS = (
    ("patient {n0} was seen today, mrn {i0}.", ("ZORVAK QUILLAN",), ("44172290",)),
    ("{n0} called about the visit on unit {i0}.", ("MERIDETH ASHGROVE",), ("7731",)),
    (
        "dr {n0} reviewed the chart today and then {i0} for {n1}.",
        ("TOVIN BRAKKE", "ELSPETH NUROVA"),
        ("90021",),
    ),
    ("guardian {n0}, member id {i0}, consented.", ("QUESNA DORVILLE",), ("5528193",)),
    ("café visit with {n0} next week 🙂.", ("ZOË HALVARDSEN",), ()),
    ("record {i0} was merged later into {i1}.", (), ("88310", "88311")),
)
GLINER_SENTINELS = tuple(
    sorted({value for _, names, ids in _GLINER_ROWS for value in (*names, *ids)})
)
_GLICLASS_ROWS = (
    ("cardiology follow up for VELTRANE ORSK about chest pain", ("cardiology",)),
    ("refill request from QUORRA BELLAMINE for a skin cream", ("dermatology",)),
    ("billing question on invoice 66120458 from the family", ("billing",)),
    ("schedule a lab draw for PELLIN ASHGROVE next week", ("scheduling",)),
)
GLICLASS_LABELS = ("cardiology", "dermatology", "billing", "scheduling")
GLICLASS_SENTINELS = (
    "VELTRANE ORSK",
    "QUORRA BELLAMINE",
    "66120458",
    "PELLIN ASHGROVE",
)
SENTINELS = tuple(sorted({*GLINER_SENTINELS, *GLICLASS_SENTINELS}))


def gliner_fixture_rows() -> list[dict[str, Any]]:
    """Return synthetic span fixtures whose gold spans match the wired stub."""
    rows: list[dict[str, Any]] = []
    for index, (template, names, ids) in enumerate(_GLINER_ROWS, start=1):
        values = {f"n{i}": name for i, name in enumerate(names)}
        values.update({f"i{i}": value for i, value in enumerate(ids)})
        text = template.format(**values)
        spans = []
        cursor = 0
        ordered = sorted(
            [(text.index(name), name, "PERSON") for name in names]
            + [(text.index(value), value, "ID_NUM") for value in ids]
        )
        for _, surface, label in ordered:
            start = text.index(surface, cursor)
            spans.append(
                {
                    "start": start,
                    "end": start + len(surface),
                    "label": label,
                    "text": surface,
                }
            )
            cursor = start + len(surface)
        rows.append(
            {
                "id": f"gliner-stub-{index:02d}",
                "language": "en",
                "text": text,
                "gold_spans": spans,
                "metadata": {"synthetic": True},
            }
        )
    return rows


def gliclass_fixture_rows() -> list[dict[str, Any]]:
    """Return synthetic zero-shot classification fixtures."""
    return [
        {
            "id": f"gliclass-stub-{index:02d}",
            "language": "en",
            "text": text,
            "candidate_labels": list(GLICLASS_LABELS),
            "gold_labels": list(gold),
            "metadata": {"synthetic": True},
        }
        for index, (text, gold) in enumerate(_GLICLASS_ROWS, start=1)
    ]


class _Encoding:
    def __init__(self, ids: list[int], word_ids: list[int | None]) -> None:
        self.ids = ids
        self.word_ids = word_ids
        self.attention_mask = [1] * len(ids)


class CharTokenizer:
    """Pure-Python stand-in for ``tokenizers.Tokenizer`` (no dependencies).

    One token per non-space character (id ``100 + ord(char)``), prompt markers
    as single tokens with the stub vocabulary ids, ``[CLS]=2`` and ``[SEP]=3``
    around every encoding. ``drop`` characters produce no token, which lets a
    test simulate a tokenizer that loses a source word.
    """

    def __init__(self, markers: tuple[str, ...], *, drop: str = "") -> None:
        self.markers = {marker: 5 + index for index, marker in enumerate(markers)}
        self.drop = set(drop)

    def token_to_id(self, token: str) -> int | None:
        """Return the id of a prompt marker."""
        return self.markers.get(token)

    def to_str(self) -> str:
        """Serialize like ``tokenizers.Tokenizer.to_str``."""
        return json.dumps(
            {"model": {"type": "CharStub", "vocab": self.markers}}, sort_keys=True
        )

    def _pieces(self, text: str) -> list[int]:
        ids: list[int] = []
        index = 0
        while index < len(text):
            marker = next((m for m in self.markers if text.startswith(m, index)), None)
            if marker is not None:
                ids.append(self.markers[marker])
                index += len(marker)
                continue
            char = text[index]
            if not char.isspace() and char not in self.drop:
                ids.append(100 + ord(char))
            index += 1
        return ids

    def encode(self, sequence: Any, is_pretokenized: bool = False) -> _Encoding:
        """Encode a string, or a list of words when *is_pretokenized*."""
        ids: list[int] = [2]
        word_ids: list[int | None] = [None]
        words = sequence if is_pretokenized else [sequence]
        for word_index, word in enumerate(words):
            pieces = self._pieces(word)
            ids.extend(pieces)
            word_ids.extend([word_index if is_pretokenized else None] * len(pieces))
        ids.append(3)
        word_ids.append(None)
        return _Encoding(ids, word_ids)


def rule_span_logits(feed: Any) -> Any:
    """Oracle GLiNER logits from a feed built with :class:`CharTokenizer`.

    A label whose first character is ``p`` matches maximal runs of words that
    start uppercase; ``i`` matches maximal runs of words that start with a
    digit. Matching cells score ``+8``, others ``-8``, masked spans ``-1e9``.
    """

    import numpy as np

    ids = [int(value) for value in feed["input_ids"][0]]
    words_mask = [int(value) for value in feed["words_mask"][0]]
    num_words = int(feed["text_lengths"][0][0])
    spans = feed["span_idx"][0]
    valid = feed["span_mask"][0]
    markers = [index for index, token in enumerate(ids) if token == 5]
    kinds = [chr(ids[index + 1] - 100) for index in markers]
    first = {
        mask - 1: chr(ids[index] - 100)
        for index, mask in enumerate(words_mask)
        if mask > 0
    }
    classes = [
        "U" if first[w].isupper() else "D" if first[w].isdigit() else "O"
        for w in range(num_words)
    ]
    width = len(spans) // max(num_words, 1)
    logits = np.full((1, num_words, width, len(kinds)), -8.0, dtype=np.float32)
    for index, ((start, end), ok) in enumerate(zip(spans, valid)):
        row, step = divmod(index, width)
        if not ok:
            logits[0, row, step, :] = -1.0e9
            continue
        for column, kind in enumerate(kinds):
            want = {"p": "U", "i": "D"}.get(kind)
            run = (
                all(classes[w] == want for w in range(start, end + 1))
                and (start == 0 or classes[start - 1] != want)
                and (end == num_words - 1 or classes[end + 1] != want)
            )
            if run:
                logits[0, row, step, column] = 8.0
    return logits


def rule_class_logits(feed: Any) -> Any:
    """Deterministic GLiClass oracle logits from a :class:`CharTokenizer` feed.

    Each label scores by how often its first character occurs in the text,
    minus a small label-specific offset, so rankings depend on both.
    """

    import numpy as np

    ids = [int(value) for value in feed["input_ids"][0]]
    markers = [index for index, token in enumerate(ids) if token == 5]
    separator = ids.index(6)
    text = ids[separator + 1 :]
    logits = []
    for index in markers:
        key = ids[index + 1]
        logits.append(text.count(key) - 1.5 + (key % 7) / 10.0)
    return np.asarray([logits], dtype=np.float32)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    """Write *rows* as JSONL and return *path*."""
    lines = [json.dumps(row, sort_keys=True, ensure_ascii=False) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def vocabulary(markers: tuple[str, ...]) -> dict[str, int]:
    """Return the character vocabulary with the prompt markers."""
    symbols = [
        *SPECIAL_TOKENS,
        *markers,
        *string.ascii_lowercase,
        *string.ascii_uppercase,
        *string.digits,
        *PUNCTUATION,
    ]
    return {symbol: index for index, symbol in enumerate(symbols)}


def build_tokenizer(directory: Path, markers: tuple[str, ...]) -> Any:
    """Save and return a character-level ``tokenizers.Tokenizer``."""
    from tokenizers import AddedToken, Regex, Tokenizer, pre_tokenizers
    from tokenizers.models import WordLevel
    from tokenizers.processors import TemplateProcessing

    vocab = vocabulary(markers)
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.WhitespaceSplit(),
            pre_tokenizers.Split(Regex("."), behavior="isolated"),
        ]
    )
    backend.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=[("[CLS]", vocab["[CLS]"]), ("[SEP]", vocab["[SEP]"])],
    )
    backend.add_special_tokens(
        [AddedToken(token, special=True) for token in (*SPECIAL_TOKENS, *markers)]
    )
    directory.mkdir(parents=True, exist_ok=True)
    backend.save(str(directory / "tokenizer.json"))
    return backend


def _deberta_config(vocab_size: int) -> dict[str, Any]:
    return {
        "model_type": "deberta-v2",
        "vocab_size": vocab_size,
        "hidden_size": 32,
        "encoder_hidden_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "intermediate_size": 64,
        "hidden_act": "gelu",
        "max_position_embeddings": 128,
        "type_vocab_size": 0,
        "layer_norm_eps": 1e-7,
        "relative_attention": True,
        "max_relative_positions": -1,
        "position_buckets": 32,
        "pos_att_type": ["p2c", "c2p"],
        "norm_rel_ebd": "layer_norm",
        "share_att_key": True,
        "position_biased_input": False,
        "pad_token_id": 0,
        "dropout": 0.0,
        "hidden_dropout_prob": 0.0,
        "attention_probs_dropout_prob": 0.0,
    }


def gliner_config(**overrides: Any) -> dict[str, Any]:
    """Return the tiny GLiNER span configuration (MLX artifact layout)."""
    vocab = vocabulary(GLINER_MARKERS)
    config = _deberta_config(len(vocab))
    config.update(
        {
            "_mlx_family": "gliner-uni-encoder-span",
            "_mlx_task": "zero-shot-ner",
            "_mlx_prompt_spec": {
                "entity_token": "<<ENT>>",
                "separator_token": "<<SEP>>",
            },
            "class_token_index": vocab["<<ENT>>"],
            "max_width": 4,
            "max_len": 96,
            "num_rnn_layers": 1,
            "embed_ent_token": False,
            "span_mode": "markerV0",
        }
    )
    config.update(overrides)
    return config


def gliclass_config(**overrides: Any) -> dict[str, Any]:
    """Return the tiny GLiClass uni-encoder configuration."""
    vocab = vocabulary(GLICLASS_MARKERS)
    config = _deberta_config(len(vocab))
    config.update(
        {
            "_mlx_family": "gliclass-uni-encoder",
            "_mlx_task": "zero-shot-sequence-classification",
            "_mlx_prompt_spec": {
                "label_token": "<<LABEL>>",
                "separator_token": "<<SEP>>",
                "prompt_first": True,
            },
            "class_token_index": vocab["<<LABEL>>"],
            "text_token_index": vocab["<<SEP>>"],
            "example_token_index": vocab["<<EXAMPLE>>"],
            "architecture_type": "uni-encoder",
            "scorer_type": "mlp",
            "pooling_strategy": "first",
            "class_token_pooling": "average",
            "embed_class_token": True,
            "use_segment_embeddings": True,
            "normalize_features": False,
            "logit_scale_init_value": 1.0,
        }
    )
    config.update(overrides)
    return config


def _mlx_parameter_shapes(config: dict[str, Any]) -> dict[str, tuple[int, ...]]:
    """Return MLX parameter names and shapes via the PyTorch reference.

    The MLX runtime loads strictly, so a wrong name here fails the MLX parity
    test instead of being silently ignored.
    """

    import torch

    from openmed.mlx.convert import remap_key
    from openmed.mlx.models import normalize_model_config
    from openmed.ner.families.zero_shot_export import (
        resolve_zero_shot_profile,
        validate_zero_shot_config,
    )
    from openmed.torch.zero_shot_export import build_zero_shot_module

    normalized = normalize_model_config(dict(config))
    profile = resolve_zero_shot_profile(normalized)
    spec = validate_zero_shot_config(profile, normalized)
    module = build_zero_shot_module(torch, normalized, spec)
    shapes: dict[str, tuple[int, ...]] = {}
    for key, tensor in module.state_dict().items():
        shape = tuple(tensor.shape)
        if key.startswith("deberta."):
            shapes[remap_key(key, "deberta-v2")] = shape
        elif key.endswith("_l0") and "_lstm." in key:
            prefix, _, name = key.rpartition(".")
            if name == "weight_ih_l0":
                shapes[f"{prefix}.Wx"] = shape
            elif name == "weight_hh_l0":
                shapes[f"{prefix}.Wh"] = shape
            elif name == "bias_ih_l0":
                shapes[f"{prefix}.bias"] = shape
        else:
            shapes[key] = shape
    return shapes


def _random_weights(
    shapes: dict[str, tuple[int, ...]], *, seed: int, scale: float
) -> dict[str, Any]:
    import numpy as np

    rng = np.random.default_rng(seed)
    weights: dict[str, Any] = {}
    for key, shape in sorted(shapes.items()):
        if key.endswith("LayerNorm.weight") or key.endswith(
            (".ln1.weight", ".ln2.weight")
        ):
            weights[key] = np.ones(shape, dtype=np.float32)
        elif key.endswith("LayerNorm.bias") or key.endswith((".ln1.bias", ".ln2.bias")):
            weights[key] = np.zeros(shape, dtype=np.float32)
        else:
            weights[key] = (rng.standard_normal(shape) * scale).astype(np.float32)
    return weights


def _quiet_backbone(weights: dict[str, Any], *, seed: int) -> None:
    """Keep attention and feed-forward updates small against the residual."""
    import numpy as np

    rng = np.random.default_rng(seed + 1)
    for key in list(weights):
        if not key.startswith("deberta.encoder.layer."):
            continue
        shape = weights[key].shape
        if key.endswith(".bias"):
            if ".ln1." not in key and ".ln2." not in key:
                weights[key] = np.zeros(shape, dtype=np.float32)
            continue
        scale = 0.03 if (".out_proj." in key or ".linear2." in key) else 0.3
        if ".ln1." not in key and ".ln2." not in key:
            weights[key] = (rng.standard_normal(shape) * scale).astype(np.float32)


def _wire_gliner(weights: dict[str, Any], vocab: dict[str, int]) -> None:
    """Hand-set the GLiNER head so capitalized and digit runs score high."""
    import numpy as np

    rng = np.random.default_rng(SEED + 7)
    embedding = (
        rng.standard_normal(weights["deberta.embeddings.word_embeddings.weight"].shape)
        * 0.05
    ).astype(np.float32)
    for symbol, index in vocab.items():
        if symbol in string.ascii_uppercase:
            embedding[index, 0] += 3.0
        elif symbol in string.digits:
            embedding[index, 1] += 3.0
        else:
            embedding[index, 2] += 3.0
        if symbol == "p":
            embedding[index, 3] += 3.0
        if symbol == "i":
            embedding[index, 4] += 3.0
    weights["deberta.embeddings.word_embeddings.weight"] = embedding

    hidden = weights["token_projection.weight"].shape[0]
    weights["token_projection.weight"] = np.eye(hidden, dtype=np.float32)
    weights["token_projection.bias"] = np.zeros(hidden, dtype=np.float32)

    half = hidden // 2
    for direction in ("forward_lstm", "backward_lstm"):
        wx = np.zeros((4 * half, hidden), dtype=np.float32)
        wh = np.zeros((4 * half, half), dtype=np.float32)
        bias = np.zeros(4 * half, dtype=np.float32)
        bias[0:half] = -10.0  # input gates closed for idle units
        bias[half : 2 * half] = -10.0  # forget gates closed: no memory
        bias[3 * half : 4 * half] = 10.0  # output gates open
        cell = 2 * half
        for unit, feature in ((0, 0), (1, 1)):
            bias[unit] = 10.0
            wx[cell + unit, feature] = 2.0
            bias[cell + unit] = -5.0
        for unit, source in ((2, 0), (3, 1)):
            bias[unit] = 10.0
            wh[cell + unit, source] = 5.0
            bias[cell + unit] = -1.0
        weights[f"rnn.{direction}.Wx"] = wx
        weights[f"rnn.{direction}.Wh"] = wh
        weights[f"rnn.{direction}.bias"] = bias

    inner = hidden * 4

    def mlp(prefix: str, first: Any, first_bias: Any, second: Any) -> None:
        weights[f"{prefix}.linear1.weight"] = first
        weights[f"{prefix}.linear1.bias"] = first_bias
        weights[f"{prefix}.linear2.weight"] = second
        weights[f"{prefix}.linear2.bias"] = np.zeros(second.shape[0], np.float32)

    def run_boundary(cur: tuple[int, int], neighbour: tuple[int, int]) -> tuple:
        first = np.zeros((inner, hidden), dtype=np.float32)
        for row, (c, n) in enumerate(zip(cur, neighbour)):
            first[row, c] = 1.0
            first[row, n] = -1.0
        second = np.zeros((hidden, inner), dtype=np.float32)
        second[0, 0] = 1.0
        second[1, 1] = 1.0
        return first, np.zeros(inner, np.float32), second

    mlp("span_rep_layer.project_start", *run_boundary((0, 1), (2, 3)))
    mlp(
        "span_rep_layer.project_end",
        *run_boundary((half, half + 1), (half + 2, half + 3)),
    )
    first = np.zeros((inner, hidden * 2), dtype=np.float32)
    first_bias = np.zeros(inner, dtype=np.float32)
    first[0, 0] = first[0, hidden] = 1.0
    first[1, 1] = first[1, hidden + 1] = 1.0
    first_bias[0:2] = -2.0
    first_bias[2] = 1.0
    second = np.zeros((hidden, inner), dtype=np.float32)
    second[0, 0] = second[1, 1] = second[2, 2] = 1.0
    mlp("span_rep_layer.out_project", first, first_bias, second)

    first = np.zeros((inner, hidden), dtype=np.float32)
    first_bias = np.zeros(inner, dtype=np.float32)
    first[0, 3] = 1.0
    first[1, 4] = 1.0
    first_bias[0:2] = -1.0
    first_bias[2] = 1.0
    second = np.zeros((hidden, inner), dtype=np.float32)
    second[0, 0] = second[1, 1] = 3.5
    second[2, 2] = -4.0
    mlp("prompt_rep_layer", first, first_bias, second)


def _write_artifact(
    directory: Path, config: dict[str, Any], weights: dict[str, Any]
) -> Path:
    from safetensors.numpy import save_file

    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True), encoding="utf-8"
    )
    save_file(weights, str(directory / "weights.safetensors"))
    return directory


def build_gliner_artifact(
    directory: Path,
    *,
    wired: bool = True,
    seed: int = SEED,
    **overrides: Any,
) -> Path:
    """Write a tiny GLiNER span MLX artifact to *directory* and return it."""
    config = gliner_config(**overrides)
    build_tokenizer(directory, GLINER_MARKERS)
    weights = _random_weights(_mlx_parameter_shapes(config), seed=seed, scale=0.2)
    if wired:
        _quiet_backbone(weights, seed=seed)
        _wire_gliner(weights, vocabulary(GLINER_MARKERS))
    return _write_artifact(directory, config, weights)


def build_gliclass_artifact(
    directory: Path,
    *,
    seed: int = SEED,
    **overrides: Any,
) -> Path:
    """Write a tiny GLiClass uni-encoder MLX artifact and return it."""
    import numpy as np

    config = gliclass_config(**overrides)
    build_tokenizer(directory, GLICLASS_MARKERS)
    weights = _random_weights(_mlx_parameter_shapes(config), seed=seed, scale=0.2)
    weights["logit_scale"] = np.asarray(
        config.get("logit_scale_init_value", 1.0), dtype=np.float32
    )
    return _write_artifact(directory, config, weights)
