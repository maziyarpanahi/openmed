"""Local MedDecider runtime for the clinical NLI classifier contract.

MedDecider models answer a question about a state with lettered options; the
answer is the next-token distribution over the option letters. This runtime
asks the three-class NLI question and exposes the result as the
``tokenizer``/``model``/``config`` mapping that :class:`EncoderNLIBackend` and
the offline NLI qualifier already consume. Selection, selective gating,
qualification and receipt binding are unchanged.

An artifact is a caller-provisioned local directory holding the LoRA adapter in
``adapter/`` and the base model it names in ``base/``. Only permissively
licensed adapters and bases load. Loading never contacts a hub, never runs
remote code and never truncates a pair.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Sequence

MEDDECIDER_NLI_LABEL_MAPPING = {"0": "entailment", "1": "neutral", "2": "contradiction"}
_LETTERS = "ABC"
_OPTIONS = (
    "Yes, it follows from the text",
    "It might be true, but the text does not say",
    "No, the text contradicts it",
)
# Both option orders are scored and averaged, cancelling letter position bias.
_ORDERS = ((0, 1, 2), (2, 1, 0))
_QUESTION = "Based only on the text above, is this statement true? Statement: {}"
_INSTRUCTION = "Answer with the letter of the correct option only."
_PERMISSIVE = frozenset({"apache-2.0", "mit", "bsd-3-clause"})
_ADAPTER_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "meddecider_config.json",
    "README.md",
)
_FLOOR = 1e-12


def meddecider_nli_prompt(premise: str, hypothesis: str, order: Sequence[int]) -> str:
    """Return the user turn asking whether ``premise`` supports ``hypothesis``.

    Args:
        premise: Source span used as the decision state.
        hypothesis: Claim to classify.
        order: Permutation of the three option indices shown as ``A``-``C``.

    Returns:
        The exact user message; the layout is fixed for released adapters.
    """
    options = "\n".join(
        f"{_LETTERS[position]}. {_OPTIONS[index]}"
        for position, index in enumerate(order)
    )
    return (
        f"{premise.strip()}\n\nQuestion: {_QUESTION.format(hypothesis.strip())}"
        f"\nOptions:\n{options}\n\n{_INSTRUCTION}"
    )


class MedDeciderReadout:
    """Three-class NLI probabilities from MedDecider option-letter logits.

    Args:
        encode: Renders one user message with the chat template and returns
            its complete token ids.
        letter_logits: Returns the next-token logits of letters ``A``-``C``
            after the supplied token ids.
        temperature: Positive calibration temperature applied to the logits.
        max_position_embeddings: Optional model context length for the
            backend's no-truncation check.
    """

    def __init__(
        self,
        encode: Callable[[str], Sequence[int]],
        letter_logits: Callable[[Sequence[int]], Sequence[float]],
        *,
        temperature: float,
        max_position_embeddings: int | None = None,
    ) -> None:
        self._encode = encode
        self._letter_logits = letter_logits
        self.temperature = _temperature(temperature)
        self.max_position_embeddings = max_position_embeddings

    def encode(self, premise: str, hypothesis: str) -> tuple[list[int], ...]:
        """Return the complete prompt token ids for each option order."""
        return tuple(
            list(self._encode(meddecider_nli_prompt(premise, hypothesis, order)))
            for order in _ORDERS
        )

    def probabilities(self, encoded: Sequence[Sequence[int]]) -> list[float]:
        """Return probabilities in :data:`MEDDECIDER_NLI_LABEL_MAPPING` order."""
        if len(encoded) != len(_ORDERS):
            raise ValueError("MedDecider readout needs one prompt per option order")
        totals = [0.0, 0.0, 0.0]
        for order, ids in zip(_ORDERS, encoded):
            logits = [float(value) for value in self._letter_logits(ids)]
            if len(logits) != len(order) or not all(map(math.isfinite, logits)):
                raise ValueError("invalid MedDecider letter logits")
            scaled = [value / self.temperature for value in logits]
            peak = max(scaled)
            exponentials = [math.exp(value - peak) for value in scaled]
            total = sum(exponentials)
            for position, index in enumerate(order):
                totals[index] += exponentials[position] / total / len(_ORDERS)
        return totals

    def classifier(self) -> dict[str, Any]:
        """Return the mapping consumed by :class:`EncoderNLIBackend`.

        The model returns ``log p`` as its three logits, so the backend's
        softmax recovers the calibrated probabilities exactly.
        """
        readout = self

        def tokenizer(premise: str, hypothesis: str, **kwargs: Any) -> dict[str, Any]:
            if kwargs.get("truncation") is not False:
                raise ValueError("MedDecider pairs are never truncated")
            orders = readout.encode(premise, hypothesis)
            return {"input_ids": [max(orders, key=len)], "meddecider_orders": orders}

        def model(*, meddecider_orders: Any, **_: Any) -> SimpleNamespace:
            logits = [
                math.log(max(value, _FLOOR))
                for value in readout.probabilities(meddecider_orders)
            ]
            return SimpleNamespace(logits=[SimpleNamespace(tolist=lambda: logits)])

        config = SimpleNamespace(max_position_embeddings=self.max_position_embeddings)
        return {"tokenizer": tokenizer, "model": model, "config": config}


def read_meddecider_artifact(directory: str | Path) -> dict[str, Any]:
    """Validate a local MedDecider artifact without loading weights.

    Args:
        directory: Directory with ``adapter/`` and ``base/`` subdirectories.

    Returns:
        The adapter and base directories and the calibration temperature.

    Raises:
        ValueError: With a fixed, path-free message when the layout, metadata or
            licence is not supported.
    """
    root = Path(directory)
    adapter, base = root / "adapter", root / "base"
    for path in (*(adapter / name for name in _ADAPTER_FILES), base / "config.json"):
        if path.is_symlink() or not path.is_file():
            raise ValueError("MedDecider artifact is incomplete")
    for card in (adapter / "README.md", base / "README.md"):
        if _card_license(card) not in _PERMISSIVE:
            raise ValueError("MedDecider artifact licence is not permissive")
    try:
        settings = json.loads((adapter / "meddecider_config.json").read_text("utf-8"))
        temperature = settings["temperature"]["choice"]
    except (OSError, UnicodeError, ValueError, KeyError, TypeError):
        raise ValueError("MedDecider configuration is invalid") from None
    return {"adapter": adapter, "base": base, "temperature": _temperature(temperature)}


def load_meddecider_classifier(
    directory: str | Path, *, device: str | None = None
) -> dict[str, Any]:
    """Load a local MedDecider artifact as a three-class NLI classifier.

    The adapter is merged into the base weights in memory; files are only read.

    Args:
        directory: Validated artifact directory (see
            :func:`read_meddecider_artifact`).
        device: Torch device preference; ``None`` uses OpenMed's device
            resolution and environment overrides.

    Returns:
        The ``tokenizer``/``model``/``config`` classifier mapping.
    """
    artifact = read_meddecider_artifact(directory)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .device import resolve_torch_device

    target = resolve_torch_device(device)
    tokenizer = AutoTokenizer.from_pretrained(
        artifact["base"], local_files_only=True, trust_remote_code=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        artifact["base"],
        dtype=torch.float32 if target == "cpu" else torch.bfloat16,
        local_files_only=True,
        trust_remote_code=False,
    )
    merge_lora_adapter(model, artifact["adapter"])
    model = model.to(target).eval()
    letters = [
        tokenizer.encode(letter, add_special_tokens=False) for letter in _LETTERS
    ]
    if any(len(ids) != 1 for ids in letters):
        raise ValueError("MedDecider option letters must be single tokens")
    letter_ids = [ids[0] for ids in letters]

    def encode(message: str) -> list[int]:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": message}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return list(tokenizer(text, add_special_tokens=False)["input_ids"])

    def letter_logits(ids: Sequence[int]) -> list[float]:
        with torch.inference_mode():
            tensor = torch.tensor([list(ids)], device=model.device)
            output = model(input_ids=tensor, logits_to_keep=1).logits[0, -1]
            return output[letter_ids].float().cpu().tolist()

    text_config = getattr(model.config, "text_config", None) or model.config
    return MedDeciderReadout(
        encode,
        letter_logits,
        temperature=artifact["temperature"],
        max_position_embeddings=getattr(text_config, "max_position_embeddings", None),
    ).classifier()


def merge_lora_adapter(model: Any, adapter: str | Path) -> None:
    """Merge a plain LoRA adapter into ``model`` in place.

    Only the plain LoRA subset is admitted (see :func:`lora_scale`), and every
    tensor must be a finite rank-``r`` ``lora_A``/``lora_B`` pair matching an
    existing two-dimensional weight. Everything is validated before any weight
    changes. Each update is accumulated in float32 and the merged weight is
    rounded to its dtype once, as PEFT does with its default float32 adapter.

    Args:
        model: Loaded base model exposing ``named_parameters()``.
        adapter: Directory with ``adapter_config.json`` and
            ``adapter_model.safetensors``.
    """
    import torch
    from safetensors.torch import load_file

    adapter = Path(adapter)
    config = json.loads((adapter / "adapter_config.json").read_text("utf-8"))
    scale = lora_scale(config)
    rank = config["r"]
    tensors = load_file(str(adapter / "adapter_model.safetensors"))
    parameters = dict(model.named_parameters())
    prefix, down_suffix, up_suffix = (
        "base_model.model.",
        ".lora_A.weight",
        ".lora_B.weight",
    )
    modules = [
        key[len(prefix) : -len(down_suffix)]
        for key in tensors
        if key.startswith(prefix) and key.endswith(down_suffix)
    ]
    expected = {
        prefix + m + suffix for m in modules for suffix in (down_suffix, up_suffix)
    }
    if not modules or set(tensors) != expected:
        raise ValueError("unsupported MedDecider adapter weights")
    updates = []
    for module in modules:
        weight = parameters.get(module + ".weight")
        down = tensors[prefix + module + down_suffix]
        up = tensors[prefix + module + up_suffix]
        if (
            weight is None
            or weight.dim() != 2
            or not all(t.is_floating_point() and t.dim() == 2 for t in (down, up))
            or tuple(down.shape) != (rank, weight.shape[1])
            or tuple(up.shape) != (weight.shape[0], rank)
            or not bool(torch.isfinite(down).all() and torch.isfinite(up).all())
        ):
            raise ValueError("MedDecider adapter does not match the base model")
        updates.append((weight, down, up))
    with torch.no_grad():
        for weight, down, up in updates:
            delta = up.to(weight.device, torch.float32) @ down.to(
                weight.device, torch.float32
            )
            weight.copy_((weight.float() + delta * scale).to(weight.dtype))


# Adapter settings that never change the merged weights of a plain LoRA.
_LORA_INERT = frozenset(
    {
        "auto_mapping",
        "base_model_name_or_path",
        "exclude_modules",
        "inference_mode",
        "layers_pattern",
        "lora_alpha",
        "lora_dropout",
        "megatron_core",
        "peft_type",
        "peft_version",
        "qalora_group_size",
        "r",
        "revision",
        "target_modules",
        "task_type",
    }
)


def lora_scale(config: dict[str, Any]) -> float:
    """Validate a plain LoRA ``adapter_config.json`` and return its scaling.

    Admits ``peft_type="LORA"`` for a causal LM with ``bias="none"``, a positive
    integer ``r``, a positive finite ``lora_alpha`` and a standard
    initialisation. Every other setting (DoRA, rsLoRA, aLoRA invocation tokens,
    layer replication, per-module rank/alpha, extra modules, trainable tokens,
    special initialisations and any unknown option) must be unset.

    Args:
        config: Parsed adapter configuration.

    Returns:
        The finite scaling ``lora_alpha / r``.

    Raises:
        ValueError: If the configuration is outside the supported subset.
    """
    rank, alpha = config.get("r"), config.get("lora_alpha")
    unset = [
        key
        for key, value in config.items()
        if key not in _LORA_INERT | {"bias", "init_lora_weights"}
        and value not in (None, False, "", [], {})
    ]
    if (
        unset
        or config.get("peft_type") != "LORA"
        or config.get("task_type") not in (None, "CAUSAL_LM")
        or config.get("bias", "none") != "none"
        or config.get("init_lora_weights", True) not in (True, False, "gaussian")
        or type(rank) is not int
        or rank <= 0
        or isinstance(alpha, bool)
        or not isinstance(alpha, (int, float))
        or not math.isfinite(alpha)
        or alpha <= 0
    ):
        raise ValueError("unsupported MedDecider adapter configuration")
    scale = float(alpha) / rank
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("unsupported MedDecider adapter configuration")
    return scale


def _temperature(value: Any) -> float:
    """Return a positive finite calibration temperature or raise ``ValueError``."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError("MedDecider temperature must be positive and finite")
    return float(value)


def _card_license(path: Path) -> str | None:
    """Return the ``license`` value from a model card's front matter."""
    try:
        lines = path.read_text("utf-8").splitlines()
    except (OSError, UnicodeError):
        return None
    if not lines or lines[0].strip() != "---":
        return None
    for line in lines[1:]:
        if line.strip() == "---":
            break
        key, _, value = line.partition(":")
        if key.strip() == "license":
            return value.strip().casefold() or None
    return None


__all__ = [
    "MEDDECIDER_NLI_LABEL_MAPPING",
    "MedDeciderReadout",
    "load_meddecider_classifier",
    "meddecider_nli_prompt",
    "lora_scale",
    "merge_lora_adapter",
    "read_meddecider_artifact",
]
