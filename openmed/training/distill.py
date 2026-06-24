"""Mode-A token-classification knowledge distillation utilities.

The module keeps the numeric contract importable without PyTorch so unit tests
and recipe validation stay offline-friendly. Real teacher/student execution
uses optional ``torch`` and ``transformers`` imports inside the runtime path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from openmed.core.decoding import (
    build_label_info,
    labels_to_token_spans,
    refine_privacy_filter_span,
    viterbi_decode,
    zero_viterbi_biases,
)
from openmed.training.recipe import TrainingRecipeConfig, load_preset

IGNORE_INDEX = -100


@dataclass(frozen=True)
class RepairedSpan:
    """Teacher span decoded through the shared BIOES repair path."""

    label: str
    token_start: int
    token_end: int
    start: int | None = None
    end: int | None = None
    logits: tuple[float, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "label": self.label,
            "token_end": self.token_end,
            "token_start": self.token_start,
        }
        if self.start is not None:
            payload["start"] = self.start
        if self.end is not None:
            payload["end"] = self.end
        if self.logits:
            payload["logits"] = list(self.logits)
        return payload


@dataclass(frozen=True)
class DistillationTargets:
    """Teacher outputs used to fit a smaller token-classification student."""

    teacher_id: str
    token_logits: Any
    token_soft_labels: Any
    span_logits: Any | None
    repaired_spans: tuple[tuple[RepairedSpan, ...], ...]


@dataclass(frozen=True)
class KDLossBreakdown:
    """Weighted Mode-A KD loss components."""

    total: Any
    hard_ce: Any
    soft_kl: Any
    span_transfer: Any
    alpha: float
    temperature: float
    span_loss_weight: float

    def to_dict(self) -> dict[str, float]:
        return {
            "alpha": self.alpha,
            "hard_ce": _scalar_float(self.hard_ce),
            "soft_kl": _scalar_float(self.soft_kl),
            "span_loss_weight": self.span_loss_weight,
            "span_transfer": _scalar_float(self.span_transfer),
            "temperature": self.temperature,
            "total": _scalar_float(self.total),
        }


@dataclass(frozen=True)
class LabelRecallDelta:
    """Per-label student recall movement relative to the teacher."""

    label: str
    teacher_recall: float
    student_recall: float
    delta: float
    critical_drop: bool

    def to_dict(self) -> dict[str, float | bool | str]:
        return {
            "critical_drop": self.critical_drop,
            "delta": self.delta,
            "label": self.label,
            "student_recall": self.student_recall,
            "teacher_recall": self.teacher_recall,
        }


@dataclass(frozen=True)
class DistillationReport:
    """Model-card evidence for one teacher-to-student distillation run."""

    teacher_id: str
    student_backbone: str
    temperature: float
    alpha: float
    per_label_recall_delta: tuple[LabelRecallDelta, ...]

    @property
    def critical_label_drops(self) -> tuple[str, ...]:
        return tuple(
            delta.label for delta in self.per_label_recall_delta if delta.critical_drop
        )

    @property
    def recall_gate_passed(self) -> bool:
        return not self.critical_label_drops

    def to_dict(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "critical_label_drops": list(self.critical_label_drops),
            "per_label_recall_delta": [
                delta.to_dict() for delta in self.per_label_recall_delta
            ],
            "recall_gate_passed": self.recall_gate_passed,
            "student_backbone": self.student_backbone,
            "teacher_id": self.teacher_id,
            "temperature": self.temperature,
        }

    def model_card_evidence(self) -> dict[str, Any]:
        """Return the report nested as a model-card evidence block."""

        return {"distillation": self.to_dict()}


def student_backbone_from_tiny_distill_preset(
    config: TrainingRecipeConfig | None = None,
) -> str:
    """Return the Mode-A student backbone declared by ``tiny_distill``."""

    recipe = config if config is not None else load_preset("tiny_distill")
    if recipe.mode != "A" or recipe.preset_name != "tiny_distill":
        raise ValueError("distillation requires the Mode-A tiny_distill preset")
    return recipe.backbone.model_ref


def soft_label_distributions(logits: Any, *, temperature: float = 1.0) -> Any:
    """Return per-token soft-label distributions from teacher logits."""

    _validate_temperature(temperature)
    if _is_torch_tensor(logits):
        torch = _import_torch()
        return torch.nn.functional.softmax(logits / temperature, dim=-1)
    return tuple(
        tuple(tuple(_softmax(token, temperature=temperature)) for token in sample)
        for sample in _as_batched_logits(logits)
    )


def compute_kd_loss(
    *,
    student_logits: Any,
    hard_labels: Any,
    teacher_logits: Any | None = None,
    teacher_soft_labels: Any | None = None,
    student_span_logits: Any | None = None,
    teacher_span_logits: Any | None = None,
    mask: Any | None = None,
    span_mask: Any | None = None,
    temperature: float = 2.0,
    alpha: float = 0.5,
    span_loss_weight: float = 1.0,
    ignore_index: int = IGNORE_INDEX,
) -> KDLossBreakdown:
    """Compute Mode-A KD loss for token classification.

    The total loss is ``(1 - alpha) * CE + alpha * (KL + span_weight * span)``.
    Setting ``alpha=0`` therefore collapses exactly to hard-label
    cross-entropy, which is the sanity invariant used by the release gate.

    Args:
        student_logits: Student token logits with shape ``batch x seq x labels``.
        hard_labels: Integer hard labels with shape ``batch x seq``.
        teacher_logits: Teacher token logits. Required unless
            ``teacher_soft_labels`` is provided.
        teacher_soft_labels: Teacher soft-label probabilities at
            ``temperature``. Used directly when supplied.
        student_span_logits: Optional student span-level logits.
        teacher_span_logits: Optional teacher span-level logits.
        mask: Optional boolean token mask. Positions with ``ignore_index`` are
            always excluded.
        span_mask: Optional boolean mask for span-logit rows.
        temperature: KD temperature. Must be positive.
        alpha: Teacher-loss mixing weight in ``[0, 1]``.
        span_loss_weight: Weight for the span-logit transfer term inside the
            teacher side of the mixture.
        ignore_index: Hard-label sentinel excluded from loss terms.
    """

    _validate_kd_weights(alpha, temperature, span_loss_weight)
    if teacher_logits is None and teacher_soft_labels is None:
        raise ValueError("teacher_logits or teacher_soft_labels is required")

    if _is_torch_tensor(student_logits):
        return _compute_torch_kd_loss(
            student_logits=student_logits,
            hard_labels=hard_labels,
            teacher_logits=teacher_logits,
            teacher_soft_labels=teacher_soft_labels,
            student_span_logits=student_span_logits,
            teacher_span_logits=teacher_span_logits,
            mask=mask,
            span_mask=span_mask,
            temperature=temperature,
            alpha=alpha,
            span_loss_weight=span_loss_weight,
            ignore_index=ignore_index,
        )

    return _compute_python_kd_loss(
        student_logits=student_logits,
        hard_labels=hard_labels,
        teacher_logits=teacher_logits,
        teacher_soft_labels=teacher_soft_labels,
        student_span_logits=student_span_logits,
        teacher_span_logits=teacher_span_logits,
        mask=mask,
        temperature=temperature,
        alpha=alpha,
        span_loss_weight=span_loss_weight,
        ignore_index=ignore_index,
    )


def decode_repaired_spans(
    token_logits: Any,
    id2label: Mapping[int | str, str],
    *,
    texts: Sequence[str] | None = None,
    offset_mapping: Any | None = None,
    transition_biases: Mapping[str, float] | None = None,
) -> tuple[tuple[RepairedSpan, ...], ...]:
    """Decode token logits into repaired BIOES spans using core decoding."""

    label_map = {int(key): value for key, value in id2label.items()}
    label_info = build_label_info(label_map)
    biases = zero_viterbi_biases()
    if transition_biases:
        biases.update(
            {
                key: float(value)
                for key, value in transition_biases.items()
                if key in biases
            }
        )

    batch_logits = _as_batched_logits(_to_list(token_logits))
    offsets = _as_optional_batch_offsets(offset_mapping)
    batches: list[tuple[RepairedSpan, ...]] = []
    for batch_index, sample_logits in enumerate(batch_logits):
        token_logprobs = [_log_softmax(row) for row in sample_logits]
        path = viterbi_decode(token_logprobs, label_info=label_info, biases=biases)
        labels_by_index = _labels_by_valid_offsets(path, offsets, batch_index)
        token_spans = labels_to_token_spans(labels_by_index, label_info)
        text = texts[batch_index] if texts is not None else None
        sample_offsets = offsets[batch_index] if offsets is not None else None

        repaired: list[RepairedSpan] = []
        for span_label, token_start, token_end in token_spans:
            label = _span_label_name(label_info.span_class_names, span_label)
            start: int | None = None
            end: int | None = None
            if text is not None and sample_offsets is not None:
                start, end = _char_offsets_for_token_span(
                    sample_offsets,
                    token_start,
                    token_end,
                )
                if start is not None and end is not None:
                    start, end = refine_privacy_filter_span(label, start, end, text)
            repaired.append(
                RepairedSpan(
                    label=label,
                    token_start=token_start,
                    token_end=token_end,
                    start=start,
                    end=end,
                    logits=_mean_token_logits(sample_logits, token_start, token_end),
                )
            )
        batches.append(tuple(repaired))
    return tuple(batches)


def span_logits_from_repaired_spans(
    repaired_spans: Sequence[Sequence[RepairedSpan]],
) -> tuple[tuple[tuple[float, ...], ...], ...]:
    """Extract span-logit rows from repaired teacher spans."""

    return tuple(
        tuple(span.logits for span in sample if span.logits)
        for sample in repaired_spans
    )


def build_distillation_report(
    *,
    teacher_id: str,
    student_backbone: str,
    temperature: float,
    alpha: float,
    teacher_recall_by_label: Mapping[str, float],
    student_recall_by_label: Mapping[str, float],
    critical_labels: Sequence[str],
    critical_drop_tolerance: float = 0.0,
) -> DistillationReport:
    """Build model-card evidence and recall-first drop flags."""

    _validate_kd_weights(alpha, temperature, span_loss_weight=0.0)
    critical = set(critical_labels)
    labels = sorted(set(teacher_recall_by_label) | set(student_recall_by_label))
    deltas = []
    for label in labels:
        teacher_recall = float(teacher_recall_by_label.get(label, 0.0))
        student_recall = float(student_recall_by_label.get(label, 0.0))
        delta = student_recall - teacher_recall
        critical_drop = label in critical and delta < -critical_drop_tolerance
        deltas.append(
            LabelRecallDelta(
                label=label,
                teacher_recall=teacher_recall,
                student_recall=student_recall,
                delta=delta,
                critical_drop=critical_drop,
            )
        )
    return DistillationReport(
        teacher_id=teacher_id,
        student_backbone=student_backbone,
        temperature=temperature,
        alpha=alpha,
        per_label_recall_delta=tuple(deltas),
    )


class ModeADistillationPipeline:
    """Optional Torch/Transformers teacher-to-student distillation runner."""

    def __init__(
        self,
        *,
        teacher_id: str,
        student_backbone: str,
        teacher_model: Any,
        student_model: Any | None,
        tokenizer: Any,
        temperature: float = 2.0,
        alpha: float = 0.5,
        span_loss_weight: float = 1.0,
        id2label: Mapping[int | str, str] | None = None,
    ) -> None:
        _validate_kd_weights(alpha, temperature, span_loss_weight)
        self.teacher_id = teacher_id
        self.student_backbone = student_backbone
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.alpha = alpha
        self.span_loss_weight = span_loss_weight
        self.id2label = _resolve_id2label(teacher_model, id2label)

    @classmethod
    def from_tiny_distill_preset(
        cls,
        teacher_checkpoint: str,
        *,
        student_checkpoint: str | None = None,
        local_files_only: bool = True,
        trust_remote_code: bool = False,
        load_student: bool = True,
        temperature: float = 2.0,
        alpha: float = 0.5,
        span_loss_weight: float = 1.0,
        device: str | None = None,
    ) -> "ModeADistillationPipeline":
        """Load teacher and tiny student declared by the Mode-A preset."""

        try:
            import torch
            from transformers import AutoModelForTokenClassification, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional runtime path
            raise ImportError(
                "Mode-A distillation requires `torch` and `transformers`. "
                "Install the Hugging Face training extras before running it."
            ) from exc

        student_backbone = (
            student_checkpoint or student_backbone_from_tiny_distill_preset()
        )
        tokenizer = AutoTokenizer.from_pretrained(
            teacher_checkpoint,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )
        teacher_model = AutoModelForTokenClassification.from_pretrained(
            teacher_checkpoint,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )
        student_model = (
            AutoModelForTokenClassification.from_pretrained(
                student_backbone,
                local_files_only=local_files_only,
                trust_remote_code=trust_remote_code,
            )
            if load_student
            else None
        )
        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        teacher_model.to(resolved_device)
        teacher_model.eval()
        if student_model is not None:
            student_model.to(resolved_device)
        return cls(
            teacher_id=teacher_checkpoint,
            student_backbone=student_backbone,
            teacher_model=teacher_model,
            student_model=student_model,
            tokenizer=tokenizer,
            temperature=temperature,
            alpha=alpha,
            span_loss_weight=span_loss_weight,
        )

    def teacher_targets(
        self,
        texts: str | Sequence[str],
        *,
        tokenizer_kwargs: Mapping[str, Any] | None = None,
    ) -> DistillationTargets:
        """Run the teacher and return soft-label plus repaired-span targets."""

        torch = _import_torch()
        text_batch = [texts] if isinstance(texts, str) else list(texts)
        options = {
            "padding": True,
            "return_offsets_mapping": True,
            "return_tensors": "pt",
            "truncation": True,
        }
        if tokenizer_kwargs:
            options.update(dict(tokenizer_kwargs))
        encoded = self.tokenizer(text_batch, **options)
        offset_mapping = encoded.pop("offset_mapping", None)
        device = next(self.teacher_model.parameters()).device
        model_inputs = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in encoded.items()
        }
        with torch.no_grad():
            outputs = self.teacher_model(**model_inputs)
        token_logits = outputs.logits
        repaired_spans = decode_repaired_spans(
            token_logits,
            self.id2label,
            texts=text_batch,
            offset_mapping=offset_mapping,
        )
        span_logits = _extract_span_logits(outputs)
        if span_logits is None:
            span_logits = span_logits_from_repaired_spans(repaired_spans)
        return DistillationTargets(
            teacher_id=self.teacher_id,
            token_logits=token_logits,
            token_soft_labels=soft_label_distributions(
                token_logits,
                temperature=self.temperature,
            ),
            span_logits=span_logits,
            repaired_spans=repaired_spans,
        )

    def student_loss(
        self,
        *,
        model_inputs: Mapping[str, Any],
        hard_labels: Any,
        teacher_targets: DistillationTargets,
        mask: Any | None = None,
    ) -> KDLossBreakdown:
        """Run the student on ``model_inputs`` and compute the KD fit loss."""

        if self.student_model is None:
            raise ValueError("student_model is not loaded")
        outputs = self.student_model(**dict(model_inputs))
        return compute_kd_loss(
            student_logits=outputs.logits,
            hard_labels=hard_labels,
            teacher_logits=teacher_targets.token_logits,
            teacher_soft_labels=teacher_targets.token_soft_labels,
            student_span_logits=_extract_span_logits(outputs),
            teacher_span_logits=teacher_targets.span_logits,
            mask=mask,
            temperature=self.temperature,
            alpha=self.alpha,
            span_loss_weight=self.span_loss_weight,
        )


def _compute_torch_kd_loss(
    *,
    student_logits: Any,
    hard_labels: Any,
    teacher_logits: Any | None,
    teacher_soft_labels: Any | None,
    student_span_logits: Any | None,
    teacher_span_logits: Any | None,
    mask: Any | None,
    span_mask: Any | None,
    temperature: float,
    alpha: float,
    span_loss_weight: float,
    ignore_index: int,
) -> KDLossBreakdown:
    torch = _import_torch()
    functional = torch.nn.functional
    labels = _torch_tensor(
        hard_labels,
        torch=torch,
        device=student_logits.device,
        dtype=torch.long,
    )
    valid = labels.ne(ignore_index)
    if mask is not None:
        valid = valid & _torch_tensor(
            mask,
            torch=torch,
            device=student_logits.device,
            dtype=torch.bool,
        )

    safe_labels = labels.masked_fill(~valid, 0)
    flat_logits = student_logits.reshape(-1, student_logits.shape[-1])
    flat_labels = safe_labels.reshape(-1)
    flat_valid = valid.reshape(-1)
    if bool(flat_valid.any()):
        ce_rows = functional.cross_entropy(flat_logits, flat_labels, reduction="none")
        hard_ce = ce_rows[flat_valid].mean()
    else:
        hard_ce = student_logits.sum() * 0.0

    if teacher_soft_labels is None:
        teacher_probs = functional.softmax(teacher_logits / temperature, dim=-1)
    else:
        teacher_probs = _torch_tensor(
            teacher_soft_labels,
            torch=torch,
            device=student_logits.device,
            dtype=student_logits.dtype,
        )
    kl_rows = functional.kl_div(
        functional.log_softmax(student_logits / temperature, dim=-1),
        teacher_probs,
        reduction="none",
    ).sum(dim=-1)
    soft_kl = (
        kl_rows[valid].mean() * (temperature * temperature)
        if bool(valid.any())
        else hard_ce * 0.0
    )
    span_transfer = _torch_span_transfer(
        student_span_logits,
        teacher_span_logits,
        span_mask=span_mask,
        torch=torch,
        like=student_logits,
    )
    total = (1.0 - alpha) * hard_ce + alpha * (
        soft_kl + span_loss_weight * span_transfer
    )
    return KDLossBreakdown(
        total=total,
        hard_ce=hard_ce,
        soft_kl=soft_kl,
        span_transfer=span_transfer,
        alpha=alpha,
        temperature=temperature,
        span_loss_weight=span_loss_weight,
    )


def _compute_python_kd_loss(
    *,
    student_logits: Any,
    hard_labels: Any,
    teacher_logits: Any | None,
    teacher_soft_labels: Any | None,
    student_span_logits: Any | None,
    teacher_span_logits: Any | None,
    mask: Any | None,
    temperature: float,
    alpha: float,
    span_loss_weight: float,
    ignore_index: int,
) -> KDLossBreakdown:
    valid_rows = _valid_token_rows(student_logits, hard_labels, mask, ignore_index)
    if not valid_rows:
        hard_ce = 0.0
        soft_kl = 0.0
    else:
        ce_values = []
        kl_values = []
        teacher_logits_batch = (
            _as_batched_logits(teacher_logits) if teacher_logits is not None else None
        )
        teacher_probs_batch = (
            _as_batched_logits(teacher_soft_labels)
            if teacher_soft_labels is not None
            else None
        )
        for batch_index, token_index, student_row, label in valid_rows:
            ce_values.append(-_log_softmax(student_row)[label])
            student_log_probs = _log_softmax(student_row, temperature=temperature)
            if teacher_probs_batch is not None:
                teacher_probs = teacher_probs_batch[batch_index][token_index]
            elif teacher_logits_batch is not None:
                teacher_probs = _softmax(
                    teacher_logits_batch[batch_index][token_index],
                    temperature=temperature,
                )
            else:  # pragma: no cover - guarded by public entrypoint
                raise ValueError("teacher logits or soft labels are required")
            kl_values.append(_kl_divergence(teacher_probs, student_log_probs))
        hard_ce = sum(ce_values) / len(ce_values)
        soft_kl = (sum(kl_values) / len(kl_values)) * (temperature * temperature)

    span_transfer = _python_span_transfer(student_span_logits, teacher_span_logits)
    total = (1.0 - alpha) * hard_ce + alpha * (
        soft_kl + span_loss_weight * span_transfer
    )
    return KDLossBreakdown(
        total=total,
        hard_ce=hard_ce,
        soft_kl=soft_kl,
        span_transfer=span_transfer,
        alpha=alpha,
        temperature=temperature,
        span_loss_weight=span_loss_weight,
    )


def _valid_token_rows(
    student_logits: Any,
    hard_labels: Any,
    mask: Any | None,
    ignore_index: int,
) -> list[tuple[int, int, Sequence[float], int]]:
    logits_batch = _as_batched_logits(student_logits)
    labels_batch = _as_batched_labels(hard_labels)
    mask_batch = _as_batched_labels(mask) if mask is not None else None
    rows = []
    for batch_index, sample_logits in enumerate(logits_batch):
        sample_labels = labels_batch[batch_index]
        sample_mask = mask_batch[batch_index] if mask_batch is not None else None
        for token_index, student_row in enumerate(sample_logits):
            label = int(sample_labels[token_index])
            if label == ignore_index:
                continue
            if sample_mask is not None and not bool(sample_mask[token_index]):
                continue
            if label < 0 or label >= len(student_row):
                raise ValueError("hard label is outside the token-label space")
            rows.append((batch_index, token_index, student_row, label))
    return rows


def _torch_span_transfer(
    student_span_logits: Any | None,
    teacher_span_logits: Any | None,
    *,
    span_mask: Any | None,
    torch: Any,
    like: Any,
) -> Any:
    if student_span_logits is None or teacher_span_logits is None:
        return like.sum() * 0.0
    student = _torch_tensor(
        student_span_logits,
        torch=torch,
        device=like.device,
        dtype=like.dtype,
    )
    teacher = _torch_tensor(
        teacher_span_logits,
        torch=torch,
        device=like.device,
        dtype=like.dtype,
    )
    squared = (student - teacher).pow(2)
    if span_mask is not None:
        mask = _torch_tensor(
            span_mask,
            torch=torch,
            device=like.device,
            dtype=torch.bool,
        )
        while mask.dim() < squared.dim():
            mask = mask.unsqueeze(-1)
        squared = squared[mask.expand_as(squared)]
    return squared.mean() if squared.numel() else like.sum() * 0.0


def _python_span_transfer(
    student_span_logits: Any | None,
    teacher_span_logits: Any | None,
) -> float:
    if student_span_logits is None or teacher_span_logits is None:
        return 0.0
    student_values = list(_flatten_numbers(student_span_logits))
    teacher_values = list(_flatten_numbers(teacher_span_logits))
    if len(student_values) != len(teacher_values):
        raise ValueError("student and teacher span logits must have the same shape")
    if not student_values:
        return 0.0
    squared = [
        (float(student) - float(teacher)) ** 2
        for student, teacher in zip(student_values, teacher_values)
    ]
    return sum(squared) / len(squared)


def _labels_by_valid_offsets(
    path: Sequence[int],
    offsets: tuple[tuple[tuple[int, int], ...], ...] | None,
    batch_index: int,
) -> dict[int, int]:
    if offsets is None:
        return {index: label for index, label in enumerate(path)}
    sample_offsets = offsets[batch_index]
    labels = {}
    for index, label in enumerate(path):
        if index >= len(sample_offsets):
            break
        start, end = sample_offsets[index]
        if start == end == 0:
            continue
        labels[index] = label
    return labels


def _char_offsets_for_token_span(
    sample_offsets: Sequence[tuple[int, int]],
    token_start: int,
    token_end: int,
) -> tuple[int | None, int | None]:
    covered = [
        offset
        for offset in sample_offsets[token_start:token_end]
        if offset[0] != offset[1]
    ]
    if not covered:
        return None, None
    return covered[0][0], covered[-1][1]


def _mean_token_logits(
    sample_logits: Sequence[Sequence[float]],
    token_start: int,
    token_end: int,
) -> tuple[float, ...]:
    rows = sample_logits[token_start:token_end]
    if not rows:
        return ()
    width = len(rows[0])
    return tuple(
        sum(float(row[index]) for row in rows) / len(rows) for index in range(width)
    )


def _span_label_name(names: Sequence[str], index: int) -> str:
    if 0 <= index < len(names):
        return names[index]
    return f"label_{index}"


def _extract_span_logits(outputs: Any) -> Any | None:
    if isinstance(outputs, Mapping):
        return outputs.get("span_logits")
    return getattr(outputs, "span_logits", None)


def _resolve_id2label(
    model: Any,
    id2label: Mapping[int | str, str] | None,
) -> dict[int, str]:
    if id2label is not None:
        return {int(key): value for key, value in id2label.items()}
    config = getattr(model, "config", None)
    raw = getattr(config, "id2label", None)
    if not isinstance(raw, Mapping):
        raise ValueError("teacher model config must expose id2label")
    return {int(key): str(value) for key, value in raw.items()}


def _as_optional_batch_offsets(
    value: Any | None,
) -> tuple[tuple[tuple[int, int], ...], ...] | None:
    if value is None:
        return None
    return tuple(
        tuple((int(start), int(end)) for start, end in sample)
        for sample in _to_list(value)
    )


def _as_batched_logits(value: Any) -> list[list[list[float]]]:
    raw = _to_list(value)
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("logits must be a batched numeric sequence")
    return [[[float(item) for item in token] for token in sample] for sample in raw]


def _as_batched_labels(value: Any) -> list[list[Any]]:
    raw = _to_list(value)
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("labels must be a batched sequence")
    return [list(sample) for sample in raw]


def _to_list(value: Any) -> Any:
    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach()
    if hasattr(value, "cpu") and callable(value.cpu):
        value = value.cpu()
    if hasattr(value, "tolist") and callable(value.tolist):
        return value.tolist()
    return value


def _log_softmax(
    values: Sequence[float], *, temperature: float = 1.0
) -> tuple[float, ...]:
    scaled = [float(value) / temperature for value in values]
    max_value = max(scaled)
    log_total = max_value + math.log(
        sum(math.exp(value - max_value) for value in scaled)
    )
    return tuple(value - log_total for value in scaled)


def _softmax(values: Sequence[float], *, temperature: float = 1.0) -> tuple[float, ...]:
    logs = _log_softmax(values, temperature=temperature)
    return tuple(math.exp(value) for value in logs)


def _kl_divergence(
    teacher_probs: Sequence[float],
    student_log_probs: Sequence[float],
) -> float:
    if len(teacher_probs) != len(student_log_probs):
        raise ValueError("teacher and student label spaces must have the same width")
    total = 0.0
    for probability, student_log_prob in zip(teacher_probs, student_log_probs):
        probability = float(probability)
        if probability <= 0.0:
            continue
        total += probability * (math.log(probability) - student_log_prob)
    return total


def _flatten_numbers(value: Any) -> Sequence[float]:
    raw = _to_list(value)
    if isinstance(raw, (int, float)):
        return [float(raw)]
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        flattened = []
        for item in raw:
            flattened.extend(_flatten_numbers(item))
        return flattened
    raise ValueError("span logits must be numeric sequences")


def _torch_tensor(value: Any, *, torch: Any, device: Any, dtype: Any) -> Any:
    if _is_torch_tensor(value):
        return value.to(device=device, dtype=dtype)
    return torch.as_tensor(value, device=device, dtype=dtype)


def _is_torch_tensor(value: Any) -> bool:
    return type(value).__module__.startswith("torch")


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional runtime path
        raise ImportError(
            "PyTorch is required for tensor distillation execution"
        ) from exc
    return torch


def _scalar_float(value: Any) -> float:
    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach()
    if hasattr(value, "cpu") and callable(value.cpu):
        value = value.cpu()
    if hasattr(value, "item") and callable(value.item):
        return float(value.item())
    return float(value)


def _validate_temperature(temperature: float) -> None:
    if temperature <= 0:
        raise ValueError("temperature must be positive")


def _validate_kd_weights(
    alpha: float,
    temperature: float,
    span_loss_weight: float,
) -> None:
    _validate_temperature(temperature)
    if alpha < 0.0 or alpha > 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if span_loss_weight < 0.0:
        raise ValueError("span_loss_weight must be non-negative")


__all__ = [
    "DistillationReport",
    "DistillationTargets",
    "IGNORE_INDEX",
    "KDLossBreakdown",
    "LabelRecallDelta",
    "ModeADistillationPipeline",
    "RepairedSpan",
    "build_distillation_report",
    "compute_kd_loss",
    "decode_repaired_spans",
    "soft_label_distributions",
    "span_logits_from_repaired_spans",
    "student_backbone_from_tiny_distill_preset",
]
