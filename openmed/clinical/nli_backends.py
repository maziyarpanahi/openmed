"""Offline sequence-classification backends for clinical NLI."""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from openmed.clinical.nli_gate import (
    EvidenceLink,
    NLIThresholds,
    evaluate_nli,
    hash_text,
)
from openmed.clinical.nli_labels import normalize_nli_label
from openmed.core.model_registry import get_default_nli_model, get_model_info

_REMOTE_NAMES = frozenset({"remote", "openai", "anthropic", "azure", "bedrock"})
_CANONICAL_SCORES = frozenset({"entailment", "contradiction", "neutral"})


class LocalNLIError(RuntimeError):
    """A value-free failure to resolve, load, or run local NLI."""


class RemoteNLIBackendError(LocalNLIError):
    """A remote NLI backend was requested at the clinical boundary."""


class EncoderNLIBackend:
    """Run a pinned, cached sequence classifier through a local runtime.

    Class-index meanings and calibrated thresholds must be supplied by the
    checkpoint release. Neither can be guessed from model output. Loading is
    lazy and guarded by :class:`ModelLoader`'s offline-only path.
    """

    backend_id = "local-encoder"

    def __init__(
        self,
        model_ref: str | Path,
        *,
        revision: str | None = None,
        runtime: str = "torch",
        label_mapping: Mapping[str, str],
        thresholds: NLIThresholds,
        loader: Any | None = None,
    ) -> None:
        reference = str(model_ref)
        if _is_remote(reference):
            raise RemoteNLIBackendError("remote NLI backends are prohibited")
        if runtime not in {"torch", "onnx"}:
            raise LocalNLIError("unsupported local NLI runtime")
        if not isinstance(thresholds, NLIThresholds):
            raise LocalNLIError("calibrated NLI thresholds are required")
        if not isinstance(label_mapping, Mapping):
            raise LocalNLIError("NLI class mapping is required")
        try:
            mapped = {
                str(key): normalize_nli_label(value).value
                for key, value in label_mapping.items()
            }
        except (TypeError, ValueError):
            raise LocalNLIError("NLI class mapping is invalid") from None
        if set(mapped.values()) != _CANONICAL_SCORES or set(mapped) != {"0", "1", "2"}:
            raise LocalNLIError("NLI class mapping must cover three model states")

        self.model_ref = reference
        self.revision = revision
        self.runtime = runtime
        self.label_mapping = mapped
        self.thresholds = thresholds
        self._loader = loader
        self._artifact: Mapping[str, Any] | None = None

    def _load(self) -> Mapping[str, Any]:
        if self._artifact is None:
            try:
                if self._loader is None:
                    from openmed.core.models import ModelLoader

                    self._loader = ModelLoader()
                artifact = self._loader.load_local_sequence_classifier(
                    self.model_ref, revision=self.revision, runtime=self.runtime
                )
                if not isinstance(artifact, Mapping):
                    raise TypeError("invalid local classifier")
            except Exception:
                raise LocalNLIError("local NLI checkpoint is unavailable") from None
            self._artifact = artifact
        return self._artifact

    def predict(self, premise: str, hypothesis: str) -> dict[str, str | float]:
        """Return a four-state, calibrated decision with no input text."""

        if not isinstance(premise, str) or not premise.strip():
            raise LocalNLIError("NLI premise is required")
        if not isinstance(hypothesis, str) or not hypothesis.strip():
            raise LocalNLIError("NLI hypothesis is required")

        artifact = self._load()
        try:
            tokenizer = artifact["tokenizer"]
            model = artifact["model"]
            # A decision must cover the entire pair, not a silently clipped
            # premise that may have lost a contradictory clause.
            encoded = tokenizer(
                premise,
                hypothesis,
                return_tensors="pt" if self.runtime == "torch" else "np",
                truncation=False,
            )
            limits = [512]
            for limit in (
                getattr(tokenizer, "model_max_length", None),
                getattr(artifact.get("config"), "max_position_embeddings", None),
            ):
                if type(limit) is int and limit > 0:
                    limits.append(limit)
            if len(encoded["input_ids"][0]) > min(limits):
                return {
                    "label": "abstention",
                    "score": 0.0,
                    "backend_id": self.backend_id,
                }
            if self.runtime == "torch":
                logits = model(**encoded).logits[0].tolist()
            else:
                feeds = {item.name: encoded[item.name] for item in model.get_inputs()}
                logits = model.run(None, feeds)[0][0].tolist()
            if len(logits) != 3 or not all(math.isfinite(float(x)) for x in logits):
                raise ValueError("invalid classifier output")
            peak = max(float(x) for x in logits)
            exponentials = [math.exp(float(x) - peak) for x in logits]
            total = sum(exponentials)
            scores = {
                self.label_mapping[str(index)]: value / total
                for index, value in enumerate(exponentials)
            }
            if set(scores) != _CANONICAL_SCORES:
                raise ValueError("invalid class mapping")
            gate = evaluate_nli(
                scores,
                EvidenceLink(
                    source_id="nli-source",
                    claim_id="nli-claim",
                    source_hash=hash_text(premise),
                    claim_hash=hash_text(hypothesis),
                ),
                thresholds=self.thresholds,
            )
        except Exception:
            raise LocalNLIError("local NLI inference failed") from None
        return {
            "label": "abstention" if gate.outcome == "abstain" else gate.outcome,
            "score": gate.selected_probability,
            "backend_id": self.backend_id,
        }


def _is_remote(value: str) -> bool:
    lower = value.strip().casefold()
    return "://" in lower or lower in _REMOTE_NAMES or lower.startswith("api.")


def resolve_nli_backend(
    backend: str | Callable[[str, str], Mapping[str, Any]] | Any = "local",
    *,
    loader: Any | None = None,
) -> Any:
    """Resolve only a local registry model, explicit heuristic, or callable."""

    if not isinstance(backend, str):
        if callable(backend) or callable(getattr(backend, "predict", None)):
            return backend
        raise LocalNLIError("NLI backend must be local or callable")
    if _is_remote(backend):
        raise RemoteNLIBackendError("remote NLI backends are prohibited")
    if backend == "heuristic":
        from openmed.clinical.nli import HEURISTIC_NLI_BACKEND

        return HEURISTIC_NLI_BACKEND
    name = get_default_nli_model() if backend == "local" else backend
    if name is None:
        raise LocalNLIError("no released local NLI checkpoint is registered")
    info = get_model_info(name)
    if (
        info is None
        or info.category != "Clinical NLI"
        or info.task not in {"text-classification", "sequence-classification"}
        or not info.released
        or not info.license
        or info.license.casefold() not in {"apache-2.0", "mit", "bsd-3-clause"}
    ):
        raise LocalNLIError("NLI model alias is not registered")
    provenance = info.provenance
    revision = provenance.get("revision")
    labels = provenance.get("nli_label_mapping")
    calibration = provenance.get("nli_calibration")
    if (
        not isinstance(revision, str)
        or re.fullmatch(r"[0-9a-fA-F]{40}", revision) is None
        or not isinstance(labels, Mapping)
        or not isinstance(calibration, Mapping)
        or not {"entailment", "contradiction", "margin", "calibration_id"}
        <= set(calibration)
    ):
        raise LocalNLIError("NLI checkpoint release metadata is incomplete")
    try:
        thresholds = NLIThresholds.from_mapping(calibration)
    except (TypeError, ValueError):
        raise LocalNLIError("NLI checkpoint calibration is invalid") from None
    return EncoderNLIBackend(
        info.model_id,
        revision=revision,
        runtime="onnx" if "onnx-int8" in info.formats else "torch",
        label_mapping=labels,
        thresholds=thresholds,
        loader=loader,
    )


__all__ = [
    "EncoderNLIBackend",
    "LocalNLIError",
    "RemoteNLIBackendError",
    "resolve_nli_backend",
]
