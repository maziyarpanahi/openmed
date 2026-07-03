"""OpenVINO inference session helpers for token-classification graphs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

OPENVINO_DEVICE_FALLBACK_ORDER = ("CPU", "GPU", "NPU")


@dataclass(frozen=True)
class OpenVinoDeviceSelection:
    """Deterministic OpenVINO device resolution evidence."""

    requested_device: str
    selected_device: str
    available_devices: tuple[str, ...]
    fallback_order: tuple[str, ...]
    fallback_used: bool

    def to_metadata(self) -> dict[str, Any]:
        """Return JSON-serializable device-selection metadata."""

        return {
            "requested_device": self.requested_device,
            "selected_device": self.selected_device,
            "available_devices": list(self.available_devices),
            "fallback_order": list(self.fallback_order),
            "fallback_used": self.fallback_used,
        }


class OpenVinoTokenClassificationSession:
    """OpenVINO compiled-model wrapper returning token-classification logits."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "CPU",
        fallback_order: Sequence[str] = OPENVINO_DEVICE_FALLBACK_ORDER,
        core: Any | None = None,
        compile_config: Mapping[str, Any] | None = None,
    ) -> None:
        if core is None:
            core = _openvino_core()

        self.model_path = Path(model_path)
        self.core = core
        self.selection = resolve_openvino_device(
            device,
            tuple(getattr(core, "available_devices", ())),
            fallback_order=fallback_order,
        )
        self.model = core.read_model(str(self.model_path))
        self.compiled_model = _compile_model(
            core,
            self.model,
            self.selection.selected_device,
            compile_config=compile_config,
        )

    @property
    def selected_device(self) -> str:
        """Return the OpenVINO device used for compilation."""

        return self.selection.selected_device

    def run(
        self,
        *,
        input_ids: Any,
        attention_mask: Any,
        token_type_ids: Any | None = None,
    ) -> Any:
        """Run one token-classification batch and return the logits array."""

        inputs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            inputs["token_type_ids"] = token_type_ids

        result = self.compiled_model(inputs)
        return _extract_logits(result)


def resolve_openvino_device(
    requested_device: str,
    available_devices: Sequence[str],
    *,
    fallback_order: Sequence[str] = OPENVINO_DEVICE_FALLBACK_ORDER,
) -> OpenVinoDeviceSelection:
    """Resolve *requested_device* against available devices deterministically."""

    requested = _normalize_device(requested_device or "CPU")
    available = _dedupe_devices(available_devices)
    fallback = tuple(_normalize_device(device) for device in fallback_order)
    if not available:
        raise RuntimeError("OpenVINO reported no available devices")

    direct = _find_available_device(requested, available)
    if direct is not None:
        return OpenVinoDeviceSelection(
            requested_device=requested,
            selected_device=direct,
            available_devices=available,
            fallback_order=fallback,
            fallback_used=False,
        )

    for fallback_device in fallback:
        selected = _find_available_device(fallback_device, available)
        if selected is not None:
            return OpenVinoDeviceSelection(
                requested_device=requested,
                selected_device=selected,
                available_devices=available,
                fallback_order=fallback,
                fallback_used=True,
            )

    return OpenVinoDeviceSelection(
        requested_device=requested,
        selected_device=sorted(available)[0],
        available_devices=available,
        fallback_order=fallback,
        fallback_used=True,
    )


def _openvino_core() -> Any:
    try:
        from openvino import Core
    except ImportError:
        try:
            from openvino.runtime import Core
        except ImportError as exc:
            raise ImportError(
                "OpenVINO runtime is required for OpenVINO inference. "
                "Install with: pip install openmed[openvino]"
            ) from exc
    return Core()


def _compile_model(
    core: Any,
    model: Any,
    device: str,
    *,
    compile_config: Mapping[str, Any] | None,
) -> Any:
    if compile_config:
        try:
            return core.compile_model(model, device, config=dict(compile_config))
        except TypeError:
            return core.compile_model(model, device, dict(compile_config))
    return core.compile_model(model, device)


def _extract_logits(result: Any) -> Any:
    if isinstance(result, Mapping):
        if "logits" in result:
            return result["logits"]
        try:
            return next(iter(result.values()))
        except StopIteration as exc:
            raise RuntimeError("OpenVINO inference returned no outputs") from exc

    if isinstance(result, Sequence) and not isinstance(result, (str, bytes)):
        if not result:
            raise RuntimeError("OpenVINO inference returned no outputs")
        return result[0]

    return result


def _dedupe_devices(devices: Sequence[str]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for device in devices:
        normalized = _normalize_device(device)
        if normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return tuple(result)


def _find_available_device(requested: str, available: Sequence[str]) -> str | None:
    for candidate in available:
        if candidate == requested or candidate.split(".", 1)[0] == requested:
            return candidate
    return None


def _normalize_device(device: str) -> str:
    return str(device).strip().upper()


__all__ = [
    "OPENVINO_DEVICE_FALLBACK_ORDER",
    "OpenVinoDeviceSelection",
    "OpenVinoTokenClassificationSession",
    "resolve_openvino_device",
]
