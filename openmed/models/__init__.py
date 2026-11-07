"""Model lifecycle helpers."""

from importlib import import_module
from typing import Any

__all__ = [
    "BootstrapReport",
    "ClinicalSLMCapabilityCheck",
    "ClinicalSLMCapabilityError",
    "ClinicalSLMCapabilityReport",
    "DiagnosticCategory",
    "bootstrap_check",
    "check_clinical_slm_capabilities",
    "check_bootstrap",
    "format_human",
    "probe_clinical_slm_capabilities",
    "render_json",
    "run_bootstrap_check",
]


def __getattr__(name: str) -> Any:
    """Load bootstrap helpers lazily, including for ``python -m`` execution."""

    exports = {
        "BootstrapReport",
        "ClinicalSLMCapabilityCheck",
        "ClinicalSLMCapabilityError",
        "ClinicalSLMCapabilityReport",
        "DiagnosticCategory",
        "check_clinical_slm_capabilities",
        "check_bootstrap",
        "format_human",
        "probe_clinical_slm_capabilities",
        "render_json",
        "run_bootstrap_check",
    }
    if name == "bootstrap_check" or name in exports:
        module_name = (
            ".clinical_slm_capabilities"
            if name
            in {
                "ClinicalSLMCapabilityCheck",
                "ClinicalSLMCapabilityError",
                "ClinicalSLMCapabilityReport",
                "check_clinical_slm_capabilities",
                "probe_clinical_slm_capabilities",
            }
            else ".bootstrap_check"
        )
        module = import_module(module_name, __name__)
        if name == "bootstrap_check":
            return module
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
