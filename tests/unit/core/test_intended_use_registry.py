"""Coverage and offline validation of the installed capability registry."""

import ast
import json
from importlib.resources import files

import jsonschema
import pytest

from openmed.core.intended_use import IntendedUseError, load_intended_use_registry


def _exports(package):
    source = files(package).joinpath("__init__.py").read_text()
    return next(
        ast.literal_eval(node.value)
        for node in ast.parse(source).body
        if isinstance(node, ast.Assign)
        and any(getattr(target, "id", None) == "__all__" for target in node.targets)
    )


def test_all_exported_clinical_functions_are_reviewed():
    exports = _exports("openmed.clinical")
    expected = {"openmed.clinical." + name for name in exports if name[0].islower()}
    mapped = load_intended_use_registry().entry_points
    assert expected == {
        name
        for name in mapped
        if name.startswith("openmed.clinical.") and name.count(".") == 2
    }


def test_all_declarations_match_schema_and_aliases():
    registry = load_intended_use_registry()
    schema = json.loads(
        files("openmed.core.schemas.json")
        .joinpath("intended_use.schema.json")
        .read_text()
    )
    validator = jsonschema.Draft202012Validator(schema)
    for code, declaration in registry.declarations.items():
        validator.validate(declaration.to_dict())
        assert registry.resolve(code) is declaration
        assert {"diagnosis", "treatment_selection", "autonomous_action"} <= set(
            declaration.excluded_uses
        )
    for entry_point, code in registry.entry_points.items():
        assert registry.resolve(entry_point) is registry.declarations[code]


@pytest.mark.parametrize(
    ("name", "tier"),
    [
        ("openmed.deidentify", "non_clinical"),
        ("openmed.extract_pii", "non_clinical"),
        ("openmed.analyze_text", "assistive_review"),
        ("openmed.clinical.summarize", "assistive_review"),
        ("openmed.ground", "assistive_review"),
        ("openmed.generate_text", "assistive_review"),
        ("ecg", "restricted"),
        ("waveform", "restricted"),
        ("visual_qa", "restricted"),
        ("mental_health", "restricted"),
        ("transcribe", "assistive_review"),
    ],
)
def test_reviewed_tiers(name, tier):
    assert load_intended_use_registry().resolve(name).regulatory_tier.value == tier


def test_unknown_capability_is_value_free():
    with pytest.raises(IntendedUseError, match="^unknown_capability$"):
        load_intended_use_registry().resolve("unreviewed")


def test_top_level_entry_points_are_reviewed():
    # Model discovery, configuration and instrumentation do not process clinical inputs.
    infrastructure = {
        "available_backends",
        "backend_status",
        "clear_cached_model",
        "coerce_budget",
        "disable_profiling",
        "discover_anonymizer_provider_plugins",
        "enable_profiling",
        "get_all_models",
        "get_default_pii_model",
        "get_india_clinical_model_route",
        "get_logger",
        "get_model_info",
        "get_model_max_length",
        "get_model_suggestions",
        "get_models_by_category",
        "get_ner_models_by_language",
        "get_peak_rss_bytes",
        "get_pii_models_by_language",
        "get_profile_report",
        "india_clinical_route_active",
        "is_backend_available",
        "list_attestation_profiles",
        "list_cached_models",
        "list_model_categories",
        "list_models",
        "load_attestation_template",
        "load_model",
        "load_onnx_model",
        "measure_peak_rss",
        "normalize_label",
        "prefetch_model",
        "profile",
        "register_clinical_provider",
        "register_label_generator",
        "require_backend",
        "resolve_repo_id",
        "sanitize_filename",
        "setup_logging",
        "timed",
        "validate_batch_size",
        "validate_confidence_threshold",
        "validate_input",
        "validate_model_name",
        "validate_output_format",
    }
    expected = {
        "openmed." + name
        for name in _exports("openmed")
        if name[0].islower() and name not in infrastructure
    }
    mapped = load_intended_use_registry().entry_points
    assert expected == {name for name in mapped if name.count(".") == 1}
