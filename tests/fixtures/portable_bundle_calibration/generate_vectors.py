"""Regenerate the decision-calibration bundle-extension golden vectors.

TEST-ONLY KEYS. Verification cases are signed with the TEST-ONLY keys of
``tests/fixtures/portable_bundle/generate_vectors.py``, which derive from public
labels. Never add them to a production trust policy, and never sign real
bundles with them.

Usage::

    python tests/fixtures/portable_bundle_calibration/generate_vectors.py
    python tests/fixtures/portable_bundle_calibration/generate_vectors.py --check

Every bundle file is a tiny synthetic text placeholder and every digest of
calibration data is a digest of a public label. No model weights, patient
data, calibration records, credentials, or restricted assets are involved.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openmed.core import portable_bundle as pb  # noqa: E402
from openmed.core import portable_bundle_calibration as pc  # noqa: E402

VECTORS_PATH = Path(__file__).with_name("vectors.json")
BASE_GENERATOR = ROOT / "tests" / "fixtures" / "portable_bundle" / "generate_vectors.py"
EXT = pc.DECISION_CALIBRATION_EXTENSION


def _base() -> Any:
    spec = importlib.util.spec_from_file_location(
        "portable_bundle_vector_generator", BASE_GENERATOR
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = _base()
NOW = BASE.NOW
SIGNER = "openmed-test-root-a"

TREE = {
    "LICENSE": "Synthetic decision-calibration fixture.\nLicensed Apache-2.0.\n",
    "config.json": '{"model_type":"bert","num_labels":3,"synthetic":true}\n',
    "coreml/model.mlpackage/Data/com.apple.CoreML/model.mlmodel": (
        "synthetic coreml decision graph placeholder\n"
    ),
    "coreml/model.mlpackage/Manifest.json": '{"synthetic":true}\n',
    "evidence/decision-calibration.json": (
        '{"contains_source_text":false,"slices":0,"synthetic":true}\n'
    ),
    "id2label.json": '{"0":"no","1":"unsure","2":"yes"}\n',
    "onnx/model.onnx": "synthetic fp32 onnx decision placeholder\n",
    "onnx/model_int8.onnx": "synthetic int8 onnx decision placeholder\n",
    "policy/default.json": '{"policy_version":1,"synthetic":true}\n',
    "tokenizer.json": '{"synthetic":true,"version":"1.0"}\n',
}
ROLES = {
    "LICENSE": "license",
    "config.json": "config",
    "coreml/model.mlpackage/Data/com.apple.CoreML/model.mlmodel": "graph",
    "coreml/model.mlpackage/Manifest.json": "graph",
    "evidence/decision-calibration.json": "evidence",
    "id2label.json": "labels",
    "onnx/model.onnx": "graph",
    "onnx/model_int8.onnx": "graph",
    "policy/default.json": "policy",
    "tokenizer.json": "tokenizer",
}
OPTIONS = ["no", "unsure", "yes"]
QUESTIONS = [
    {"mode": "boolean_choice", "options": ["no", "yes"], "question_id": "follow_up"},
    {
        "mode": "fixed_choice",
        "options": ["low", "medium", "high"],
        "question_id": "urgency",
    },
]
STALE_ONNX = "synthetic fp32 onnx decision placeholder, re-exported\n"


def label_digest(label: str) -> str:
    """Return the digest of a public synthetic label."""

    return "sha256:" + hashlib.sha256(label.encode("ascii")).hexdigest()


def file_entry(path: str, text: str) -> dict:
    """Return a manifest ``files`` entry for a synthetic text file."""

    data = text.encode("utf-8")
    media = "application/json" if path.endswith(".json") else "text/plain"
    if path.endswith(".onnx"):
        media = "application/onnx"
    elif path.endswith(".mlmodel"):
        media = "application/vnd.apple.coreml"
    return {
        "executable": False,
        "media_type": media,
        "path": path,
        "role": ROLES[path],
        "sha256": "sha256:" + hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
    }


def _platform(name: str, min_os: str | None) -> dict:
    return {"max_os_exclusive": None, "min_os": min_os, "platform": name}


def bare_manifest() -> dict:
    """Return the golden decision-model manifest without the extension."""

    coreml = "coreml/model.mlpackage"
    return {
        "bundle": {
            "expires_at": "2027-05-01T00:00:00Z",
            "id": "openmed.synthetic.tiny-decision",
            "issued_at": "2026-10-01T00:00:00Z",
            "sequence": 3,
            "version": "1.0.0",
        },
        "calibration": None,
        "compatibility": {
            "sdks": [
                {"max_exclusive": "5.0.0", "min": "4.0.0", "name": name}
                for name in (
                    "openmed-python",
                    "openmedkit-android",
                    "openmedkit-swift",
                    "openmedkit-web",
                )
            ]
        },
        "critical_extensions": [],
        "decoder": {"contract_version": 1, "kind": "fixed-option"},
        "evidence": [
            {
                "kind": "decision-calibration",
                "path": "evidence/decision-calibration.json",
            }
        ],
        "extensions": {},
        "files": [
            file_entry(path, TREE[path])
            for path in sorted(TREE, key=lambda item: item.encode("utf-8"))
        ],
        "labels": {"count": 3, "path": "id2label.json", "scheme": "fixed-option"},
        "licenses": [{"files": ["LICENSE"], "id": "apache-2.0"}],
        "model": {
            "architecture": "bert",
            "family": "decision",
            "id": "OpenMed/synthetic-tiny-decision",
            "languages": ["en"],
            "revision": "89abcdef0123456789abcdef0123456789abcdef",
            "task": "text-classification",
        },
        "offsets": {"contract_version": 1, "unit": "unicode_scalar"},
        "policy": {
            "max_version_exclusive": 2,
            "min_version": 1,
            "path": "policy/default.json",
        },
        "provenance": {
            "export_record": None,
            "source_digest": label_digest("synthetic-decision-source"),
        },
        "runtimes": [
            {
                "entrypoint": f"{coreml}/Manifest.json",
                "fallback": "onnx-fp32",
                "files": [
                    f"{coreml}/Data/com.apple.CoreML/model.mlmodel",
                    f"{coreml}/Manifest.json",
                    "config.json",
                ],
                "format": "coreml",
                "id": "coreml-fp16",
                "min_memory_bytes": 268435456,
                "min_storage_bytes": 1048576,
                "platforms": [_platform("ios", "17.0"), _platform("macos", "14.0")],
                "precision": "fp16",
            },
            {
                "entrypoint": "onnx/model_int8.onnx",
                "fallback": "onnx-fp32",
                "files": ["config.json", "onnx/model_int8.onnx"],
                "format": "onnx",
                "id": "onnx-int8",
                "min_memory_bytes": 134217728,
                "min_storage_bytes": 1048576,
                "platforms": [
                    _platform("android", "26"),
                    _platform("web", None),
                    _platform("linux", None),
                ],
                "precision": "int8",
            },
            {
                "entrypoint": "onnx/model.onnx",
                "fallback": None,
                "files": ["config.json", "onnx/model.onnx"],
                "format": "onnx",
                "id": "onnx-fp32",
                "min_memory_bytes": 268435456,
                "min_storage_bytes": 2097152,
                "platforms": [
                    _platform("android", "26"),
                    _platform("ios", "17.0"),
                    _platform("macos", "14.0"),
                    _platform("linux", None),
                    _platform("web", None),
                ],
                "precision": "fp32",
            },
        ],
        "schema": pb.PORTABLE_BUNDLE_SCHEMA,
        "schema_version": pb.PORTABLE_BUNDLE_SCHEMA_VERSION,
        "segmenter": None,
        "tokenizer": {
            "contract_version": 1,
            "files": ["tokenizer.json"],
            "kind": "wordpiece",
        },
    }


def golden_profiles(manifest: dict) -> list[dict]:
    """Return the four golden profiles covering every method and scope kind."""

    common = {
        "dataset_digest": label_digest("synthetic-decision-calibration-dataset"),
        "in_domain_digest": label_digest("synthetic-in-domain-slice"),
        "out_of_domain_digest": label_digest("synthetic-out-of-domain-slice"),
    }
    options = pc.options_scope_digest(OPTIONS)
    return [
        pc.build_calibration_profile(
            manifest,
            calibration_id="openmed.synthetic.triage.v1",
            version="1.0.0",
            runtime_ids=["coreml-fp16", "onnx-fp32"],
            scope_kind="options",
            scope_digest=options,
            decision_schema_min="1.0.0",
            decision_schema_max_exclusive="2.0.0",
            minimum_confidence_ppm=700000,
            minimum_margin_ppm=150000,
            multi_label_threshold_ppm=700000,
            evaluation_path="evidence/decision-calibration.json",
            **common,
        ),
        pc.build_calibration_profile(
            manifest,
            calibration_id="openmed.synthetic.triage.v1",
            version="1.1.0",
            runtime_ids=["onnx-int8"],
            scope_kind="options",
            scope_digest=options,
            decision_schema_min="1.0.0",
            decision_schema_max_exclusive="2.0.0",
            method="temperature_scaling",
            parameters={"temperature_ppm": 1250000},
            minimum_confidence_ppm=750000,
            minimum_margin_ppm=200000,
            multi_label_threshold_ppm=750000,
            **common,
        ),
        pc.build_calibration_profile(
            manifest,
            calibration_id="openmed.synthetic.screening.v1",
            version="2.0.0",
            runtime_ids=["onnx-fp32"],
            scope_kind="questions",
            scope_digest=pc.questions_scope_digest(QUESTIONS),
            decision_schema_min="1.1.0",
            decision_schema_max_exclusive="2.0.0",
            method="platt_scaling",
            parameters={"intercept_ppm": 250000, "slope_ppm": -1800000},
            minimum_confidence_ppm=650000,
            minimum_margin_ppm=100000,
            multi_label_threshold_ppm=600000,
            **common,
        ),
        pc.build_calibration_profile(
            manifest,
            calibration_id="openmed.synthetic.labels.v1",
            version="1.2.3",
            runtime_ids=["onnx-fp32", "onnx-int8"],
            scope_kind="labels",
            decision_schema_min="1.0.0",
            decision_schema_max_exclusive="2.0.0",
            method="isotonic_regression",
            parameters={
                "points": [
                    {"input_ppm": 0, "output_ppm": 0},
                    {"input_ppm": 250000, "output_ppm": 100000},
                    {"input_ppm": 500000, "output_ppm": 450000},
                    {"input_ppm": 750000, "output_ppm": 800000},
                    {"input_ppm": 1000000, "output_ppm": 1000000},
                ]
            },
            minimum_confidence_ppm=800000,
            minimum_margin_ppm=250000,
            multi_label_threshold_ppm=800000,
            **common,
        ),
    ]


def golden_manifest() -> dict:
    """Return the golden manifest with the critical calibration extension."""

    manifest = bare_manifest()
    return pc.with_decision_calibration(manifest, golden_profiles(manifest))


def text(value: Any) -> str:
    """Return canonical JSON text."""

    return pb.canonical_json_bytes(value).decode("utf-8")


def mutated(edit: Callable[[dict], None]) -> str:
    """Return canonical text of the golden manifest after ``edit``."""

    value = golden_manifest()
    edit(value)
    return text(value)


def _profiles(value: dict) -> list[dict]:
    return value["extensions"][EXT]["profiles"]


def _redigest(profile: dict) -> None:
    profile["profile_digest"] = pc.calibration_profile_digest(profile)


def profile_edit(
    index: int, edit: Callable[[dict], None], *, redigest: bool = True
) -> Callable[[dict], None]:
    """Return a manifest edit that changes one profile and re-derives its digest.

    Re-deriving the self digest isolates the check under test; the digest
    mismatch cases skip it.
    """

    def apply(value: dict) -> None:
        profile = _profiles(value)[index]
        edit(profile)
        if redigest:
            _redigest(profile)

    return apply


def _file(value: dict, path: str) -> dict:
    return next(item for item in value["files"] if item["path"] == path)


def _runtime(value: dict, runtime_id: str) -> dict:
    return next(item for item in value["runtimes"] if item["id"] == runtime_id)


def _stale_weights(value: dict) -> None:
    _file(value, "onnx/model.onnx").update(
        {
            "sha256": "sha256:" + hashlib.sha256(STALE_ONNX.encode()).hexdigest(),
            "size_bytes": len(STALE_ONNX.encode()),
        }
    )


def verified(manifest_text: str, *, files: bool = True, ok: bool = True) -> dict:
    """Return a synthetic verifier result for ``manifest_text``."""

    return {
        "content_id": pb.content_id(manifest_text.encode("utf-8")),
        "eligible_runtimes": [1, 2] if ok else [],
        "files_checked": len(TREE) if ok and files else 0,
        "files_verified": ok and files,
        "ok": ok,
        "reason": None if ok else "signature_invalid",
        "valid_signatures": 1 if ok else 0,
    }


def request(**overrides: Any) -> dict:
    """Return a fully explicit resolution request."""

    value = {
        "calibration_id": "openmed.synthetic.triage.v1",
        "decision_schema_version": "1.1.0",
        "model_id": None,
        "model_revision": None,
        "profile_digest": None,
        "runtime_digest": None,
        "runtime_id": "onnx-fp32",
        "scope_digest": None,
        "scope_kind": None,
    }
    value.update(overrides)
    return value


def build_cases() -> tuple[list[dict], list[dict], list[dict]]:
    """Return the verification, read, and resolve cases."""

    golden = text(golden_manifest())
    golden_value = golden_manifest()
    profiles = _profiles(golden_value)
    digests = [item["profile_digest"] for item in profiles]
    noncritical = mutated(lambda value: value.update(critical_extensions=[]))
    bare = text(bare_manifest())

    verification: list[dict] = []

    def verify_case(
        case_id: str,
        reason: str | None,
        manifest: str,
        *,
        understood: tuple[str, ...] = (EXT,),
        envelope: str | None = None,
        tree_set: dict[str, str] | None = None,
    ) -> None:
        verification.append(
            {
                "envelope": envelope or BASE.sign(manifest, [SIGNER]),
                "id": case_id,
                "intended_reason": reason,
                "manifest": manifest,
                "tree_set": [
                    {"path": path, "text": value}
                    for path, value in sorted((tree_set or {}).items())
                ],
                "understood_extensions": list(understood),
            }
        )

    malformed = mutated(lambda value: _profiles(value).clear())
    malformed_noncritical = mutated(
        lambda value: (_profiles(value).clear(), value.update(critical_extensions=[]))
    )
    tampered = golden.replace(
        '"minimum_confidence_ppm":700000', '"minimum_confidence_ppm":500000', 1
    )
    assert tampered != golden
    golden_envelope = BASE.sign(golden, [SIGNER])
    verify_case("understood_valid", None, golden, envelope=golden_envelope)
    verify_case(
        "critical_not_understood",
        "unknown_critical_extension",
        golden,
        understood=(),
        envelope=golden_envelope,
    )
    verify_case("noncritical_not_understood", None, noncritical, understood=())
    verify_case("noncritical_understood", None, noncritical)
    verify_case("malformed_understood", "extension_invalid", malformed)
    verify_case(
        "malformed_noncritical_not_understood",
        None,
        malformed_noncritical,
        understood=(),
    )
    verify_case(
        "malformed_noncritical_understood",
        "extension_invalid",
        malformed_noncritical,
    )
    verify_case(
        "profile_digest_mismatch_understood",
        "extension_invalid",
        mutated(
            profile_edit(
                0,
                lambda item: item["thresholds"].update(minimum_confidence_ppm=500000),
                redigest=False,
            )
        ),
    )
    verify_case(
        "model_revision_mismatch_understood",
        "extension_invalid",
        mutated(lambda value: value["model"].update(revision="f" * 40)),
    )
    verify_case(
        "stale_weights_understood",
        "extension_invalid",
        mutated(_stale_weights),
        tree_set={"onnx/model.onnx": STALE_ONNX},
    )
    verify_case(
        "critical_extension_absent",
        "manifest_malformed",
        mutated(lambda value: value["extensions"].pop(EXT)),
    )
    verify_case(
        "threshold_edited_after_signing",
        "content_id_mismatch",
        tampered,
        envelope=golden_envelope,
    )
    verify_case(
        "threshold_edited_relabelled",
        "signature_invalid",
        tampered,
        envelope=BASE.edit_envelope(
            golden_envelope,
            lambda value: value.update(
                content_id=pb.content_id(tampered.encode("utf-8"))
            ),
        ),
    )

    reads: list[dict] = []

    def read_case(
        case_id: str,
        reason: str | None,
        manifest: str,
        *,
        result: dict | None = None,
    ) -> None:
        reads.append(
            {
                "id": case_id,
                "intended_reason": reason,
                "manifest": manifest,
                "verification": result or verified(manifest),
            }
        )

    def edit_ext(edit: Callable[[dict], None]) -> str:
        return mutated(lambda value: edit(value["extensions"][EXT]))

    def edit_profile(index: int, edit: Callable[[dict], None]) -> str:
        return mutated(profile_edit(index, edit))

    def set_path(path: tuple[str, ...], item: Any) -> Callable[[dict], None]:
        def edit(profile: dict) -> None:
            target = profile
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = item

        return edit

    read_case("valid", None, golden)
    read_case("valid_noncritical", None, noncritical)
    read_case(
        "valid_files_not_verified", None, golden, result=verified(golden, files=False)
    )
    read_case(
        "verification_failed",
        "calibration_bundle_unverified",
        golden,
        result=verified(golden, ok=False),
    )
    read_case(
        "verification_for_other_manifest",
        "calibration_bundle_unverified",
        golden,
        result=verified(noncritical),
    )
    read_case(
        "manifest_not_canonical",
        "calibration_bundle_unverified",
        json.dumps(golden_manifest(), indent=1, sort_keys=True),
    )
    read_case("extension_missing", "calibration_extension_missing", bare)
    read_case(
        "extension_not_object",
        "calibration_malformed",
        mutated(lambda value: value["extensions"].update({EXT: []})),
    )
    read_case(
        "version_missing",
        "calibration_malformed",
        edit_ext(lambda ext: ext.pop("schema_version")),
    )
    read_case(
        "version_string",
        "calibration_malformed",
        edit_ext(lambda ext: ext.update(schema_version="1")),
    )
    read_case(
        "version_unsupported",
        "calibration_version_unsupported",
        edit_ext(lambda ext: ext.update(schema_version=2)),
    )
    read_case(
        "extension_unknown_member",
        "calibration_malformed",
        edit_ext(lambda ext: ext.update(default_profile="openmed.synthetic.triage.v1")),
    )
    read_case(
        "profiles_empty",
        "calibration_malformed",
        edit_ext(lambda ext: ext.update(profiles=[])),
    )
    read_case(
        "profiles_not_array",
        "calibration_malformed",
        edit_ext(lambda ext: ext.update(profiles={})),
    )
    read_case(
        "profiles_too_many",
        "calibration_malformed",
        edit_ext(
            lambda ext: ext.update(
                profiles=[copy.deepcopy(ext["profiles"][0]) for _ in range(17)]
            )
        ),
    )
    read_case(
        "profile_not_object",
        "calibration_malformed",
        edit_ext(lambda ext: ext["profiles"].append("openmed.synthetic.triage.v1")),
    )
    read_case(
        "profile_unknown_field",
        "calibration_malformed",
        edit_profile(0, lambda item: item.update(notes="synthetic")),
    )
    read_case(
        "profile_missing_field",
        "calibration_malformed",
        edit_profile(0, lambda item: item.pop("scope")),
    )
    read_case(
        "calibration_id_invalid",
        "calibration_malformed",
        edit_profile(0, set_path(("calibration_id",), "Triage")),
    )
    read_case(
        "version_not_semver",
        "calibration_malformed",
        edit_profile(0, set_path(("version",), "1.0")),
    )
    read_case(
        "method_unknown",
        "calibration_malformed",
        edit_profile(0, set_path(("method",), "histogram_binning")),
    )
    read_case(
        "parameters_for_other_method",
        "calibration_malformed",
        edit_profile(0, set_path(("parameters",), {"temperature_ppm": 1000000})),
    )
    read_case(
        "temperature_zero",
        "calibration_malformed",
        edit_profile(1, set_path(("parameters", "temperature_ppm"), 0)),
    )
    read_case(
        "temperature_too_large",
        "calibration_malformed",
        edit_profile(1, set_path(("parameters", "temperature_ppm"), 100000001)),
    )
    read_case(
        "platt_slope_zero",
        "calibration_malformed",
        edit_profile(2, set_path(("parameters", "slope_ppm"), 0)),
    )
    read_case(
        "platt_intercept_out_of_range",
        "calibration_malformed",
        edit_profile(2, set_path(("parameters", "intercept_ppm"), -100000001)),
    )
    read_case(
        "isotonic_single_point",
        "calibration_malformed",
        edit_profile(
            3,
            set_path(("parameters", "points"), [{"input_ppm": 0, "output_ppm": 0}]),
        ),
    )
    read_case(
        "isotonic_inputs_not_increasing",
        "calibration_malformed",
        edit_profile(
            3,
            lambda item: item["parameters"]["points"][2].update(input_ppm=250000),
        ),
    )
    read_case(
        "isotonic_outputs_decreasing",
        "calibration_malformed",
        edit_profile(
            3,
            lambda item: item["parameters"]["points"][2].update(output_ppm=50000),
        ),
    )
    read_case(
        "isotonic_point_out_of_range",
        "calibration_malformed",
        edit_profile(
            3,
            lambda item: item["parameters"]["points"][4].update(output_ppm=1000001),
        ),
    )
    read_case(
        "threshold_above_scale",
        "calibration_malformed",
        edit_profile(0, set_path(("thresholds", "multi_label_threshold_ppm"), 1000001)),
    )
    read_case(
        "threshold_negative",
        "calibration_malformed",
        edit_profile(0, set_path(("thresholds", "minimum_margin_ppm"), -1)),
    )
    read_case(
        "threshold_boolean",
        "calibration_malformed",
        edit_profile(0, set_path(("thresholds", "minimum_confidence_ppm"), True)),
    )
    read_case(
        "margin_exceeds_confidence",
        "calibration_malformed",
        edit_profile(0, set_path(("thresholds", "minimum_margin_ppm"), 700001)),
    )
    read_case(
        "scope_kind_unknown",
        "calibration_malformed",
        edit_profile(0, set_path(("scope", "kind"), "labelset")),
    )
    read_case(
        "scope_digest_invalid",
        "calibration_malformed",
        edit_profile(0, set_path(("scope", "digest"), "sha256:XYZ")),
    )
    read_case(
        "model_revision_invalid",
        "calibration_malformed",
        edit_profile(0, set_path(("model", "revision"), "89ABCDEF" * 5)),
    )
    read_case(
        "model_runtimes_empty",
        "calibration_malformed",
        edit_profile(0, set_path(("model", "runtimes"), [])),
    )
    read_case(
        "model_runtime_duplicate",
        "calibration_malformed",
        edit_profile(
            0,
            lambda item: item["model"]["runtimes"].append(
                dict(item["model"]["runtimes"][0])
            ),
        ),
    )
    read_case(
        "decision_schema_inverted",
        "calibration_malformed",
        edit_profile(0, set_path(("decision_schema", "max_exclusive"), "0.9.0")),
    )
    read_case(
        "decision_schema_empty_range",
        "calibration_malformed",
        edit_profile(0, set_path(("decision_schema", "max_exclusive"), "1.0.0")),
    )
    read_case(
        "dataset_digest_invalid",
        "calibration_malformed",
        edit_profile(0, set_path(("provenance", "dataset_digest"), "sha256:00")),
    )
    read_case(
        "evaluation_path_empty",
        "calibration_malformed",
        edit_profile(0, set_path(("provenance", "evaluation", "path"), "")),
    )
    read_case(
        "evaluation_path_outside_bundle",
        "calibration_malformed",
        edit_profile(
            0, set_path(("provenance", "evaluation", "path"), "../outside.json")
        ),
    )
    read_case(
        "binding_duplicated_across_profiles",
        "calibration_malformed",
        edit_profile(
            1,
            lambda item: item["model"]["runtimes"].append(
                dict(profiles[0]["model"]["runtimes"][1])
            ),
        ),
    )
    read_case(
        "profile_digest_mismatch",
        "calibration_digest_mismatch",
        mutated(
            profile_edit(
                2,
                lambda item: item["thresholds"].update(minimum_confidence_ppm=900000),
                redigest=False,
            )
        ),
    )
    read_case(
        "model_id_mismatch",
        "calibration_model_mismatch",
        edit_profile(0, set_path(("model", "id"), "OpenMed/synthetic-other-model")),
    )
    read_case(
        "model_revision_mismatch",
        "calibration_model_mismatch",
        mutated(lambda value: value["model"].update(revision="f" * 40)),
    )
    read_case(
        "model_revision_unpinned",
        "calibration_model_mismatch",
        edit_profile(3, set_path(("model", "revision"), None)),
    )
    read_case(
        "runtime_undeclared",
        "calibration_runtime_mismatch",
        edit_profile(
            1,
            lambda item: item["model"]["runtimes"][0].update(id="tflite-int8"),
        ),
    )
    read_case(
        "runtime_weights_stale", "calibration_runtime_mismatch", mutated(_stale_weights)
    )
    read_case(
        "runtime_precision_changed",
        "calibration_runtime_mismatch",
        mutated(lambda value: _runtime(value, "onnx-int8").update(precision="int4")),
    )
    read_case(
        "labels_scope_mismatch",
        "calibration_scope_mismatch",
        edit_profile(3, set_path(("scope", "digest"), label_digest("other-labels"))),
    )
    read_case(
        "labels_scope_without_label_map",
        "calibration_scope_mismatch",
        mutated(lambda value: value.update(labels=None)),
    )
    read_case(
        "evaluation_undeclared",
        "calibration_evidence_mismatch",
        edit_profile(
            0, set_path(("provenance", "evaluation", "path"), "evidence/missing.json")
        ),
    )
    read_case(
        "evaluation_digest_mismatch",
        "calibration_evidence_mismatch",
        edit_profile(
            0,
            set_path(
                ("provenance", "evaluation", "sha256"), label_digest("other-report")
            ),
        ),
    )

    resolves: list[dict] = []

    def resolve_case(
        case_id: str,
        reason: str | None,
        req: dict,
        *,
        manifest: str = golden,
        result: dict | None = None,
        selects: int | None = None,
    ) -> None:
        resolves.append(
            {
                "id": case_id,
                "intended_reason": reason,
                "intended_profile": selects,
                "manifest": manifest,
                "request": req,
                "verification": result or verified(manifest),
            }
        )

    fp32 = pc.calibration_runtime_digest(golden_value, "onnx-fp32")
    coreml = pc.calibration_runtime_digest(golden_value, "coreml-fp16")
    int8 = pc.calibration_runtime_digest(golden_value, "onnx-int8")
    labels = _file(golden_value, "id2label.json")["sha256"]
    model = golden_value["model"]
    resolve_case("triage_fp32", None, request(), selects=0)
    resolve_case(
        "triage_coreml",
        None,
        request(runtime_id="coreml-fp16", runtime_digest=coreml),
        selects=0,
    )
    resolve_case(
        "triage_int8_quantized_profile",
        None,
        request(runtime_id="onnx-int8"),
        selects=1,
    )
    resolve_case(
        "screening_questions",
        None,
        request(
            calibration_id="openmed.synthetic.screening.v1",
            scope_kind="questions",
            scope_digest=pc.questions_scope_digest(QUESTIONS),
        ),
        selects=2,
    )
    resolve_case(
        "labels_fully_pinned",
        None,
        request(
            calibration_id="openmed.synthetic.labels.v1",
            runtime_id="onnx-int8",
            model_id=model["id"],
            model_revision=model["revision"],
            runtime_digest=int8,
            scope_kind="labels",
            scope_digest=labels,
            profile_digest=digests[3],
            decision_schema_version="1.0.0",
        ),
        selects=3,
    )
    resolve_case(
        "files_not_verified",
        "calibration_bundle_unverified",
        request(),
        result=verified(golden, files=False),
    )
    resolve_case(
        "verification_failed",
        "calibration_bundle_unverified",
        request(),
        result=verified(golden, ok=False),
    )
    resolve_case(
        "extension_missing",
        "calibration_extension_missing",
        request(),
        manifest=bare,
    )
    resolve_case(
        "extension_digest_mismatch",
        "calibration_digest_mismatch",
        request(),
        manifest=mutated(
            profile_edit(
                0,
                lambda item: item["thresholds"].update(minimum_confidence_ppm=500000),
                redigest=False,
            )
        ),
    )
    resolve_case(
        "profile_missing",
        "calibration_profile_missing",
        request(calibration_id="openmed.synthetic.unknown.v1"),
    )
    resolve_case(
        "runtime_unbound",
        "calibration_runtime_unbound",
        request(
            calibration_id="openmed.synthetic.screening.v1", runtime_id="onnx-int8"
        ),
    )
    resolve_case(
        "runtime_not_declared",
        "calibration_runtime_unbound",
        request(runtime_id="tflite-int8"),
    )
    resolve_case(
        "loaded_model_id_mismatch",
        "calibration_model_mismatch",
        request(model_id="OpenMed/synthetic-other-model"),
    )
    resolve_case(
        "loaded_model_revision_mismatch",
        "calibration_model_mismatch",
        request(model_revision="f" * 40),
    )
    resolve_case(
        "loaded_runtime_digest_mismatch",
        "calibration_model_mismatch",
        request(runtime_digest=coreml),
    )
    resolve_case(
        "decision_schema_below_range",
        "calibration_schema_incompatible",
        request(
            calibration_id="openmed.synthetic.screening.v1",
            decision_schema_version="1.0.9",
        ),
    )
    resolve_case(
        "decision_schema_at_max",
        "calibration_schema_incompatible",
        request(decision_schema_version="2.0.0"),
    )
    resolve_case(
        "decision_schema_numeric_order",
        "calibration_schema_incompatible",
        request(decision_schema_version="10.0.0"),
    )
    resolve_case(
        "scope_digest_mismatch",
        "calibration_scope_mismatch",
        request(
            scope_kind="options",
            scope_digest=pc.options_scope_digest(["yes", "no"]),
        ),
    )
    resolve_case(
        "scope_kind_mismatch",
        "calibration_scope_mismatch",
        request(scope_kind="questions", scope_digest=profiles[0]["scope"]["digest"]),
    )
    resolve_case(
        "pin_mismatch",
        "calibration_pin_mismatch",
        request(profile_digest=digests[1]),
    )
    resolve_case(
        "pin_match",
        None,
        request(
            profile_digest=digests[0],
            scope_kind="options",
            scope_digest=pc.options_scope_digest(OPTIONS),
            runtime_digest=fp32,
        ),
        selects=0,
    )
    for case_id, overrides in (
        ("request_calibration_id_invalid", {"calibration_id": "Triage"}),
        ("request_runtime_id_invalid", {"runtime_id": "ONNX"}),
        ("request_schema_version_invalid", {"decision_schema_version": "1.1"}),
        ("request_model_revision_invalid", {"model_revision": "main"}),
        ("request_scope_without_digest", {"scope_kind": "options"}),
        (
            "request_scope_kind_invalid",
            {"scope_kind": "labelset", "scope_digest": labels},
        ),
        ("request_profile_digest_invalid", {"profile_digest": "sha256:00"}),
    ):
        resolve_case(case_id, "calibration_request_invalid", request(**overrides))
    return verification, reads, resolves


def _reference_verify(item: dict) -> dict:
    files = {path: value.encode("utf-8") for path, value in TREE.items()}
    for entry in item["tree_set"]:
        files[entry["path"]] = entry["text"].encode("utf-8")
    result = pb.verify_portable_bundle(
        item["manifest"].encode("utf-8"),
        item["envelope"].encode("utf-8"),
        trust_policy=pb.TrustPolicy.from_mapping(BASE.trust_policies()["default"]),
        context=pb.VerificationContext(
            now=NOW,
            target=pb.VerificationTarget(**BASE.ANDROID),
            understood_extensions=frozenset(item["understood_extensions"]),
        ),
        files=files,
    )
    payload = result.to_dict()
    payload.pop("schema")
    payload.pop("schema_version")
    return payload


def _verification(raw: dict) -> pb.PortableBundleVerification:
    return pb.PortableBundleVerification(
        ok=raw["ok"],
        reason=raw["reason"],
        content_id=raw["content_id"],
        valid_signatures=raw["valid_signatures"],
        eligible_runtimes=tuple(raw["eligible_runtimes"]),
        files_checked=raw["files_checked"],
        files_verified=raw["files_verified"],
    )


def _profile_texts(profiles: tuple[pc.BundleCalibrationProfile, ...]) -> list[str]:
    return [item.canonical_bytes().decode("utf-8") for item in profiles]


def build_vectors() -> dict:
    """Return the complete vector document."""

    golden_value = golden_manifest()
    golden = text(golden_value)
    if not golden.isascii():
        raise SystemExit("golden manifest must be ASCII so splices are byte offsets")
    manifests: dict[str, dict] = {"golden": {"text": golden}}
    keys = {golden: "golden"}

    def manifest_key(case_id: str, manifest_text: str) -> str:
        if not manifest_text.isascii():
            raise SystemExit(f"case {case_id}: manifest must be ASCII")
        if manifest_text not in keys:
            keys[manifest_text] = case_id
            manifests[case_id] = BASE.splice(golden, manifest_text)
        return keys[manifest_text]

    verification, reads, resolves = build_cases()
    verification_cases = []
    for item in verification:
        expected = _reference_verify(item)
        intended = item.pop("intended_reason")
        if expected["reason"] != intended:
            raise SystemExit(
                f"verification {item['id']}: expected {intended}, "
                f"got {expected['reason']}"
            )
        item["manifest"] = manifest_key(item["id"], item["manifest"])
        item["expected"] = expected
        verification_cases.append(item)

    read_cases = []
    for item in reads:
        manifest_bytes = item["manifest"].encode("utf-8")
        result = pc.read_decision_calibration(
            manifest_bytes, _verification(item["verification"])
        )
        intended = item.pop("intended_reason")
        if result.reason != intended:
            raise SystemExit(
                f"read {item['id']}: expected {intended}, got {result.reason}"
            )
        item["manifest"] = manifest_key(item["id"], item["manifest"])
        item["expected"] = {
            "content_id": result.content_id,
            "ok": result.ok,
            "profiles": _profile_texts(result.profiles),
            "reason": result.reason,
        }
        read_cases.append(item)

    resolve_cases = []
    for item in resolves:
        intended = item.pop("intended_reason")
        selects = item.pop("intended_profile")
        try:
            req = pc.DecisionCalibrationRequest(**item["request"])
        except pc.DecisionCalibrationError as error:
            if error.reason.value != intended:
                raise SystemExit(f"resolve {item['id']}: unexpected request error")
            expected: dict[str, Any] = {"error": error.reason.value}
        else:
            selection = pc.resolve_decision_calibration(
                item["manifest"].encode("utf-8"),
                _verification(item["verification"]),
                req,
            )
            if selection.reason != intended:
                raise SystemExit(
                    f"resolve {item['id']}: expected {intended}, got {selection.reason}"
                )
            profile = selection.profile
            if (selects is None) != (profile is None) or (
                profile is not None
                and profile.profile_digest
                != _profiles(golden_value)[selects]["profile_digest"]
            ):
                raise SystemExit(f"resolve {item['id']}: wrong profile selected")
            expected = {
                "content_id": selection.content_id,
                "ok": selection.ok,
                "profile_digest": None if profile is None else profile.profile_digest,
                "reason": selection.reason,
            }
        item["manifest"] = manifest_key(item["id"], item["manifest"])
        item["expected"] = expected
        resolve_cases.append(item)

    golden_profiles_read = pc.validate_decision_calibration_extension(golden_value)
    return {
        "extension": EXT,
        "extension_version": pc.DECISION_CALIBRATION_EXTENSION_VERSION,
        "golden": {
            "content_id": pb.content_id(golden.encode("utf-8")),
            "manifest": "golden",
            "profile_digests": [item.profile_digest for item in golden_profiles_read],
            "runtime_digests": {
                runtime["id"]: pc.calibration_runtime_digest(
                    golden_value, runtime["id"]
                )
                for runtime in golden_value["runtimes"]
            },
        },
        "manifests": manifests,
        "notice": (
            "TEST-ONLY vectors. Verification cases are signed with the TEST-ONLY "
            "keys of tests/fixtures/portable_bundle; never trust them in production."
        ),
        "now": NOW,
        "read_cases": read_cases,
        "reason_codes": list(pc.REASON_CODES),
        "resolve_cases": resolve_cases,
        "schema": "openmed.portable_bundle.decision_calibration.vectors",
        "synthetic_only": True,
        "target": copy.deepcopy(BASE.ANDROID),
        "test_only_keys": True,
        "tree": [
            {"path": path, "text": TREE[path]}
            for path in sorted(TREE, key=lambda item: item.encode("utf-8"))
        ],
        "trust_policy": BASE.trust_policies()["default"],
        "verification_cases": verification_cases,
        "version": 1,
    }


def render() -> str:
    """Return the serialized vector document."""

    return json.dumps(build_vectors(), indent=2, sort_keys=True) + "\n"


def main() -> int:
    """Write or check ``vectors.json``."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rendered = render()
    if args.check:
        if VECTORS_PATH.read_text(encoding="utf-8") != rendered:
            print("decision calibration vectors are stale")
            return 1
        return 0
    VECTORS_PATH.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
