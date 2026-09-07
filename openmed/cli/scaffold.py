"""Deterministic, offline project scaffolding for the OpenMed CLI."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent
from typing import Mapping

PERSONA_PRESETS = ("researcher", "app-developer", "data-engineer")
"""Persona names accepted by :func:`scaffold_project`."""

MANAGED_FILES = (
    "openmed.toml",
    "pipeline.py",
    ".env.example",
    ".gitignore",
    "README.md",
)
"""Files owned by a generated OpenMed project scaffold."""


class ScaffoldError(ValueError):
    """Base error for invalid or unsafe scaffold operations."""


class ScaffoldConflictError(ScaffoldError):
    """Raised when existing managed files differ from rendered templates."""

    def __init__(self, paths: tuple[str, ...]) -> None:
        self.paths = paths
        rendered = ", ".join(paths)
        super().__init__(
            "Refusing to overwrite existing scaffold files: "
            f"{rendered}. Re-run with --force to replace only these files."
        )


@dataclass(frozen=True)
class ScaffoldResult:
    """Summary of one project-scaffold operation.

    Attributes:
        destination: Project directory supplied by the caller.
        preset: Persona preset used to render the files.
        created: Relative paths created by this operation.
        overwritten: Relative paths replaced after explicit force approval.
        unchanged: Relative paths already matching the rendered scaffold.
    """

    destination: Path
    preset: str
    created: tuple[str, ...]
    overwritten: tuple[str, ...]
    unchanged: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-compatible operation summary.

        Returns:
            Destination, preset, and status-grouped relative paths.
        """
        return {
            "destination": str(self.destination),
            "preset": self.preset,
            "created": list(self.created),
            "overwritten": list(self.overwritten),
            "unchanged": list(self.unchanged),
        }


@dataclass(frozen=True)
class _Preset:
    title: str
    policy: str
    config: Mapping[str, object]
    pipeline: str


_RESEARCHER_PIPELINE = '''\
"""Researcher starter using synthetic input only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from openmed import deidentify
from openmed.core.config import load_config_from_file


PROJECT_ROOT = Path(__file__).resolve().parent
PRESET = "researcher"
POLICY = "research_limited_dataset"
SYNTHETIC_NOTES = (
    "SYNTHETIC_PATIENT has record ID DEMO-0001 and phone 555-0100.",
)


def main(argv: Sequence[str] | None = None) -> int:
    """Validate the project or de-identify the bundled synthetic note."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate local configuration without loading a model.",
    )
    args = parser.parse_args(argv)
    config = load_config_from_file(PROJECT_ROOT / "openmed.toml")

    if args.check:
        print(
            json.dumps(
                {
                    "local_only": config.local_only,
                    "policy": POLICY,
                    "preset": PRESET,
                    "synthetic_records": len(SYNTHETIC_NOTES),
                },
                sort_keys=True,
            )
        )
        return 0

    result = deidentify(
        SYNTHETIC_NOTES[0],
        method="mask",
        config=config,
        policy=POLICY,
    )
    print(result.deidentified_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''

_APP_DEVELOPER_PIPELINE = '''\
"""Application-developer starter using a synthetic request."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from openmed import deidentify
from openmed.core.config import load_config_from_file


PROJECT_ROOT = Path(__file__).resolve().parent
PRESET = "app-developer"
POLICY = "strict_no_leak"
SYNTHETIC_REQUEST = {
    "text": "Contact SYNTHETIC_USER at demo.user@example.invalid or 555-0101.",
    "method": "mask",
}


def deidentify_request(
    payload: Mapping[str, Any],
    *,
    config: Any,
) -> dict[str, object]:
    """Adapt a request mapping to OpenMed's local de-identification API."""
    result = deidentify(
        str(payload["text"]),
        method=str(payload.get("method", "mask")),
        config=config,
        policy=POLICY,
    )
    return {
        "deidentified_text": result.deidentified_text,
        "entity_count": len(result.pii_entities),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Validate the project or process the bundled synthetic request."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate local configuration without loading a model.",
    )
    args = parser.parse_args(argv)
    config = load_config_from_file(PROJECT_ROOT / "openmed.toml")

    if args.check:
        print(
            json.dumps(
                {
                    "local_only": config.local_only,
                    "policy": POLICY,
                    "preset": PRESET,
                    "synthetic_records": 1,
                },
                sort_keys=True,
            )
        )
        return 0

    response = deidentify_request(SYNTHETIC_REQUEST, config=config)
    print(json.dumps(response, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''

_DATA_ENGINEER_PIPELINE = '''\
"""Data-engineer starter using a synthetic in-memory batch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from openmed import BatchProcessor
from openmed.core.config import load_config_from_file


PROJECT_ROOT = Path(__file__).resolve().parent
PRESET = "data-engineer"
POLICY = "strict_no_leak"
SYNTHETIC_NOTES = (
    "SYNTHETIC_PATIENT has record ID DEMO-1001 and phone 555-0102.",
    "Email SYNTHETIC_USER at batch.user@example.invalid.",
)


def main(argv: Sequence[str] | None = None) -> int:
    """Validate the project or process the bundled synthetic batch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate local configuration without loading a model.",
    )
    args = parser.parse_args(argv)
    config = load_config_from_file(PROJECT_ROOT / "openmed.toml")

    if args.check:
        print(
            json.dumps(
                {
                    "local_only": config.local_only,
                    "policy": POLICY,
                    "preset": PRESET,
                    "synthetic_records": len(SYNTHETIC_NOTES),
                },
                sort_keys=True,
            )
        )
        return 0

    processor = BatchProcessor(
        operation="deidentify",
        model_name="pii_detection",
        batch_size=config.batch_size or 8,
        config=config,
        method="mask",
        policy=POLICY,
    )
    result = processor.process_texts(SYNTHETIC_NOTES)
    for item in result.get_successful_results():
        print(item.result.deidentified_text)
    print(f"Processed {result.successful_items}/{result.total_items} synthetic records")
    return 0 if result.failed_items == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
'''

_PRESETS: dict[str, _Preset] = {
    "researcher": _Preset(
        title="Researcher",
        policy="research_limited_dataset",
        config={
            "device": "cpu",
            "local_only": True,
            "log_level": "WARNING",
            "timeout": 300,
            "use_medical_tokenizer": True,
        },
        pipeline=_RESEARCHER_PIPELINE,
    ),
    "app-developer": _Preset(
        title="Application developer",
        policy="strict_no_leak",
        config={
            "device": "cpu",
            "lazy_model_loading": True,
            "local_only": True,
            "log_level": "WARNING",
            "timeout": 300,
            "use_medical_tokenizer": True,
        },
        pipeline=_APP_DEVELOPER_PIPELINE,
    ),
    "data-engineer": _Preset(
        title="Data engineer",
        policy="strict_no_leak",
        config={
            "batch_size": 16,
            "device": "cpu",
            "lazy_model_loading": True,
            "local_only": True,
            "log_level": "WARNING",
            "num_workers": 1,
            "timeout": 900,
            "use_medical_tokenizer": True,
        },
        pipeline=_DATA_ENGINEER_PIPELINE,
    ),
}

_ENV_TEMPLATE = """\
# Offline defaults only. Never store credentials in this committed template.
OPENMED_CONFIG=./openmed.toml
OPENMED_OFFLINE=1
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
"""

_GITIGNORE_TEMPLATE = """\
.env
.venv/
__pycache__/
*.py[cod]
outputs/
"""


def _config_schema_path() -> Path:
    return Path(__file__).resolve().parents[1] / "core" / "config.schema.json"


def _json_type_matches(value: object, expected: str) -> bool:
    if expected == "null":
        return value is None
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "string":
        return isinstance(value, str)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, Mapping)
    return False


def _validate_config_against_schema(config: Mapping[str, object]) -> None:
    """Validate a rendered preset against the bundled schema subset it uses."""
    try:
        schema = json.loads(_config_schema_path().read_text(encoding="utf-8"))
        properties = schema["properties"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ScaffoldError("The bundled OpenMedConfig schema is invalid.") from exc

    if not isinstance(properties, Mapping):
        raise ScaffoldError("The bundled OpenMedConfig schema is invalid.")

    errors: list[str] = []
    if schema.get("additionalProperties") is False:
        errors.extend(sorted(set(config) - set(properties)))

    for key, value in config.items():
        field_schema = properties.get(key)
        if not isinstance(field_schema, Mapping):
            continue
        expected = field_schema.get("type")
        expected_types = (expected,) if isinstance(expected, str) else expected
        if not isinstance(expected_types, list | tuple) or not any(
            isinstance(item, str) and _json_type_matches(value, item)
            for item in expected_types
        ):
            errors.append(key)
            continue
        if "enum" in field_schema and value not in field_schema["enum"]:
            errors.append(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if "minimum" in field_schema and value < field_schema["minimum"]:
                errors.append(key)
            if "maximum" in field_schema and value > field_schema["maximum"]:
                errors.append(key)
        if isinstance(value, str) and len(value) < field_schema.get("minLength", 0):
            errors.append(key)

    if errors:
        fields = ", ".join(sorted(set(errors)))
        raise ScaffoldError(
            "The bundled project configuration does not satisfy "
            f"config.schema.json: {fields}."
        )


def _toml_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=True)
    raise ScaffoldError("A scaffold template contains an unsupported config value.")


def _config_template(preset_name: str, preset: _Preset) -> str:
    _validate_config_against_schema(preset.config)
    lines = [
        "# OpenMed project configuration",
        f"# Persona preset: {preset_name}",
        "# This file intentionally contains no credentials or patient data.",
        "",
    ]
    lines.extend(
        f"{key} = {_toml_value(value)}" for key, value in preset.config.items()
    )
    return "\n".join(lines) + "\n"


def _readme_template(preset_name: str, preset: _Preset) -> str:
    return dedent(
        f"""\
        # OpenMed {preset.title} starter

        This deterministic scaffold uses the `{preset_name}` persona preset and
        the `{preset.policy}` policy. All bundled inputs are explicitly synthetic.
        The scaffold contains no credentials, patient data, telemetry, or model
        downloads.

        ## Files

        - `openmed.toml` — cache-only OpenMedConfig values validated against the
          bundled JSON Schema.
        - `pipeline.py` — a persona-specific starter over synthetic input.
        - `.env.example` — offline environment defaults with no secret fields.
        - `.gitignore` — excludes local environments, outputs, and `.env`.

        ## Validate locally

        Install OpenMed, then run the model-free configuration check:

        ```bash
        python pipeline.py --check
        ```

        This check does not load or download a model. To run the actual synthetic
        pipeline, first place the required PII model in the OpenMed cache from an
        explicitly approved connected environment. Then return offline and run:

        ```bash
        python pipeline.py
        ```

        The committed config sets `local_only = true`, so a missing model fails
        closed instead of attempting a network download. Keep real clinical data
        out of source control and review residual disclosure risk before release.

        See the OpenMed project-scaffold and persona-quickstart documentation for
        cache preparation, service deployment, and batch-processing next steps.
        """
    )


def render_project_scaffold(preset: str = "researcher") -> dict[str, str]:
    """Render a persona scaffold without touching the filesystem.

    Args:
        preset: One of ``researcher``, ``app-developer``, or ``data-engineer``.

    Returns:
        Ordered mapping of relative managed paths to UTF-8 text content.

    Raises:
        ScaffoldError: If the preset is unknown or a bundled template is invalid.
    """
    selected = _PRESETS.get(preset)
    if selected is None:
        choices = ", ".join(PERSONA_PRESETS)
        raise ScaffoldError(
            f"Unknown persona preset {preset!r}. Choose from: {choices}."
        )

    rendered = {
        "openmed.toml": _config_template(preset, selected),
        "pipeline.py": selected.pipeline,
        ".env.example": _ENV_TEMPLATE,
        ".gitignore": _GITIGNORE_TEMPLATE,
        "README.md": _readme_template(preset, selected),
    }
    if tuple(rendered) != MANAGED_FILES:
        raise ScaffoldError("The bundled scaffold manifest is inconsistent.")
    if any(not content.endswith("\n") for content in rendered.values()):
        raise ScaffoldError("A bundled scaffold template is not newline-terminated.")
    return rendered


def _atomic_write(path: Path, content: str) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.openmed-init-",
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.chmod(0o644)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def scaffold_project(
    destination: str | os.PathLike[str],
    *,
    preset: str = "researcher",
    force: bool = False,
) -> ScaffoldResult:
    """Create a deterministic OpenMed project scaffold.

    Existing identical managed files are left untouched. Differing managed files
    cause a conflict before any write unless ``force`` is true; forced operation
    replaces only the five paths in :data:`MANAGED_FILES` and never removes
    unrelated files.

    Args:
        destination: Directory to create or populate.
        preset: Persona-specific starter preset.
        force: Replace differing regular managed files when true.

    Returns:
        A stable summary of created, overwritten, and unchanged paths.

    Raises:
        ScaffoldConflictError: If managed files differ and force is false.
        ScaffoldError: If the destination or a managed path is unsafe.
    """
    target = Path(destination).expanduser()
    rendered = render_project_scaffold(preset)

    if target.is_symlink():
        raise ScaffoldError("Refusing to scaffold through a symbolic-link directory.")
    if target.exists() and not target.is_dir():
        raise ScaffoldError("The project destination exists and is not a directory.")

    created: list[str] = []
    overwritten: list[str] = []
    unchanged: list[str] = []
    conflicts: list[str] = []

    for relative_path, content in rendered.items():
        path = target / relative_path
        if path.is_symlink():
            raise ScaffoldError(
                f"Refusing to replace symbolic-link scaffold path: {relative_path}."
            )
        if not path.exists():
            created.append(relative_path)
            continue
        if not path.is_file():
            raise ScaffoldError(
                f"Managed scaffold path is not a regular file: {relative_path}."
            )
        try:
            current = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise ScaffoldError(
                f"Unable to inspect managed scaffold path: {relative_path}."
            ) from exc
        if current == content:
            unchanged.append(relative_path)
        elif force:
            overwritten.append(relative_path)
        else:
            conflicts.append(relative_path)

    if conflicts:
        raise ScaffoldConflictError(tuple(conflicts))

    try:
        target.mkdir(parents=True, exist_ok=True)
        write_paths = set(created) | set(overwritten)
        for relative_path, content in rendered.items():
            if relative_path in write_paths:
                _atomic_write(target / relative_path, content)
    except OSError as exc:
        raise ScaffoldError("Unable to write the project scaffold.") from exc

    return ScaffoldResult(
        destination=target,
        preset=preset,
        created=tuple(created),
        overwritten=tuple(overwritten),
        unchanged=tuple(unchanged),
    )


__all__ = [
    "MANAGED_FILES",
    "PERSONA_PRESETS",
    "ScaffoldConflictError",
    "ScaffoldError",
    "ScaffoldResult",
    "render_project_scaffold",
    "scaffold_project",
]
