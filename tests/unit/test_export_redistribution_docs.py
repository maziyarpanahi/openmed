"""Offline coherence checks for the export toolchain and redistribution page.

The page must map every export command in the export and runtime guides to a
toolchain row, give every kit runtime library a license row, cite only
repository paths that exist, and keep the never-bundle list aligned with the
repository rules. The checks deliberately avoid asserting dependency versions.
"""

from __future__ import annotations

import json
import posixpath
import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
PAGE = DOCS / "security" / "export-redistribution.md"
PAGE_ROUTE = "security/export-redistribution.md"
EXPORT_GUIDES = (
    "coreml-export.md",
    "export-onnx-webgpu.md",
    "export-onnx-android.md",
    "export-transformersjs.md",
    "mlx-backend.md",
    "export-mlx-quant.md",
    "export-gguf.md",
    "export-awq.md",
    "export-gptq.md",
    "maple-on-device.md",
    "on-device-segmenters.md",
    "runtimes/openvino.md",
    "runtimes/tensorrt.md",
    "runtimes/gguf-grounding.md",
    "runtimes/awq-grounding.md",
)
LINKING_PAGES = (
    "export-matrix.md",
    "swift-openmedkit.md",
    "android-integration.md",
    "runtimes/android-accelerators.md",
    "runtimes/webgpu.md",
    "security/license-inventory.md",
)
REPOSITORY_ROOTS = (
    "android/",
    "docs/",
    "examples/",
    "js/",
    "openmed/",
    "scripts/",
    "swift/",
    "tests/",
)
REPOSITORY_FILES = frozenset(
    {
        "AGENTS.md",
        "LICENSE",
        "NOTICE",
        "Package.resolved",
        "Package.swift",
        "models.jsonl",
        "pyproject.toml",
        "uv.lock",
    }
)
NEVER_BUNDLE_TERMS = (
    "GPL",
    "Source-available",
    "Proprietary",
    "DUA-gated",
    "UMLS",
    "SNOMED CT",
    "CPT",
    "MIMIC",
    "i2b2",
    "n2c2",
)
LICENSE_CELL = re.compile(r"^(`[^`]+`(, `[^`]+`)*( .*)?|To confirm.*|Proprietary .+)$")
_BACKTICK = re.compile(r"`([^`\n]+)`")
_IMPORT = re.compile(r"from openmed[\w.]* import (?:\(([^)]*)\)|([^\n]+))")


def _page() -> str:
    return PAGE.read_text(encoding="utf-8")


def _section(markdown: str, heading: str) -> str:
    marker = f"\n## {heading}\n"
    assert marker in markdown, f"missing section: {heading}"
    return markdown.split(marker, 1)[1].split("\n## ", 1)[0]


def _backtick_tokens(text: str) -> set[str]:
    return {token.strip() for token in _BACKTICK.findall(text)}


def _table_rows(section: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    header: list[str] | None = None
    for line in section.splitlines():
        if not line.startswith("|"):
            header = None
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if header is None:
            header = cells
        elif not set("".join(cells)) <= set("- :"):
            assert len(cells) == len(header), f"malformed table row: {line}"
            rows.append(dict(zip(header, cells, strict=True)))
    return rows


def _cited_repository_paths(markdown: str) -> set[str]:
    paths = set()
    for token in _backtick_tokens(markdown):
        if " " in token or "*" in token:
            continue
        if token in REPOSITORY_FILES or token.startswith(REPOSITORY_ROOTS):
            paths.add(token.rstrip("/"))
    return paths


def _assert_repository_paths_exist(markdown: str, *, root: Path) -> None:
    paths = _cited_repository_paths(markdown)
    assert paths, "the page cites no repository paths"
    for path in sorted(paths):
        relative = Path(path)
        assert ".." not in relative.parts and not relative.is_absolute()
        assert (root / relative).exists(), f"missing repository path: {path}"


def _export_entry_points(markdown: str) -> set[str]:
    entry_points = {
        f"python -m {module}"
        for module in re.findall(r"-m (openmed(?:\.\w+)+)", markdown)
    }
    entry_points |= {
        f"--profile {name}" for name in re.findall(r"--profile (\w+)", markdown)
    }
    if "--include-transformersjs" in markdown:
        entry_points.add("--include-transformersjs")
    for match in _IMPORT.finditer(markdown):
        names = (match.group(1) or match.group(2)).replace("\n", ",").split(",")
        entry_points |= {
            f"`{name.strip()}`"
            for name in names
            if re.match(r"(quantize|export|build)_\w+$", name.strip())
        }
    return entry_points


def _assert_export_commands_mapped(guides: dict[str, str], page: str) -> None:
    toolchains = _section(page, "Export toolchains")
    missing = {
        f"{guide}: {entry_point}"
        for guide, text in guides.items()
        for entry_point in _export_entry_points(text)
        if entry_point not in toolchains
    }
    assert not missing, f"export commands without a toolchain row: {sorted(missing)}"


def _assert_kit_license_cells(section: str) -> None:
    rows = [row for row in _table_rows(section) if "License" in row]
    assert rows, "kit runtime library tables are missing"
    for row in rows:
        assert LICENSE_CELL.match(row["License"]), f"no license for row: {row}"


def _assert_kit_libraries_licensed(section: str, libraries: set[str]) -> None:
    rows = [
        " | ".join(row.values()).lower()
        for row in _table_rows(section)
        if LICENSE_CELL.match(row.get("License", ""))
    ]
    missing = {
        name
        for name in libraries
        if not any(
            re.search(rf"(?<![\w-]){re.escape(name.lower())}(?![\w-])", row)
            for row in rows
        )
    }
    assert not missing, f"kit libraries without a license row: {sorted(missing)}"


def _guide_texts() -> dict[str, str]:
    return {
        guide: (DOCS / guide).read_text(encoding="utf-8") for guide in EXPORT_GUIDES
    }


def _swift_packages() -> set[str]:
    package_url = re.compile(r'url: "https://github\.com/[^/]+/([^"/]+?)(?:\.git)?"')
    names = set()
    for manifest in (ROOT / "Package.swift", ROOT / "swift/OpenMedKit/Package.swift"):
        names |= set(package_url.findall(manifest.read_text(encoding="utf-8")))
    return names


def _android_modules() -> set[str]:
    catalog = tomllib.loads(
        (ROOT / "android/gradle/libs.versions.toml").read_text(encoding="utf-8")
    )["libraries"]
    build = (ROOT / "android/openmedkit/build.gradle.kts").read_text(encoding="utf-8")
    aliases = re.findall(r"^\s*(?:implementation|api)\(libs\.([\w.]+)\)", build, re.M)
    return {catalog[alias.replace(".", "-")]["module"] for alias in aliases}


def _javascript_packages() -> set[str]:
    names = set()
    for kit in ("web", "electron", "react-native", "tauri"):
        manifest = json.loads(
            (ROOT / f"js/openmedkit-{kit}/package.json").read_text(encoding="utf-8")
        )
        for field in ("dependencies", "peerDependencies"):
            names |= set(manifest.get(field) or {})
    return names - {"openmed"}


def _flutter_and_tauri_packages() -> set[str]:
    flutter = ROOT / "js/openmedkit-flutter"
    pubspec = (flutter / "pubspec.yaml").read_text(encoding="utf-8")
    dependencies = pubspec.split("\ndependencies:\n", 1)[1].split("\n\n", 1)[0]
    names = {
        name
        for name in re.findall(r"^  ([\w-]+):", dependencies, re.M)
        if name != "flutter"
    }
    for podspec in flutter.glob("*/openmedkit_flutter.podspec"):
        names |= set(
            re.findall(r"s\.dependency '([^']+)'", podspec.read_text(encoding="utf-8"))
        ) - {"Flutter", "FlutterMacOS"}
    gradle = (flutter / "android/build.gradle").read_text(encoding="utf-8")
    names |= set(re.findall(r'^\s*implementation "([\w.]+:[\w-]+):', gradle, re.M))
    cargo = tomllib.loads(
        (ROOT / "js/openmedkit-tauri/src-tauri/Cargo.toml").read_text(encoding="utf-8")
    )
    return names | set(cargo["dependencies"])


def test_page_is_navigated_and_linked_from_export_and_platform_guides() -> None:
    nav = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    publication = (DOCS / "brand/system/publication.yml").read_text(encoding="utf-8")
    assert f"Export Toolchains and Redistribution: {PAGE_ROUTE}" in nav
    assert f"    - {PAGE_ROUTE}\n" in publication
    for page in LINKING_PAGES:
        text = (DOCS / page).read_text(encoding="utf-8")
        target = posixpath.relpath(PAGE_ROUTE, posixpath.dirname(page) or ".")
        assert f"]({target})" in text, f"{page} must link to the redistribution page"


def test_page_states_versions_as_pinned_at_the_time_of_writing() -> None:
    text = _page()
    assert "as pinned at the time of writing" in text
    assert "To confirm" in text


def test_cited_repository_paths_exist() -> None:
    _assert_repository_paths_exist(_page(), root=ROOT)


def test_repository_path_check_rejects_missing_paths() -> None:
    with pytest.raises(AssertionError, match="missing repository path"):
        _assert_repository_paths_exist(
            "See `openmed/onnx/missing_exporter.py`.", root=ROOT
        )


def test_relative_documentation_links_resolve() -> None:
    for target in re.findall(r"\]\(([^)\s]+)\)", _page()):
        if "://" in target:
            continue
        path = (PAGE.parent / target.split("#", 1)[0]).resolve()
        assert path.is_file(), f"broken documentation link: {target}"


def test_every_export_command_in_the_guides_maps_to_a_toolchain_row() -> None:
    guides = _guide_texts()
    assert any(_export_entry_points(text) for text in guides.values())
    _assert_export_commands_mapped(guides, _page())


def test_export_command_check_rejects_an_unmapped_command() -> None:
    guide = "```bash\npython -m openmed.onnx.unlisted_export --model x\n```\n"
    with pytest.raises(AssertionError, match="without a toolchain row"):
        _assert_export_commands_mapped({"synthetic.md": guide}, _page())


def test_cited_export_modules_exist() -> None:
    toolchains = _section(_page(), "Export toolchains")
    modules = re.findall(r"python -m (openmed(?:\.\w+)+)", toolchains)
    assert modules
    for module in modules:
        assert (ROOT / f"{module.replace('.', '/')}.py").is_file(), module


def test_every_kit_runtime_library_has_a_license_row() -> None:
    section = _section(_page(), "Kit runtime libraries")
    libraries = (
        _swift_packages()
        | _android_modules()
        | _javascript_packages()
        | _flutter_and_tauri_packages()
    )
    assert {"swift-transformers", "onnxruntime-web", "ffi", "tauri"} <= libraries
    _assert_kit_libraries_licensed(section, libraries)
    _assert_kit_license_cells(section)


def test_kit_license_check_rejects_a_row_without_a_license() -> None:
    section = (
        "| Library | Pinned | License | Verified from |\n"
        "| --- | --- | --- | --- |\n"
        "| `example-runtime` | 1.0.0 |  | manifest |\n"
    )
    with pytest.raises(AssertionError, match="no license for row"):
        _assert_kit_license_cells(section)
    with pytest.raises(AssertionError, match="without a license row"):
        _assert_kit_libraries_licensed(section, {"example-runtime"})


def test_restricted_components_have_a_default_posture_and_opt_in_path() -> None:
    rows = _table_rows(_section(_page(), "Restricted and vendor components"))
    components = " ".join(row["Component"] for row in rows)
    for name in ("ML Kit", "QNN", "TensorRT", "llama.cpp", "Hugging Face Hub"):
        assert name in components
    for row in rows:
        assert row["Default posture"] and row["Opt-in path"], row


def test_never_bundle_list_matches_the_repository_rules() -> None:
    section = _section(_page(), "Never bundle")
    rules = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    for term in NEVER_BUNDLE_TERMS:
        assert term in section, f"never-bundle list is missing {term}"
    for term in ("UMLS", "SNOMED CT", "CPT", "MIMIC", "i2b2", "n2c2", "DUA-gated"):
        assert term in rules


def test_offline_inputs_cover_each_runtime_surface() -> None:
    rows = _table_rows(_section(_page(), "Offline inputs by runtime"))
    runtimes = " ".join(row["Runtime"] for row in rows)
    for surface in (
        "Python ONNX Runtime",
        "Python MLX",
        "Swift Core ML",
        "Swift MLX",
        "Android",
        "Web",
        "Electron",
        "React Native",
        "Flutter",
        "Tauri",
    ):
        assert surface in runtimes, f"missing offline inputs for {surface}"
    assert all(row["Files it loads"] for row in rows)


def test_page_contains_no_credentials_or_private_paths() -> None:
    text = _page()
    for pattern in (
        r"/Users/",
        r"/home/",
        r"~/",
        r"\b[A-Za-z]:\\[A-Za-z]",
        r"\bhf_[A-Za-z0-9]{20,}",
        r"BEGIN [A-Z ]*PRIVATE KEY",
        r"HF_WRITE_TOKEN=\S+",
    ):
        assert not re.search(pattern, text), pattern
