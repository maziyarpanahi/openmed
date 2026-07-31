"""Documentation publication, localization, and custom-surface contracts."""

from __future__ import annotations

import importlib.util
import posixpath
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import unquote, urlsplit

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
MKDOCS = ROOT / "mkdocs.yml"
PUBLICATION = DOCS / "brand" / "system" / "publication.yml"
LOCALES = ("hi", "zh")


def _load_docs_hooks() -> Any:
    pytest.importorskip(
        "mkdocs",
        reason="documentation hooks require the optional docs dependency set",
    )
    spec = importlib.util.spec_from_file_location(
        "openmed_docs_hooks", DOCS / "_hooks.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _HookConfig(dict[str, Any]):
    def __init__(self, site_dir: Path, plugins: dict[str, Any]) -> None:
        super().__init__(
            extra_css=["shared.css", "shared.css", "docs.css"],
            extra_javascript=["shared.js", "shared.js", "docs.js"],
        )
        self.site_dir = str(site_dir)
        self.plugins = plugins


def test_localized_builds_restore_canonical_feeds_and_english_404(
    tmp_path: Path,
) -> None:
    hooks = _load_docs_hooks()
    i18n = SimpleNamespace(is_default_language_build=True)
    llmstxt = SimpleNamespace(config=SimpleNamespace(full_output="llms-full.txt"))
    config = _HookConfig(tmp_path, {"i18n": i18n, "llmstxt": llmstxt})
    canonical = {
        "404.html": b'<!doctype html><html lang="en">English 404</html>',
        "llms.txt": b"# OpenMed\n\n## Guides\n\n- [Guide](/docs/guide.md)\n",
        "llms-full.txt": b"# OpenMed full documentation\n\nComplete content.\n",
    }
    for name, payload in canonical.items():
        (tmp_path / name).write_bytes(payload)

    hooks.on_post_build(config)

    i18n.is_default_language_build = False
    for name in canonical:
        (tmp_path / name).write_bytes(b"localized overwrite")
    hooks.on_post_build(config)
    assert {name: (tmp_path / name).read_bytes() for name in canonical} == canonical

    hooks._DEFAULT_BUILD_ARTIFACTS.clear()
    with pytest.raises(RuntimeError, match="cannot restore.*404.html"):
        hooks.on_post_build(config)


def test_generated_template_assets_are_deduplicated(tmp_path: Path) -> None:
    hooks = _load_docs_hooks()
    config = _HookConfig(tmp_path, {})

    hooks.on_template_context({}, config)

    assert config["extra_css"] == ["shared.css", "docs.css"]
    assert config["extra_javascript"] == ["shared.js", "docs.js"]


def _load_yaml(path: Path, *, base: bool = False) -> dict[str, Any]:
    loader = yaml.BaseLoader if base else yaml.SafeLoader
    payload = yaml.load(path.read_text(encoding="utf-8"), Loader=loader)
    assert isinstance(payload, dict)
    return payload


def _nav_paths(node: object) -> list[str]:
    if isinstance(node, str):
        return [node]
    if isinstance(node, list):
        return [path for child in node for path in _nav_paths(child)]
    if isinstance(node, dict):
        return [path for child in node.values() for path in _nav_paths(child)]
    return []


def _public_default_markdown() -> set[str]:
    paths: set[str] = set()
    for path in DOCS.rglob("*.md"):
        relative = path.relative_to(DOCS).as_posix()
        if relative.startswith(("brand/", "website/")):
            continue
        if relative == "demo/web/README.md":
            continue
        if any(relative.endswith(f".{locale}.md") for locale in LOCALES):
            continue
        paths.add(relative)
    return paths


def _resolve_docs_link(
    source: str,
    href: str,
    *,
    published: set[str],
) -> str | None:
    parsed = urlsplit(href)
    if parsed.scheme or parsed.netloc:
        return None
    raw_path = unquote(parsed.path)
    if not raw_path:
        return source
    if raw_path.startswith("/docs/"):
        candidate = raw_path.removeprefix("/docs/")
    elif raw_path.startswith("/"):
        return None
    else:
        candidate = posixpath.normpath(
            posixpath.join(posixpath.dirname(source), raw_path)
        )
    candidate = candidate.lstrip("./")
    choices = [candidate]
    if candidate.endswith("/"):
        choices.append(f"{candidate}index.md")
    elif not candidate.endswith(".md"):
        choices.extend((f"{candidate}.md", f"{candidate}/index.md"))
    for choice in choices:
        if choice in published:
            return choice
    return None


def _markdown_targets(source: str, *, published: set[str]) -> set[str]:
    markdown = (DOCS / source).read_text(encoding="utf-8")
    hrefs = re.findall(r"!?\[[^\]]*\]\(([^)\s]+)", markdown)
    return {
        target
        for href in hrefs
        if (target := _resolve_docs_link(source, href, published=published)) is not None
    }


def test_publication_classifies_every_default_markdown_page_exactly_once() -> None:
    config = _load_yaml(MKDOCS, base=True)
    publication = _load_yaml(PUBLICATION)
    classification = publication["classification"]
    navigated = classification["navigated"]
    link_only = classification["link_only"]

    assert publication["version"] == 1
    assert publication["fallback_policy"] == "disabled"
    assert config["plugins"][1]["i18n"]["fallback_to_default"] == "false"
    assert navigated == _nav_paths(config["nav"])
    assert len(navigated) == len(set(navigated))
    assert not set(navigated) & set(link_only)
    assert set(navigated) | set(link_only) == _public_default_markdown()
    assert classification["excluded"] == [
        "_hooks.py",
        "__pycache__/**",
        "overrides/**",
        "brand/**",
        "website/**",
        "demo/web/README.md",
    ]


def test_every_link_only_page_is_reachable_from_the_documentation_nav() -> None:
    publication = _load_yaml(PUBLICATION)
    classification = publication["classification"]
    published = set(classification["navigated"]) | set(classification["link_only"])
    reachable = set(classification["navigated"])
    pending = list(reachable)

    while pending:
        source = pending.pop()
        for target in _markdown_targets(source, published=published):
            if target not in reachable:
                reachable.add(target)
                pending.append(target)

    assert set(classification["link_only"]) <= reachable


def test_translation_groups_are_complete_without_false_fallback_routes() -> None:
    config = _load_yaml(MKDOCS, base=True)
    publication = _load_yaml(PUBLICATION)
    groups = publication["translation_groups"]
    declared: set[str] = set()

    for group in groups.values():
        source = group["source"]
        assert (DOCS / source).is_file()
        for locale, translation in group["translations"].items():
            assert locale in LOCALES
            assert translation.endswith(f".{locale}.md")
            assert (DOCS / translation).is_file()
            declared.add(translation)

    actual = {
        path.relative_to(DOCS).as_posix()
        for path in DOCS.rglob("*.md")
        if any(path.name.endswith(f".{locale}.md") for locale in LOCALES)
    }
    assert declared == actual
    assert {
        "docs/zh/index.html",
        "docs/zh/getting-started/index.html",
        "docs/zh/onboarding-china/index.html",
        "docs/hi/index.html",
        "docs/hi/getting-started/index.html",
        "docs/hi/onboarding-india/index.html",
    } <= set(publication["expected_routes"])

    language_configs = config["plugins"][1]["i18n"]["languages"]
    localized_nav = {
        language["locale"]: set(_nav_paths(language["nav"]))
        for language in language_configs
        if language["locale"] in LOCALES
    }
    assert localized_nav == {
        "zh": {"index.md", "getting-started.md", "onboarding-china.md"},
        "hi": {"index.md", "getting-started.md", "onboarding-india.md"},
    }


def test_localized_entry_pages_keep_content_and_link_parity() -> None:
    english_index = (DOCS / "index.md").read_text(encoding="utf-8")
    chinese_index = (DOCS / "index.zh.md").read_text(encoding="utf-8")
    hindi_index = (DOCS / "index.hi.md").read_text(encoding="utf-8")

    for localized in (chinese_index, hindi_index):
        assert "/docs/export-onnx-webgpu/" in localized
        assert "34" in localized
        assert re.search(r"(?m)^    - ", localized)
        assert re.search(r"(?m)^5\. ", localized)
    assert re.search(r"(?m)^    - ", english_index)
    assert re.search(r"(?m)^5\. ", english_index)

    for locale in LOCALES:
        getting_started = (DOCS / f"getting-started.{locale}.md").read_text(
            encoding="utf-8"
        )
        assert "/docs/low-bandwidth-install/" in getting_started
        assert "/docs/anonymization/#quickstart-choosing-a-method" in getting_started

    china = (DOCS / "onboarding-china.md").read_text(encoding="utf-8")
    china_zh = (DOCS / "onboarding-china.zh.md").read_text(encoding="utf-8")
    india = (DOCS / "onboarding-india.md").read_text(encoding="utf-8")
    india_hi = (DOCS / "onboarding-india.hi.md").read_text(encoding="utf-8")
    assert "/docs/zh/onboarding-china/" in china
    assert "/docs/onboarding-china/" in china_zh
    assert "/docs/hi/onboarding-india/" in india
    assert "/docs/onboarding-india/" in india_hi


def test_docs_consume_the_shared_system_and_repository_owned_fonts() -> None:
    config = _load_yaml(MKDOCS, base=True)
    main_override = (DOCS / "overrides" / "main.html").read_text(encoding="utf-8")
    docs_script = (DOCS / "javascripts" / "openmed-docs.js").read_text(encoding="utf-8")
    assert config["theme"]["font"] == "false"
    assert config["theme"]["custom_dir"] == "docs/overrides"
    assert config["extra_css"][:2] == [
        "stylesheets/openmed-system.css",
        "stylesheets/openmed-brand.css",
    ]
    assert "javascripts/openmed-docs.js" in config["extra_javascript"]
    assert "page.file.alternates.items()" in main_override
    assert "alternate_language != page_language" in main_override
    assert "alternate.dataset.openmedHreflangRel" in main_override
    assert 'alternate.removeAttribute("rel")' in main_override
    assert "restoreHreflangLinks()" in docs_script
    assert "link[data-openmed-hreflang-rel]" in docs_script


def test_published_theme_controls_offer_only_light_and_dark() -> None:
    config = _load_yaml(MKDOCS, base=True)
    website = (DOCS / "website" / "index.html").read_text(encoding="utf-8")
    website_script = (DOCS / "website" / "assets" / "script.js").read_text(
        encoding="utf-8"
    )
    standalone_script = (DOCS / "javascripts" / "openmed-standalone.js").read_text(
        encoding="utf-8"
    )
    standalone_pages = [
        (DOCS / "demo" / "web" / "index.html").read_text(encoding="utf-8"),
        (DOCS / "demo" / "rtl" / "index.html").read_text(encoding="utf-8"),
        (DOCS / "eval" / "benchmark-leaderboard" / "index.html").read_text(
            encoding="utf-8"
        ),
    ]

    palette = config["theme"]["palette"]
    assert [entry["scheme"] for entry in palette] == ["default", "slate"]
    assert [entry["toggle"]["name"] for entry in palette] == [
        "Switch to dark mode",
        "Switch to light mode",
    ]
    assert 'const preferences = ["light", "dark"]' in website_script
    assert 'const modes = ["light", "dark"]' in standalone_script
    assert "Color theme: system" not in website
    assert "OpenMed provides technical controls, not legal compliance." not in website
    assert all("Theme: system" not in page for page in standalone_pages)


def test_website_preserves_the_approved_head_to_head_matrix() -> None:
    website = (DOCS / "website" / "index.html").read_text(encoding="utf-8")

    expected_copy = (
        "Every other option asks you to give something up.",
        "What we don't claim",
        "AWS Comprehend Medical · Azure Health · Google Healthcare NLP",
        "Enterprise NLP suites, per-server subscription",
        "medspaCy · MedCAT · cTAKES",
        "Where inference runs",
        "Patient data leaves the network",
        "What it costs at 10M notes",
        "Languages supported",
        "Runs on iPhone and Android",
        "Apple Silicon acceleration",
        "Runs in the browser",
        "CPU-optimised ONNX builds",
        "Release cadence",
        "Benchmarks you can rerun",
        "Capability and cadence rows describe publicly documented positions",
    )
    for text in expected_copy:
        assert text in website
    assert website.count("<tr>") >= 11


def test_custom_surfaces_have_metadata_shared_chrome_and_rtl_fixture_policy() -> None:
    publication = _load_yaml(PUBLICATION)
    demo = (DOCS / "demo" / "web" / "index.html").read_text(encoding="utf-8")
    demo_app = (DOCS / "demo" / "web" / "app.js").read_text(encoding="utf-8")
    rtl = (DOCS / "demo" / "rtl" / "index.html").read_text(encoding="utf-8")

    for html in (demo, rtl):
        assert "../../stylesheets/openmed-system.css" in html
        assert "../../stylesheets/openmed-standalone.css" in html
        assert "../../javascripts/openmed-standalone.js" in html
        assert 'class="om-site-header"' in html
        assert 'class="om-site-footer"' in html
        assert "<main" in html and "<h1" in html

    assert '<html lang="ar" dir="rtl">' in rtl
    assert 'name="robots" content="noindex,nofollow"' in rtl
    assert "SYNTH-AR-0042" in rtl
    assert "https://" not in demo_app
    assert "import(runtimeUrl.href)" in demo_app
    assert "resolved.origin !== window.location.origin" in demo_app
    assert publication["fixtures"] == [
        {
            "source": "demo/rtl/index.html",
            "route": "docs/demo/rtl/index.html",
            "indexing": "noindex,nofollow",
            "data_policy": "synthetic_only",
            "social_metadata": "prohibited",
            "purpose": "rtl_layout_and_accessibility",
        }
    ]
