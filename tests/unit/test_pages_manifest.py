"""Tests for the unified GitHub Pages staging contract."""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
from pathlib import Path
from urllib.parse import urlsplit
from xml.etree import ElementTree

import pytest

from scripts.docs import stage_pages as stage_pages_module
from scripts.docs.stage_pages import (
    DEFAULT_PUBLICATION,
    MANIFEST_NAME,
    SCHEMA_ID_HOST,
    SCHEMA_INDEX_PATH,
    SCHEMA_PENDING_ID_HOSTS,
    SCHEMA_SOURCE_DIR,
    SCHEMA_UNPUBLISHED_REASON,
    PageStagingError,
    artifact_path_for_url,
    build_manifest,
    changed_snapshot_paths,
    compact_search_index,
    copy_locale_sitemaps,
    ensure_safe_output_dir,
    find_overlay_collisions,
    load_publication,
    markdown_output_path,
    normalized_expected_paths,
    owner_for_path,
    packaged_schema_files,
    plan_schema_publication,
    prune_source_maps,
    public_route_for_path,
    resolve_release_tag,
    schema_publication_path,
    stage_schemas,
    translated_output_path,
    translated_output_paths,
    validate_internal_references,
    validate_llm_feeds,
    validate_publication_contract,
    validate_publication_metadata,
    verify_staged_schemas,
)

ROOT = Path(__file__).resolve().parents[2]


def _publication() -> dict[str, object]:
    return {
        "version": 1,
        "fallback_policy": "disabled",
        "expected_routes": [
            "docs/index.html",
            "docs/zh/index.html",
            "docs/hi/getting-started/index.html",
        ],
        "expected_assets": [],
        "translation_groups": {
            "home": {
                "source": "index.md",
                "translations": {"zh": "index.zh.md"},
            },
            "getting-started": {
                "source": "getting-started.md",
                "translations": {"hi": "getting-started.hi.md"},
            },
        },
        "classification": {
            "navigated": ["index.md", "getting-started.md"],
            "link_only": [],
            "excluded": [],
        },
        "metadata_policy": {
            "indexable_pages": {
                "title": "required",
                "description": "required",
                "canonical": "required",
                "favicon": "required_local",
                "open_graph": "required",
                "twitter_card": "required",
            },
            "localized_pages": {
                "alternates": "real_translations_only",
                "default_language_fallback": "prohibited",
            },
        },
        "fixtures": [],
    }


def test_public_route_for_path(subtests) -> None:
    for path, route in [
        ("index.html", "/"),
        ("docs/index.html", "/docs/"),
        ("docs/guide/index.html", "/docs/guide/"),
        ("docs/llms.txt", "/docs/llms.txt"),
    ]:
        with subtests.test(path=path, route=route):
            assert public_route_for_path(path) == route


def test_owner_for_path(subtests) -> None:
    for path, owner in [
        ("index.html", "marketing"),
        ("docs/index.html", "mkdocs"),
        ("docs/demo/web/index.html", "browser-demo"),
        ("docs/eval/benchmark-leaderboard/index.html", "leaderboard"),
        (MANIFEST_NAME, "staging"),
        (SCHEMA_INDEX_PATH, "schemas"),
        ("schemas/compliance/control_evidence.schema.json", "schemas"),
    ]:
        with subtests.test(path=path, owner=owner):
            assert owner_for_path(path) == owner


def test_translation_groups_map_only_real_sources() -> None:
    publication = _publication()

    assert translated_output_path("index.zh.md", "zh") == "docs/zh/index.html"
    assert translated_output_path("guide/setup.hi.md", "hi") == (
        "docs/hi/guide/setup/index.html"
    )
    assert translated_output_paths(publication) == [
        "docs/hi/getting-started/index.html",
        "docs/zh/index.html",
    ]
    assert markdown_output_path("index.md") == "docs/index.html"
    assert markdown_output_path("guide/index.md") == "docs/guide/index.html"
    assert markdown_output_path("guide/setup.md") == "docs/guide/setup/index.html"


def test_translation_source_requires_matching_suffix() -> None:
    with pytest.raises(PageStagingError, match=r"\.hi\.md"):
        translated_output_path("getting-started.zh.md", "hi")


def test_expected_paths_reject_duplicates_and_traversal() -> None:
    publication = _publication()
    assert normalized_expected_paths(publication) == sorted(
        publication["expected_routes"]
    )

    publication["expected_routes"] = ["docs/index.html", "/docs/index.html"]
    with pytest.raises(PageStagingError, match="Unsafe"):
        normalized_expected_paths(publication)

    publication["expected_routes"] = ["../outside.html"]
    with pytest.raises(PageStagingError, match="Unsafe"):
        normalized_expected_paths(publication)


def test_overlay_collision_detection_rejects_replacement_and_symlink(
    tmp_path: Path,
) -> None:
    website = tmp_path / "website"
    artifact = tmp_path / "site"
    website.mkdir()
    artifact.mkdir()
    (website / "index.html").write_text("marketing", encoding="utf-8")
    (website / "docs").mkdir()
    (website / "docs" / "asset.css").write_text("x", encoding="utf-8")
    (artifact / "docs").mkdir()
    (artifact / "docs" / "asset.css").write_text("docs", encoding="utf-8")
    (website / "linked").symlink_to(website / "index.html")

    assert find_overlay_collisions(website, artifact) == [
        "docs",
        "docs/asset.css",
        "linked (symlink is not allowed)",
    ]


def test_changed_snapshot_paths_covers_add_remove_and_modify() -> None:
    assert changed_snapshot_paths(
        {"removed": "a", "same": "b", "updated": "c"},
        {"added": "d", "same": "b", "updated": "e"},
    ) == ["added", "removed", "updated"]


def test_manifest_hashes_files_and_assigns_unique_routes(tmp_path: Path) -> None:
    (tmp_path / "docs" / "demo" / "web").mkdir(parents=True)
    (tmp_path / "docs" / "eval" / "benchmark-leaderboard").mkdir(parents=True)
    (tmp_path / "index.html").write_text("home", encoding="utf-8")
    (tmp_path / "docs" / "index.html").write_text("docs", encoding="utf-8")
    (tmp_path / "docs" / "demo" / "web" / "index.html").write_text(
        "demo", encoding="utf-8"
    )
    (tmp_path / "docs" / "eval" / "benchmark-leaderboard" / "index.html").write_text(
        "leaderboard", encoding="utf-8"
    )

    manifest = build_manifest(
        tmp_path,
        release_tag="v2.0.0",
        publication=_publication(),
    )

    assert manifest["schema_version"] == 1
    assert manifest["release_tag"] == "v2.0.0"
    routes = [entry["route"] for entry in manifest["files"]]
    assert len(routes) == len(set(routes))
    assert {entry["owner"] for entry in manifest["files"]} == {
        "browser-demo",
        "leaderboard",
        "marketing",
        "mkdocs",
    }
    assert all(len(entry["sha256"]) == 64 for entry in manifest["files"])
    json.dumps(manifest)


def test_manifest_rejects_symlinks(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_text("target", encoding="utf-8")
    (tmp_path / "linked.txt").symlink_to(target)

    with pytest.raises(PageStagingError, match="symlinks"):
        build_manifest(
            tmp_path,
            release_tag="v2.0.0",
            publication=_publication(),
        )


def test_artifact_path_for_url_supports_files_and_pretty_routes(
    tmp_path: Path,
) -> None:
    (tmp_path / "docs" / "guide").mkdir(parents=True)
    (tmp_path / "docs" / "guide" / "index.html").write_text("guide", encoding="utf-8")
    (tmp_path / "asset.css").write_text("x", encoding="utf-8")

    assert artifact_path_for_url(tmp_path, "/docs/guide/") == (
        tmp_path / "docs" / "guide" / "index.html"
    )
    assert artifact_path_for_url(tmp_path, "/docs/guide") == (
        tmp_path / "docs" / "guide" / "index.html"
    )
    assert artifact_path_for_url(tmp_path, "/asset.css") == tmp_path / "asset.css"
    assert artifact_path_for_url(tmp_path, "/missing") is None


def test_llm_feed_validation_requires_curated_links_and_full_content(
    tmp_path: Path,
) -> None:
    config = tmp_path / "mkdocs.yml"
    config.write_text(
        """
site_name: OpenMed
site_url: https://openmed.life/docs/
plugins:
  - llmstxt:
      full_output: llms-full.txt
      sections:
        Getting started:
          - index.md: Documentation home
        Guides:
          - guide/setup.md: Setup guide
markdown_extensions:
  - pymdownx.emoji:
      emoji_index: !!python/name:material.extensions.emoji.twemoji
""".lstrip(),
        encoding="utf-8",
    )

    docs = tmp_path / "docs"
    guide = docs / "guide" / "setup"
    guide.mkdir(parents=True)
    home_content = (
        "# Documentation home\n\n"
        "This synthetic rendered page contains enough substantive content to "
        "prove that the complete documentation body is embedded in the feed."
    )
    guide_content = (
        "# Setup guide\n\n"
        "This synthetic setup guide also contains enough substantive content "
        "to exercise exact full-feed inclusion without relying on a build."
    )
    (docs / "index.md").write_text(home_content + "\n", encoding="utf-8")
    (guide / "index.md").write_text(guide_content + "\n", encoding="utf-8")
    index_feed = """
# OpenMed

## Getting started

- [Documentation home](https://openmed.life/docs/index.md): Documentation home

## Guides

- [Setup guide](https://openmed.life/docs/guide/setup/index.md): Setup guide
""".lstrip()
    full_feed = (
        "# OpenMed\n\n"
        "# Getting started\n\n"
        f"{home_content}\n"
        "# Guides\n\n"
        f"{guide_content}\n"
    )
    (docs / "llms.txt").write_text(index_feed, encoding="utf-8")
    (docs / "llms-full.txt").write_text(full_feed, encoding="utf-8")

    validate_llm_feeds(tmp_path, config_path=config)

    (docs / "llms.txt").write_text("# OpenMed\n", encoding="utf-8")
    with pytest.raises(PageStagingError, match="section coverage"):
        validate_llm_feeds(tmp_path, config_path=config)

    (docs / "llms.txt").write_text(index_feed, encoding="utf-8")
    (docs / "llms-full.txt").write_text("# OpenMed\n", encoding="utf-8")
    with pytest.raises(PageStagingError, match="missing section"):
        validate_llm_feeds(tmp_path, config_path=config)


def test_internal_reference_crawler_checks_links_assets_and_fragments(
    tmp_path: Path,
) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "style.css").write_text(
        "body { background: url('../image.svg'); }", encoding="utf-8"
    )
    (tmp_path / "image.svg").write_text("<svg/>", encoding="utf-8")
    (tmp_path / "index.html").write_text(
        """
        <html><body><main id="main">
          <a href="/docs/#guide">Guide</a>
          <img src="/image.svg" alt="">
          <link rel="stylesheet" href="/assets/style.css">
        </main></body></html>
        """,
        encoding="utf-8",
    )
    (tmp_path / "docs" / "index.html").write_text(
        '<html><body><h1 id="guide">Guide</h1></body></html>',
        encoding="utf-8",
    )

    validate_internal_references(tmp_path)

    (tmp_path / "index.html").write_text(
        '<html><body><a href="/docs/#missing">Broken</a></body></html>',
        encoding="utf-8",
    )
    with pytest.raises(PageStagingError, match="missing fragment"):
        validate_internal_references(tmp_path)


def test_internal_reference_crawler_rejects_duplicate_scripts_and_stylesheets(
    tmp_path: Path,
) -> None:
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "app.js").write_text("", encoding="utf-8")
    (tmp_path / "assets" / "style.css").write_text("", encoding="utf-8")
    (tmp_path / "index.html").write_text(
        """
        <html><head>
          <link rel="stylesheet" href="/assets/style.css">
          <link rel="stylesheet" href="/assets/style.css">
          <script src="/assets/app.js"></script>
          <script src="/assets/app.js"></script>
        </head><body></body></html>
        """,
        encoding="utf-8",
    )

    with pytest.raises(PageStagingError, match="duplicate (script|stylesheet)"):
        validate_internal_references(tmp_path)


def test_locale_sitemaps_contain_only_real_translated_routes(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "sitemap.xml").write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
        <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"
                xmlns:xhtml="http://www.w3.org/1999/xhtml">
          <url>
            <loc>https://openmed.life/docs/</loc>
            <xhtml:link rel="alternate" hreflang="en"
                        href="https://openmed.life/docs/"/>
            <xhtml:link rel="alternate" hreflang="zh"
                        href="https://openmed.life/docs/zh/"/>
          </url>
          <url>
            <loc>https://openmed.life/docs/zh/</loc>
            <xhtml:link rel="alternate" hreflang="en"
                        href="https://openmed.life/docs/"/>
            <xhtml:link rel="alternate" hreflang="zh"
                        href="https://openmed.life/docs/zh/"/>
          </url>
          <url>
            <loc>https://openmed.life/docs/hi/getting-started/</loc>
            <xhtml:link rel="alternate" hreflang="en"
                        href="https://openmed.life/docs/getting-started/"/>
            <xhtml:link rel="alternate" hreflang="hi"
                        href="https://openmed.life/docs/hi/getting-started/"/>
          </url>
        </urlset>
        """,
        encoding="utf-8",
    )

    copy_locale_sitemaps(tmp_path, _publication())

    expected = {
        "zh": {"https://openmed.life/docs/zh/"},
        "hi": {"https://openmed.life/docs/hi/getting-started/"},
    }
    for locale, expected_urls in expected.items():
        tree = ElementTree.parse(docs / locale / "sitemap.xml")
        urls = {
            element.text.strip()
            for element in tree.iter()
            if element.tag.endswith("loc") and element.text
        }
        assert urls == expected_urls


def test_repository_publication_contract_is_complete() -> None:
    publication = load_publication(DEFAULT_PUBLICATION)
    assert publication["metadata_policy"]["localized_pages"] == {
        "alternates": "real_translations_only",
        "default_language_fallback": "prohibited",
    }

    invalid = copy.deepcopy(publication)
    invalid["classification"]["link_only"].append(
        invalid["classification"]["navigated"][0]
    )
    with pytest.raises(PageStagingError, match="both navigated and link-only"):
        validate_publication_contract(invalid)


def _page_metadata_html(
    *,
    canonical: str,
    language: str,
    title: str,
    alternates: dict[str, str],
    robots: str | None = None,
    direction: str | None = None,
    social: bool = True,
) -> str:
    description = (
        f"{title} provides useful synthetic release metadata for validation tests."
    )
    alternate_html = "".join(
        f'<link rel="alternate" hreflang="{locale}" href="{href}">'
        for locale, href in alternates.items()
    )
    robots_html = (
        f'<meta name="robots" content="{robots}">' if robots is not None else ""
    )
    direction_html = f' dir="{direction}"' if direction else ""
    social_html = ""
    if social:
        social_html = f"""
        <meta property="og:type" content="website">
        <meta property="og:title" content="{title}">
        <meta property="og:description" content="{description}">
        <meta property="og:url" content="{canonical}">
        <meta property="og:image" content="https://openmed.life/og.png">
        <meta property="og:image:alt" content="Synthetic OpenMed metadata image">
        <meta name="twitter:card" content="summary_large_image">
        <meta name="twitter:title" content="{title}">
        <meta name="twitter:description" content="{description}">
        <meta name="twitter:image" content="https://openmed.life/og.png">
        <meta name="twitter:image:alt" content="Synthetic OpenMed metadata image">
        """
    return f"""
    <!doctype html>
    <html lang="{language}"{direction_html}>
      <head>
        <title>{title}</title>
        <meta name="description" content="{description}">
        {robots_html}
        <link rel="canonical" href="{canonical}">
        <link rel="icon" href="/favicon.svg">
        {alternate_html}
        {social_html}
      </head>
      <body><main><h1>{title}</h1></main></body>
    </html>
    """


def _write_metadata_fixture(tmp_path: Path) -> dict[str, object]:
    publication: dict[str, object] = {
        "version": 1,
        "fallback_policy": "disabled",
        "expected_routes": [
            "docs/index.html",
            "docs/zh/index.html",
            "docs/demo/rtl/index.html",
        ],
        "expected_assets": [],
        "translation_groups": {
            "home": {
                "source": "index.md",
                "translations": {"zh": "index.zh.md"},
            }
        },
        "classification": {
            "navigated": ["index.md"],
            "link_only": [],
            "excluded": [],
        },
        "metadata_policy": _publication()["metadata_policy"],
        "fixtures": [
            {
                "source": "demo/rtl/index.html",
                "route": "docs/demo/rtl/index.html",
                "indexing": "noindex,nofollow",
                "data_policy": "synthetic_only",
                "social_metadata": "prohibited",
                "purpose": "rtl_layout_and_accessibility",
            }
        ],
    }
    (tmp_path / "docs" / "zh").mkdir(parents=True)
    (tmp_path / "docs" / "demo" / "rtl").mkdir(parents=True)
    (tmp_path / "og.png").write_bytes(b"synthetic image")
    (tmp_path / "favicon.svg").write_text("<svg/>", encoding="utf-8")
    alternates = {
        "en": "https://openmed.life/docs/",
        "zh": "https://openmed.life/docs/zh/",
    }
    (tmp_path / "docs" / "index.html").write_text(
        _page_metadata_html(
            canonical="https://openmed.life/docs/",
            language="en",
            title="English documentation",
            alternates=alternates,
        ),
        encoding="utf-8",
    )
    (tmp_path / "docs" / "zh" / "index.html").write_text(
        _page_metadata_html(
            canonical="https://openmed.life/docs/zh/",
            language="zh",
            title="Chinese documentation",
            alternates=alternates,
        ),
        encoding="utf-8",
    )
    (tmp_path / "docs" / "demo" / "rtl" / "index.html").write_text(
        _page_metadata_html(
            canonical="https://openmed.life/docs/demo/rtl/",
            language="ar",
            title="RTL fixture",
            alternates={},
            robots="noindex,nofollow",
            direction="rtl",
            social=False,
        ),
        encoding="utf-8",
    )
    (tmp_path / "docs" / "sitemap.xml").write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
        <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"
                xmlns:xhtml="http://www.w3.org/1999/xhtml">
          <url>
            <loc>https://openmed.life/docs/</loc>
            <xhtml:link rel="alternate" hreflang="en"
                        href="https://openmed.life/docs/"/>
            <xhtml:link rel="alternate" hreflang="zh"
                        href="https://openmed.life/docs/zh/"/>
          </url>
          <url>
            <loc>https://openmed.life/docs/zh/</loc>
            <xhtml:link rel="alternate" hreflang="en"
                        href="https://openmed.life/docs/"/>
            <xhtml:link rel="alternate" hreflang="zh"
                        href="https://openmed.life/docs/zh/"/>
          </url>
        </urlset>
        """,
        encoding="utf-8",
    )
    (tmp_path / "docs" / "zh" / "sitemap.xml").write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
        <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"
                xmlns:xhtml="http://www.w3.org/1999/xhtml">
          <url>
            <loc>https://openmed.life/docs/zh/</loc>
            <xhtml:link rel="alternate" hreflang="en"
                        href="https://openmed.life/docs/"/>
            <xhtml:link rel="alternate" hreflang="zh"
                        href="https://openmed.life/docs/zh/"/>
          </url>
        </urlset>
        """,
        encoding="utf-8",
    )
    return publication


def test_publication_metadata_enforces_canonical_hreflang_and_fixture_policy(
    tmp_path: Path,
) -> None:
    publication = _write_metadata_fixture(tmp_path)
    metadata = validate_publication_metadata(tmp_path, publication)
    assert metadata["/docs/"]["hreflang"] == {
        "en": "https://openmed.life/docs/",
        "zh": "https://openmed.life/docs/zh/",
    }
    assert metadata["/docs/demo/rtl/"]["indexing"] == "noindex,nofollow"

    chinese = tmp_path / "docs" / "zh" / "index.html"
    chinese.write_text(
        chinese.read_text(encoding="utf-8").replace(
            "https://openmed.life/docs/zh/",
            "https://openmed.life/docs/",
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(PageStagingError, match="does not self-match"):
        validate_publication_metadata(tmp_path, publication)


def test_publication_metadata_rejects_unsafe_canonical_urls(subtests, tmp_path) -> None:
    case_root = tmp_path
    for unsafe in [
        "http://openmed.life/docs/",
        "https://user@openmed.life/docs/",
        "https://openmed.life/docs/?preview=1",
    ]:
        with subtests.test(unsafe=unsafe):
            tmp_path = case_root / str(len(list(case_root.iterdir())))
            tmp_path.mkdir()
            publication = _write_metadata_fixture(tmp_path)
            index = tmp_path / "docs" / "index.html"
            index.write_text(
                index.read_text(encoding="utf-8").replace(
                    "https://openmed.life/docs/",
                    unsafe,
                    1,
                ),
                encoding="utf-8",
            )
            with pytest.raises(PageStagingError, match="HTTPS|does not self-match"):
                validate_publication_metadata(tmp_path, publication)


def test_publication_metadata_rejects_fixture_in_sitemap(tmp_path: Path) -> None:
    publication = _write_metadata_fixture(tmp_path)
    sitemap = tmp_path / "docs" / "sitemap.xml"
    sitemap.write_text(
        sitemap.read_text(encoding="utf-8").replace(
            "</urlset>",
            """
            <url><loc>https://openmed.life/docs/demo/rtl/</loc></url>
            </urlset>
            """,
        ),
        encoding="utf-8",
    )
    with pytest.raises(PageStagingError, match="classification|fixture"):
        validate_publication_metadata(tmp_path, publication)


def test_publication_metadata_requires_english_canonical_404(tmp_path: Path) -> None:
    publication = _write_metadata_fixture(tmp_path)
    not_found = tmp_path / "docs" / "404.html"
    not_found.write_text(
        '<!doctype html><html lang="hi"><head><title>Not found</title></head>'
        "<body></body></html>",
        encoding="utf-8",
    )

    with pytest.raises(PageStagingError, match="404.*lang=en"):
        validate_publication_metadata(tmp_path, publication)

    not_found.write_text(
        '<!doctype html><html lang="en"><head><title>Not found</title></head>'
        "<body></body></html>",
        encoding="utf-8",
    )
    validate_publication_metadata(tmp_path, publication)


def test_publication_metadata_rejects_global_entries_in_locale_sitemap(
    tmp_path: Path,
) -> None:
    publication = _write_metadata_fixture(tmp_path)
    locale_sitemap = tmp_path / "docs" / "zh" / "sitemap.xml"
    locale_sitemap.write_text(
        (tmp_path / "docs" / "sitemap.xml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    with pytest.raises(PageStagingError, match="zh sitemap.*real translations"):
        validate_publication_metadata(tmp_path, publication)


def test_source_maps_are_pruned_from_the_staged_artifact(tmp_path: Path) -> None:
    source_map = tmp_path / "assets" / "bundle.js.map"
    source_map.parent.mkdir()
    source_map.write_text("{}", encoding="utf-8")
    runtime = source_map.with_name("bundle.js")
    runtime.write_text("console.log('runtime');", encoding="utf-8")

    assert prune_source_maps(tmp_path) == ["assets/bundle.js.map"]
    assert not source_map.exists()
    assert runtime.is_file()


def test_search_compaction_preserves_multilingual_data_and_is_idempotent(
    tmp_path: Path,
) -> None:
    index = tmp_path / "docs" / "search" / "search_index.json"
    index.parent.mkdir(parents=True)
    payload = {
        "config": {"lang": ["en", "zh", "hi"], "separator": r"[\s\-]+"},
        "docs": [
            {
                "location": "zh/",
                "title": "中文 दस्तावेज़ 🩺",
                "text": '<p>中文\nclinical "evidence"</p>',
            }
        ],
    }
    index.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")
    before = index.stat().st_size

    saved = compact_search_index(tmp_path)

    assert saved > 0
    assert index.stat().st_size == before - saved
    assert json.loads(index.read_bytes()) == payload
    assert "中文" in index.read_text(encoding="utf-8")
    assert compact_search_index(tmp_path) == 0


def test_search_compaction_fails_closed_on_missing_or_invalid_index(
    subtests, tmp_path
) -> None:
    case_root = tmp_path
    for contents in [None, b"{invalid", b"\xff"]:
        with subtests.test(contents=contents):
            tmp_path = case_root / str(len(list(case_root.iterdir())))
            tmp_path.mkdir()
            index = tmp_path / "docs" / "search" / "search_index.json"
            index.parent.mkdir(parents=True)
            if contents is not None:
                index.write_bytes(contents)

            with pytest.raises(PageStagingError, match="compact staged search index"):
                compact_search_index(tmp_path)


def test_search_compaction_rejects_an_index_symlink(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    index = artifact / "docs" / "search" / "search_index.json"
    index.parent.mkdir(parents=True)
    outside = tmp_path / "outside.json"
    outside.write_text('{"text": "untouched"}', encoding="utf-8")
    try:
        index.symlink_to(outside)
    except OSError:
        pytest.skip("symlink creation is unavailable on this platform")

    with pytest.raises(PageStagingError, match="inside the artifact"):
        compact_search_index(artifact)
    assert outside.read_text(encoding="utf-8") == '{"text": "untouched"}'


def test_release_tag_is_normalized_and_rejects_non_versions() -> None:
    assert resolve_release_tag("2.0.0") == "v2.0.0"
    assert resolve_release_tag("v2.0.0-rc.1") == "v2.0.0-rc.1"
    with pytest.raises(PageStagingError, match="semantic version"):
        resolve_release_tag("main")


def test_output_directory_guard_rejects_non_staging_paths() -> None:
    assert ensure_safe_output_dir(ROOT / "site") == ROOT / "site"
    assert ensure_safe_output_dir(ROOT / "site-preview") == ROOT / "site-preview"
    with pytest.raises(PageStagingError, match="site-"):
        ensure_safe_output_dir(ROOT / "docs")


def test_makefile_and_pages_workflow_use_only_shared_staging_entrypoint() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    workflow = (ROOT / ".github" / "workflows" / "pages.yml").read_text(
        encoding="utf-8"
    )

    for content in (makefile, workflow):
        assert "scripts/docs/stage_pages.py" in content
        assert "rsync " not in content
    assert "mkdocs build --strict" not in workflow
    assert "openmed.eval.leaderboard" not in workflow


def _write_schema(root: Path, relative: str, schema: dict[str, object]) -> Path:
    path = root / "openmed" / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
    return path


def _synthetic_schema(schema_id: str, **extra: object) -> dict[str, object]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": schema_id,
        "title": "Synthetic record",
        "type": "object",
        **extra,
    }


def test_schema_id_host_is_the_pages_custom_domain() -> None:
    cname = (ROOT / "docs" / "website" / "CNAME").read_text(encoding="utf-8")

    assert SCHEMA_ID_HOST == cname.strip() == "openmed.life"
    assert SCHEMA_ID_HOST not in SCHEMA_PENDING_ID_HOSTS
    assert SCHEMA_SOURCE_DIR == ROOT / "openmed"


def test_every_packaged_schema_is_staged_byte_identical_at_its_id_path(
    tmp_path: Path,
) -> None:
    output = tmp_path / "site"
    output.mkdir()

    index_path = stage_schemas(output)
    verify_staged_schemas(output)

    index = json.loads(index_path.read_text(encoding="utf-8"))
    packaged = {
        path.relative_to(ROOT).as_posix(): path
        for path in (ROOT / "openmed").rglob("*.schema.json")
        if "__pycache__" not in path.parts
    }
    sources = [entry["source"] for entry in index["schemas"]]
    assert sorted(sources) == sources
    assert len(sources) == len(set(sources))
    assert set(sources) == set(packaged)
    assert index["host"] == SCHEMA_ID_HOST
    assert index["schema_version"] == 1
    assert index["summary"]["total"] == len(packaged)

    published = 0
    for entry in index["schemas"]:
        source_bytes = packaged[entry["source"]].read_bytes()
        source = json.loads(source_bytes)
        assert entry["$id"] == source["$id"]
        assert entry["sha256"] == hashlib.sha256(source_bytes).hexdigest()
        host = urlsplit(entry["$id"]).hostname
        if host == SCHEMA_ID_HOST:
            published += 1
            assert entry["published"] is True
            assert entry["reason"] is None
            assert entry["path"] == urlsplit(entry["$id"]).path.lstrip("/")
            assert (output / entry["path"]).read_bytes() == source_bytes
        else:
            assert host in SCHEMA_PENDING_ID_HOSTS
            assert entry["published"] is False
            assert entry["path"] is None
            assert entry["reason"] == SCHEMA_UNPUBLISHED_REASON
    assert published > 0
    assert index["summary"]["published"] == published
    assert index["summary"]["unpublished"] == len(packaged) - published

    staged = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file()
    }
    expected = {entry["path"] for entry in index["schemas"] if entry["published"]}
    assert staged == expected | {SCHEMA_INDEX_PATH}
    assert {owner_for_path(path) for path in staged} == {"schemas"}


def test_index_records_declared_schema_versions() -> None:
    entries = {entry["source"]: entry for entry in plan_schema_publication()}

    assert entries["openmed/core/schemas/json/span.schema.json"]["schema_version"] == 1
    assert (
        entries["openmed/core/schemas/json/care_gap.schema.json"]["schema_version"]
        == "1.0.0"
    )
    assert (
        entries["openmed/core/schemas/json/decision_result.schema.json"][
            "schema_version"
        ]
        == "1.2.0"
    )
    assert entries["openmed/core/config.schema.json"]["schema_version"] is None


def test_off_host_schema_id_fails_staging_naming_only_the_file(
    tmp_path: Path,
) -> None:
    source = tmp_path / "openmed"
    _write_schema(
        tmp_path,
        "synthetic/record.schema.json",
        _synthetic_schema("https://schemas.example.invalid/schemas/record.json"),
    )
    output = tmp_path / "site"
    output.mkdir()

    with pytest.raises(PageStagingError) as excinfo:
        stage_schemas(output, source_dir=source)

    message = str(excinfo.value)
    assert message.endswith(": openmed/synthetic/record.schema.json")
    assert "example.invalid" not in message
    assert str(tmp_path) not in message
    assert not (output / SCHEMA_INDEX_PATH).exists()


@pytest.mark.parametrize(
    "schema_id",
    [
        None,
        "",
        "http://openmed.life/schemas/record.schema.json",
        "https://user@openmed.life/schemas/record.schema.json",
        "https://openmed.life:8443/schemas/record.schema.json",
        "https://openmed.life/schemas/record.schema.json?v=1",
        "https://openmed.life/schemas/record.schema.json#root",
        "https://openmed.life/record.schema.json",
        "https://openmed.life/schemas/../index.html",
        "https://openmed.life/schemas//record.schema.json",
        "https://openmed.life/schemas/record",
        "https://openmed.life/schemas/index.json",
        "https://www.openmed.life/schemas/record.schema.json",
        "urn:openmed:record",
    ],
)
def test_unpublishable_schema_ids_are_rejected(schema_id: object) -> None:
    label = "openmed/synthetic/record.schema.json"

    with pytest.raises(PageStagingError) as excinfo:
        schema_publication_path(schema_id, label=label)

    assert str(excinfo.value).endswith(f": {label}")
    if isinstance(schema_id, str) and schema_id:
        assert schema_id not in str(excinfo.value)


def test_pending_id_hosts_are_listed_but_not_copied(tmp_path: Path) -> None:
    source = tmp_path / "openmed"
    _write_schema(
        tmp_path,
        "a/published.schema.json",
        _synthetic_schema("https://openmed.life/schemas/synthetic/a.schema.json"),
    )
    _write_schema(
        tmp_path,
        "b/pending.schema.json",
        _synthetic_schema("https://openmed.dev/schemas/synthetic/b/1.0.0"),
    )
    _write_schema(tmp_path, "b/__pycache__/ignored.schema.json", {"$id": "x"})
    output = tmp_path / "site"
    output.mkdir()

    index = json.loads(stage_schemas(output, source_dir=source).read_text("utf-8"))
    verify_staged_schemas(output, source_dir=source)

    assert [entry["source"] for entry in index["schemas"]] == [
        "openmed/a/published.schema.json",
        "openmed/b/pending.schema.json",
    ]
    pending = index["schemas"][1]
    assert pending["published"] is False
    assert pending["path"] is None
    assert pending["schema_version"] == "1.0.0"
    assert (output / "schemas" / "synthetic" / "a.schema.json").is_file()
    assert not (output / "schemas" / "synthetic" / "b").exists()
    assert index["summary"] == {"published": 1, "total": 2, "unpublished": 1}


def test_schema_staging_refuses_duplicates_and_replacements(tmp_path: Path) -> None:
    source = tmp_path / "openmed"
    schema_id = "https://openmed.life/schemas/synthetic.schema.json"
    _write_schema(tmp_path, "a/first.schema.json", _synthetic_schema(schema_id))
    output = tmp_path / "site"
    (output / "schemas").mkdir(parents=True)
    (output / "schemas" / "synthetic.schema.json").write_text("{}", encoding="utf-8")

    with pytest.raises(PageStagingError, match="would replace staged output"):
        stage_schemas(output, source_dir=source)

    _write_schema(tmp_path, "b/second.schema.json", _synthetic_schema(schema_id))
    with pytest.raises(PageStagingError, match="share one \\$id path"):
        plan_schema_publication(source)


def test_schema_staging_rejects_invalid_json_and_symlinks(tmp_path: Path) -> None:
    source = tmp_path / "openmed"
    broken = source / "broken.schema.json"
    broken.parent.mkdir(parents=True)
    broken.write_text("{not json", encoding="utf-8")

    with pytest.raises(PageStagingError, match="not valid JSON: openmed/broken"):
        plan_schema_publication(source)

    broken.write_text("[]", encoding="utf-8")
    with pytest.raises(PageStagingError, match="JSON object: openmed/broken"):
        plan_schema_publication(source)

    broken.unlink()
    target = tmp_path / "outside.json"
    target.write_text("{}", encoding="utf-8")
    (source / "linked.schema.json").symlink_to(target)
    with pytest.raises(PageStagingError, match="symlink: openmed/linked"):
        packaged_schema_files(source)


def test_staged_schema_verification_detects_drift(tmp_path: Path) -> None:
    source = tmp_path / "openmed"
    _write_schema(
        tmp_path,
        "record.schema.json",
        _synthetic_schema("https://openmed.life/schemas/record.schema.json"),
    )
    output = tmp_path / "site"
    output.mkdir()
    stage_schemas(output, source_dir=source)
    staged = output / "schemas" / "record.schema.json"

    staged.write_text("{}", encoding="utf-8")
    with pytest.raises(PageStagingError, match="differs openmed/record"):
        verify_staged_schemas(output, source_dir=source)

    staged.unlink()
    with pytest.raises(PageStagingError, match="missing openmed/record"):
        verify_staged_schemas(output, source_dir=source)

    _write_schema(
        tmp_path,
        "late.schema.json",
        _synthetic_schema("https://openmed.life/schemas/late.schema.json"),
    )
    with pytest.raises(PageStagingError, match="every packaged schema once"):
        verify_staged_schemas(output, source_dir=source)

    (output / SCHEMA_INDEX_PATH).write_text("{", encoding="utf-8")
    with pytest.raises(PageStagingError, match="unreadable"):
        verify_staged_schemas(output, source_dir=source)


def test_stage_pages_publishes_and_verifies_schemas_before_the_manifest() -> None:
    source = inspect.getsource(stage_pages_module.stage_pages)

    copy_website = source.index("copy_website(DEFAULT_WEBSITE_DIR, output_dir)")
    staging = source.index("stage_schemas(output_dir)")
    verification = source.index("verify_staged_schemas(output_dir)")
    manifest = source.index("write_manifest(")
    assert copy_website < staging < verification < manifest
    assert "SCHEMA_INDEX_PATH" in source
