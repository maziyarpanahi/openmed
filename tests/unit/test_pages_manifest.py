"""Tests for the unified GitHub Pages staging contract."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from xml.etree import ElementTree

import pytest

from scripts.docs.stage_pages import (
    DEFAULT_PUBLICATION,
    MANIFEST_NAME,
    PageStagingError,
    artifact_path_for_url,
    build_manifest,
    changed_snapshot_paths,
    copy_locale_sitemaps,
    ensure_safe_output_dir,
    find_overlay_collisions,
    load_publication,
    markdown_output_path,
    normalized_expected_paths,
    owner_for_path,
    prune_source_maps,
    public_route_for_path,
    resolve_release_tag,
    translated_output_path,
    translated_output_paths,
    validate_internal_references,
    validate_llm_feeds,
    validate_publication_contract,
    validate_publication_metadata,
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


@pytest.mark.parametrize(
    ("path", "route"),
    [
        ("index.html", "/"),
        ("docs/index.html", "/docs/"),
        ("docs/guide/index.html", "/docs/guide/"),
        ("docs/llms.txt", "/docs/llms.txt"),
    ],
)
def test_public_route_for_path(path: str, route: str) -> None:
    assert public_route_for_path(path) == route


@pytest.mark.parametrize(
    ("path", "owner"),
    [
        ("index.html", "marketing"),
        ("docs/index.html", "mkdocs"),
        ("docs/demo/web/index.html", "browser-demo"),
        ("docs/eval/benchmark-leaderboard/index.html", "leaderboard"),
        (MANIFEST_NAME, "staging"),
    ],
)
def test_owner_for_path(path: str, owner: str) -> None:
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


@pytest.mark.parametrize(
    "unsafe",
    [
        "http://openmed.life/docs/",
        "https://user@openmed.life/docs/",
        "https://openmed.life/docs/?preview=1",
    ],
)
def test_publication_metadata_rejects_unsafe_canonical_urls(
    tmp_path: Path,
    unsafe: str,
) -> None:
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
