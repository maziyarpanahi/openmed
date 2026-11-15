"""Tests for the OpenMedKit Swift/Kotlin public API baselines and migration gate."""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
RELEASE = ROOT / "scripts" / "release"
SWIFT_BASELINES = ROOT / "swift" / "OpenMedKit" / "api"
KOTLIN_BASELINE = ROOT / "android" / "openmedkit" / "api" / "openmedkit.api"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, RELEASE / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


native_api = _load("check_native_api_migration")
swift_api = _load("swift_api_baseline")

ASYNC_MANGLED = "$s7KitDemo6ClientC5fetch2idSSSi_tYaKF"
SYNC_MANGLED = "$s7KitDemo6ClientC4load_5limitSaySSGSS_SitKF"
DEMANGLED = {
    ASYNC_MANGLED: "KitDemo.Client.fetch(id: Swift.Int) async throws -> Swift.String",
    SYNC_MANGLED: (
        "KitDemo.Client.load(_: Swift.String, limit: Swift.Int) throws -> "
        "[Swift.String]"
    ),
}


def fake_demangler(names):
    return {name: DEMANGLED.get(name, "") for name in names}


def _type(printed: str, **extra):
    return {"kind": "TypeNominal", "name": printed, "printedName": printed, **extra}


def _conformance(name: str):
    return {"kind": "Conformance", "name": name, "printedName": name}


def synthetic_dump() -> dict:
    """A hand-written digester dump for a fictional ``KitDemo`` module."""

    client = {
        "kind": "TypeDecl",
        "name": "Client",
        "printedName": "Client",
        "declKind": "Class",
        "moduleName": "KitDemo",
        "declAttributes": ["Final"],
        "conformances": [
            _conformance("Copyable"),
            _conformance("Escapable"),
            _conformance("SendableMetatype"),
            _conformance("Sendable"),
        ],
        "children": [
            {
                "kind": "Function",
                "name": "fetch",
                "printedName": "fetch(id:)",
                "declKind": "Func",
                "moduleName": "KitDemo",
                "mangledName": ASYNC_MANGLED,
                "throwing": True,
                "children": [_type("Swift.String"), _type("Swift.Int")],
            },
            {
                "kind": "Function",
                "name": "load",
                "printedName": "load(_:limit:)",
                "declKind": "Func",
                "moduleName": "KitDemo",
                "mangledName": SYNC_MANGLED,
                "throwing": True,
                "static": True,
                "children": [
                    _type("[Swift.String]"),
                    _type("Swift.String"),
                    _type("Swift.Int", hasDefaultArg=True),
                ],
            },
            {
                "kind": "Constructor",
                "name": "init",
                "printedName": "init(name:)",
                "declKind": "Constructor",
                "moduleName": "KitDemo",
                "init_kind": "Convenience",
                "children": [_type("KitDemo.Client"), _type("Swift.String")],
            },
            {
                "kind": "Var",
                "name": "name",
                "printedName": "name",
                "declKind": "Var",
                "moduleName": "KitDemo",
                "children": [_type("Swift.String")],
                "accessors": [
                    {"kind": "Accessor", "accessorKind": "get"},
                    {"kind": "Accessor", "accessorKind": "set"},
                ],
            },
            {
                "kind": "Function",
                "name": "hash",
                "printedName": "hash(into:)",
                "declKind": "Func",
                "moduleName": "KitDemo",
                "implicit": True,
                "children": [_type("()"), _type("Swift.Hasher")],
            },
        ],
    }
    status = {
        "kind": "TypeDecl",
        "name": "Status",
        "printedName": "Status",
        "declKind": "Enum",
        "moduleName": "KitDemo",
        "enumRawTypeName": "String",
        "conformances": [_conformance("RawRepresentable")],
        "children": [
            {
                "kind": "Var",
                "name": "ready",
                "printedName": "ready",
                "declKind": "EnumElement",
                "moduleName": "KitDemo",
                "children": [
                    {
                        "kind": "TypeFunc",
                        "printedName": "(KitDemo.Status.Type) -> KitDemo.Status",
                        "children": [_type("KitDemo.Status")],
                    }
                ],
            },
            {
                "kind": "Var",
                "name": "failed",
                "printedName": "failed(_:)",
                "declKind": "EnumElement",
                "moduleName": "KitDemo",
                "children": [
                    {
                        "kind": "TypeFunc",
                        "printedName": (
                            "(KitDemo.Status.Type) -> (Swift.Int) -> KitDemo.Status"
                        ),
                        "children": [
                            {
                                "kind": "TypeFunc",
                                "printedName": "(Swift.Int) -> KitDemo.Status",
                            }
                        ],
                    }
                ],
            },
        ],
    }
    external = {
        "kind": "TypeDecl",
        "name": "String",
        "printedName": "String",
        "declKind": "Struct",
        "moduleName": "Swift",
        "children": [
            {
                "kind": "Var",
                "name": "kitDemoTag",
                "printedName": "kitDemoTag",
                "declKind": "Var",
                "moduleName": "KitDemo",
                "children": [_type("Swift.String")],
                "accessors": [{"kind": "Accessor", "accessorKind": "get"}],
            }
        ],
    }
    return {
        "ABIRoot": {
            "kind": "Root",
            "name": "KitDemo",
            "printedName": "KitDemo",
            "tool_arguments": ["-sdk", "/Users/someone/private/path"],
            "children": [
                {"kind": "Import", "name": "Foundation", "declKind": "Import"},
                client,
                status,
                external,
                {
                    "kind": "TypeAlias",
                    "name": "Handler",
                    "printedName": "Handler",
                    "declKind": "TypeAlias",
                    "moduleName": "KitDemo",
                    "children": [_type("(Swift.String) -> Swift.Void")],
                },
            ],
        }
    }


def normalized(dump: dict | None = None) -> dict:
    return swift_api.normalize_dump(dump or synthetic_dump(), "KitDemo", fake_demangler)


def test_normalize_dump_renders_stable_declarations():
    symbols = normalized()["symbols"]

    assert symbols == {
        "KitDemo.Client": [
            "conformance KitDemo.Client: Sendable",
            "final class KitDemo.Client",
        ],
        "KitDemo.Client.fetch": [
            "func KitDemo.Client.fetch(id: Swift.Int) async throws -> Swift.String"
        ],
        "KitDemo.Client.init": [
            "convenience KitDemo.Client.init(name: Swift.String) -> KitDemo.Client"
        ],
        "KitDemo.Client.load": [
            "static func KitDemo.Client.load(_: Swift.String, limit: Swift.Int = "
            "default) throws -> [Swift.String]"
        ],
        "KitDemo.Client.name": ["var KitDemo.Client.name: Swift.String { get set }"],
        "KitDemo.Handler": ["typealias KitDemo.Handler = (Swift.String) -> Swift.Void"],
        "KitDemo.Status": [
            "conformance KitDemo.Status: RawRepresentable",
            "enum KitDemo.Status raw(String)",
        ],
        "KitDemo.Status.failed": [
            "case KitDemo.Status.failed(_:): (Swift.Int) -> KitDemo.Status"
        ],
        "KitDemo.Status.ready": ["case KitDemo.Status.ready"],
        "Swift.String.kitDemoTag": [
            "var Swift.String.kitDemoTag: Swift.String { get }"
        ],
    }


def test_normalized_baseline_drops_paths_and_toolchain_markers():
    rendered = swift_api.render_baseline(normalized())

    assert "/Users/" not in rendered
    assert "tool_arguments" not in rendered
    for marker in ("Copyable", "Escapable", "SendableMetatype"):
        assert marker not in rendered
    assert "hash(into:)" not in rendered  # implicit, synthesized members
    assert rendered == swift_api.render_baseline(normalized())  # deterministic


def test_normalize_dump_rejects_a_dump_for_another_module():
    with pytest.raises(swift_api.SwiftApiError):
        swift_api.normalize_dump(synthetic_dump(), "OtherModule", fake_demangler)


@pytest.mark.parametrize(
    ("demangled", "expected"),
    [
        ("M.T.f(_: Swift.Int) async throws -> ()", True),
        ("M.T.f() async -> ()", True),
        ("M.T.f(_: (Swift.Int) async throws -> ()) throws -> ()", False),
        ("M.f<A where A: Swift.Sendable>(_: A) async -> A", True),
        ("(extension in M):Swift.String.f() async -> ()", True),
        ("static M.T.f() -> ()", False),
        ("", False),
        (None, False),
    ],
)
def test_is_async_reads_only_the_declarations_own_effects(demangled, expected):
    assert swift_api.is_async(demangled) is expected


def _committed_from(document: dict) -> dict[str, str]:
    return {"KitDemo": swift_api.render_baseline(document)}


def test_swift_check_names_removed_and_changed_symbols(tmp_path, capsys):
    baseline_dir = tmp_path / "macos"
    swift_api.write_baselines({"KitDemo": normalized()}, baseline_dir)
    dump = synthetic_dump()
    client = dump["ABIRoot"]["children"][1]
    client["children"] = [
        child for child in client["children"] if child["name"] != "name"
    ]  # removes KitDemo.Client.name
    client["children"][1]["children"][2].pop("hasDefaultArg")  # load(limit:)

    exit_code = swift_api.run_check({"KitDemo": normalized(dump)}, baseline_dir)
    output = capsys.readouterr()

    assert exit_code == 1
    assert "removed: KitDemo.Client.name" in output.out
    assert "changed: KitDemo.Client.load" in output.out
    assert "was: static func KitDemo.Client.load" in output.out
    assert "--platform macos --update" in output.err


def test_swift_check_reports_async_change_as_breaking(tmp_path, capsys):
    baseline_dir = tmp_path / "macos"
    swift_api.write_baselines({"KitDemo": normalized()}, baseline_dir)

    def sync_demangler(names):
        return {name: "" for name in names}

    current = swift_api.normalize_dump(synthetic_dump(), "KitDemo", sync_demangler)

    assert swift_api.run_check({"KitDemo": current}, baseline_dir) == 1
    assert "changed: KitDemo.Client.fetch" in capsys.readouterr().out


def test_swift_check_passes_additions_and_lists_them(tmp_path, capsys):
    baseline_dir = tmp_path / "macos"
    swift_api.write_baselines({"KitDemo": normalized()}, baseline_dir)
    dump = synthetic_dump()
    status = dump["ABIRoot"]["children"][2]
    status["conformances"].append(_conformance("Sendable"))
    added_case = copy.deepcopy(status["children"][0])
    added_case.update(name="pending", printedName="pending")
    status["children"].append(added_case)

    exit_code = swift_api.run_check({"KitDemo": normalized(dump)}, baseline_dir)
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "+ KitDemo.Status.pending" in output
    assert "+ KitDemo.Status\n" in output  # new conformance only
    assert "removed" not in output


def test_swift_check_matches_and_emits_github_annotations(
    tmp_path, capsys, monkeypatch
):
    baseline_dir = tmp_path / "ios"
    swift_api.write_baselines({"KitDemo": normalized()}, baseline_dir)
    assert swift_api.run_check({"KitDemo": normalized()}, baseline_dir, "ios") == 0
    assert "match for 1 module(s) on ios" in capsys.readouterr().out

    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    dump = synthetic_dump()
    dump["ABIRoot"]["children"].pop()  # removes KitDemo.Handler
    assert swift_api.run_check({"KitDemo": normalized(dump)}, baseline_dir, "ios") == 1
    assert (
        "::error title=Swift API removed::KitDemo.Handler (ios)"
        in capsys.readouterr().out
    )


def test_cli_normalizes_existing_digester_json(tmp_path, monkeypatch, capsys):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    dump = synthetic_dump()
    dump["ABIRoot"]["name"] = "OpenMedKit"
    (raw_dir / "OpenMedKit.json").write_text(json.dumps(dump), encoding="utf-8")
    monkeypatch.setattr(swift_api, "demangle", fake_demangler)
    baseline_dir = tmp_path / "baselines"

    args = [
        "--module",
        "OpenMedKit",
        "--digester-json",
        str(raw_dir),
        "--baseline-dir",
        str(baseline_dir),
    ]
    assert swift_api.main([*args, "--update"]) == 0
    assert swift_api.main([*args, "--check"]) == 0
    written = json.loads((baseline_dir / "OpenMedKit.json").read_text("utf-8"))
    assert written["module"] == "OpenMedKit"
    assert "updated swift api baselines" in capsys.readouterr().out.lower()


def test_cli_reports_missing_build_without_paths(tmp_path, capsys):
    exit_code = swift_api.main(
        ["--module", "OpenMedKit", "--modules-dir", str(tmp_path / "missing")]
    )
    error = capsys.readouterr().err

    assert exit_code == 2
    assert "OpenMedKit.swiftmodule was not found for macos" in error
    assert str(tmp_path) not in error


def test_default_modules_dir_falls_back_to_swiftpm_5_layout(tmp_path):
    spec = swift_api.PLATFORM_SPECS["macos"]
    legacy = tmp_path / ".build" / "debug"
    (legacy / "OpenMedKit.swiftmodule").mkdir(parents=True)

    assert swift_api.default_modules_dir(spec, "OpenMedKit", tmp_path) == legacy
    (legacy / "Modules" / "OpenMedKit.swiftmodule").mkdir(parents=True)
    assert (
        swift_api.default_modules_dir(spec, "OpenMedKit", tmp_path)
        == legacy / "Modules"
    )


KOTLIN_BEFORE = """\
public final class com/example/kit/Client : java/io/Closeable {
\tpublic fun <init> (Ljava/lang/String;)V
\tpublic fun close ()V
\tpublic final fun fetch (I)Ljava/lang/String;
\tpublic static synthetic fun fetch$default (Lcom/example/kit/Client;IILjava/lang/Object;)Ljava/lang/String;
\tpublic final fun getName ()Ljava/lang/String;
}

public final class com/example/kit/Client$Companion {
\tpublic final fun create ()Lcom/example/kit/Client;
}

public abstract interface class com/example/kit/Listener {
\tpublic abstract fun onEvent (Ljava/lang/String;)V
}

"""


def kotlin_table(text: str = KOTLIN_BEFORE):
    return native_api.parse_kotlin_api_dump(text)


def test_kotlin_parser_keys_members_by_class_and_merges_default_bridges():
    table = kotlin_table()

    assert table["com.example.kit.Client"] == frozenset(
        {"public final class com/example/kit/Client", "supertype java/io/Closeable"}
    )
    assert table["com.example.kit.Client.fetch"] == frozenset(
        {
            "public final fun fetch (I)Ljava/lang/String;",
            "public static synthetic fun fetch$default "
            "(Lcom/example/kit/Client;IILjava/lang/Object;)Ljava/lang/String;",
        }
    )
    assert "com.example.kit.Client.Companion.create" in table
    assert "com.example.kit.Client.<init>" in table
    assert table["com.example.kit.Listener"] == frozenset(
        {"public abstract interface class com/example/kit/Listener"}
    )


def test_kotlin_parser_rejects_members_outside_a_class():
    with pytest.raises(native_api.NativeApiError):
        native_api.parse_kotlin_api_dump("\tpublic fun stray ()V\n")


def _changes(before: str, after: str):
    return {
        change.symbol: change.change
        for change in native_api.diff_tables(
            kotlin_table(before),
            kotlin_table(after),
            platform="kotlin",
            baseline="android/openmedkit/api/openmedkit.api",
        )
    }


def test_kotlin_diff_classifies_removals_changes_and_additions():
    after = (
        KOTLIN_BEFORE.replace(
            "\tpublic final fun getName ()Ljava/lang/String;\n", ""
        )  # removed
        .replace("fetch (I)", "fetch (J)")  # changed parameter type
        .replace(
            "com/example/kit/Client : java/io/Closeable {",
            "com/example/kit/Client : java/io/Closeable, java/io/Serializable {",
        )  # added supertype
        .replace(
            "\tpublic final fun create ()",
            "\tpublic final fun createDefault ()Lcom/example/kit/Client;\n"
            "\tpublic final fun create ()",
        )  # added member
    )

    assert _changes(KOTLIN_BEFORE, after) == {
        "com.example.kit.Client": "added",
        "com.example.kit.Client.Companion.createDefault": "added",
        "com.example.kit.Client.fetch": "changed",
        "com.example.kit.Client.getName": "removed",
    }


def test_kotlin_diff_treats_class_modifier_changes_as_changes():
    after = KOTLIN_BEFORE.replace(
        "public abstract interface class com/example/kit/Listener",
        "public abstract interface class com/example/kit/Listener : "
        "java/util/EventListener",
    ).replace(
        "public final class com/example/kit/Client :",
        "public abstract class com/example/kit/Client :",
    )

    changes = _changes(KOTLIN_BEFORE, after)

    assert changes == {
        "com.example.kit.Client": "changed",
        "com.example.kit.Listener": "added",
    }


@pytest.mark.parametrize(
    ("document", "documented"),
    [
        ("Replace `OpenMedKit.OpenMed.analyzeText` with ...", True),
        ("Call `OpenMedKit.OpenMed.analyzeText(_:confidenceThreshold:)` now.", True),
        ("We removed OpenMedKit.OpenMed.analyzeText.", True),
        ("Use `OpenMedKit.OpenMed.analyzeTextChunked` instead.", False),
        ("Use `OpenMedKit.OpenMed.analyzeText.Options` instead.", False),
        ("Use `XOpenMedKit.OpenMed.analyzeText` instead.", False),
        ("Nothing to see here.", False),
    ],
)
def test_migration_symbols_match_with_identifier_boundaries(document, documented):
    change = native_api.BaselineChange(
        platform="swift",
        baseline="swift/OpenMedKit/api/macos/OpenMedKit.json",
        symbol="OpenMedKit.OpenMed.analyzeText",
        change="removed",
    )

    missing = native_api.missing_migration_symbols([change], {"doc.md": document})

    assert (missing == ()) is documented


def test_missing_migration_symbols_ignores_additions():
    added = native_api.BaselineChange(
        platform="kotlin", baseline="x.api", symbol="com.example.New", change="added"
    )

    assert native_api.missing_migration_symbols([added], {}) == ()


def test_swift_baseline_parser_rejects_unknown_schema():
    with pytest.raises(native_api.NativeApiError):
        native_api.parse_swift_baseline('{"schema_version": 99, "symbols": {}}')
    with pytest.raises(native_api.NativeApiError):
        native_api.parse_swift_baseline("not json")


# --- Git-level migration gate --------------------------------------------------

GIT_ENV = {
    "GIT_AUTHOR_NAME": "Test Maintainer",
    "GIT_AUTHOR_EMAIL": "maintainer@example.org",
    "GIT_COMMITTER_NAME": "Test Maintainer",
    "GIT_COMMITTER_EMAIL": "maintainer@example.org",
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
}


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), "-c", "commit.gpgsign=false", *args],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, **GIT_ENV},
    )
    return result.stdout


SWIFT_REL = "swift/OpenMedKit/api/macos/KitDemo.json"
KOTLIN_REL = "android/openmedkit/api/openmedkit.api"


def _write(repo: Path, relative: str, text: str) -> None:
    path = repo / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture
def gate_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _write(repo, SWIFT_REL, swift_api.render_baseline(normalized()))
    _write(repo, KOTLIN_REL, KOTLIN_BEFORE)
    _write(repo, "docs/migration/3.2-to-3.3.md", "# Old guide\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "base")
    _git(repo, "switch", "-q", "-c", "feature")
    return repo


def _commit(repo: Path, message: str = "change") -> None:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message)


def _gate(repo: Path, *args: str) -> int:
    return native_api.main(["--repo-root", str(repo), "--base", "main", *args])


def _remove_swift_symbol(repo: Path) -> None:
    document = normalized()
    del document["symbols"]["KitDemo.Client.name"]
    _write(repo, SWIFT_REL, swift_api.render_baseline(document))


def test_gate_passes_additions_without_a_migration_entry(gate_repo, capsys):
    document = normalized()
    document["symbols"]["KitDemo.Client.reset"] = ["func KitDemo.Client.reset() -> ()"]
    _write(gate_repo, SWIFT_REL, swift_api.render_baseline(document))
    _write(
        gate_repo,
        KOTLIN_REL,
        KOTLIN_BEFORE.replace(
            "\tpublic fun close ()V\n",
            "\tpublic fun close ()V\n\tpublic final fun reset ()V\n",
        ),
    )
    _commit(gate_repo)

    assert _gate(gate_repo, "--head", "feature") == 0
    output = capsys.readouterr().out
    assert "0 breaking symbol(s), 2 added symbol(s)" in output


def test_gate_fails_when_no_migration_entry_changed(gate_repo, capsys):
    _remove_swift_symbol(gate_repo)
    _commit(gate_repo)

    assert _gate(gate_repo) == 1
    output = capsys.readouterr()
    assert "removed: KitDemo.Client.name (swift/OpenMedKit/api/macos/KitDemo.json)" in (
        output.out
    )
    assert "no docs/migration/*.md file was added or modified" in output.err


def test_gate_fails_when_the_migration_entry_omits_a_symbol(gate_repo, capsys):
    _remove_swift_symbol(gate_repo)
    _write(
        gate_repo,
        KOTLIN_REL,
        KOTLIN_BEFORE.replace("\tpublic final fun getName ()Ljava/lang/String;\n", ""),
    )
    _write(
        gate_repo,
        "docs/migration/3.3-to-4.0.md",
        "# 4.0\n\n- `KitDemo.Client.name` was removed.\n",
    )
    _commit(gate_repo)

    assert _gate(gate_repo) == 1
    error = capsys.readouterr().err
    assert "  - com.example.kit.Client.getName" in error
    assert "KitDemo.Client.name\n" not in error
    assert "docs/migration/3.3-to-4.0.md" in error


def test_gate_passes_when_the_migration_entry_names_every_symbol(gate_repo, capsys):
    _remove_swift_symbol(gate_repo)
    _write(
        gate_repo,
        KOTLIN_REL,
        KOTLIN_BEFORE.replace("fetch (I)", "fetch (J)"),
    )
    _write(
        gate_repo,
        "docs/migration/3.2-to-3.3.md",
        "# Old guide\n\n- `KitDemo.Client.name` is gone.\n"
        "- `com.example.kit.Client.fetch` takes a `Long`.\n",
    )
    _commit(gate_repo)

    assert _gate(gate_repo) == 0
    assert "names every breaking symbol" in capsys.readouterr().out


def test_gate_platform_filter_and_merge_commit_base(gate_repo):
    _write(
        gate_repo,
        KOTLIN_REL,
        KOTLIN_BEFORE.replace("\tpublic final fun getName ()Ljava/lang/String;\n", ""),
    )
    _commit(gate_repo)
    _git(gate_repo, "switch", "-q", "main")
    _git(gate_repo, "merge", "-q", "--no-ff", "-m", "merge", "feature")

    # CI checks out the PR merge commit and compares it with its first parent.
    def merge_gate(platform: str) -> int:
        return native_api.main(
            ["--repo-root", str(gate_repo), "--base", "HEAD^1", "--platform", platform]
        )

    assert merge_gate("swift") == 0
    assert merge_gate("kotlin") == 1
    assert merge_gate("all") == 1


def test_gate_worktree_mode_reads_uncommitted_files(gate_repo):
    _remove_swift_symbol(gate_repo)
    assert _gate(gate_repo, "--head", "WORKTREE") == 1

    _write(gate_repo, "docs/migration/3.3-to-4.0.md", "`KitDemo.Client.name`\n")
    assert _gate(gate_repo, "--head", "WORKTREE") == 0


def test_gate_treats_a_deleted_baseline_as_removed_symbols(gate_repo, capsys):
    (gate_repo / SWIFT_REL).unlink()
    _commit(gate_repo)

    assert _gate(gate_repo) == 1
    assert "removed: KitDemo.Client.fetch" in capsys.readouterr().out


def test_gate_reports_unreadable_refs_with_exit_code_2(gate_repo, capsys):
    assert native_api.main(["--repo-root", str(gate_repo), "--base", "nope"]) == 2
    assert "check-native-api-migration:" in capsys.readouterr().err


# --- Committed baselines and CI wiring ----------------------------------------


def _library_products(manifest: Path) -> list[str]:
    text = manifest.read_text(encoding="utf-8")
    return re.findall(r'\.library\(\s*name:\s*"([^"]+)"', text)


@pytest.mark.parametrize("platform_name", ["macos", "ios"])
def test_every_library_product_has_a_committed_swift_baseline(platform_name):
    products = _library_products(ROOT / "Package.swift")

    assert sorted(products) == sorted(swift_api.MODULES)
    assert sorted(
        _library_products(ROOT / "swift" / "OpenMedKit" / "Package.swift")
    ) == sorted(products)
    for module in products:
        path = SWIFT_BASELINES / platform_name / f"{module}.json"
        document = json.loads(path.read_text(encoding="utf-8"))
        assert document["module"] == module
        assert document["schema_version"] == native_api.SWIFT_SCHEMA_VERSION
        native_api.parse_swift_baseline(path.read_text(encoding="utf-8"))
        assert path.read_text(encoding="utf-8") == swift_api.render_baseline(document)


def test_committed_swift_baselines_cover_the_parity_checklist():
    macos = json.loads(
        (SWIFT_BASELINES / "macos" / "OpenMedKit.json").read_text("utf-8")
    )["symbols"]
    ios_extensions = {
        module: json.loads(
            (SWIFT_BASELINES / "ios" / f"{module}.json").read_text("utf-8")
        )["symbols"]
        for module in ("OpenMedShareExtension", "OpenMedActionExtension")
    }

    for symbol in (
        "OpenMedKit.OpenMed.analyzeText",
        "OpenMedKit.OpenMed.extractPII",
        "OpenMedKit.OpenMed.extractPIIChunked",
        "OpenMedKit.EntityPrediction.label",
        "OpenMedKit.OpenMedModelStore.downloadMLXModel",
        "OpenMedKit.OpenMedMLXModelCacheState.ready",
    ):
        assert symbol in macos
    assert any(
        "async throws" in line
        for line in macos["OpenMedKit.OpenMedModelStore.downloadMLXModel"]
    )
    assert (
        "OpenMedShareExtension.RedactionShareViewController"
        in (ios_extensions["OpenMedShareExtension"])
    )
    assert (
        "OpenMedActionExtension.RedactionActionRequestHandler"
        in (ios_extensions["OpenMedActionExtension"])
    )


def test_committed_baselines_contain_no_local_paths():
    files = [*SWIFT_BASELINES.rglob("*.json"), KOTLIN_BASELINE]
    assert len(files) == 2 * len(swift_api.MODULES) + 1
    for path in files:
        text = path.read_text(encoding="utf-8")
        assert "/Users/" not in text
        assert "/home/" not in text
        assert "tool_arguments" not in text


def test_committed_kotlin_baseline_covers_the_parity_checklist():
    table = native_api.parse_kotlin_api_dump(KOTLIN_BASELINE.read_text("utf-8"))

    for symbol in (
        "com.openmed.openmedkit.OpenMed.analyzeText",
        "com.openmed.openmedkit.OpenMed.extractPII",
        "com.openmed.openmedkit.OpenMed.extractPIIChunked",
        "com.openmed.openmedkit.EntityPrediction.getLabel",
        "com.openmed.openmedkit.OpenMedModelStore.isMLXModelCached",
    ):
        assert symbol in table


def test_kotlin_binary_compatibility_validator_is_pinned_and_applied():
    catalog = tomllib.loads(
        (ROOT / "android" / "gradle" / "libs.versions.toml").read_text("utf-8")
    )
    plugin = catalog["plugins"]["kotlinx-binary-compatibility-validator"]
    build = (ROOT / "android" / "openmedkit" / "build.gradle.kts").read_text("utf-8")

    assert plugin == {
        "id": "org.jetbrains.kotlinx.binary-compatibility-validator",
        "version": {"ref": "binary-compatibility-validator"},
    }
    assert re.fullmatch(
        r"\d+\.\d+\.\d+", catalog["versions"]["binary-compatibility-validator"]
    )
    assert "alias(libs.plugins.kotlinx.binary.compatibility.validator)" in build


def _steps(workflow: str, job: str) -> list[dict]:
    document = yaml.safe_load(
        (ROOT / ".github" / "workflows" / workflow).read_text(encoding="utf-8")
    )
    return document["jobs"][job]["steps"]


def _runs(steps: list[dict]) -> str:
    return "\n".join(str(step.get("run", "")) for step in steps)


def test_swift_workflow_checks_both_platform_baselines_and_the_migration_gate():
    build = _runs(_steps("swift-test.yml", "build-and-test"))
    ios_steps = _steps("swift-test.yml", "api-baseline-ios")
    policy = _steps("swift-test.yml", "api-migration-policy")

    assert build.index("swift build") < build.index(
        "swift_api_baseline.py --platform macos --check"
    )
    assert "-derivedDataPath .build/xcode-ios" in _runs(ios_steps)
    assert "swift_api_baseline.py --platform ios --check" in _runs(ios_steps)
    assert policy[0]["with"]["fetch-depth"] == 2
    assert "--platform swift --base HEAD^1 --head HEAD" in _runs(policy)


def test_android_workflow_runs_api_check_and_the_migration_gate():
    build = _runs(_steps("android-ci.yml", "build-and-test"))
    policy = _steps("android-ci.yml", "api-migration-policy")

    assert "./gradlew :openmedkit:apiCheck" in build
    assert policy[0]["with"]["fetch-depth"] == 2
    assert "--platform kotlin --base HEAD^1 --head HEAD" in _runs(policy)


@pytest.mark.parametrize(
    ("workflow", "scripts"),
    [
        (
            "swift-test.yml",
            {
                "scripts/release/swift_api_baseline.py",
                "scripts/release/check_native_api_migration.py",
            },
        ),
        ("android-ci.yml", {"scripts/release/check_native_api_migration.py"}),
    ],
)
def test_workflows_rerun_when_the_baseline_tooling_changes(workflow, scripts):
    document = yaml.safe_load(
        (ROOT / ".github" / "workflows" / workflow).read_text(encoding="utf-8")
    )
    triggers = document.get("on", document.get(True))
    for event in ("push", "pull_request"):
        assert scripts <= set(triggers[event]["paths"])
