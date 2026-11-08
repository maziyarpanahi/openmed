"""R8-minified consumer gate tests with synthetic dex, APK and R8 output fixtures."""

from __future__ import annotations

import io
import json
import struct
import zipfile
from pathlib import Path

import pytest

from scripts.android.check_r8_consumer import (
    APK_NATIVE_LIBRARY_MISSING,
    CONSUMER_RULE_NOT_APPLIED,
    DESKTOP_NATIVE_RESOURCE_PACKAGED,
    EXIT_FAIL,
    EXIT_INPUT_ERROR,
    EXIT_PASS,
    JNI_CLASS_NOT_KEPT,
    JNI_MEMBER_NOT_KEPT,
    JNI_NATIVE_BINDING_RENAMED,
    NETWORK_PERMISSION_DECLARED,
    R8_DONTWARN_NOT_ALLOWLISTED,
    R8_WARNINGS_IGNORED,
    SERVICE_PROVIDER_MISSING,
    SIZE_BUDGET_EXCEEDED,
    Expectation,
    Finding,
    GateInputError,
    demangle_jni_symbol,
    dontwarn_patterns,
    drop_keep_rule,
    jni_class_references,
    main,
    measure_sizes,
    merged_manifest_permissions,
    parse_mapping,
    parse_usage,
    read_allowlist,
    read_dex,
    read_output_metadata,
    read_rules,
    verify,
)

ACC_PUBLIC = 0x1
ACC_NATIVE = 0x100
NO_INDEX = 0xFFFFFFFF

ORT_KEEP = "-keep class ai.onnxruntime.** { *; }"
CONSUMER_RULES = (
    f"# JNI\n{ORT_KEEP}\n-keep class ai.djl.engine.EngineException {{ *; }}\n"
)
JNI_BLOB = (
    b"\x7fELF\0ai/onnxruntime/TensorInfo\0(JJLai/onnxruntime/TensorInfo;)V\0"
    b"Java_ai_onnxruntime_OrtSession_run__JJ\0"
    # Rust concatenates literals without separators.
    b"Couldn't get java string!ai/djl/engine/EngineExceptionsrc/ndarray/cmp.rs\0"
)
MANIFEST = """<manifest xmlns:android="http://schemas.android.com/apk/res/android">
    <uses-permission android:name="android.permission.ACCESS_NETWORK_STATE" />
    <application android:label="smoke" />
</manifest>
"""


def _uleb(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        out.append(byte | (0x80 if value else 0))
        if not value:
            return bytes(out)


def _dex(classes: dict[str, dict[str, list[str]]]) -> bytes:
    """Build a minimal dex: string, type, field, method ids, class defs and data.

    Each class maps to ``fields``, ``methods`` and ``native`` member names.
    """
    strings: list[str] = []

    def string_index(value: str) -> int:
        if value not in strings:
            strings.append(value)
        return strings.index(value)

    names = list(classes)
    descriptors = [string_index("L" + name.replace(".", "/") + ";") for name in names]
    fields: list[tuple[int, int]] = []
    methods: list[tuple[int, int, int]] = []
    for class_index, name in enumerate(names):
        members = classes[name]
        for field_name in members.get("fields", []):
            fields.append((class_index, string_index(field_name)))
        for method_name in members.get("methods", []):
            methods.append((class_index, string_index(method_name), ACC_PUBLIC))
        for method_name in members.get("native", []):
            access = ACC_PUBLIC | ACC_NATIVE
            methods.append((class_index, string_index(method_name), access))

    header_size = 0x70
    string_ids_off = header_size
    type_ids_off = string_ids_off + 4 * len(strings)
    field_ids_off = type_ids_off + 4 * len(names)
    method_ids_off = field_ids_off + 8 * len(fields)
    class_defs_off = method_ids_off + 8 * len(methods)
    data_off = class_defs_off + 32 * len(names)

    data = bytearray()
    string_offsets = []
    for value in strings:
        string_offsets.append(data_off + len(data))
        encoded = value.encode("utf-8")
        data += _uleb(len(value)) + encoded + b"\0"
    class_data_offsets = []
    for class_index in range(len(names)):
        class_fields = [i for i, item in enumerate(fields) if item[0] == class_index]
        class_methods = [i for i, item in enumerate(methods) if item[0] == class_index]
        class_data_offsets.append(data_off + len(data))
        data += _uleb(0) + _uleb(len(class_fields)) + _uleb(len(class_methods))
        data += _uleb(0)
        previous = 0
        for index in class_fields:
            data += _uleb(index - previous) + _uleb(ACC_PUBLIC)
            previous = index
        previous = 0
        for index in class_methods:
            data += _uleb(index - previous) + _uleb(methods[index][2]) + _uleb(0)
            previous = index

    body = bytearray()
    body += b"".join(struct.pack("<I", offset) for offset in string_offsets)
    body += b"".join(struct.pack("<I", index) for index in descriptors)
    body += b"".join(
        struct.pack("<HHI", class_index, class_index, name_index)
        for class_index, name_index in fields
    )
    body += b"".join(
        struct.pack("<HHI", class_index, 0, name_index)
        for class_index, name_index, _ in methods
    )
    for class_index in range(len(names)):
        body += struct.pack(
            "<IIIIIIII",
            class_index,
            ACC_PUBLIC,
            NO_INDEX,
            0,
            NO_INDEX,
            0,
            class_data_offsets[class_index],
            0,
        )
    header = bytearray(header_size)
    header[0:8] = b"dex\n035\0"
    struct.pack_into(
        "<IIII", header, 0x38, len(strings), string_ids_off, len(names), type_ids_off
    )
    struct.pack_into(
        "<IIII", header, 0x50, len(fields), field_ids_off, len(methods), method_ids_off
    )
    struct.pack_into("<II", header, 0x60, len(names), class_defs_off)
    return bytes(header + body + data)


KEPT_CLASSES: dict[str, dict[str, list[str]]] = {
    "ai.onnxruntime.TensorInfo": {"fields": ["shape"], "methods": ["<init>"]},
    "ai.onnxruntime.OrtSession": {"methods": ["close"], "native": ["run"]},
    "ai.djl.engine.Engine": {"methods": ["getEngine"]},
    "ai.djl.engine.EngineException": {"methods": ["<init>"]},
    "ai.djl.engine.EngineProvider": {},
    "ai.djl.engine.rust.RsEngineProvider": {"methods": ["<init>"]},
}
MAPPING = """# compiler: R8
ai.onnxruntime.TensorInfo -> ai.onnxruntime.TensorInfo:
    long[] shape -> shape
    0:3:void <init>(long[]):10:10 -> <init>
ai.onnxruntime.OrtSession -> ai.onnxruntime.OrtSession:
    0:2:void close():5:5 -> close
ai.djl.engine.Engine -> ai.djl.engine.Engine:
ai.djl.engine.EngineException -> ai.djl.engine.EngineException:
"""


def _apk(
    path: Path,
    *,
    classes: dict[str, dict[str, list[str]]] | None = None,
    natives: dict[str, bytes] | None = None,
    extra: dict[str, bytes] | None = None,
) -> Path:
    natives = (
        {"onnxruntime4j_jni": JNI_BLOB, "djl_tokenizer": b"\0"}
        if natives is None
        else natives
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "classes.dex", _dex(KEPT_CLASSES if classes is None else classes)
        )
        for library, payload in natives.items():
            info = zipfile.ZipInfo(f"lib/arm64-v8a/lib{library}.so")
            archive.writestr(info, payload)
        archive.writestr(
            "META-INF/services/ai.djl.engine.EngineProvider",
            "# providers\nai.djl.engine.rust.RsEngineProvider\n",
        )
        for name, payload in (extra or {}).items():
            archive.writestr(name, payload)
    return path


def _verify(apk: Path, **overrides: object):
    arguments: dict[str, object] = {
        "apks": {"arm64-v8a": apk},
        "mapping_text": MAPPING,
        "usage_text": "",
        "configuration_text": "# section\n"
        + CONSUMER_RULES
        + "-dontwarn androidx.**\n",
        "merged_manifest": MANIFEST,
        "consumer_rules_text": CONSUMER_RULES,
        "dontwarn_allowlist_text": "# reviewed\nandroidx.**\n",
        "jni_packages": ["ai/onnxruntime", "ai/djl", "com/openmed"],
        "required_native_libraries": ["onnxruntime4j_jni", "djl_tokenizer"],
    }
    arguments.update(overrides)
    return verify(**arguments)  # type: ignore[arg-type]


def _codes(report) -> set[str]:
    return {finding.code for finding in report.findings}


def test_read_dex_returns_defined_classes_and_member_names() -> None:
    classes = read_dex(_dex(KEPT_CLASSES))

    assert set(classes) == set(KEPT_CLASSES)
    session = classes["ai.onnxruntime.OrtSession"]
    assert session.methods == {"close", "run"}
    assert session.native_methods == {"run"}
    assert classes["ai.onnxruntime.TensorInfo"].fields == {"shape"}


@pytest.mark.parametrize("payload", [b"", b"PK\x03\x04" + bytes(200)])
def test_read_dex_rejects_non_dex_bytes(payload: bytes) -> None:
    with pytest.raises(GateInputError):
        read_dex(payload)


def test_read_dex_rejects_truncated_tables() -> None:
    with pytest.raises(GateInputError):
        read_dex(_dex(KEPT_CLASSES)[:0x90])


def test_kept_jni_surface_passes(tmp_path: Path) -> None:
    report = _verify(_apk(tmp_path / "a.apk"))

    assert report.passed, report.findings
    assert report.counts["jniClassReferences"] == 2
    assert report.counts["jniNativeBindings"] == 1
    assert report.counts["serviceProviders"] == 1


def test_longest_universe_name_wins_over_rust_literal_concatenation() -> None:
    universe = {
        "ai/djl/engine/Engine",
        "ai/djl/engine/EngineException",
        "ai/onnxruntime/TensorInfo",
    }

    references = jni_class_references(JNI_BLOB, universe, ["ai/djl", "ai/onnxruntime"])

    assert references == {"ai.djl.engine.EngineException", "ai.onnxruntime.TensorInfo"}


def test_renamed_jni_class_fails(tmp_path: Path) -> None:
    classes = dict(KEPT_CLASSES)
    classes["a.b"] = classes.pop("ai.djl.engine.EngineException")
    mapping = MAPPING.replace(
        "ai.djl.engine.EngineException -> ai.djl.engine.EngineException:",
        "ai.djl.engine.EngineException -> a.b:",
    )

    report = _verify(_apk(tmp_path / "a.apk", classes=classes), mapping_text=mapping)

    assert report.findings == [
        Finding(JNI_CLASS_NOT_KEPT, "ai.djl.engine.EngineException", "renamed")
    ]


def test_removed_jni_class_fails(tmp_path: Path) -> None:
    classes = dict(KEPT_CLASSES)
    del classes["ai.onnxruntime.TensorInfo"]
    mapping = MAPPING.replace(
        "ai.onnxruntime.TensorInfo -> ai.onnxruntime.TensorInfo:\n"
        "    long[] shape -> shape\n"
        "    0:3:void <init>(long[]):10:10 -> <init>\n",
        "",
    )

    report = _verify(
        _apk(tmp_path / "a.apk", classes=classes),
        mapping_text=mapping,
        usage_text="ai.onnxruntime.TensorInfo\n",
    )

    assert Finding(JNI_CLASS_NOT_KEPT, "ai.onnxruntime.TensorInfo", "removed") in (
        report.findings
    )


def test_renamed_and_removed_members_of_jni_classes_fail(tmp_path: Path) -> None:
    classes = dict(KEPT_CLASSES)
    classes["ai.onnxruntime.TensorInfo"] = {"fields": ["a"], "methods": []}
    mapping = MAPPING.replace("long[] shape -> shape", "long[] shape -> a")
    usage = "ai.onnxruntime.TensorInfo:\n    public void <init>(long[])\n"

    report = _verify(
        _apk(tmp_path / "a.apk", classes=classes),
        mapping_text=mapping,
        usage_text=usage,
    )

    assert set(report.findings) == {
        Finding(JNI_MEMBER_NOT_KEPT, "ai.onnxruntime.TensorInfo.shape", "renamed"),
        Finding(JNI_MEMBER_NOT_KEPT, "ai.onnxruntime.TensorInfo.<init>", "removed"),
    }


def test_inline_frames_and_r8_synthetics_are_not_renames(tmp_path: Path) -> None:
    mapping = MAPPING.replace(
        "    0:3:void <init>(long[]):10:10 -> <init>\n",
        "    0:3:void <init>(long[]):10:10 -> <init>\n"
        "    4:9:void <init>(long[]):10:10 -> a\n"
        "    0:4:java.lang.String $r8$lambda$abc(java.lang.String):0:0 -> b\n",
    )

    assert _verify(_apk(tmp_path / "a.apk"), mapping_text=mapping).passed


def test_native_binding_breaks_only_when_renamed(tmp_path: Path) -> None:
    renamed_method = dict(KEPT_CLASSES)
    renamed_method["ai.onnxruntime.OrtSession"] = {
        "methods": ["close"],
        "native": ["a"],
    }
    mapping = MAPPING.replace(
        "    0:2:void close():5:5 -> close\n",
        "    0:2:void close():5:5 -> close\n    void run(long,long) -> a\n",
    )
    method_report = _verify(
        _apk(tmp_path / "method.apk", classes=renamed_method), mapping_text=mapping
    )

    renamed_class = dict(KEPT_CLASSES)
    renamed_class["c.d"] = renamed_class.pop("ai.onnxruntime.OrtSession")
    class_report = _verify(
        _apk(tmp_path / "class.apk", classes=renamed_class),
        mapping_text=MAPPING.replace(
            "ai.onnxruntime.OrtSession -> ai.onnxruntime.OrtSession:",
            "ai.onnxruntime.OrtSession -> c.d:",
        ),
    )

    removed = dict(KEPT_CLASSES)
    removed["ai.onnxruntime.OrtSession"] = {"methods": ["close"]}
    removed_report = _verify(
        _apk(tmp_path / "removed.apk", classes=removed),
        usage_text="ai.onnxruntime.OrtSession:\n    private native void run(long,long)\n",
    )

    assert method_report.findings == [
        Finding(JNI_NATIVE_BINDING_RENAMED, "ai.onnxruntime.OrtSession.run", "method")
    ]
    assert class_report.findings == [
        Finding(JNI_NATIVE_BINDING_RENAMED, "ai.onnxruntime.OrtSession", "class")
    ]
    assert removed_report.passed


def test_demangles_jni_exports() -> None:
    assert demangle_jni_symbol("Java_ai_onnxruntime_OrtSession_run__JJ") == (
        "ai.onnxruntime.OrtSession",
        "run",
    )
    assert demangle_jni_symbol(
        "Java_ai_djl_huggingface_tokenizers_jni_TokenizersLibrary_encode"
    ) == ("ai.djl.huggingface.tokenizers.jni.TokenizersLibrary", "encode")
    assert demangle_jni_symbol("Java_a_b_1c_Outer_00024Inner_do_1it") == (
        "a.b_c.Outer$Inner",
        "do_it",
    )
    assert demangle_jni_symbol("Java_run") is None
    assert demangle_jni_symbol("JNI_OnLoad") is None


def test_desktop_jni_resources_and_missing_android_libraries_fail(
    tmp_path: Path,
) -> None:
    apk = _apk(
        tmp_path / "a.apk",
        natives={"onnxruntime4j_jni": JNI_BLOB},
        extra={
            "native/lib/osx-aarch64/cpu/libtokenizers.dylib": b"\0",
            "com/sun/jna/win32-x86-64/jnidispatch.dll": b"\0",
            "lib/arm64-v8a/notes.txt": b"",
        },
    )

    report = _verify(apk)

    assert set(report.findings) == {
        Finding(
            DESKTOP_NATIVE_RESOURCE_PACKAGED,
            "native/lib/osx-aarch64/cpu/libtokenizers.dylib",
        ),
        Finding(
            DESKTOP_NATIVE_RESOURCE_PACKAGED, "com/sun/jna/win32-x86-64/jnidispatch.dll"
        ),
        Finding(APK_NATIVE_LIBRARY_MISSING, "arm64-v8a:libdjl_tokenizer.so"),
    }


def test_internet_permission_fails_unless_removed() -> None:
    declared = MANIFEST.replace(
        "<application",
        '<uses-permission android:name="android.permission.INTERNET" />\n    <application',
    )
    removed = MANIFEST.replace(
        "<application",
        '<uses-permission android:name="android.permission.INTERNET"\n'
        '        tools:node="remove" />\n    <application',
    )

    assert "android.permission.INTERNET" in merged_manifest_permissions(declared)
    assert "android.permission.INTERNET" not in merged_manifest_permissions(removed)
    assert merged_manifest_permissions(MANIFEST) == {
        "android.permission.ACCESS_NETWORK_STATE"
    }


def test_network_permission_finding(tmp_path: Path) -> None:
    manifest = MANIFEST.replace(
        "<application",
        '<uses-permission android:name="android.permission.INTERNET" />\n<application',
    )

    report = _verify(_apk(tmp_path / "a.apk"), merged_manifest=manifest)

    assert _codes(report) == {NETWORK_PERMISSION_DECLARED}


def test_missing_class_suppression_fails(tmp_path: Path) -> None:
    configuration = (
        CONSUMER_RULES + "-dontwarn androidx.**\n-dontwarn ai.djl.**, java.awt.**\n"
        "-ignorewarnings\n-dontwarn\n"
    )

    report = _verify(_apk(tmp_path / "a.apk"), configuration_text=configuration)

    assert set(report.findings) == {
        Finding(R8_WARNINGS_IGNORED, "-ignorewarnings"),
        Finding(R8_DONTWARN_NOT_ALLOWLISTED, "-dontwarn ai.djl.**"),
        Finding(R8_DONTWARN_NOT_ALLOWLISTED, "-dontwarn java.awt.**"),
        Finding(R8_DONTWARN_NOT_ALLOWLISTED, "-dontwarn"),
    }


def test_dontwarn_parsing_and_allowlist_limits() -> None:
    rules = read_rules(
        "-dontwarn a.**,b.C # trailing\n  -dontwarn   c.D\n-keep class x\n"
    )

    assert dontwarn_patterns(rules) == ["a.**", "b.C", "c.D"]
    assert read_allowlist("# comment\nsun.misc.Unsafe\n\n") == {"sun.misc.Unsafe"}
    for entry in ("**", "*", "com.openmed.**"):
        with pytest.raises(GateInputError):
            read_allowlist(entry + "\n")


def test_consumer_rules_must_reach_r8_ignoring_whitespace(tmp_path: Path) -> None:
    spaced = CONSUMER_RULES.replace(ORT_KEEP, "-keep   class ai.onnxruntime.**  { *; }")
    configuration = spaced.replace(
        "-keep class ai.djl.engine.EngineException { *; }", ""
    )

    report = _verify(
        _apk(tmp_path / "a.apk"),
        configuration_text=configuration + "-dontwarn androidx.**\n",
    )

    assert report.findings == [
        Finding(
            CONSUMER_RULE_NOT_APPLIED,
            "-keep class ai.djl.engine.EngineException { *; }",
        )
    ]


def test_service_provider_must_survive_when_its_service_does(tmp_path: Path) -> None:
    classes = dict(KEPT_CLASSES)
    del classes["ai.djl.engine.rust.RsEngineProvider"]
    without_service = dict(classes)
    del without_service["ai.djl.engine.EngineProvider"]

    report = _verify(_apk(tmp_path / "a.apk", classes=classes))
    unused = _verify(_apk(tmp_path / "b.apk", classes=without_service))

    assert report.findings == [
        Finding(
            SERVICE_PROVIDER_MISSING,
            "ai.djl.engine.rust.RsEngineProvider",
            "ai.djl.engine.EngineProvider",
        )
    ]
    assert unused.passed


def test_mapping_and_usage_parsers() -> None:
    mapping = parse_mapping(
        [
            "# comment",
            "a.B -> a.B:",
            '# {"id":"sourceFile","fileName":"B.java"}',
            "    int count -> a",
            "    1:2:void run():3:4 -> run",
            "    5:6:void other.Klass.inlined():7:7 -> b",
            "c.D -> e.f:",
        ]
    )
    usage = parse_usage(
        [
            "x.Removed",
            "a.B:",
            "    private static final int CONSTANT",
            "    public static a.B create(java.lang.String,long[])",
        ]
    )

    assert mapping.classes == {"a.B": "a.B", "c.D": "e.f"}
    assert mapping.renamed_members == {"a.B": {"count"}}
    assert usage.removed_classes == {"x.Removed"}
    assert usage.removed_members == {"a.B": {"CONSTANT", "create"}}


def _write_variant(tmp_path: Path, *, classes=None) -> list[str]:
    apk_dir = tmp_path / "apk"
    apk_dir.mkdir()
    _apk(apk_dir / "smoke-arm64-v8a-release.apk", classes=classes)
    _metadata(apk_dir, {"arm64-v8a": "smoke-arm64-v8a-release.apk"})
    mapping_dir = tmp_path / "mapping"
    mapping_dir.mkdir()
    (mapping_dir / "mapping.txt").write_text(MAPPING, encoding="utf-8")
    (mapping_dir / "usage.txt").write_text("", encoding="utf-8")
    (mapping_dir / "configuration.txt").write_text(
        CONSUMER_RULES + "-dontwarn androidx.**\n", encoding="utf-8"
    )
    (tmp_path / "AndroidManifest.xml").write_text(MANIFEST, encoding="utf-8")
    (tmp_path / "consumer-rules.pro").write_text(CONSUMER_RULES, encoding="utf-8")
    (tmp_path / "allowlist.txt").write_text("androidx.**\n", encoding="utf-8")
    return [
        "verify",
        "--apk-dir",
        str(apk_dir),
        "--mapping",
        str(mapping_dir / "mapping.txt"),
        "--merged-manifest",
        str(tmp_path / "AndroidManifest.xml"),
        "--consumer-rules",
        str(tmp_path / "consumer-rules.pro"),
        "--dontwarn-allowlist",
        str(tmp_path / "allowlist.txt"),
        "--jni-package",
        "ai/onnxruntime",
        "--jni-package",
        "ai/djl",
        "--require-native-library",
        "onnxruntime4j_jni",
        "--report",
        str(tmp_path / "report.properties"),
    ]


def _metadata(apk_dir: Path, outputs: dict[str, str]) -> None:
    elements = [
        {
            "type": "ONE_OF_MANY",
            "filters": [{"filterType": "ABI", "value": abi}],
            "outputFile": name,
        }
        for abi, name in outputs.items()
    ]
    (apk_dir / "output-metadata.json").write_text(
        json.dumps({"version": 3, "elements": elements}), encoding="utf-8"
    )


def _run(argv: list[str]) -> tuple[int, str, str]:
    stdout, stderr = io.StringIO(), io.StringIO()
    code = main(argv, stdout=stdout, stderr=stderr)
    return code, stdout.getvalue(), stderr.getvalue()


def test_cli_verify_passes_and_writes_a_value_free_report(tmp_path: Path) -> None:
    code, stdout, _ = _run(_write_variant(tmp_path))

    assert code == EXIT_PASS
    assert stdout.startswith("PASS R8 consumer gate: 0 findings")
    assert str(tmp_path) not in stdout
    report = (tmp_path / "report.properties").read_text(encoding="utf-8")
    assert "status=PASS\nfindings=0\n" in report
    assert str(tmp_path) not in report


def test_cli_negative_control_requires_the_expected_finding(tmp_path: Path) -> None:
    classes = dict(KEPT_CLASSES)
    classes["e.f"] = classes.pop("ai.onnxruntime.TensorInfo")
    argv = _write_variant(tmp_path, classes=classes)
    mapping = tmp_path / "mapping" / "mapping.txt"
    mapping.write_text(
        MAPPING.replace(
            "ai.onnxruntime.TensorInfo -> ai.onnxruntime.TensorInfo:",
            "ai.onnxruntime.TensorInfo -> e.f:",
        ),
        encoding="utf-8",
    )

    failing_code, failing_stdout, _ = _run(argv)
    expected_code, expected_stdout, _ = _run(
        argv + ["--expect-failure", "jni_class_not_kept:ai.onnxruntime."]
    )
    wrong_code, wrong_stdout, _ = _run(
        argv + ["--expect-failure", "jni_class_not_kept:ai.djl."]
    )

    assert failing_code == EXIT_FAIL
    assert "FAIL jni_class_not_kept ai.onnxruntime.TensorInfo (renamed)" in (
        failing_stdout
    )
    assert expected_code == EXIT_PASS
    assert "EXPECTED jni_class_not_kept ai.onnxruntime.TensorInfo" in expected_stdout
    assert "PASS negative control: the gate failed as expected" in expected_stdout
    assert wrong_code == EXIT_FAIL
    assert "FAIL negative control: expected findings not reported" in wrong_stdout
    assert "status=FAIL" in (tmp_path / "report.properties").read_text()


def test_cli_negative_control_fails_when_the_gate_passes(tmp_path: Path) -> None:
    code, stdout, _ = _run(
        _write_variant(tmp_path) + ["--expect-failure", "jni_class_not_kept"]
    )

    assert code == EXIT_FAIL
    assert "FAIL negative control" in stdout


def test_cli_verify_reports_missing_inputs_without_paths(tmp_path: Path) -> None:
    argv = _write_variant(tmp_path)
    (tmp_path / "mapping" / "usage.txt").unlink()

    code, stdout, stderr = _run(argv)

    assert code == EXIT_INPUT_ERROR
    assert stdout == ""
    assert stderr == "error: R8 usage.txt is missing or unreadable\n"


def test_expectation_parsing() -> None:
    expectation = Expectation.parse("jni_class_not_kept:ai.onnxruntime.")

    assert expectation.matches(Finding(JNI_CLASS_NOT_KEPT, "ai.onnxruntime.A"))
    assert not expectation.matches(Finding(JNI_CLASS_NOT_KEPT, "ai.djl.A"))
    assert Expectation.parse("code").describe() == "code"
    with pytest.raises(GateInputError):
        Expectation.parse(":subject")


# --------------------------------------------------------------------------
# size


def _split_apks(directory: Path, sizes: dict[str, int], *, native: bool) -> None:
    directory.mkdir()
    outputs = {}
    for abi, padding in sizes.items():
        name = f"app-{abi}-release.apk"
        with zipfile.ZipFile(directory / name, "w") as archive:
            archive.writestr("classes.dex", bytes(16))
            if native:
                info = zipfile.ZipInfo(f"lib/{abi}/libonnxruntime.so")
                archive.writestr(info, bytes(padding))
        outputs[abi] = name
    _metadata(directory, outputs)


def _budgets(path: Path, values: dict[str, int]) -> Path:
    path.write_text(
        "# budgets\naar.maxBytes=1\n"
        + "".join(
            f"consumerApk.{abi}.maxDeltaBytes={v}\n" for abi, v in values.items()
        ),
        encoding="utf-8",
    )
    return path


def test_size_reports_per_abi_deltas_against_the_baseline(tmp_path: Path) -> None:
    _split_apks(tmp_path / "kit", {"arm64-v8a": 4000, "x86_64": 6000}, native=True)
    _split_apks(tmp_path / "base", {"arm64-v8a": 0, "x86_64": 0}, native=False)
    budgets = _budgets(
        tmp_path / "b.properties", {"arm64-v8a": 10_000, "x86_64": 10_000}
    )

    code, stdout, _ = _run(
        [
            "size",
            "--apk-dir",
            str(tmp_path / "kit"),
            "--baseline-apk-dir",
            str(tmp_path / "base"),
            "--budgets",
            str(budgets),
            "--report",
            str(tmp_path / "size.properties"),
        ]
    )

    report = (tmp_path / "size.properties").read_text(encoding="utf-8")
    assert code == EXIT_PASS
    assert stdout.startswith("PASS arm64-v8a: OpenMedKit adds ")
    assert "native libraries: libonnxruntime.so 4000" in stdout
    assert "status=PASS\nabis=arm64-v8a,x86_64\n" in report
    assert "x86_64.maxDeltaBytes=10000\n" in report
    assert "x86_64.nativeLibraryBytes=6000\n" in report
    assert str(tmp_path) not in stdout + report


def test_size_budget_overrun_fails(tmp_path: Path) -> None:
    _split_apks(tmp_path / "kit", {"arm64-v8a": 4000}, native=True)
    _split_apks(tmp_path / "base", {"arm64-v8a": 0}, native=False)
    budgets = _budgets(tmp_path / "b.properties", {"arm64-v8a": 100})

    code, stdout, _ = _run(
        [
            "size",
            "--apk-dir",
            str(tmp_path / "kit"),
            "--baseline-apk-dir",
            str(tmp_path / "base"),
            "--budgets",
            str(budgets),
        ]
    )

    assert code == EXIT_FAIL
    assert f"FAIL {SIZE_BUDGET_EXCEEDED} arm64-v8a" in stdout


@pytest.mark.parametrize(
    "budgets",
    [
        {"arm64-v8a": 10_000},
        {"arm64-v8a": 10_000, "x86_64": 10_000, "x86": 10_000},
    ],
)
def test_size_budgets_must_cover_exactly_the_built_abis(
    tmp_path: Path, budgets: dict[str, int]
) -> None:
    _split_apks(tmp_path / "kit", {"arm64-v8a": 1, "x86_64": 1}, native=True)
    _split_apks(tmp_path / "base", {"arm64-v8a": 0, "x86_64": 0}, native=False)
    budget_file = _budgets(tmp_path / "b.properties", budgets)

    with pytest.raises(GateInputError):
        measure_sizes(
            read_output_metadata(tmp_path / "kit"),
            read_output_metadata(tmp_path / "base"),
            dict(
                line.split("=", 1)
                for line in budget_file.read_text().splitlines()
                if "=" in line
            ),
        )


def test_size_rejects_mismatched_splits_and_bad_budgets(tmp_path: Path) -> None:
    _split_apks(tmp_path / "kit", {"arm64-v8a": 1}, native=True)
    _split_apks(tmp_path / "base", {"x86": 0}, native=False)
    kit = read_output_metadata(tmp_path / "kit")

    with pytest.raises(GateInputError):
        measure_sizes(kit, read_output_metadata(tmp_path / "base"), {})
    with pytest.raises(GateInputError):
        measure_sizes(kit, kit, {"consumerApk.arm64-v8a.maxDeltaBytes": "-5"})


def test_output_metadata_must_split_by_abi(tmp_path: Path) -> None:
    (tmp_path / "output-metadata.json").write_text(
        json.dumps({"elements": [{"filters": [], "outputFile": "app.apk"}]})
    )

    with pytest.raises(GateInputError):
        read_output_metadata(tmp_path)
    with pytest.raises(GateInputError):
        read_output_metadata(tmp_path / "missing")


# --------------------------------------------------------------------------
# drop-keep-rule


def _aar(path: Path, proguard: str | None) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("AndroidManifest.xml", "<manifest/>")
        info = zipfile.ZipInfo("classes.jar")
        info.compress_type = zipfile.ZIP_DEFLATED
        archive.writestr(info, bytes(64))
        if proguard is not None:
            archive.writestr("proguard.txt", proguard)
    return path


def test_drop_keep_rule_removes_exactly_one_rule(tmp_path: Path) -> None:
    source = _aar(tmp_path / "in.aar", CONSUMER_RULES)
    output = tmp_path / "nested" / "out.aar"

    code, stdout, _ = _run(
        [
            "drop-keep-rule",
            "--aar",
            str(source),
            "--output",
            str(output),
            "--rule",
            "-keep  class ai.onnxruntime.** { *; }",
        ]
    )

    with zipfile.ZipFile(source) as before, zipfile.ZipFile(output) as after:
        assert before.namelist() == after.namelist()
        assert after.getinfo("classes.jar").compress_type == zipfile.ZIP_DEFLATED
        assert after.read("classes.jar") == before.read("classes.jar")
        rules = after.read("proguard.txt").decode()
    assert code == EXIT_PASS
    assert (
        stdout
        == "wrote out.aar without 1 line(s) of -keep  class ai.onnxruntime.** { *; }\n"
    )
    assert ORT_KEEP not in rules
    assert "-keep class ai.djl.engine.EngineException { *; }" in rules


@pytest.mark.parametrize("proguard", [None, "-keep class other.** { *; }\n"])
def test_drop_keep_rule_refuses_a_no_op(tmp_path: Path, proguard: str | None) -> None:
    source = _aar(tmp_path / "in.aar", proguard)

    with pytest.raises(GateInputError):
        drop_keep_rule(source, tmp_path / "out.aar", ORT_KEEP)
    assert not (tmp_path / "out.aar").exists()
