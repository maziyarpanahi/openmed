"""Check release history without importing candidate code or contacting GitHub."""

import json
import subprocess

import pytest

from scripts.release.deprecation_gate import run_gate


def git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], capture_output=True, check=True)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "no-global"))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.name", "Test Maintainer")
    git(tmp_path, "config", "user.email", "test@example.test")
    (tmp_path / "openmed").mkdir()
    api_dir = tmp_path / "docs/api"
    api_dir.mkdir(parents=True)
    (api_dir / "openapi.json").write_text('{"openapi":"3.1.0","paths":{}}')
    git(tmp_path, "add", "docs/api/openapi.json")
    return tmp_path


def commit(repo, source, tag=None):
    (repo / "openmed" / "__init__.py").write_text(source)
    git(repo, "add", "openmed/__init__.py")
    git(repo, "commit", "--allow-empty", "-qm", "synthetic release")
    if tag:
        git(repo, "tag", tag)


def test_major_searches_all_stable_previous_tags(repo):
    source = "def legacy(): pass\ndef current(): pass\n"
    commit(repo, '@deprecated(since="4.3",remove_in="5.0")\n' + source, "v4.3.0")
    commit(repo, source, "v4.10.0")
    commit(repo, "")
    assert run_gate(repo, "5.0.0", "HEAD") == ["openmed.current"]


@pytest.mark.parametrize(
    "notice",
    [
        '@deprecated(since="4.3",remove_in="6.0")',
        '@deprecated(since="4.3",remove_in=unknown)',
    ],
)
def test_candidate_only_and_late_notices_fail(repo, notice):
    commit(
        repo,
        notice + "\ndef late(): pass\ndef candidate_only(value=None): pass\n",
        "v4.3.0",
    )
    commit(
        repo,
        '@deprecated(since="4.3",remove_in="5.0")\ndef candidate_only(value): pass\n',
    )
    assert run_gate(repo, "5.0.0", "HEAD") == ["openmed.candidate_only", "openmed.late"]


def test_minor_patch_skip_and_major_requires_release(repo):
    assert run_gate(repo, "4.3.0", "missing") == []
    assert run_gate(repo, "4.3.1", "missing") == []
    with pytest.raises(ValueError, match="previous_major_tags_missing"):
        run_gate(repo, "5.0.0", "HEAD")


def test_released_openapi_notices_bind_operation_and_property(repo):
    def spec(tag, paths, fields):
        payload = {
            "paths": paths,
            "components": {
                "schemas": {"Result": {"type": "object", "properties": fields}}
            },
        }
        (repo / "docs/api/openapi.json").write_text(json.dumps(payload))
        git(repo, "add", "docs/api/openapi.json")
        commit(repo, "", tag)

    earlier = {
        "/warned": {"delete": {"deprecated": True}},
        "/unchanged": {"get": {}},
        "/silent": {"post": {}},
    }
    spec("v4.2.0", earlier, {"old": {"deprecated": True}, "required_review": {}})
    earlier["/warned"]["delete"] = {}
    spec("v4.10.0", earlier, {"old": {}, "required_review": {}})
    spec(None, {"/unchanged": {"get": {}}}, {})
    assert set(run_gate(repo, "5.0.0", "HEAD")) == {
        "REST operation POST /silent",
        "REST schema Result required_review",
    }
    assert run_gate(repo, "5.1.0", "HEAD") == []


def test_inline_response_properties_and_removed_operation_scope(repo):
    def save(paths, tag):
        (repo / "docs/api/openapi.json").write_text(json.dumps({"paths": paths}))
        git(repo, "add", "docs/api/openapi.json")
        commit(repo, "", tag)

    response = {
        "200": {
            "content": {"application/json": {"schema": {"properties": {"unused": {}}}}}
        }
    }
    save(
        {
            "/retire": {"get": {"deprecated": True, "responses": response}},
            "/stay": {"post": {"responses": response}},
        },
        "v4.3.0",
    )
    save({"/stay": {"post": {}}}, None)
    assert run_gate(repo, "5.0.0", "HEAD") == [
        "REST operation POST /stay response 200 application/json unused"
    ]
