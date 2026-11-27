"""Plan a labeled OpenMed 4.3 backport and optionally cherry-pick it locally."""

from __future__ import annotations

import argparse
import difflib
import json
import re
import subprocess
import tempfile
from pathlib import Path

if __package__:
    from . import api_surface_diff, check_repo_policy
else:
    import api_surface_diff
    import check_repo_policy


class BackportError(ValueError):
    """Controlled policy or local Git refusal reason."""


def _git(root, *args):
    reply = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True
    )
    if reply.returncode:
        raise BackportError("git_operation_failed")
    return reply.stdout.strip()


def _sha(root, ref):
    return _git(root, "rev-parse", "--verify", "--end-of-options", f"{ref}^{{commit}}")


def github_lookup(endpoint, *, paginate=False):
    """Read candidate provenance from the canonical repository."""
    command = ["gh", "api", f"repos/maziyarpanahi/openmed/{endpoint}"]
    if paginate:
        command.extend(["--paginate", "--slurp"])
    reply = subprocess.run(command, capture_output=True, text=True)
    if reply.returncode:
        raise BackportError("github_lookup_failed")
    data = json.loads(reply.stdout)
    return [row for page in data for row in page] if paginate else data


def _eligible(pr):
    return pr.get("merged_at") and any(
        label["name"] == "backport-4.3" for label in pr["labels"]
    )


def _changelog_lines(text):
    section = ""
    result = []
    for line in text.splitlines():
        if line.startswith("## "):
            section = ""
        if line.startswith("### "):
            section = line[4:].strip()
        result.append((section, line))
    return result


def _check_changelog(before, after):
    left, right = map(_changelog_lines, (before, after))
    entry = False
    for tag, i, j, k, m in difflib.SequenceMatcher(
        None, left, right, autojunk=False
    ).get_opcodes():
        if tag == "equal":
            continue
        for section, line in left[i:j] + right[k:m]:
            if line.strip() and section not in {"Fixed", "Security"}:
                raise BackportError("changelog_section_refused")
        entry |= any(line.lstrip().startswith(("- ", "* ")) for _, line in right[k:m])
    if not entry:
        raise BackportError("fix_changelog_entry_required")


def backport(
    root: Path, *, target: str, commits=(), pr=None, apply=False, lookup=github_lookup
):
    """Check a labeled fix against the latest reachable LTS tag in a scratch clone.

    Args:
        root: Local non-shallow repository with source commits and LTS tags.
        target: Local LTS branch to use as the prospective base.
        commits: Explicit commits from merged, eligible pull requests in pick order.
        pr: Alternatively, one merged, eligible pull request number.
        apply: Cherry-pick only in a clean checkout already on the target.
        lookup: Injectable read-only GitHub REST lookup.
    """
    if bool(commits) == (pr is not None) or not target or target.startswith("-"):
        raise BackportError("invalid_inputs")
    base = _sha(root, f"refs/heads/{target}")
    if target in {"master", "main"}:
        raise BackportError("maintenance_branch_required")
    tags = _git(root, "tag", "--merged", base, "--list", "v4.3.*").splitlines()
    versions = [t for t in tags if re.fullmatch(r"v4\.3\.(0|[1-9][0-9]*)", t)]
    if not versions:
        raise BackportError("lts_tag_missing")
    tag = max(versions, key=lambda t: int(t.rsplit(".", 1)[1]))
    if pr is not None:
        if type(pr) is not int or pr < 1 or not _eligible(lookup(f"pulls/{pr}")):
            raise BackportError("eligible_merged_pr_required")
        commits = [
            item["sha"]
            for item in lookup(f"pulls/{pr}/commits?per_page=100", paginate=True)
        ]
    picks = [_sha(root, commit) for commit in commits]
    if not picks or len(set(picks)) != len(picks):
        raise BackportError("invalid_commits")
    for sha in picks:
        if pr is None and not any(
            _eligible(p)
            for p in lookup(f"commits/{sha}/pulls?per_page=100", paginate=True)
        ):
            raise BackportError("eligible_merged_pr_required")
        parents = _git(root, "show", "-s", "--format=%P", sha).split()
        if len(parents) != 1:
            raise BackportError("single_parent_commit_required")
        if check_repo_policy.check_commit_range(parents[0], sha, repo_root=root):
            raise BackportError("attribution_refused")

    with tempfile.TemporaryDirectory() as temp:
        scratch = Path(temp) / "candidate"
        _git(
            root,
            "clone",
            "--shared",
            "--no-checkout",
            "--quiet",
            "--",
            str(root.resolve()),
            str(scratch),
        )
        _git(scratch, "checkout", "--detach", base)
        # A disposable preview never requires a signing key. Real picks inherit
        # the operator's normal signing configuration.
        _git(scratch, "config", "commit.gpgsign", "false")
        for field in ("user.name", "user.email"):
            _git(scratch, "config", field, _git(root, "config", "--get", field))
        _git(scratch, "cherry-pick", "-x", *picks)
        _check_changelog(
            _git(root, "show", f"{base}:CHANGELOG.md"),
            _git(scratch, "show", "HEAD:CHANGELOG.md"),
        )
        if check_repo_policy.check_commit_range(base, "HEAD", repo_root=scratch):
            raise BackportError("attribution_refused")
        if api_surface_diff.compare_refs(scratch, tag, "HEAD").added:
            raise BackportError("public_api_added")
    if apply:
        if (
            _git(root, "symbolic-ref", "--short", "HEAD") != target
            or _sha(root, "HEAD") != base
            or _git(root, "status", "--porcelain")
        ):
            raise BackportError("clean_target_checkout_required")
        try:
            _git(root, "cherry-pick", "-x", *picks)
        except BackportError:
            # A conflict can leave an in-progress sequence. Git's abort restores
            # its pre-sequence state, which was checked clean immediately above.
            subprocess.run(
                ["git", "-C", str(root), "cherry-pick", "--abort"], capture_output=True
            )
            raise
    return {
        "target": target,
        "base": base,
        "lts_tag": tag,
        "commits": picks,
        "applied": apply,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("commits", nargs="*")
    parser.add_argument("--pr", type=int)
    parser.add_argument("--target", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    try:
        plan = backport(
            Path(_git(Path.cwd(), "rev-parse", "--show-toplevel")),
            target=args.target,
            commits=args.commits,
            pr=args.pr,
            apply=args.apply,
        )
    except BackportError as exc:
        parser.exit(2, f"{exc}\n")
    print(json.dumps(plan, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
