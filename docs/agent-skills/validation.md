# Agent Skills validation

The repository skill catalog is validated locally and in CI by:

```bash
python scripts/skills/validate.py
```

The gate is deterministic and local-only. It does not fetch content from the
network, resolve web links, import the OpenMed package, or make network
requests.

It checks:

- YAML frontmatter, kebab-case folder/name identifiers, descriptions, metadata,
  body length, and duplicate identifiers;
- relative Markdown links and repository-bound referenced files;
- membership of every `skills/<name>` directory in the committed
  `.claude-plugin/marketplace.json` pack, with no duplicate or unknown entries;
- the `--help` exit status of executable helpers under `skills/` and
  `scripts/skills/` with a temporary home, offline environment flags, proxy
  blocking, discarded output, and no ambient credential variables.

Failures contain only a repository path, an optional line number, and a fixed
diagnostic. Skill bodies, parser details, and helper output are intentionally
not included in logs. Symlinked helpers and skill directories are rejected
rather than followed.

Every executable helper added to the skill workflow must have a successful,
offline `--help` command and a focused test. The focused gate is
`tests/unit/skills/test_validation.py`; broad repository and platform testing
remains owned by the other CI workflows.

Use `--help` to inspect the command without validating or writing anything:

```bash
python scripts/skills/validate.py --help
```
