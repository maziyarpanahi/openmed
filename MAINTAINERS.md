# Maintainers

OpenMed is currently maintained by:

| Maintainer | GitHub | Areas of ownership |
|---|---|---|
| Maziyar Panahi | [@maziyarpanahi](https://github.com/maziyarpanahi) | Project lead; privacy and de-identification; model registry and release engineering; documentation, brand, and website; community and Code of Conduct enforcement |

## Review and merge process

- Pull requests follow the [contributing guide](CONTRIBUTING.md) and the
  [pull request template](.github/PULL_REQUEST_TEMPLATE.md), and should link an
  accepted issue unless a maintainer has confirmed the scope directly.
- CI must pass before merge: lint, format, tests, and any scoped gates
  (`make lint`, `make format-check`, and `.venv/bin/python -m pytest tests/ -q`).
- Maintainers may add focused follow-up commits to a contributor branch when a
  change needs tests, hardening, or conflict resolution. Contributor credit is
  retained when the completed pull request is squash-merged.
- Pull requests stay focused on one feature or fix; unrelated formatting churn
  is kept out of the merge.
- Privacy-sensitive paths (PII extraction, de-identification, logging, service
  request handling) get extra scrutiny for direct-identifier recall, critical
  leakage, and span integrity, per the no-raw-PHI rule.
- Releases are tag-driven; see the [release process](docs/contributing.md#release-outline).
