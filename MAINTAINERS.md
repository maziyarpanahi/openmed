# Maintainers

OpenMed is currently maintained by:

| Maintainer | GitHub | Areas of ownership |
|---|---|---|
| Maziyar Panahi | [@maziyarpanahi](https://github.com/maziyarpanahi) | Project lead; privacy and de-identification; model registry and release engineering; documentation, brand, and website; community and Code of Conduct enforcement |

## Review and merge process

- Pull requests follow the [contributing guide](CONTRIBUTING.md) and the
  [pull request template](.github/PULL_REQUEST_TEMPLATE.md).
- CI must pass before merge: lint, format, tests, and any scoped gates
  (`make lint`, `make format-check`, and `.venv/bin/python -m pytest tests/ -q`).
- A change is merged by a maintainer after self-review or, for non-trivial
  changes, review by a second set of eyes. PRs are kept small and focused on a
  single feature or fix; unrelated formatting churn is rejected.
- Privacy-sensitive paths (PII extraction, de-identification, logging, service
  request handling) get extra scrutiny for direct-identifier recall, critical
  leakage, and span integrity, per the no-raw-PHI rule.
- Releases are tag-driven; see the [release process](docs/contributing.md#release-outline).
