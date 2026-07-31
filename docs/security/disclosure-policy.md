# Security & Responsible Disclosure

OpenMed de-identifies PHI and other personal data, so a **redaction bypass or
PHI/PII leak is a security defect** — report it **privately**, never as a public
issue.

- **Report privately:** [open a GitHub security advisory](https://github.com/maziyarpanahi/openmed/security/advisories/new).
- **Full policy:** [SECURITY.md](https://github.com/maziyarpanahi/openmed/blob/master/SECURITY.md)
  — supported versions, response targets, scope, and safe-harbor terms.

!!! danger "Never include real PHI/PII in a report"
    OpenMed exists to keep such data private. Reproduce issues with **synthetic
    data** and redact any sample text. A report that leaks real data is itself an
    incident.

The canonical policy lives in
[`SECURITY.md`](https://github.com/maziyarpanahi/openmed/blob/master/SECURITY.md)
at the repository root; this page is a pointer so it stays a single source of
truth.
