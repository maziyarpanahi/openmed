# Security Policy

OpenMed de-identifies protected health information (PHI) and other personal data.
A defect that causes identifiers to leak — a redaction bypass — is a **security**
defect, not an ordinary bug. Please report it **privately** so it can be fixed
before it is disclosed publicly.

The structured analysis of *how* the redactor can fail — adversary model, trust
boundaries, and a catalog of leakage-bypass abuse cases with their mitigations
and known gaps — lives in the
[redactor threat model](docs/security/threat-model.md). Mitigated abuse classes
are backed by synthetic regression tests
(`tests/unit/security/test_redactor_leakage_bypass.py`). The current public docs
and tests intentionally omit actionable reproduction details for unmitigated
bypasses. Report new findings through the vulnerability-reporting process below.

## Reporting a vulnerability

**Use GitHub Private Vulnerability Reporting:**

➡️ **https://github.com/maziyarpanahi/openmed/security/advisories/new**

This opens a private security advisory visible only to you and the maintainers.
It is the only supported channel for security reports.

Please **do not** open a public GitHub issue, pull request, or discussion for a
suspected vulnerability, and do not post details on social media or mailing lists
before a fix is released.

If you do not receive an acknowledgement within **14 calendar days**, treat the
report as unhandled: you may escalate to a maintainer through their public GitHub
profile and, absent a remediation plan, you are released from the coordination
embargo after the disclosure window in [Response targets](#response-targets).

> [!CAUTION]
> **Never include real sensitive data in a report.**
> No PHI, PII, patient records, credentials, API tokens (for example `HF_TOKEN`),
> secrets, database dumps, production exports, or logs/screenshots that contain
> any of these. OpenMed exists to keep such data private — reproduce every issue
> with **synthetic data only** (for example with the bundled
> [`Faker`](https://faker.readthedocs.io/) tooling), and scrub tokens and secrets.
> The private advisory thread is the secure transfer channel — do not email
> artifacts. A report that leaks real data is itself an incident and must be
> flagged as one.

A good report includes:

- The OpenMed version (`python -c "import openmed; print(openmed.__version__)"`),
  Python version, and OS.
- The policy profile in use, if relevant (for example `hipaa_safe_harbor`).
- A minimal, **synthetic** reproduction and the observed vs. expected behavior.
- The impact you believe it has (which identifiers leak, under what conditions).

## If something is disclosed by accident

- **You opened a public issue, PR, or discussion by mistake:** do not add further
  detail. Delete it if you can, then report privately through the advisory link
  above and tell us what happened so we can request removal of cached copies.
- **A report exposed real PHI/PII:** treat it as an incident. Notify us
  immediately through a private advisory and do not re-post the data. Maintainers
  will redact or delete the content, scrub issue and edit history, and request
  search-cache removal.

## What to report privately (in scope)

These classes are security-relevant and must go through private disclosure, never
a public issue:

- **Redaction bypass / PHI or PII leakage** — input that causes a direct
  identifier to survive de-identification, including systematic identifier
  false-negatives that leak data under a documented policy profile.
- **Model and pipeline leakage** — prompt injection that defeats
  de-identification, training-data or memorization disclosure, or
  retrieval/context (RAG) exfiltration that surfaces identifiers. Treated as
  redaction-bypass class.
- **Audit integrity** — forging or tampering with signed audit reports, or
  defeating the reproducibility/HMAC signatures.
- **Surrogate / pseudonym reversibility** — unintended recovery or linkage of
  original values through `replace` / `reversible_id`, or defects in mapping,
  vault, and key handling. Seeded replacement is not encryption and is not
  claimed to provide cryptographic non-invertibility.
- **Secret or credential handling defects** — leakage of tokens, keys, or
  patient data into logs, caches, temp files, or audit artifacts.
- **Supply-chain compromise** — malicious or compromised dependencies, build,
  CI/CD, or release artifacts affecting OpenMed. See
  [`docs/security/dependency-policy.md`](docs/security/dependency-policy.md).
- **Dependency vulnerabilities** that enable remote code execution or code
  execution within OpenMed.

## Out of scope

- The public [redactor threat model](docs/security/threat-model.md) documents
  architecture, threat classes, and mitigations. It does not exempt newly found
  bypasses from this reporting policy: keep actionable details for suspected or
  unmitigated issues in a private advisory until coordinated disclosure.
- Signing-key custody and automated secret-scanning configuration are tracked
  separately and are not part of this policy.
- Model accuracy or quality requests that are **not** a redaction bypass — please
  file those as a normal [issue](https://github.com/maziyarpanahi/openmed/issues).
- Testing of, or findings against, hosted endpoints (for example `openmed.life`
  or Hugging Face–hosted models). This policy covers the OpenMed software only.

## Severity

We triage with [CVSS v4.0](https://www.first.org/cvss/) where it applies, plus a
privacy-impact override.

| Severity | Examples |
|---|---|
| **Critical** | Real PHI/PII leakage or redaction bypass; surrogate/pseudonym reversal without the key; audit-signature forgery |
| **High** | Systematic identifier false-negatives under a documented profile; secret or credential leakage into artifacts |
| **Medium** | Limited-condition leakage; dependency RCE reachable only under non-default configuration |
| **Low** | Hardening gaps with no direct identifier exposure |

Privacy-impacting defects are never rated below **High**.

## Supported versions

Security fixes target the **latest released minor version** (currently the
`1.7.x` line); we do not backport to end-of-life lines. Reproduce on a supported
version before reporting.

| Version | Supported |
|---|---|
| Latest minor (`1.7.x`) | Yes |
| Older releases | Upgrade first |

## Response targets

We aim for the following timeline. "Business days" follow the maintainer's local
calendar.

| Stage | Target |
|---|---|
| Acknowledge your report | within **3 business days** |
| Initial severity assessment | within **7 calendar days** |
| Status updates while a report is open | at least every **7 days** |
| Coordinated public disclosure | within **90 days**, or sooner once a fix ships |

For qualifying vulnerabilities we request a CVE through GitHub and publish a
GitHub Security Advisory when a fix ships. If we do not provide a remediation plan
within the disclosure window, you may disclose publicly after **90 days** without
losing the safe-harbor protection below.

## Conduct and recognition

Unless you ask otherwise, we credit good-faith reporters in the published
advisory. OpenMed does **not** operate a paid bug bounty. Extortion, ransom
demands, or threats to disclose for payment forfeit safe-harbor protection and
will be reported to the relevant authorities. Automated, duplicate, or
no-proof-of-concept reports may be closed without full triage.

## Safe harbor

We support good-faith security research. If you make a good-faith effort to follow
this policy, we will not pursue or support legal action against you for your
research, and we will work with you to understand and resolve the issue quickly.

Good faith means: you only test the OpenMed software (the library, CLI, and
source in this repository) using data you are authorized to use; you do **not**
access, exfiltrate, or expose real PHI/PII; you avoid privacy violations and
service disruption; and you give us reasonable time to remediate — per the
disclosure timeline under [Response targets](#response-targets) above — before any
public disclosure.

To the extent permitted by law, we consider good-faith research conducted under
this policy to be authorized access — including authorized circumvention of
technical measures for research purposes — and if a third party brings action
against you for such research, we will make clear that it was authorized. This
safe harbor covers the software in this repository only, not third-party or hosted
services; it does not bind third parties and does not waive rights against
bad-faith conduct.

Thank you for helping keep OpenMed and the people whose data it protects safe.
