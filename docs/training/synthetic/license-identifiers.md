# SPDX license identifiers in synthetic lineage

`normalize_spdx_identifier` gives synthetic-lineage manifests one stable,
deterministic answer for candidate SPDX short identifiers: canonical,
normalized, deprecated-alias, unknown, or malformed. It never guesses an
unknown license, never performs a network lookup, and never promotes an
unknown value to permissive.

```python
from openmed.training.synthetic import normalize_spdx_identifier

result = normalize_spdx_identifier("apache-2.0")
print(result.status.value, result.normalized, result.is_permissive)
```

## Supported boundary

Normalization applies only to the committed allowlist
`PERMISSIVE_SPDX_IDENTIFIERS` (0BSD, Apache-2.0, BlueOak-1.0.0,
BSD-2-Clause, BSD-2-Clause-Patent, BSD-3-Clause, BSL-1.0, ISC, MIT, MPL-2.0,
Python-2.0, Unlicense, and Zlib), their case and surrounding-whitespace
variants, the committed deprecated-alias table, and `LicenseRef-` values
from the SPDX grammar. Extending the allowlist is a reviewed policy change,
never an inference from input data.

The deprecated-alias table is deliberately minimal and evidence-based:
`BSD-2-Clause-NetBSD` (deprecated in SPDX license list 3.9 and documented by
SPDX as a match to `BSD-2-Clause`) maps to its permissive successor. No other
alias is inferred.

`LicenseRef-` identifiers must match the SPDX production
`LicenseRef-[A-Za-z0-9.-]+`; the prefix is case sensitive, surrounding
whitespace is trimmed, and the value is never treated as permissive because
its terms are user-defined and outside any registry. The caller owns that
decision.

## Outcomes

Each outcome carries a stable `category`, and results serialize through
`to_dict()`/`to_json()` with a fixed field order and schema version 1:

- `canonical`: an exact allowlist member, or a well-formed `LicenseRef-` value.
- `normalized`: a case variant (`spdx_identifier_normalized_case`), a
  whitespace variant (`spdx_identifier_normalized_whitespace`), or a trimmed
  `LicenseRef-` value (`spdx_identifier_normalized_licenseref_whitespace`).
  `is_permissive` is true only for allowlist members.
- `deprecated_alias`: a committed deprecated identifier with its documented,
  allowlisted successor in `normalized`.
- `unknown`: a syntactically valid SPDX short identifier that is not in the
  allowlist or alias table. `normalized` is `None` and `is_permissive` is
  always false.
- `malformed`: values that are not SPDX short identifiers: non-string input
  (`spdx_identifier_malformed_type`), empty or whitespace-only input
  (`spdx_identifier_malformed_empty`), SPDX expressions such as `A OR B`,
  `A AND B`, `A WITH B`, or the deprecated `+` "or later" shorthand
  (`spdx_identifier_malformed_expression`), and invalid characters or
  `LicenseRef-` syntax (`spdx_identifier_malformed_characters`).

Malformed results never echo the input value. This module is not an SPDX
expression parser, performs no license compatibility analysis, and is not
legal advice.
