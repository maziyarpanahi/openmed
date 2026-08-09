# Anonymizer provider plugins

OpenMed can load locally installed Faker providers and surrogate generators
through the `openmed.providers` Python entry-point group. Discovery is lazy:
the group is enumerated once, on the first anonymizer registry lookup. It does
not download packages or make network calls.

## Entry-point declaration

Add a registrar to your package's `pyproject.toml`:

```toml
[project.entry-points."openmed.providers"]
acme_ids = "acme_openmed_provider:register"
```

The entry-point target is a zero-argument callable with this signature:

```python
def register() -> None:
    """Register this distribution's OpenMed anonymizer extensions."""
```

The registrar runs only after its distribution has been installed. It should
use the existing registration APIs for every provider or generator it exposes:

```python
from faker.providers import BaseProvider

from openmed.core.anonymizer import (
    register_clinical_provider,
    register_label_generator,
)


class AcmeProvider(BaseProvider):
    """Generate synthetic identifiers for the Acme fixture system."""

    def acme_id(self) -> str:
        return self.generator.numerify("ACME-#####")


def generate_acme_id(faker, original: str, *, locale: str) -> str:
    """Return an Acme identifier using the active Faker instance."""

    del original, locale
    return faker.acme_id()


def register() -> None:
    register_clinical_provider(AcmeProvider)
    register_label_generator("ID_NUM", generate_acme_id)
```

Label generators receive `(faker, original, *, locale)` and return a string.
They must not retain or log `original`. A provider-only package may instead
publish a `faker.providers.BaseProvider` subclass directly as its entry-point
target; the registrar form is recommended when a package supplies both a
provider and one or more label generators.

## Runtime behavior and failures

`Anonymizer.surrogate()` triggers discovery before resolving a generator. A
successful registrar affects newly created Faker instances and the process-wide
label registry. Discovery is idempotent, so repeated anonymization does not
reload entry points.

If an entry point cannot be imported or its registrar raises, OpenMed logs a
warning containing only the entry-point name and exception type, then continues
with the remaining plugins and built-in fallbacks. Plugin code runs in the
OpenMed process; install only packages you trust. Keep plugin behavior
local-first and offline for protected health information workflows.
