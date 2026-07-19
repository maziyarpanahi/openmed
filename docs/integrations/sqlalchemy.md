# SQLAlchemy Write-time Redaction

OpenMed can de-identify clinical free text at the ORM persistence boundary. The
integration is opt-in: use the `RedactedText` type for declarative column-level
protection, or install a SQLAlchemy event hook when an existing schema must keep
ordinary `Text` columns.

Install the optional dependency:

```bash
pip install "openmed[sqlalchemy]"
```

## Redacting text type

`RedactedText` processes non-null values during SQL parameter binding. Values
loaded from the database are returned unchanged because only already-redacted
text is persisted.

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from openmed.integrations.sqlalchemy_redact import RedactedText


class Base(DeclarativeBase):
    pass


class ClinicalNote(Base):
    __tablename__ = "clinical_notes"

    id: Mapped[int] = mapped_column(primary_key=True)
    body: Mapped[str] = mapped_column(
        RedactedText(policy_profile="hipaa_safe_harbor")
    )


engine = create_engine("sqlite+pysqlite:///clinical.db")
Base.metadata.create_all(engine)

with Session(engine) as session:
    session.add(ClinicalNote(body="Patient identifiers are redacted on commit"))
    session.commit()
```

`policy` is accepted as an alias for `policy_profile`.

## Existing `Text` columns

For an existing mapped schema, install a `before_flush` listener on a session
instance, session subclass, or another target accepted by SQLAlchemy's session
event API. The mapping is explicit so unrelated model attributes are never
examined or changed.

```python
from sqlalchemy import Text
from sqlalchemy.orm import Mapped, Session, mapped_column

from openmed.integrations.sqlalchemy_redact import install_session_redaction


class ImportedNote(Base):
    __tablename__ = "imported_notes"

    id: Mapped[int] = mapped_column(primary_key=True)
    body: Mapped[str] = mapped_column(Text)


with Session(engine) as session:
    registration = install_session_redaction(
        session,
        {ImportedNote: ("body",)},
        policy_profile="strict_no_leak",
    )
    try:
        session.add(ImportedNote(body="Clinical free text"))
        session.commit()
    finally:
        registration.remove()
```

New objects are redacted before their first flush. On dirty objects, configured
columns are processed only when their own value changed.

For applications that prefer mapper events, `install_mapper_redaction` attaches
the same behavior to `before_insert` for one mapped class:

```python
from openmed.integrations.sqlalchemy_redact import install_mapper_redaction

registration = install_mapper_redaction(
    ImportedNote,
    columns="body",
    policy_profile="hipaa_safe_harbor",
)
```

Keep the returned registration for as long as the hook is needed, then call
`registration.remove()` during application shutdown or test cleanup.

## Reusing one pipeline

The default registry is cached by immutable redaction configuration. For
explicit application or engine scoping, construct one registry and pass it to
every protected type and event hook. The OpenMed pipeline is created lazily on
the first non-empty write and reused instead of being instantiated per row.

```python
from openmed.integrations.sqlalchemy_redact import (
    RedactedText,
    RedactionPipelineRegistry,
    SQLAlchemyRedactionConfig,
    install_session_redaction,
)

config = SQLAlchemyRedactionConfig(
    policy_profile="strict_no_leak",
    confidence_threshold=0.65,
    lang="en",
)
registry = RedactionPipelineRegistry(config)

# Use RedactedText(registry=registry) on declarative columns, and share it with
# event-based mappings in the same persistence boundary.
registration = install_session_redaction(
    session,
    {ImportedNote: "body"},
    registry=registry,
)
```

Pipeline execution is synchronized because model backends may not be safe to
invoke concurrently through one instance. Create separate registries when
different policies or independent concurrency domains are required.

## Boundaries

- Values must be `str` or `None`; other values fail before persistence.
- Redaction is one-way. This integration does not persist reversible pseudonym
  mappings and never re-identifies values during reads.
- Do not configure both `RedactedText` and an event hook for the same column;
  choose one write boundary per field.
- SQLAlchemy's synchronous ORM events are supported. Native async event wiring
  is outside this integration; `AsyncSession` users may deliberately install a
  hook on the synchronous session class they control after reviewing SQLAlchemy's
  async event guidance.
