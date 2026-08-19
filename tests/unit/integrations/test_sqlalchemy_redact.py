from __future__ import annotations

from datetime import datetime

import pytest
from sqlalchemy import Integer, Text, create_engine, select, text
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from openmed.core.pipeline import Pipeline
from openmed.integrations.sqlalchemy_redact import (
    RedactedText,
    RedactionPipelineRegistry,
    SQLAlchemyRedactionConfig,
    install_mapper_redaction,
    install_session_redaction,
)
from openmed.processing.outputs import PredictionResult


class Base(DeclarativeBase):
    pass


class TypedNote(Base):
    __tablename__ = "typed_notes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    body: Mapped[str | None] = mapped_column(RedactedText(), nullable=True)


class HookedNote(Base):
    __tablename__ = "hooked_notes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    body: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(Text, default="new")


class InsertHookNote(Base):
    __tablename__ = "insert_hook_notes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    body: Mapped[str | None] = mapped_column(Text, nullable=True)


class _CountingPipelineFactory:
    def __init__(self) -> None:
        self.created = 0
        self.pipeline: _CountingPipeline | None = None

    def __call__(self, config: SQLAlchemyRedactionConfig) -> _CountingPipeline:
        self.created += 1
        self.pipeline = _CountingPipeline(
            Pipeline(
                model_name=config.model_name,
                confidence_threshold=config.confidence_threshold,
                use_smart_merging=config.use_smart_merging,
                lang=config.lang,
                normalize_accents=config.normalize_accents,
                use_safety_sweep=config.use_safety_sweep,
                policy=config.policy_profile,
                model_detector=_empty_model_detector,
            )
        )
        return self.pipeline


class _CountingPipeline:
    def __init__(self, pipeline: Pipeline) -> None:
        self._pipeline = pipeline
        self.run_calls = 0

    def run(self, *args, **kwargs):
        self.run_calls += 1
        return self._pipeline.run(*args, **kwargs)


def _empty_model_detector(text_value: str, **kwargs) -> PredictionResult:
    return PredictionResult(
        text=text_value,
        entities=[],
        model_name=kwargs["model_name"],
        timestamp=datetime.now().isoformat(),
    )


def _registry() -> tuple[RedactionPipelineRegistry, _CountingPipelineFactory]:
    factory = _CountingPipelineFactory()
    registry = RedactionPipelineRegistry(
        SQLAlchemyRedactionConfig(policy_profile="hipaa_safe_harbor"),
        pipeline_factory=factory,
    )
    return registry, factory


def test_type_and_before_flush_hook_redact_sqlite_writes_and_reuse_pipeline():
    registry, factory = _registry()
    TypedNote.__table__.c.body.type.registry = registry
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    seeded_phi = "SSN 123-45-6789; email patient842@example.test; IP 203.0.113.42"

    with Session(engine) as session:
        registration = install_session_redaction(
            session,
            {HookedNote: "body"},
            registry=registry,
        )
        try:
            session.add_all(
                [
                    TypedNote(body=seeded_phi),
                    TypedNote(body="Account 555-55-1234"),
                    HookedNote(body=seeded_phi),
                    HookedNote(body=None),
                ]
            )
            session.commit()

            hooked = session.scalar(select(HookedNote).where(HookedNote.id == 1))
            assert hooked is not None
            hooked.status = "reviewed"
            session.commit()
        finally:
            registration.remove()

    assert factory.created == 1
    assert factory.pipeline is registry.pipeline
    assert factory.pipeline.run_calls == 3

    with engine.connect() as connection:
        typed_values = (
            connection.execute(text("SELECT body FROM typed_notes ORDER BY id"))
            .scalars()
            .all()
        )
        hooked_values = (
            connection.execute(text("SELECT body FROM hooked_notes ORDER BY id"))
            .scalars()
            .all()
        )

    assert typed_values[0] == hooked_values[0]
    assert typed_values[0] != seeded_phi
    assert typed_values[1] != "Account 555-55-1234"
    assert hooked_values[1] is None
    for stored in (typed_values[0], typed_values[1], hooked_values[0]):
        assert "123-45-6789" not in stored
        assert "patient842@example.test" not in stored
        assert "203.0.113.42" not in stored

    with Session(engine) as session:
        typed_read = session.scalar(select(TypedNote).where(TypedNote.id == 1))
        hooked_read = session.scalar(select(HookedNote).where(HookedNote.id == 1))

    assert typed_read is not None
    assert hooked_read is not None
    assert typed_read.body == typed_values[0]
    assert hooked_read.body == hooked_values[0]
    assert factory.created == 1
    assert factory.pipeline.run_calls == 3


def test_before_insert_hook_redacts_configured_columns():
    registry, factory = _registry()
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    registration = install_mapper_redaction(
        InsertHookNote,
        columns=("body",),
        registry=registry,
    )
    try:
        with Session(engine) as session:
            session.add_all(
                [
                    InsertHookNote(body="SSN 123-45-6789"),
                    InsertHookNote(body="Email insert842@example.test"),
                ]
            )
            session.commit()
    finally:
        registration.remove()

    with engine.connect() as connection:
        values = (
            connection.execute(text("SELECT body FROM insert_hook_notes ORDER BY id"))
            .scalars()
            .all()
        )

    assert factory.created == 1
    assert factory.pipeline is not None
    assert factory.pipeline.run_calls == 2
    assert "123-45-6789" not in values[0]
    assert "insert842@example.test" not in values[1]


def test_configuration_validation_and_non_string_values():
    with pytest.raises(ValueError, match="must match"):
        RedactedText(policy="strict_no_leak", policy_profile="hipaa_safe_harbor")

    registry, _ = _registry()
    with pytest.raises(TypeError, match="str or None"):
        registry.redact(42)  # type: ignore[arg-type]
