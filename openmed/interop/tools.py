"""Canonical callable definitions for OpenMed's public agent tools.

The registry in this module deliberately depends only on the Python library and
Pydantic.  Service and transport layers may reuse these argument models, but
importing the registry never imports FastAPI or :mod:`openmed.service`.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Final, Literal, Optional

from openmed.core.policy import canonical_policy_name
from openmed.utils.gateway import normalize_text, validate_language
from openmed.utils.validation import (
    validate_confidence_threshold,
    validate_model_name,
)

try:
    from pydantic import (
        BaseModel,
        ConfigDict,
        Field,
        RootModel,
        field_validator,
        model_validator,
    )

    PYDANTIC_V2 = True
except ImportError:  # pragma: no cover - retained for Pydantic v1 service users
    from pydantic import BaseModel, Field, root_validator, validator

    ConfigDict = None  # type: ignore[assignment]
    RootModel = None  # type: ignore[assignment]
    field_validator = None  # type: ignore[assignment]
    model_validator = None  # type: ignore[assignment]
    PYDANTIC_V2 = False


DEFAULT_PII_MODEL: Final = "OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1"
PIILanguage = Literal[
    "am",
    "as",
    "bn",
    "en",
    "fr",
    "de",
    "it",
    "es",
    "nl",
    "hi",
    "gu",
    "kn",
    "ml",
    "mr",
    "or",
    "pa",
    "ta",
    "te",
    "pt",
    "ar",
    "he",
    "ja",
    "tr",
    "id",
    "th",
    "ko",
    "ro",
    "ru",
    "sv",
    "da",
    "no",
    "sw",
    "zu",
    "xh",
    "zh",
    "uk",
    "cs",
    "el",
]


def _normalize_confidence(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return validate_confidence_threshold(value)


def _normalize_policy(value: Any) -> Optional[str]:
    if value is None:
        return None
    return canonical_policy_name(str(value))


def _normalize_shift_dates(values: dict[str, Any]) -> dict[str, Any]:
    method = values.get("method", "mask")
    shift_dates = values.get("shift_dates")
    date_shift_days = values.get("date_shift_days")

    if shift_dates is True and method != "shift_dates":
        values["method"] = "shift_dates"
        method = "shift_dates"
    elif shift_dates is False and method == "shift_dates":
        raise ValueError("shift_dates=false conflicts with method='shift_dates'")

    if date_shift_days is not None and method != "shift_dates":
        raise ValueError("date_shift_days requires method='shift_dates'")
    return values


class ToolArgsModel(BaseModel):
    """Base class for JSON-serializable tool arguments."""

    if ConfigDict is not None:
        model_config = ConfigDict(extra="forbid")
    else:  # pragma: no cover

        class Config:
            extra = "forbid"

    if PYDANTIC_V2:

        @field_validator("lang", mode="before", check_fields=False)
        @classmethod
        def _validate_language(cls, value: Any) -> str:
            return validate_language(value, include_national_id=False)

    else:  # pragma: no cover

        @validator("lang", pre=True, check_fields=False)
        def _validate_language(cls, value: Any) -> str:
            return validate_language(value, include_national_id=False)


if PYDANTIC_V2:

    class AnalyzeTextArgs(ToolArgsModel):
        """Arguments for :func:`analyze_text`."""

        text: str
        model_name: str = "disease_detection_superclinical"
        confidence_threshold: Optional[float] = Field(default=0.0, ge=0.0, le=1.0)
        group_entities: bool = False
        aggregation_strategy: Optional[Literal["simple", "first", "average", "max"]] = (
            "simple"
        )
        sentence_detection: bool = True
        sentence_language: str = "en"
        sentence_clean: bool = False
        use_fast_tokenizer: bool = True

        @field_validator("text", mode="before")
        @classmethod
        def _validate_text(cls, value: Any) -> str:
            return normalize_text(value)

        @field_validator("model_name")
        @classmethod
        def _validate_model_name(cls, value: str) -> str:
            return validate_model_name(value)

        @field_validator("confidence_threshold")
        @classmethod
        def _validate_confidence_threshold(
            cls, value: Optional[float]
        ) -> Optional[float]:
            return _normalize_confidence(value)

    class ExtractPIIArgs(ToolArgsModel):
        """Arguments for :func:`extract_pii`."""

        text: str
        model_name: str = DEFAULT_PII_MODEL
        confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
        use_smart_merging: bool = True
        lang: PIILanguage = "en"
        normalize_accents: Optional[bool] = None

        @field_validator("text", mode="before")
        @classmethod
        def _validate_text(cls, value: Any) -> str:
            return normalize_text(value)

        @field_validator("model_name")
        @classmethod
        def _validate_model_name(cls, value: str) -> str:
            return validate_model_name(value)

        @field_validator("confidence_threshold")
        @classmethod
        def _validate_confidence_threshold(cls, value: float) -> float:
            normalized = _normalize_confidence(value)
            if normalized is None:
                raise ValueError("confidence_threshold must be a valid number")
            return normalized

    class DeidentifyArgs(ToolArgsModel):
        """Arguments for :func:`deidentify`."""

        text: str
        method: Literal["mask", "remove", "replace", "hash", "shift_dates"] = "mask"
        model_name: str = DEFAULT_PII_MODEL
        confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
        keep_year: bool = False
        shift_dates: Optional[bool] = None
        date_shift_days: Optional[int] = None
        keep_mapping: bool = False
        policy: Optional[str] = None
        use_smart_merging: bool = True
        use_safety_sweep: bool = True
        lang: PIILanguage = "en"
        normalize_accents: Optional[bool] = None

        @field_validator("text", mode="before")
        @classmethod
        def _validate_text(cls, value: Any) -> str:
            return normalize_text(value)

        @field_validator("model_name")
        @classmethod
        def _validate_model_name(cls, value: str) -> str:
            return validate_model_name(value)

        @field_validator("confidence_threshold")
        @classmethod
        def _validate_confidence_threshold(cls, value: float) -> float:
            normalized = _normalize_confidence(value)
            if normalized is None:
                raise ValueError("confidence_threshold must be a valid number")
            return normalized

        @field_validator("policy", mode="before")
        @classmethod
        def _validate_policy(cls, value: Any) -> Optional[str]:
            return _normalize_policy(value)

        @model_validator(mode="after")
        def _validate_shift_dates(self) -> "DeidentifyArgs":
            values = _normalize_shift_dates(self.model_dump())
            for field_name, value in values.items():
                setattr(self, field_name, value)
            return self

    class UnloadModelArgs(ToolArgsModel):
        """Arguments for :func:`unload_model`."""

        model_name: Optional[str] = None
        all: bool = False

        @field_validator("model_name")
        @classmethod
        def _validate_model_name(cls, value: Optional[str]) -> Optional[str]:
            if value is None:
                return None
            return validate_model_name(value)

        @model_validator(mode="after")
        def _validate_target(self) -> "UnloadModelArgs":
            if not self.all and self.model_name is None:
                raise ValueError("model_name is required unless all=true")
            return self

else:  # pragma: no cover

    class AnalyzeTextArgs(ToolArgsModel):
        """Arguments for :func:`analyze_text`."""

        text: str
        model_name: str = "disease_detection_superclinical"
        confidence_threshold: Optional[float] = Field(default=0.0, ge=0.0, le=1.0)
        group_entities: bool = False
        aggregation_strategy: Optional[Literal["simple", "first", "average", "max"]] = (
            "simple"
        )
        sentence_detection: bool = True
        sentence_language: str = "en"
        sentence_clean: bool = False
        use_fast_tokenizer: bool = True

        @validator("text", pre=True)
        def _validate_text(cls, value: Any) -> str:
            return normalize_text(value)

        @validator("model_name")
        def _validate_model_name(cls, value: str) -> str:
            return validate_model_name(value)

        @validator("confidence_threshold")
        def _validate_confidence_threshold(
            cls, value: Optional[float]
        ) -> Optional[float]:
            return _normalize_confidence(value)

    class ExtractPIIArgs(ToolArgsModel):
        """Arguments for :func:`extract_pii`."""

        text: str
        model_name: str = DEFAULT_PII_MODEL
        confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
        use_smart_merging: bool = True
        lang: PIILanguage = "en"
        normalize_accents: Optional[bool] = None

        @validator("text", pre=True)
        def _validate_text(cls, value: Any) -> str:
            return normalize_text(value)

        @validator("model_name")
        def _validate_model_name(cls, value: str) -> str:
            return validate_model_name(value)

        @validator("confidence_threshold")
        def _validate_confidence_threshold(cls, value: float) -> float:
            normalized = _normalize_confidence(value)
            if normalized is None:
                raise ValueError("confidence_threshold must be a valid number")
            return normalized

    class DeidentifyArgs(ToolArgsModel):
        """Arguments for :func:`deidentify`."""

        text: str
        method: Literal["mask", "remove", "replace", "hash", "shift_dates"] = "mask"
        model_name: str = DEFAULT_PII_MODEL
        confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
        keep_year: bool = False
        shift_dates: Optional[bool] = None
        date_shift_days: Optional[int] = None
        keep_mapping: bool = False
        policy: Optional[str] = None
        use_smart_merging: bool = True
        use_safety_sweep: bool = True
        lang: PIILanguage = "en"
        normalize_accents: Optional[bool] = None

        @validator("text", pre=True)
        def _validate_text(cls, value: Any) -> str:
            return normalize_text(value)

        @validator("model_name")
        def _validate_model_name(cls, value: str) -> str:
            return validate_model_name(value)

        @validator("confidence_threshold")
        def _validate_confidence_threshold(cls, value: float) -> float:
            normalized = _normalize_confidence(value)
            if normalized is None:
                raise ValueError("confidence_threshold must be a valid number")
            return normalized

        @validator("policy", pre=True)
        def _validate_policy(cls, value: Any) -> Optional[str]:
            return _normalize_policy(value)

        @root_validator
        def _validate_shift_dates(cls, values: dict[str, Any]) -> dict[str, Any]:
            return _normalize_shift_dates(values)

    class UnloadModelArgs(ToolArgsModel):
        """Arguments for :func:`unload_model`."""

        model_name: Optional[str] = None
        all: bool = False

        @validator("model_name")
        def _validate_model_name(cls, value: Optional[str]) -> Optional[str]:
            if value is None:
                return None
            return validate_model_name(value)

        @root_validator
        def _validate_target(cls, values: dict[str, Any]) -> dict[str, Any]:
            if not values.get("all") and values.get("model_name") is None:
                raise ValueError("model_name is required unless all=true")
            return values


class ListModelsArgs(ToolArgsModel):
    """Arguments for :func:`list_models`."""

    include_registry: bool = True
    include_remote: bool = True


class NoArgs(ToolArgsModel):
    """Empty argument object for tools that take no parameters."""


class _StructuredOutput(BaseModel):
    """Base for forward-compatible structured results."""

    if ConfigDict is not None:
        model_config = ConfigDict(extra="allow")
    else:  # pragma: no cover

        class Config:
            extra = "allow"


class PredictionOutput(_StructuredOutput):
    """JSON-ready output from analysis or PII extraction."""

    text: str
    entities: list[dict[str, Any]]
    model_name: Optional[str] = None


class DeidentifyOutput(_StructuredOutput):
    """JSON-ready de-identification output."""

    deidentified_text: str
    pii_entities: list[dict[str, Any]] = Field(default_factory=list)
    method: Optional[str] = None


class PIILanguageOutput(_StructuredOutput):
    """One supported PII language and its default model."""

    code: str
    name: str
    default_pii_model: str
    model_count: int


class PIILanguagesOutput(_StructuredOutput):
    """Supported PII language catalog."""

    count: int
    languages: list[PIILanguageOutput]


class UnloadModelOutput(_StructuredOutput):
    """Counts of model resources released from memory."""

    model_name: Optional[str] = None
    models: int
    tokenizers: int
    pipelines: int


if PYDANTIC_V2:

    class ModelListOutput(RootModel[list[str]]):
        """Available model identifiers."""

    class LoadedModelsOutput(RootModel[dict[str, dict[str, int]]]):
        """Loaded model resource counts keyed by resolved model identifier."""

else:  # pragma: no cover

    class ModelListOutput(BaseModel):
        """Available model identifiers."""

        __root__: list[str]

    class LoadedModelsOutput(BaseModel):
        """Loaded model resource counts keyed by resolved model identifier."""

        __root__: dict[str, dict[str, int]]


@lru_cache(maxsize=1)
def _get_model_loader() -> Any:
    """Create the shared local loader only when a callable needs it."""

    import openmed

    return openmed.ModelLoader()


def _json_ready(result: Any) -> Any:
    if isinstance(result, (dict, list, str, int, float, bool)) or result is None:
        return result
    model_dump = getattr(result, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    to_dict = getattr(result, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    raise TypeError("OpenMed tool results must be JSON-serializable")


def analyze_text(
    text: str,
    model_name: str = "disease_detection_superclinical",
    confidence_threshold: Optional[float] = 0.0,
    group_entities: bool = False,
    aggregation_strategy: Optional[
        Literal["simple", "first", "average", "max"]
    ] = "simple",
    sentence_detection: bool = True,
    sentence_language: str = "en",
    sentence_clean: bool = False,
    use_fast_tokenizer: bool = True,
) -> dict[str, Any]:
    """Invoke the public clinical text analysis API with JSON-ready output."""

    import openmed

    result = openmed.analyze_text(
        text,
        model_name=model_name,
        loader=_get_model_loader(),
        aggregation_strategy=aggregation_strategy,
        output_format="dict",
        confidence_threshold=confidence_threshold,
        group_entities=group_entities,
        sentence_detection=sentence_detection,
        sentence_language=sentence_language,
        sentence_clean=sentence_clean,
        use_fast_tokenizer=use_fast_tokenizer,
    )
    return _json_ready(result)


def extract_pii(
    text: str,
    model_name: str = DEFAULT_PII_MODEL,
    confidence_threshold: float = 0.5,
    use_smart_merging: bool = True,
    lang: PIILanguage = "en",
    normalize_accents: Optional[bool] = None,
) -> dict[str, Any]:
    """Invoke the public PII extraction API with JSON-ready output."""

    import openmed

    result = openmed.extract_pii(
        text,
        model_name=model_name,
        confidence_threshold=confidence_threshold,
        use_smart_merging=use_smart_merging,
        lang=lang,
        normalize_accents=normalize_accents,
        loader=_get_model_loader(),
    )
    return _json_ready(result)


def deidentify(
    text: str,
    method: Literal["mask", "remove", "replace", "hash", "shift_dates"] = "mask",
    model_name: str = DEFAULT_PII_MODEL,
    confidence_threshold: float = 0.7,
    keep_year: bool = False,
    shift_dates: Optional[bool] = None,
    date_shift_days: Optional[int] = None,
    keep_mapping: bool = False,
    policy: Optional[str] = None,
    use_smart_merging: bool = True,
    use_safety_sweep: bool = True,
    lang: PIILanguage = "en",
    normalize_accents: Optional[bool] = None,
) -> dict[str, Any]:
    """Invoke the public de-identification API with JSON-ready output."""

    import openmed

    result = openmed.deidentify(
        text,
        method=method,
        model_name=model_name,
        confidence_threshold=confidence_threshold,
        keep_year=keep_year,
        shift_dates=shift_dates,
        date_shift_days=date_shift_days,
        keep_mapping=keep_mapping,
        policy=policy,
        use_smart_merging=use_smart_merging,
        use_safety_sweep=use_safety_sweep,
        lang=lang,
        normalize_accents=normalize_accents,
        loader=_get_model_loader(),
    )
    return _json_ready(result)


def list_models(
    include_registry: bool = True,
    include_remote: bool = True,
) -> list[str]:
    """Invoke the public model-listing API."""

    import openmed

    return openmed.list_models(
        include_registry=include_registry,
        include_remote=include_remote,
        config=_get_model_loader().config,
    )


def list_pii_languages() -> dict[str, Any]:
    """Return supported PII languages and their default model identifiers."""

    import openmed
    from openmed.core.pii_i18n import (
        DEFAULT_PII_MODELS,
        INDIC_NER_LANGUAGES,
        LANGUAGE_NAMES,
        SUPPORTED_LANGUAGES,
    )

    languages = [
        {
            "code": code,
            "name": LANGUAGE_NAMES.get(code, code),
            "default_pii_model": DEFAULT_PII_MODELS[code],
            "model_count": len(openmed.get_pii_models_by_language(code)),
        }
        for code in sorted(SUPPORTED_LANGUAGES | INDIC_NER_LANGUAGES)
    ]
    return {"count": len(languages), "languages": languages}


def loaded_models() -> dict[str, dict[str, int]]:
    """Return resource counts for models loaded by registry callables."""

    return _get_model_loader().loaded_models()


def unload_model(
    model_name: Optional[str] = None,
    all: bool = False,
) -> dict[str, Any]:
    """Unload one model, or every model when ``all`` is true."""

    loader = _get_model_loader()
    if all:
        return loader.unload_all_models()
    if model_name is None:  # guarded by UnloadModelArgs for ToolDefinition.invoke
        raise ValueError("model_name is required unless all=true")
    return loader.unload_model(model_name)


def _model_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    if hasattr(model, "model_json_schema"):
        return model.model_json_schema()
    return model.schema()  # pragma: no cover


def _model_validate(model: type[BaseModel], value: Any) -> BaseModel:
    if hasattr(model, "model_validate"):
        return model.model_validate(value)
    return model.parse_obj(value)  # pragma: no cover


def _model_dump(model: BaseModel) -> Any:
    if hasattr(model, "model_dump"):
        return model.model_dump(mode="json")
    return model.dict()  # pragma: no cover


@dataclass(frozen=True)
class ToolDefinition:
    """Name, schemas, and invokable Python function for one public tool."""

    name: str
    description: str
    args_model: type[BaseModel]
    output_model: type[BaseModel]
    callable: Callable[..., Any]

    @property
    def input_model(self) -> type[BaseModel]:
        """Return the Pydantic input model."""

        return self.args_model

    @property
    def function(self) -> Callable[..., Any]:
        """Return the registered callable."""

        return self.callable

    def input_json_schema(self) -> dict[str, Any]:
        """Return this tool's Pydantic-generated input JSON Schema."""

        return _model_json_schema(self.args_model)

    def output_json_schema(self) -> dict[str, Any]:
        """Return this tool's Pydantic-generated output JSON Schema."""

        return _model_json_schema(self.output_model)

    def invoke(self, **arguments: Any) -> Any:
        """Validate arguments, call the tool, and validate its JSON-ready result."""

        parsed = _model_validate(self.args_model, arguments)
        result = self.callable(**_model_dump(parsed))
        return _model_dump(_model_validate(self.output_model, result))


_TOOL_DEFINITIONS = (
    ToolDefinition(
        name="analyze_text",
        description="Analyze clinical text with an OpenMed token-classification model.",
        args_model=AnalyzeTextArgs,
        output_model=PredictionOutput,
        callable=analyze_text,
    ),
    ToolDefinition(
        name="extract_pii",
        description="Extract personally identifiable information from clinical text.",
        args_model=ExtractPIIArgs,
        output_model=PredictionOutput,
        callable=extract_pii,
    ),
    ToolDefinition(
        name="deidentify",
        description="De-identify clinical text using an explicit privacy method.",
        args_model=DeidentifyArgs,
        output_model=DeidentifyOutput,
        callable=deidentify,
    ),
    ToolDefinition(
        name="list_models",
        description="List model identifiers available to the local OpenMed runtime.",
        args_model=ListModelsArgs,
        output_model=ModelListOutput,
        callable=list_models,
    ),
    ToolDefinition(
        name="list_pii_languages",
        description="List supported PII languages and their default model identifiers.",
        args_model=NoArgs,
        output_model=PIILanguagesOutput,
        callable=list_pii_languages,
    ),
    ToolDefinition(
        name="loaded_models",
        description="List model resources loaded by the local tool runtime.",
        args_model=NoArgs,
        output_model=LoadedModelsOutput,
        callable=loaded_models,
    ),
    ToolDefinition(
        name="unload_model",
        description="Release one or all model resources from the local tool runtime.",
        args_model=UnloadModelArgs,
        output_model=UnloadModelOutput,
        callable=unload_model,
    ),
)

TOOLS: Final[dict[str, ToolDefinition]] = {
    tool.name: tool for tool in _TOOL_DEFINITIONS
}


def list_tools() -> tuple[ToolDefinition, ...]:
    """Return all registered tools in stable declaration order."""

    return tuple(TOOLS.values())


def get_tool(name: str) -> ToolDefinition:
    """Return a registered tool by its stable name.

    Raises:
        KeyError: If *name* is not registered.
    """

    try:
        return TOOLS[name]
    except KeyError as exc:
        raise KeyError(f"unknown OpenMed tool {name!r}") from exc


__all__ = [
    "TOOLS",
    "AnalyzeTextArgs",
    "DeidentifyArgs",
    "DeidentifyOutput",
    "ExtractPIIArgs",
    "ListModelsArgs",
    "LoadedModelsOutput",
    "ModelListOutput",
    "NoArgs",
    "PIILanguage",
    "PIILanguagesOutput",
    "PredictionOutput",
    "ToolArgsModel",
    "ToolDefinition",
    "UnloadModelArgs",
    "UnloadModelOutput",
    "get_tool",
    "list_tools",
]
