"""Opt-in public clinical-trial synchronization and offline storage."""

from .contracts import (
    TRIAL_COMPATIBILITY_POLICY,
    TRIAL_SCHEMA_VERSION,
    TrialCacheCorruptionError,
    TrialContractError,
    TrialIntervention,
    TrialLocation,
    TrialSchemaDriftError,
    TrialSourceUnavailableError,
    TrialStudyRecord,
    TrialUnsupportedError,
    load_trial_study_schema,
)
from .source import (
    OFFICIAL_TRIAL_API_URL,
    ClinicalTrialSource,
    TrialSourcePage,
    TrialSourceTransport,
    UrlLibTrialSourceTransport,
    parse_trial_source_page,
)
from .store import LocalTrialStore, TrialQuery, TrialSyncReport

__all__ = [
    "OFFICIAL_TRIAL_API_URL",
    "TRIAL_COMPATIBILITY_POLICY",
    "TRIAL_SCHEMA_VERSION",
    "ClinicalTrialSource",
    "LocalTrialStore",
    "TrialCacheCorruptionError",
    "TrialContractError",
    "TrialIntervention",
    "TrialLocation",
    "TrialQuery",
    "TrialSchemaDriftError",
    "TrialSourcePage",
    "TrialSourceTransport",
    "TrialSourceUnavailableError",
    "TrialStudyRecord",
    "TrialSyncReport",
    "TrialUnsupportedError",
    "UrlLibTrialSourceTransport",
    "load_trial_study_schema",
    "parse_trial_source_page",
]
