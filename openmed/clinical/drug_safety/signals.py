"""Descriptive report-level disproportionality calculations."""

from __future__ import annotations

from collections.abc import Sequence

from openmed.clinical.journey_contracts import derived_opaque_id

from .contracts import (
    DRUG_SAFETY_ADVISORY,
    DRUG_SAFETY_METHOD_ID,
    DRUG_SAFETY_METHOD_VERSION,
    DescriptiveSignal,
    DrugSafetyContractError,
    DrugSafetyDataset,
    NormalizedSafetyTerm,
    SafetyCase,
    SignalContingencyTable,
    SignalFilter,
    SignalPolicy,
    SignalState,
    _calculate_table_outcome,
)

_BASE_CAVEATS = (
    "confounding_not_adjusted",
    "duplicate_source_reports_may_remain",
    "reporting_bias_possible",
    "signal_is_not_causal",
    "signal_is_not_incidence",
)


def compute_descriptive_signal(
    dataset: DrugSafetyDataset,
    *,
    drug: NormalizedSafetyTerm | str,
    event: NormalizedSafetyTerm | str,
    signal_filter: SignalFilter | None = None,
    policy: SignalPolicy | None = None,
) -> DescriptiveSignal:
    """Compute PRR and ROR or return an explicit suppressed/insufficient state."""

    if not isinstance(dataset, DrugSafetyDataset):
        raise TypeError("dataset must be DrugSafetyDataset")
    drug_term = (
        drug
        if isinstance(drug, NormalizedSafetyTerm)
        else NormalizedSafetyTerm(kind="drug", code=drug)
    )
    event_term = (
        event
        if isinstance(event, NormalizedSafetyTerm)
        else NormalizedSafetyTerm(kind="event", code=event)
    )
    if drug_term.kind != "drug" or event_term.kind != "event":
        raise DrugSafetyContractError("signal requires one drug and one event")
    active_filter = signal_filter or SignalFilter()
    active_policy = policy or SignalPolicy()
    cases = tuple(item for item in dataset.cases if _filter_case(item, active_filter))
    table = _contingency(cases, drug_term=drug_term, event_term=event_term)
    state, reasons, prr, ror = _calculate_table_outcome(table, active_policy)
    signal_id = derived_opaque_id(
        "safetysignal",
        dataset.dataset_digest,
        drug_term.to_dict(),
        event_term.to_dict(),
        active_filter.digest,
        active_policy.digest,
        DRUG_SAFETY_METHOD_ID,
        DRUG_SAFETY_METHOD_VERSION,
    )
    caveats = (*_BASE_CAVEATS,)
    return DescriptiveSignal(
        signal_id=signal_id,
        drug=drug_term,
        event=event_term,
        state=state,
        reason_codes=reasons,
        table=table,
        proportional_reporting_ratio=prr,
        reporting_odds_ratio=ror,
        dataset_id=dataset.manifest.dataset_id,
        dataset_version=dataset.manifest.version,
        license_id=dataset.manifest.license_id,
        source_digest=dataset.manifest.source_digest,
        dataset_digest=dataset.dataset_digest,
        adapter_version=dataset.manifest.adapter_version,
        filter=active_filter,
        filter_digest=active_filter.digest,
        policy=active_policy,
        policy_digest=active_policy.digest,
        duplicate_row_count=dataset.duplicate_row_count,
        caveats=caveats,
    )


def _filter_case(case: SafetyCase, signal_filter: SignalFilter) -> bool:
    if signal_filter.seriousness and case.seriousness not in signal_filter.seriousness:
        return False
    if (
        signal_filter.minimum_exposure_start_day is None
        and signal_filter.maximum_exposure_end_day is None
    ):
        return True
    windows = tuple(
        exposure.window for exposure in case.exposures if exposure.window is not None
    )
    if not windows:
        return False
    return any(
        (
            signal_filter.minimum_exposure_start_day is None
            or window.start_day >= signal_filter.minimum_exposure_start_day
        )
        and (
            signal_filter.maximum_exposure_end_day is None
            or window.end_day <= signal_filter.maximum_exposure_end_day
        )
        for window in windows
    )


def _contingency(
    cases: Sequence[SafetyCase],
    *,
    drug_term: NormalizedSafetyTerm,
    event_term: NormalizedSafetyTerm,
) -> SignalContingencyTable:
    a = b = c = d = 0
    for case in cases:
        has_drug = any(item.drug == drug_term for item in case.exposures)
        has_event = event_term in case.events
        if has_drug and has_event:
            a += 1
        elif has_drug:
            b += 1
        elif has_event:
            c += 1
        else:
            d += 1
    return SignalContingencyTable(
        drug_event=a,
        drug_other_event=b,
        other_drug_event=c,
        other_drug_other_event=d,
    )


__all__ = ["DRUG_SAFETY_ADVISORY", "compute_descriptive_signal"]
