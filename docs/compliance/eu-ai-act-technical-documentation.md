# AI Act Technical Documentation: Mapping OpenMed Evidence

> **Template — requires legal review. Not legal advice.** This page maps
> OpenMed artifacts to the technical documentation listed in Annex IV of the
> AI Act. It does not classify any system, demonstrate conformity, or replace
> the documentation a provider must draw up and keep.

Regulation (EU) 2024/1689 (the AI Act) requires the provider of a high-risk AI
system to draw up technical documentation before the system is placed on the
market or put into service, keep it up to date, and include at least the
elements set out in Annex IV (Article 11(1)). Providers keep that
documentation for 10 years after the system is placed on the market or put
into service (Article 18(1)(a)).

This page is for teams that build an AI system on top of OpenMed components.
For each Annex IV section it names the OpenMed artifact that can support it,
links its documentation, and states who owns the rest. Many sections are
entirely the team's own.

## Scope and roles

- These docs do not classify OpenMed components, or any system built with
  them, as medical devices or as high-risk AI systems. Classification follows
  Article 6 and Annexes I and III of the AI Act and, for medical devices,
  Regulation (EU) 2017/745 or Regulation (EU) 2017/746. Both of those are
  listed in Annex I, Section A (points 11 and 12).
- A deployer, distributor, importer or other third party becomes the provider
  of a high-risk AI system, with the provider's obligations, if it puts its
  name or trademark on the system, substantially modifies it, or changes its
  intended purpose so that it becomes high-risk (Article 25(1)). A team that
  builds a system on OpenMed should establish which role it holds.
- Article 25(4) requires a written agreement between the provider of a
  high-risk AI system and a third party that supplies tools, services,
  components or processes used in it. That requirement does not apply to third
  parties that make them publicly available under a free and open-source
  licence, except general-purpose AI models. Whether that exception covers a
  particular OpenMed component and model is for the provider to assess.
- Deployers of high-risk AI systems have separate duties under Article 26,
  for example keeping automatically generated logs under their control for at
  least six months (Article 26(6)) and using the provider's information to
  carry out a data protection impact assessment where applicable
  (Article 26(9)). See the
  [Data Protection Impact Assessment template](dpia-template.md).

## Application dates

The dates below are from Article 113 of the AI Act in the consolidated text on
EUR-Lex, which includes the amendments made by Regulation (EU) 2026/1744 (the
Digital Omnibus on AI). They were checked on 2026-11-16. Later amendments may
change them, so check the current consolidated text before relying on them.

| Provision | Applies from |
| --- | --- |
| The Regulation in general | 2 August 2026 ("It shall apply from 2 August 2026.") |
| Chapters I and II | 2 February 2025, except Article 5(1), first subparagraph, points (ba) and (bb), and Article 5(1a) and (1b), which apply from 2 December 2026 |
| Chapter III, Section 4, Chapter V, Chapter VII, Chapter XII and Article 78 | 2 August 2025, except Article 101 |
| Chapter III, Sections 1 to 3 (including Articles 11, 18 and 26), except Article 6(5), for systems that are high-risk under Article 6(2) and Annex III | 2 December 2027 |
| Chapter III, Sections 1 to 3, except Article 6(5), for systems that are high-risk under Article 6(1) and Annex I | 2 August 2028 |
| Articles 102 to 110 | 27 July 2026 |

## Annex IV mapping

Annex IV lists the minimum content of the technical documentation, as
applicable to the AI system concerned. The left column follows its numbering. "Provider" means
the provider of the AI system built on OpenMed, which may be the deploying
team (see Article 25(1) above).

| Annex IV section | What it asks for | Supporting OpenMed artifact | Who owns the remainder |
| --- | --- | --- | --- |
| 1(a) | Intended purpose, provider name, and the system version and its relation to previous versions | [Model manifest](../model-manifest.md) rows record each model's `repo_id`, `task`, `released` date and `reproducibility_hash`. The [EU AI Act and GDPR model-card fields](model-card-eu-ai-act-fields.md) give an intended-purpose block to complete. | Provider: the system's intended purpose, provider identity and versioning. |
| 1(b) | Interaction with hardware, software or other AI systems | [REST service](../rest-service.md) operations and the [OpenMedKit support matrix](../openmedkit-support-matrix.md) describe OpenMed's own interfaces and supported runtimes. | Provider: how the whole system integrates with other software and AI systems. |
| 1(c) | Versions of relevant software or firmware and version-update requirements | [Release streams, SemVer and channels](../release/semver-and-channels.md), the [public API deprecation policy](api-deprecation-policy.md) and [reproducible release provenance](reproducible-release.md). | Provider: the system's own version and update policy. |
| 1(d) | Forms in which the system is placed on the market or put into service | The [export format support matrix](../export-matrix.md) lists the artifact formats OpenMed supports. | Provider: the system's distribution forms. |
| 1(e) | Hardware on which the system is intended to run | [Device tiers and SLOs](../tiers.md) and the [OpenMedKit support matrix](../openmedkit-support-matrix.md). | Provider: the system's target hardware. |
| 1(f) | Photographs or illustrations of external features, marking and internal layout of products that contain the system | None. | Provider. |
| 1(g) | Basic description of the user interface provided to the deployer | None. | Provider. |
| 1(h) | Instructions for use for the deployer | None. OpenMed's documentation describes components, not a system's instructions for use. | Provider. |
| 2(a) | Development methods and steps, including pre-trained systems or third-party tools and how they were used or modified | The [model manifest](../model-manifest.md) `base_model`, `architecture` and `license` fields, and [model-provenance diff reports](../evaluation/model-provenance-diff.md) for evaluation runs. | Provider: the system's development process, including any fine-tuning of OpenMed models. |
| 2(b) | Design specifications: general logic, algorithms, key design choices, rationale and trade-offs | Partial: [privacy policy as data](../security/policy-as-data.md) records de-identification policy intent with a digest. | Provider: the system's design specification. |
| 2(c) | System architecture and computational resources used to develop, train, test and validate | Partial: [device tiers and SLOs](../tiers.md) record OpenMed's runtime resource classes. | Provider: architecture and compute records. |
| 2(d) | Data requirements: datasheets on training methods and data sets, provenance, scope, selection, labelling and cleaning | [OpenMed synthetic clinical-PHI eval corpus](../datasets/openmed-synth-phi.md) and the [synthetic privacy regression corpus](../evaluation/privacy-corpus.md) describe OpenMed's own synthetic evaluation data. | Provider: datasheets for every data set used to train, validate or test the system. |
| 2(e) | Assessment of human oversight measures under Article 14, and technical measures that help deployers interpret outputs | The human-oversight fields in the [model-card field specification](model-card-eu-ai-act-fields.md), and the [v3.1 clinician-review protocol](v3.1-clinician-review.md) for agent workflow previews. | Provider: the oversight assessment for the system. |
| 2(f) | Pre-determined changes to the system and its performance, and the technical solutions that keep it compliant | None. | Provider. |
| 2(g) | Validation and testing procedures, data, metrics for accuracy, robustness and discriminatory impacts, and dated and signed test logs and reports | `BenchmarkReport` metrics in the [eval harness metric bundle](../eval-harness.md#metric-bundle), the [clinical validation protocol](../clinical-validation-protocol.md) for user-supplied labelled data, the [privacy release-gate aggregator](privacy-release-gate.md), the [evidence freshness gate](evidence-freshness.md), and the [tamper-evident audit log](../security/tamper-evident-audit-log.md). | Provider: validation of the whole system in its intended setting, and the signed test reports. |
| 2(h) | Cybersecurity measures | The [redactor threat model](../security/threat-model.md), [supply chain controls](../security/supply-chain.md), [SLSA build provenance](../supply-chain/provenance.md), the [no-telemetry guarantee](../security/no-telemetry.md) and the [no-raw-PHI logging policy](../security/no-raw-phi-logging.md). | Provider: security of the deployed system. |
| 3 | Monitoring, functioning and control: capabilities and limitations, accuracy for specific persons or groups, foreseeable unintended outcomes and risk sources, human oversight measures under Article 14, and input data specifications | The known-limitations and accuracy and leakage blocks of the [model-card field specification](model-card-eu-ai-act-fields.md), [per-language PII de-identification](../languages.md) and [model tokenizer script coverage](../model-tokenizer-script-coverage.md). | Provider: these descriptions for the system and its intended population. |
| 4 | Why the chosen performance metrics are appropriate for the system | Metric definitions in the [eval harness metric bundle](../eval-harness.md#metric-bundle) and the [clinical validation protocol](../clinical-validation-protocol.md). | Provider: the justification for the system's purpose. |
| 5 | The risk management system under Article 9 | Component-level input only: the [redactor threat model](../security/threat-model.md). | Provider: the risk management system. |
| 6 | Relevant changes made by the provider through the lifecycle | Change records for OpenMed itself: [release streams, SemVer and channels](../release/semver-and-channels.md), [model-provenance diff reports](../evaluation/model-provenance-diff.md) and the [public API deprecation policy](api-deprecation-policy.md). | Provider: the system's change history. |
| 7 | Harmonised standards applied, or a description of the solutions adopted to meet Chapter III, Section 2, with other standards and specifications applied | None. The [ISO 27701/27001 control-evidence pack](iso-27701-control-evidence.md) is a crosswalk to ISO controls, not a harmonised standard under the AI Act. | Provider. |
| 8 | A copy of the EU declaration of conformity under Article 47 | None. | Provider. |
| 9 | The system for evaluating performance in the post-market phase under Article 72, including the post-market monitoring plan under Article 72(3) | None as a system. Building blocks: the [evidence freshness gate](evidence-freshness.md) and the [tamper-evident audit log](../security/tamper-evident-audit-log.md). | Provider: the post-market monitoring system and plan. |

Article 11(1) allows SMEs, including start-ups, and small mid-cap enterprises
to provide the Annex IV elements in a simplified manner, using a form the
Commission establishes. Article 11(2) requires a single set of technical
documentation when a high-risk AI system is related to a product covered by
the legislation in Annex I, Section A, such as the medical device regulations.
OpenMed provides neither the simplified form nor that combined documentation.

## Handling OpenMed evidence

- Copy measured values from immutable evidence without inferring, averaging,
  or turning a missing value into zero, as the rendering rules of the
  [model-card field specification](model-card-eu-ai-act-fields.md) require.
- Record the OpenMed version, model `repo_id` and `reproducibility_hash`, and
  the policy digest for each piece of evidence.
- Do not put raw clinical text, identifiers or reversible mappings into
  technical documentation. OpenMed evidence artifacts use codes, counts,
  offsets and digests for this reason.
- OpenMed evidence describes OpenMed components. It does not show that the
  system built on them meets the requirements of Chapter III, Section 2.

## Related pages

- [Compliance posture](../compliance.md) lists OpenMed's evidence and the
  framework split.
- [EU AI Act and GDPR model-card field specification](model-card-eu-ai-act-fields.md)
  defines the model-card block that several rows above refer to.
- [Data Protection Impact Assessment template](dpia-template.md) for the GDPR
  assessment.

OpenMed does not generate technical-documentation packs today. Pack generation
and regulatory-tier fields are tracked for a later release.
