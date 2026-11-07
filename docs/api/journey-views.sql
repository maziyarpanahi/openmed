-- OpenMed Journey read-only views
-- schema_version: 1.0.0
-- compatibility_policy: same_major
-- Base table is caller-owned; grant analytics roles SELECT on views only.

CREATE VIEW journey_artifacts AS
SELECT resource_id, resource_type, schema_version,
       compatibility_policy, state, version, revision,
       namespace, data_json, extensions_json
FROM journey_resource_records
WHERE resource_type = 'artifact';

CREATE VIEW journey_facts AS
SELECT resource_id, resource_type, schema_version,
       compatibility_policy, state, version, revision,
       namespace, data_json, extensions_json
FROM journey_resource_records
WHERE resource_type = 'fact';

CREATE VIEW journey_evidence AS
SELECT resource_id, resource_type, schema_version,
       compatibility_policy, state, version, revision,
       namespace, data_json, extensions_json
FROM journey_resource_records
WHERE resource_type = 'evidence';

CREATE VIEW journey_current_facts AS
SELECT resource_id, resource_type, schema_version,
       compatibility_policy, state, version, revision,
       namespace, data_json, extensions_json
FROM journey_resource_records
WHERE resource_type = 'current_fact';

CREATE VIEW journey_events AS
SELECT resource_id, resource_type, schema_version,
       compatibility_policy, state, version, revision,
       namespace, data_json, extensions_json
FROM journey_resource_records
WHERE resource_type = 'journey_event';

CREATE VIEW journey_mappings AS
SELECT resource_id, resource_type, schema_version,
       compatibility_policy, state, version, revision,
       namespace, data_json, extensions_json
FROM journey_resource_records
WHERE resource_type = 'mapping';

CREATE VIEW journey_cohort_runs AS
SELECT resource_id, resource_type, schema_version,
       compatibility_policy, state, version, revision,
       namespace, data_json, extensions_json
FROM journey_resource_records
WHERE resource_type = 'cohort_run';

CREATE VIEW journey_dataset_manifests AS
SELECT resource_id, resource_type, schema_version,
       compatibility_policy, state, version, revision,
       namespace, data_json, extensions_json
FROM journey_resource_records
WHERE resource_type = 'dataset_manifest';
