-- Replace every uppercase placeholder before running this statement.
CREATE OR REPLACE FUNCTION `PROJECT_ID.DATASET_ID.openmed_deidentify`(
  text STRING,
  policy STRING
)
RETURNS STRING
REMOTE WITH CONNECTION `PROJECT_ID.REGION.CONNECTION_ID`
OPTIONS (
  endpoint = 'ENDPOINT_URL',
  max_batching_rows = 256
);

-- Synthetic smoke query:
-- SELECT `PROJECT_ID.DATASET_ID.openmed_deidentify`(
--   'Patient Jane Roe called 555-0101.',
--   'hipaa_safe_harbor'
-- );
