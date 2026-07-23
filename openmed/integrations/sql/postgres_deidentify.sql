-- Install OpenMed de-identification functions for PostgreSQL PL/Python.
--
-- Contract:
--   openmed_deidentify(text) -> text
--   openmed_deidentify_batch(text[]) -> SETOF text
--
-- Prerequisite: OpenMed must be installed in the Python environment used by
-- PostgreSQL's plpython3u extension. Creating the extension normally requires
-- a PostgreSQL superuser. Re-running this migration replaces both functions.

BEGIN;

CREATE EXTENSION IF NOT EXISTS plpython3u;

CREATE OR REPLACE FUNCTION openmed_deidentify(input_text text)
RETURNS text
LANGUAGE plpython3u
STRICT
VOLATILE
PARALLEL UNSAFE
AS $openmed$
from openmed.integrations.postgres_plpython import deidentify_text

return deidentify_text(input_text, GD)
$openmed$;

CREATE OR REPLACE FUNCTION openmed_deidentify_batch(input_texts text[])
RETURNS SETOF text
LANGUAGE plpython3u
STRICT
VOLATILE
PARALLEL UNSAFE
AS $openmed$
from openmed.integrations.postgres_plpython import deidentify_batch

return deidentify_batch(input_texts, GD)
$openmed$;

COMMIT;
